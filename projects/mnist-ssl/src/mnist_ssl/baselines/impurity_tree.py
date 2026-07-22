"""Routing, measurement, and fingerprint helpers for neural impurity trees."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from mnist_ssl.baselines.impurity_convnet import (
    N_CLASSES,
    ExperimentConfig,
    _class_impurity,
    leaf_memberships,
    split_statistics,
)


ROOT_LEAVES = ("left", "right")
DEPTH_TWO_LEAVES = (
    "root_left/child_left",
    "root_left/child_right",
    "root_right/child_left",
    "root_right/child_right",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    """Hash tensor names, metadata, and bytes in a model state dict."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def freeze_splitter(model: nn.Module) -> nn.Module:
    """Freeze one completed splitter before any descendants are trained."""

    model.requires_grad_(False)
    model.eval()
    return model


def load_mnist(dataset_dir: Path, train: bool) -> Dataset:
    return datasets.MNIST(
        root=str(dataset_dir),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )


def make_loader(
    dataset: Dataset,
    config: ExperimentConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )


@torch.no_grad()
def collect_dataset_memberships(
    model: nn.Module,
    dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect memberships and labels in stable dataset order."""

    model.eval()
    memberships = []
    labels = []
    for images, batch_labels in make_loader(dataset, config, shuffle=False):
        memberships.append(leaf_memberships(model(images.to(device))).cpu())
        labels.append(batch_labels)
    return torch.cat(memberships), torch.cat(labels)


def hard_routes(memberships: torch.Tensor) -> torch.Tensor:
    if memberships.ndim != 2 or memberships.shape[1] != 2:
        raise ValueError("binary memberships must have shape (examples, 2)")
    return (memberships[:, 1] >= 0.5).long()


def assemble_tree_memberships(
    root_routes: torch.Tensor,
    left_child_memberships: torch.Tensor,
    right_child_memberships: torch.Tensor,
) -> torch.Tensor:
    """Map hard root routes plus two child gates to four final leaves."""

    if root_routes.ndim != 1:
        raise ValueError("root_routes must have one entry per example")
    expected = (len(root_routes), 2)
    if tuple(left_child_memberships.shape) != expected:
        raise ValueError(f"left child memberships must have shape {expected}")
    if tuple(right_child_memberships.shape) != expected:
        raise ValueError(f"right child memberships must have shape {expected}")
    if not torch.all((root_routes == 0) | (root_routes == 1)):
        raise ValueError("root routes must contain only 0 or 1")

    output = left_child_memberships.new_zeros((len(root_routes), 4))
    routed_left = root_routes == 0
    output[routed_left, :2] = left_child_memberships[routed_left]
    output[~routed_left, 2:] = right_child_memberships[~routed_left]
    return output


def _multi_leaf_partition_statistics(
    memberships: torch.Tensor,
    labels: torch.Tensor,
    criterion: str,
) -> dict:
    if memberships.ndim != 2 or memberships.shape[1] < 2:
        raise ValueError("memberships must contain at least two leaves")
    if labels.ndim != 1 or len(labels) != len(memberships):
        raise ValueError("labels must contain one entry per example")

    targets = F.one_hot(labels, num_classes=N_CLASSES).to(memberships.dtype)
    counts = memberships.transpose(0, 1) @ targets
    masses = counts.sum(dim=1)
    total_mass = masses.sum().clamp_min(1e-8)
    distributions = counts / masses.unsqueeze(1).clamp_min(1e-8)
    leaf_impurities = _class_impurity(distributions, criterion)
    leaf_impurities = torch.where(
        masses > 0, leaf_impurities, torch.zeros_like(leaf_impurities)
    )
    child = (masses / total_mass * leaf_impurities).sum()

    parent_counts = torch.bincount(labels, minlength=N_CLASSES).to(torch.float32)
    parent_distribution = parent_counts / parent_counts.sum().clamp_min(1.0)
    parent = _class_impurity(parent_distribution, criterion)
    reduction = parent - child

    per_digit_leaf_fraction = []
    for digit in range(N_CLASSES):
        selected = labels == digit
        per_digit_leaf_fraction.append(
            memberships[selected].mean(dim=0).tolist() if selected.any() else None
        )
    return {
        "parent_impurity": parent.item(),
        "child_weighted_impurity": child.item(),
        "impurity_reduction": reduction.item(),
        "relative_impurity_reduction": (
            reduction / parent.clamp_min(1e-8)
        ).item(),
        "leaf_mass_fraction": (masses / total_mass).tolist(),
        "leaf_impurity": leaf_impurities.tolist(),
        "leaf_class_distribution": distributions.tolist(),
        "per_digit_leaf_fraction": per_digit_leaf_fraction,
    }


def tree_statistics(
    memberships: torch.Tensor,
    labels: torch.Tensor,
    criterion: str,
    leaf_names: tuple[str, ...],
) -> dict:
    """Report soft-gate and hard-route statistics for a multi-leaf tree."""

    if len(leaf_names) != memberships.shape[1]:
        raise ValueError("leaf_names must contain one name per membership column")

    soft = _multi_leaf_partition_statistics(memberships, labels, criterion)
    hard_leaf = memberships.argmax(dim=1)
    hard = F.one_hot(hard_leaf, num_classes=memberships.shape[1]).to(
        memberships.dtype
    )
    return {
        "leaf_names": leaf_names,
        "soft": soft,
        "hard": _multi_leaf_partition_statistics(hard, labels, criterion),
    }


def hierarchy_statistics(
    root_memberships: torch.Tensor,
    tree_memberships: torch.Tensor,
    labels: torch.Tensor,
    criterion: str,
) -> dict:
    """Compare one two-leaf root with its resulting four-leaf tree."""

    root_hard = F.one_hot(hard_routes(root_memberships), num_classes=2).to(
        tree_memberships.dtype
    )
    root = split_statistics(root_hard, labels, criterion)["hard"]
    depth_two = tree_statistics(
        tree_memberships,
        labels,
        criterion,
        DEPTH_TWO_LEAVES,
    )
    for mode in ("soft", "hard"):
        final = depth_two[mode]
        incremental = (
            root["child_weighted_impurity"]
            - final["child_weighted_impurity"]
        )
        final["incremental_reduction_from_root"] = incremental
        final["relative_incremental_reduction_from_root"] = (
            incremental / max(root["child_weighted_impurity"], 1e-8)
        )
    return {"root_hard": root, "depth_two": depth_two}
