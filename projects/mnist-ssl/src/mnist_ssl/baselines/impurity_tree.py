"""Grow one frozen neural decision stump to four leaves without end-to-end training.

For each requested criterion, this module loads the matching original
three-convolution root checkpoint and freezes it.  MNIST train images are
hard-routed through that root.  A fresh three-convolution binary splitter is
then trained independently on each of the two resulting subsets, using only
the same leaf-impurity objective as its root.  The root is never placed in an
optimizer and its state fingerprint is checked after child training.

All child models for all criteria are trained before the canonical test split
is loaded.  Test images follow the frozen root and then the appropriate child;
no threshold, epoch, architecture, or parameter is selected on test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from mnist_ssl.baselines.impurity_convnet import (
    CRITERIA,
    N_CLASSES,
    ExperimentConfig,
    OriginalConvSplitter,
    _class_impurity,
    leaf_memberships,
    pick_device,
    set_seed,
    split_statistics,
    train_splitter,
)
from mnist_ssl.paths import DATASET_DIR, OUT_DIR


ROOT_LEAVES = ("left", "right")
FINAL_LEAVES = (
    "root_left/child_left",
    "root_left/child_right",
    "root_right/child_left",
    "root_right/child_right",
)
ORIGINAL_PARAMETER_COUNT = 23_361


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


def load_frozen_root(path: Path, criterion: str, device: torch.device) -> nn.Module:
    """Load one original stump and make mutation through training impossible."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("criterion") != criterion:
        raise ValueError(
            f"root criterion mismatch: expected {criterion}, "
            f"found {payload.get('criterion')}"
        )
    if payload.get("model_kind") != "small_conv_binary_splitter":
        raise ValueError(
            "depth-two growth requires the original three-convolution root"
        )
    root = OriginalConvSplitter()
    root.load_state_dict(payload["model_state_dict"], strict=True)
    root.requires_grad_(False)
    root.eval()
    return root.to(device)


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
    routed_right = ~routed_left
    output[routed_left, :2] = left_child_memberships[routed_left]
    output[routed_right, 2:] = right_child_memberships[routed_right]
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
    leaf_names: tuple[str, ...] = FINAL_LEAVES,
) -> dict:
    """Report soft-child and hard-child statistics for a multi-leaf tree."""

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
    """Compare the frozen two-leaf root with the resulting four-leaf tree."""

    root_hard = F.one_hot(hard_routes(root_memberships), num_classes=2).to(
        tree_memberships.dtype
    )
    root = split_statistics(root_hard, labels, criterion)["hard"]
    depth_two = tree_statistics(tree_memberships, labels, criterion)
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


def _train_criterion(
    criterion: str,
    root_path: Path,
    train_dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> dict:
    """Train both children for one criterion without reading test data."""

    root = load_frozen_root(root_path, criterion, device)
    root_fingerprint_before = model_state_sha256(root)
    root_train_memberships, train_labels = collect_dataset_memberships(
        root, train_dataset, config, device
    )
    train_routes = hard_routes(root_train_memberships)

    children = {}
    child_records = []
    child_train_memberships = {}
    started = time.monotonic()
    for parent_index, parent_name in enumerate(ROOT_LEAVES):
        indices = torch.nonzero(train_routes == parent_index, as_tuple=False).flatten()
        if len(indices) < 2:
            raise ValueError(f"root {parent_name} leaf has too few training examples")
        subset = Subset(train_dataset, indices.tolist())
        set_seed(config.seed)
        child = OriginalConvSplitter().to(device)
        print(
            f"{criterion} root_{parent_name}: "
            f"training_examples={len(subset)}",
            flush=True,
        )
        history = train_splitter(
            child,
            make_loader(subset, config, shuffle=True),
            criterion,
            config,
            device,
        )
        memberships, labels = collect_dataset_memberships(
            child, train_dataset, config, device
        )
        if not torch.equal(labels, train_labels):
            raise RuntimeError("train label order changed during child evaluation")
        local_memberships = memberships[train_routes == parent_index]
        local_labels = train_labels[train_routes == parent_index]
        children[parent_name] = child
        child_train_memberships[parent_name] = memberships
        child_records.append(
            {
                "parent_leaf": parent_name,
                "parameter_count": sum(p.numel() for p in child.parameters()),
                "training_examples": len(local_labels),
                "history": history,
                "metrics": {
                    "train": split_statistics(
                        local_memberships, local_labels, criterion
                    )
                },
            }
        )

    root_fingerprint_after = model_state_sha256(root)
    if root_fingerprint_after != root_fingerprint_before:
        raise RuntimeError("frozen root changed while training child splitters")

    train_tree_memberships = assemble_tree_memberships(
        train_routes,
        child_train_memberships["left"],
        child_train_memberships["right"],
    )
    return {
        "criterion": criterion,
        "root_path": root_path,
        "root_checkpoint_sha256": file_sha256(root_path),
        "root_fingerprint_before": root_fingerprint_before,
        "root_fingerprint_after": root_fingerprint_after,
        "root": root,
        "children": children,
        "child_records": child_records,
        "train_metrics": hierarchy_statistics(
            root_train_memberships,
            train_tree_memberships,
            train_labels,
            criterion,
        ),
        "elapsed_training_seconds": time.monotonic() - started,
    }


def _finalize_criterion(
    state: dict,
    test_dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Evaluate test once and persist the already-trained child networks."""

    criterion = state["criterion"]
    root_test_memberships, test_labels = collect_dataset_memberships(
        state["root"], test_dataset, config, device
    )
    test_routes = hard_routes(root_test_memberships)
    child_test_memberships = {}

    for child_record in state["child_records"]:
        parent_name = child_record["parent_leaf"]
        parent_index = ROOT_LEAVES.index(parent_name)
        child = state["children"][parent_name]
        memberships, labels = collect_dataset_memberships(
            child, test_dataset, config, device
        )
        if not torch.equal(labels, test_labels):
            raise RuntimeError("test label order changed during child evaluation")
        local_memberships = memberships[test_routes == parent_index]
        local_labels = test_labels[test_routes == parent_index]
        child_record["test_examples"] = len(local_labels)
        child_record["metrics"]["canonical_test"] = split_statistics(
            local_memberships, local_labels, criterion
        )
        child_test_memberships[parent_name] = memberships

        checkpoint_path = output_dir / f"{criterion}_root_{parent_name}.pt"
        torch.save(
            {
                "criterion": criterion,
                "parent_leaf": parent_name,
                "model_kind": "original_three_conv_child_splitter",
                "root_checkpoint": state["root_path"].name,
                "root_checkpoint_sha256": state["root_checkpoint_sha256"],
                "root_state_sha256": state["root_fingerprint_before"],
                "config": asdict(config),
                "model_state_dict": child.state_dict(),
                "history": child_record.pop("history"),
                "metrics": child_record["metrics"],
            },
            checkpoint_path,
        )
        child_record["checkpoint"] = checkpoint_path.name
        child_record["checkpoint_sha256"] = file_sha256(checkpoint_path)

    test_tree_memberships = assemble_tree_memberships(
        test_routes,
        child_test_memberships["left"],
        child_test_memberships["right"],
    )
    test_metrics = hierarchy_statistics(
        root_test_memberships,
        test_tree_memberships,
        test_labels,
        criterion,
    )
    root_fingerprint_final = model_state_sha256(state["root"])
    if root_fingerprint_final != state["root_fingerprint_before"]:
        raise RuntimeError("frozen root changed during final evaluation")

    hard = test_metrics["depth_two"]["hard"]
    print(
        f"{criterion} depth-two hard test reduction: "
        f"total={hard['relative_impurity_reduction']:.2%} "
        f"incremental={hard['relative_incremental_reduction_from_root']:.2%}",
        flush=True,
    )
    return {
        "criterion": criterion,
        "root_checkpoint": state["root_path"].name,
        "root_checkpoint_sha256": state["root_checkpoint_sha256"],
        "root_parameter_count": ORIGINAL_PARAMETER_COUNT,
        "root_state_sha256_before": state["root_fingerprint_before"],
        "root_state_sha256_after": root_fingerprint_final,
        "root_unchanged": True,
        "child_parameter_count_each": ORIGINAL_PARAMETER_COUNT,
        "elapsed_training_seconds": state["elapsed_training_seconds"],
        "children": state["child_records"],
        "metrics": {
            "train": state["train_metrics"],
            "canonical_test": test_metrics,
        },
    }


def run_experiment(
    criteria: Iterable[str],
    config: ExperimentConfig,
    root_checkpoint_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> dict:
    criteria = tuple(criteria)
    unknown = sorted(set(criteria) - set(CRITERIA))
    if unknown:
        raise ValueError(f"unknown criteria: {', '.join(unknown)}")
    if not criteria:
        raise ValueError("at least one criterion is required")

    root_paths = {
        criterion: root_checkpoint_dir / f"{criterion}.pt"
        for criterion in criteria
    }
    missing = [str(path) for path in root_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing root checkpoints: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(config.device)
    print(f"device={device} output={output_dir}", flush=True)
    train_dataset = load_mnist(dataset_dir, train=True)
    trained = [
        _train_criterion(
            criterion,
            root_paths[criterion],
            train_dataset,
            config,
            device,
        )
        for criterion in criteria
    ]

    print("all child training complete; loading canonical test split", flush=True)
    test_dataset = load_mnist(dataset_dir, train=False)
    results = [
        _finalize_criterion(state, test_dataset, config, device, output_dir)
        for state in trained
    ]
    summary = {
        "experiment": "mnist_frozen_neural_impurity_tree_depth_two",
        "architecture": (
            "one frozen three-convolution root and two independently trained "
            "three-convolution child splitters per criterion; four final leaves"
        ),
        "training_policy": (
            "hard-route MNIST train through the frozen root; train each child "
            "only on its routed subset; no end-to-end gradients"
        ),
        "test_policy": (
            "train every child before loading canonical MNIST test; test follows "
            "the frozen root and corresponding child once; no test selection"
        ),
        "final_leaf_names": FINAL_LEAVES,
        "config": asdict(config),
        "device": str(device),
        "root_checkpoint_dir": str(root_checkpoint_dir),
        "dataset_dir": str(dataset_dir),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary -> {summary_path}", flush=True)
    return summary


def parse_criteria(value: str) -> tuple[str, ...]:
    criteria = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(criteria) - set(CRITERIA))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown criteria {', '.join(unknown)}; choose from {CRITERIA}"
        )
    return criteria


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", type=parse_criteria, default=CRITERIA)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--root-checkpoint-dir",
        type=Path,
        default=OUT_DIR / "neural_impurity_stump_2026-07-21",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR / "neural_impurity_tree_depth_two",
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        balance_weight=args.balance_weight,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    run_experiment(
        args.criteria,
        config,
        args.root_checkpoint_dir,
        args.dataset_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
