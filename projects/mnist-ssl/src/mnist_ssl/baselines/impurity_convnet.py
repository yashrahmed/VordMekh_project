"""Train one convolutional binary splitter from class impurity alone.

This experiment is a neural decision *stump*, not a digit classifier.  A small
CNN emits one scalar per image.  Its sigmoid is the differentiable probability
of routing the image to the right leaf; the complement routes it left.  There
is no ten-class head and neither leaf predicts a digit.

Training minimizes the sample-weighted normalized Shannon entropy of the two
leaf class distributions.  A small label-free balance penalty discourages the
trivial all-left/all-right solution.  Evaluation reports how much the split
reduces entropy, its balance, and the label distribution it induces.  Test
labels are used only for the final measurement after the fixed training
horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_ssl.paths import DATASET_DIR, OUT_DIR


N_CLASSES = 10


@dataclass(frozen=True)
class ExperimentConfig:
    """Fixed settings for one entropy splitter run."""

    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    balance_weight: float = 0.05
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResidualConvSplitter(nn.Module):
    """A two-convolution residual CNN with one learned routing output."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.split = nn.Linear(16, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unconstrained right-routing logit per image."""

        features = self.pool(F.relu(self.conv1(images)))
        features = F.relu(self.conv2(features) + features)
        return self.split(self.global_pool(features).flatten(1)).squeeze(1)


# Retain the public name used by the original stump entry point while making
# every newly trained small splitter use the residual architecture.
SmallConvSplitter = ResidualConvSplitter


def leaf_memberships(split_logits: torch.Tensor) -> torch.Tensor:
    """Convert one split logit into ``[P(left), P(right)]`` per image."""

    if split_logits.ndim != 1:
        raise ValueError(
            f"expected one split logit per image, got {tuple(split_logits.shape)}"
        )
    right = split_logits.sigmoid()
    return torch.stack((1.0 - right, right), dim=1)


def _class_entropy(distribution: torch.Tensor) -> torch.Tensor:
    """Normalized Shannon entropy, or self-cross-entropy, over digit labels."""

    safe = distribution.clamp_min(1e-8)
    return -(distribution * safe.log()).sum(dim=-1) / math.log(N_CLASSES)


def entropy_partition_tensors(
    memberships: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted entropy, masses, distributions, and leaf entropies."""

    if memberships.ndim != 2 or memberships.shape[1] != 2:
        raise ValueError("memberships must have shape (examples, 2)")
    if labels.ndim != 1 or len(labels) != len(memberships):
        raise ValueError("labels must contain one entry per example")

    targets = F.one_hot(labels, num_classes=N_CLASSES).to(memberships.dtype)
    counts = memberships.transpose(0, 1) @ targets
    masses = counts.sum(dim=1)
    distributions = counts / masses.unsqueeze(1).clamp_min(1e-8)
    leaf_impurities = _class_entropy(distributions)
    weighted = (
        masses / masses.sum().clamp_min(1e-8) * leaf_impurities
    ).sum()
    return weighted, masses, distributions, leaf_impurities


def weighted_leaf_entropy(
    memberships: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Differentiable normalized entropy of this two-leaf partition."""

    return entropy_partition_tensors(memberships, labels)[0]


def _parent_entropy(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=N_CLASSES).to(torch.float32)
    distribution = counts / counts.sum().clamp_min(1.0)
    return _class_entropy(distribution)


def _partition_statistics(
    memberships: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    child, masses, distributions, leaf_impurities = entropy_partition_tensors(
        memberships, labels
    )
    parent = _parent_entropy(labels)
    reduction = parent - child
    per_class_right = []
    for digit in range(N_CLASSES):
        selected = labels == digit
        per_class_right.append(
            memberships[selected, 1].mean().item() if selected.any() else None
        )
    return {
        "parent_impurity": parent.item(),
        "child_weighted_impurity": child.item(),
        "impurity_reduction": reduction.item(),
        "relative_impurity_reduction": (
            reduction / parent.clamp_min(1e-8)
        ).item(),
        "leaf_mass_fraction": (masses / masses.sum()).tolist(),
        "leaf_impurity": leaf_impurities.tolist(),
        "leaf_class_distribution": distributions.tolist(),
        "per_digit_right_fraction": per_class_right,
    }


def split_statistics(
    memberships: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Summarize both differentiable and thresholded versions of one split."""

    soft = _partition_statistics(memberships, labels)
    hard_right = (memberships[:, 1] >= 0.5).long()
    hard = F.one_hot(hard_right, num_classes=2).to(memberships.dtype)
    return {
        "soft": soft,
        "hard": _partition_statistics(hard, labels),
    }


def mnist_loaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    train = datasets.MNIST(
        root=str(DATASET_DIR), train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root=str(DATASET_DIR), train=False, download=True, transform=transform
    )
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "persistent_workers": config.num_workers > 0,
    }
    return (
        DataLoader(train, shuffle=True, generator=generator, **common),
        DataLoader(test, shuffle=False, **common),
    )


def train_splitter(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "impurity": 0.0, "balance_penalty": 0.0}
        seen = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            memberships = leaf_memberships(model(images))
            impurity = weighted_leaf_entropy(memberships, labels)
            balance = (memberships[:, 1].mean() - 0.5).square()
            loss = impurity + config.balance_weight * balance

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = len(labels)
            totals["loss"] += loss.item() * batch_size
            totals["impurity"] += impurity.item() * batch_size
            totals["balance_penalty"] += balance.item() * batch_size
            seen += batch_size
        scheduler.step()
        record = {
            "epoch": epoch,
            **{name: total / seen for name, total in totals.items()},
        }
        history.append(record)
        if epoch == 1 or epoch == config.epochs or epoch % 5 == 0:
            print(
                f"entropy epoch {epoch:02d}/{config.epochs}  "
                f"loss={record['loss']:.6f}  "
                f"impurity={record['impurity']:.6f}  "
                f"balance={record['balance_penalty']:.6f}"
            )
    return history


@torch.no_grad()
def collect_memberships(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    memberships = []
    labels = []
    for images, batch_labels in loader:
        memberships.append(leaf_memberships(model(images.to(device))).cpu())
        labels.append(batch_labels)
    return torch.cat(memberships), torch.cat(labels)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_entropy(
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> dict:
    set_seed(config.seed)
    train_loader, test_loader = mnist_loaders(config)
    model = ResidualConvSplitter().to(device)
    started = time.monotonic()
    history = train_splitter(model, train_loader, config, device)

    train_memberships, train_labels = collect_memberships(
        model, train_loader, device
    )
    test_memberships, test_labels = collect_memberships(model, test_loader, device)
    metrics = {
        "train": split_statistics(train_memberships, train_labels),
        "canonical_test": split_statistics(test_memberships, test_labels),
    }

    checkpoint_path = output_dir / "entropy.pt"
    torch.save(
        {
            "criterion": "entropy",
            "model_kind": "two_conv_residual_binary_splitter",
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "history": history,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    result = {
        "criterion": "entropy",
        "model_kind": "two_conv_residual_binary_splitter",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_training": history[-1],
        "metrics": metrics,
    }
    test_soft = metrics["canonical_test"]["soft"]
    test_hard = metrics["canonical_test"]["hard"]
    print(
        "entropy test impurity reduction: "
        f"soft={test_soft['impurity_reduction']:.6f} "
        f"({test_soft['relative_impurity_reduction']:.2%}), "
        f"hard={test_hard['impurity_reduction']:.6f} "
        f"({test_hard['relative_impurity_reduction']:.2%})"
    )
    return result


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(config.device)
    print(f"device={device}  output={output_dir}")
    result = run_entropy(config, device, output_dir)
    summary = {
        "experiment": "mnist_two_conv_binary_entropy_splitter",
        "architecture": (
            "two convolutions, one identity residual addition around the "
            "second convolution, one sigmoid binary decision, two leaves, "
            "and no classifier head"
        ),
        "objective": (
            "sample-weighted normalized Shannon entropy across two leaves"
        ),
        "test_policy": (
            "fixed-horizon training on MNIST train; canonical MNIST test used "
            "once for final impurity-reduction measurement"
        ),
        "config": asdict(config),
        "device": str(device),
        "result": result,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary -> {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Training device (default: choose the best available device).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR / "neural_impurity_stump_2conv",
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
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
