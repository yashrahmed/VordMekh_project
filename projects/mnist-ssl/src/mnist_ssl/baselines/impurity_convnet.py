"""A small convolutional neural decision tree for MNIST.

The experiment in this module asks whether the two class-purity criteria used by
ordinary decision trees can train a neural router.  A compact convolutional
trunk emits one binary routing logit for every internal node of a complete tree.
The resulting soft path probabilities define differentiable leaf memberships;
the objective is the sample-weighted Gini impurity or Shannon entropy of the
true-label distribution in those leaves.

This is deliberately different from minimizing ``1 - sum(p**2)`` for each
classifier prediction.  That expression contains no target label and is
minimized by any confident (including completely collapsed) predictor.  Here,
labels enter through each leaf's class histogram, matching CART's criterion.
At evaluation time the leaves are assigned class distributions using the MNIST
training split only, then the fixed tree is scored on the held-out test split.
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
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_ssl.paths import DATASET_DIR, OUT_DIR


N_CLASSES = 10
TREE_CRITERIA = ("gini", "entropy")
ALL_CRITERIA = (*TREE_CRITERIA, "cross_entropy")


@dataclass(frozen=True)
class ExperimentConfig:
    """The fixed choices needed to reproduce one comparison run."""

    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    depth: int = 4
    balance_weight: float = 0.05
    seed: int = 0
    num_workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SmallConvTrunk(nn.Module):
    """A roughly 24K-parameter MNIST feature extractor shared by all runs."""

    output_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, self.output_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images).flatten(1)


class SmallConvClassifier(nn.Module):
    """Cross-entropy control using the same convolutional trunk."""

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.trunk = SmallConvTrunk()
        self.head = nn.Linear(self.trunk.output_dim, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(images))


def path_probabilities(
    routing_logits: torch.Tensor, depth: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert breadth-first internal-node logits into soft leaf probabilities.

    Returns ``(leaves, balance_penalty)``.  The penalty measures how far each
    reached node's conditional left/right traffic is from 50/50, averaged over
    tree depths.  It prevents dead branches without using class labels.
    """

    expected_nodes = 2**depth - 1
    if routing_logits.ndim != 2 or routing_logits.shape[1] != expected_nodes:
        raise ValueError(
            f"expected routing logits shaped (batch, {expected_nodes}), "
            f"got {tuple(routing_logits.shape)}"
        )

    paths = routing_logits.new_ones((routing_logits.shape[0], 1))
    balance = routing_logits.new_zeros(())
    offset = 0
    for level in range(depth):
        width = 2**level
        right = routing_logits[:, offset : offset + width].sigmoid()
        offset += width

        node_mass = paths.sum(dim=0).clamp_min(1e-8)
        right_fraction = (paths * right).sum(dim=0) / node_mass
        node_weight = node_mass / node_mass.sum()
        balance = balance + (
            node_weight * (right_fraction - 0.5).square()
        ).sum()

        paths = torch.stack((paths * (1.0 - right), paths * right), dim=-1)
        paths = paths.flatten(1)

    return paths, balance / depth


class NeuralImpurityTree(nn.Module):
    """A CNN whose outputs are the internal decisions of a complete soft tree."""

    def __init__(self, depth: int = 4) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("tree depth must be at least one")
        self.depth = depth
        self.trunk = SmallConvTrunk()
        self.router = nn.Linear(self.trunk.output_dim, 2**depth - 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(self.trunk(images))
        return path_probabilities(logits, self.depth)


def leaf_impurity(
    leaf_probabilities: torch.Tensor,
    labels: torch.Tensor,
    criterion: str,
    n_classes: int = N_CLASSES,
) -> torch.Tensor:
    """Return sample-weighted class impurity across differentiable leaves."""

    if criterion not in TREE_CRITERIA:
        raise ValueError(f"unknown tree criterion: {criterion}")
    if leaf_probabilities.ndim != 2:
        raise ValueError("leaf probabilities must be a two-dimensional tensor")
    if labels.ndim != 1 or len(labels) != len(leaf_probabilities):
        raise ValueError("labels must contain one entry per sample")

    targets = F.one_hot(labels, num_classes=n_classes).to(
        dtype=leaf_probabilities.dtype
    )
    counts = leaf_probabilities.transpose(0, 1) @ targets
    mass = counts.sum(dim=1)
    distribution = counts / mass.unsqueeze(1).clamp_min(1e-8)

    if criterion == "gini":
        impurity = 1.0 - distribution.square().sum(dim=1)
    else:
        safe_distribution = distribution.clamp_min(1e-8)
        impurity = -(distribution * safe_distribution.log()).sum(dim=1)
        impurity = impurity / math.log(n_classes)

    return (mass / mass.sum().clamp_min(1e-8) * impurity).sum()


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


def _display_epoch(epoch: int, epochs: int) -> bool:
    return epoch == 1 or epoch == epochs or epoch % 5 == 0


def train_classifier(
    model: SmallConvClassifier,
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
        loss_sum = 0.0
        seen = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        scheduler.step()
        record = {"epoch": epoch, "loss": loss_sum / seen}
        history.append(record)
        if _display_epoch(epoch, config.epochs):
            print(
                f"cross_entropy epoch {epoch:02d}/{config.epochs}  "
                f"loss={record['loss']:.6f}"
            )
    return history


def train_tree(
    model: NeuralImpurityTree,
    loader: DataLoader,
    criterion: str,
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
        impurity_sum = 0.0
        balance_sum = 0.0
        loss_sum = 0.0
        seen = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            leaves, balance = model(images)
            impurity = leaf_impurity(leaves, labels, criterion)
            loss = impurity + config.balance_weight * balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = len(labels)
            impurity_sum += impurity.item() * batch_size
            balance_sum += balance.item() * batch_size
            loss_sum += loss.item() * batch_size
            seen += batch_size
        scheduler.step()
        record = {
            "epoch": epoch,
            "loss": loss_sum / seen,
            "impurity": impurity_sum / seen,
            "balance_penalty": balance_sum / seen,
        }
        history.append(record)
        if _display_epoch(epoch, config.epochs):
            print(
                f"{criterion:>7s} epoch {epoch:02d}/{config.epochs}  "
                f"loss={record['loss']:.6f}  "
                f"impurity={record['impurity']:.6f}  "
                f"balance={record['balance_penalty']:.6f}"
            )
    return history


@torch.no_grad()
def calibrate_leaves(
    model: NeuralImpurityTree,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate leaf class distributions using training labels only."""

    model.eval()
    counts = torch.zeros(2**model.depth, N_CLASSES, device=device)
    hard_counts = torch.zeros(2**model.depth, device=device, dtype=torch.long)
    for images, labels in loader:
        leaves, _ = model(images.to(device))
        targets = F.one_hot(labels.to(device), N_CLASSES).to(leaves.dtype)
        counts += leaves.transpose(0, 1) @ targets
        hard_counts += torch.bincount(
            leaves.argmax(dim=1), minlength=2**model.depth
        )
    distributions = counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return distributions, counts.sum(dim=1), hard_counts


@torch.no_grad()
def evaluate_classifier(
    model: SmallConvClassifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        correct += (predictions == labels).sum().item()
        total += len(labels)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_tree(
    model: NeuralImpurityTree,
    leaf_distributions: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Score both soft-mixture and conventional hard-leaf predictions."""

    model.eval()
    soft_correct = 0
    hard_correct = 0
    total = 0
    for images, labels in loader:
        leaves, _ = model(images.to(device))
        soft = (leaves @ leaf_distributions).argmax(dim=1).cpu()
        hard_leaf = leaves.argmax(dim=1)
        hard = leaf_distributions[hard_leaf].argmax(dim=1).cpu()
        soft_correct += (soft == labels).sum().item()
        hard_correct += (hard == labels).sum().item()
        total += len(labels)
    return {
        "soft_mixture_accuracy": 100.0 * soft_correct / total,
        "hard_leaf_accuracy": 100.0 * hard_correct / total,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_criterion(
    criterion: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Train, evaluate, and checkpoint one prespecified criterion."""

    set_seed(config.seed)
    started = time.monotonic()
    if criterion == "cross_entropy":
        model: nn.Module = SmallConvClassifier().to(device)
        history = train_classifier(model, train_loader, config, device)
        train_accuracy = evaluate_classifier(model, train_loader, device)
        test_accuracy = evaluate_classifier(model, test_loader, device)
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
        leaf_distributions = None
        leaf_masses = None
        model_kind = "small_conv_classifier"
    else:
        model = NeuralImpurityTree(depth=config.depth).to(device)
        history = train_tree(
            model, train_loader, criterion, config, device
        )
        leaf_distributions, leaf_masses, hard_leaf_counts = calibrate_leaves(
            model, train_loader, device
        )
        train_metrics = evaluate_tree(
            model, leaf_distributions, train_loader, device
        )
        test_metrics = evaluate_tree(
            model, leaf_distributions, test_loader, device
        )
        metrics = {
            "train_soft_mixture_accuracy": train_metrics[
                "soft_mixture_accuracy"
            ],
            "train_hard_leaf_accuracy": train_metrics["hard_leaf_accuracy"],
            "test_soft_mixture_accuracy": test_metrics[
                "soft_mixture_accuracy"
            ],
            "test_hard_leaf_accuracy": test_metrics["hard_leaf_accuracy"],
            "active_hard_leaves": int((hard_leaf_counts > 0).sum().item()),
        }
        model_kind = "small_conv_neural_tree"

    elapsed = time.monotonic() - started
    checkpoint_path = output_dir / f"{criterion}.pt"
    checkpoint = {
        "criterion": criterion,
        "model_kind": model_kind,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "leaf_class_distributions": (
            None if leaf_distributions is None else leaf_distributions.cpu()
        ),
        "history": history,
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)

    result = {
        "criterion": criterion,
        "model_kind": model_kind,
        "parameter_count": _parameter_count(model),
        "elapsed_seconds": elapsed,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_training": history[-1],
        "metrics": metrics,
    }
    print(f"{criterion} result: {json.dumps(metrics, sort_keys=True)}")
    return result


def run_experiment(
    criteria: Iterable[str],
    config: ExperimentConfig,
    output_dir: Path,
) -> dict:
    criteria = tuple(criteria)
    unknown = sorted(set(criteria) - set(ALL_CRITERIA))
    if unknown:
        raise ValueError(f"unknown criteria: {', '.join(unknown)}")
    if not criteria:
        raise ValueError("at least one criterion is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  output={output_dir}")
    results = []
    for criterion in criteria:
        # Rebuild the loaders so every criterion sees the same seed-0 shuffle
        # trajectory instead of inheriting the previous run's generator state.
        train_loader, test_loader = mnist_loaders(config)
        results.append(
            run_criterion(
                criterion,
                train_loader,
                test_loader,
                config,
                device,
                output_dir,
            )
        )
    summary = {
        "experiment": "mnist_small_conv_neural_impurity_tree",
        "criterion_definition": {
            "gini": "sample-weighted CART Gini impurity across soft leaves",
            "entropy": (
                "sample-weighted normalized Shannon entropy across soft leaves"
            ),
            "cross_entropy": (
                "standard supervised classifier control with the same CNN trunk"
            ),
        },
        "test_policy": (
            "fixed-horizon training on MNIST train; leaf class distributions "
            "fit on train labels; one final evaluation on canonical MNIST test"
        ),
        "config": asdict(config),
        "device": str(device),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary -> {summary_path}")
    return summary


def parse_criteria(value: str) -> tuple[str, ...]:
    criteria = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(criteria) - set(ALL_CRITERIA))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown criteria {', '.join(unknown)}; choose from {ALL_CRITERIA}"
        )
    return criteria


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criteria",
        type=parse_criteria,
        default=ALL_CRITERIA,
        help="Comma-separated subset of gini,entropy,cross_entropy.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR / "neural_impurity_convnet",
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        depth=args.depth,
        balance_weight=args.balance_weight,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    run_experiment(args.criteria, config, args.output_dir)


if __name__ == "__main__":
    main()
