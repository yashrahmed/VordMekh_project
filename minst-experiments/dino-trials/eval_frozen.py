"""Evaluate a frozen DINOv2 teacher with weighted k-NN and a linear probe.

Both evaluations operate on cached features. The backbone has ``requires_grad``
disabled, remains in eval mode, and is fingerprinted before and after probe
training so an accidental update fails the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from data import EvaluationTransform
from eval_knn import build_teacher, weighted_knn_accuracy
from train import DATASET_DIR, pick_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def backbone_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_backbone(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, object, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_teacher(checkpoint, device)
    transform = EvaluationTransform(
        config.get("global_size", 56), config.get("preprocess", True)
    )
    checkpoint_epoch = checkpoint.get("completed_epoch", config.get("epochs"))

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    metadata = {
        "family": "dinov2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_config": config,
    }
    return model, transform, metadata


def make_loader(
    train: bool,
    transform: object,
    batch_size: int,
    workers: int,
) -> DataLoader:
    dataset = datasets.MNIST(
        str(DATASET_DIR), train=train, download=True, transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pool: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for images, target in loader:
        features.append(model.encode(images.to(device), pool=pool).float().cpu())
        labels.append(target)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def accuracy(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, device) -> float:
    head.eval()
    correct = 0
    for start in range(0, len(features), 1024):
        logits = head(features[start : start + 1024].to(device))
        target = labels[start : start + 1024].to(device)
        correct += (logits.argmax(dim=1) == target).sum().item()
    return correct / len(labels)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_every: int,
    output: Path,
) -> tuple[nn.Linear, list[dict]]:
    head = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    partial = output.with_name(f"{output.stem}_resume{output.suffix}")
    start_epoch = 1
    history: list[dict] = []
    if partial.exists():
        saved = torch.load(partial, map_location=device, weights_only=False)
        if saved.get("target_epochs") != epochs:
            raise ValueError(f"probe partial {partial} targets a different epoch count")
        head.load_state_dict(saved["head_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history = saved["history"]
        start_epoch = saved["completed_epoch"] + 1
        print(f"resumed_probe={partial} start_epoch={start_epoch}", flush=True)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, epochs + 1):
        head.train()
        total_loss = 0.0
        seen = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        metrics = {"epoch": epoch, "loss": total_loss / seen}
        history.append(metrics)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"probe_epoch={epoch}/{epochs} loss={metrics['loss']:.6f}", flush=True)
        if checkpoint_every and epoch % checkpoint_every == 0 and epoch < epochs:
            torch.save(
                {
                    "target_epochs": epochs,
                    "completed_epoch": epoch,
                    "history": history,
                    "head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                partial,
            )
            print(f"probe_checkpoint={partial}", flush=True)

    partial.unlink(missing_ok=True)
    return head, history


def evaluate(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    model, transform, metadata = load_backbone(args.model, device)
    if args.pool not in ("cls", "mean", "concat"):
        raise ValueError("DINOv2 pool must be cls, mean, or concat")

    before = backbone_fingerprint(model.teacher_backbone)
    print(
        f"device={device} family=dinov2 pool={args.pool} "
        f"backbone_frozen=true checkpoint={args.model}",
        flush=True,
    )
    train_loader = make_loader(True, transform, args.feature_batch_size, args.workers)
    test_loader = make_loader(False, transform, args.feature_batch_size, args.workers)
    train_features, train_labels = extract_features(model, train_loader, device, args.pool)
    test_features, test_labels = extract_features(model, test_loader, device, args.pool)
    print(
        f"train_features={tuple(train_features.shape)} "
        f"test_features={tuple(test_features.shape)}",
        flush=True,
    )

    knn_accuracy = weighted_knn_accuracy(
        torch.nn.functional.normalize(train_features, dim=-1),
        train_labels,
        torch.nn.functional.normalize(test_features, dim=-1),
        test_labels,
        k=args.k,
        temperature=args.knn_temperature,
        query_batch_size=args.knn_batch_size,
    )
    print(f"knn_test_accuracy={knn_accuracy:.2%}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    head, history = train_linear_probe(
        train_features,
        train_labels,
        device,
        args.linear_epochs,
        args.linear_batch_size,
        args.linear_lr,
        args.linear_weight_decay,
        args.checkpoint_every,
        args.output,
    )
    train_accuracy = accuracy(head, train_features, train_labels, device)
    test_accuracy = accuracy(head, test_features, test_labels, device)
    after = backbone_fingerprint(model.teacher_backbone)
    if before != after:
        raise RuntimeError("backbone changed during frozen evaluation")

    result = {
        **metadata,
        "pool": args.pool,
        "feature_dim": train_features.shape[1],
        "backbone_frozen": True,
        "backbone_sha256_before": before,
        "backbone_sha256_after": after,
        "knn": {
            "k": args.k,
            "temperature": args.knn_temperature,
            "test_accuracy": knn_accuracy,
        },
        "linear_probe": {
            "epochs": args.linear_epochs,
            "learning_rate": args.linear_lr,
            "weight_decay": args.linear_weight_decay,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "history": history,
        },
    }
    torch.save(
        {
            "result": result,
            "head_state_dict": head.state_dict(),
            "in_dim": train_features.shape[1],
            "n_classes": 10,
        },
        args.output,
    )
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"linear_train_accuracy={train_accuracy:.2%} "
        f"linear_test_accuracy={test_accuracy:.2%} output={args.output}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pool", choices=("cls", "mean", "concat"), default="cls")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--knn-temperature", type=float, default=0.07)
    parser.add_argument("--knn-batch-size", type=int, default=256)
    parser.add_argument("--linear-epochs", type=int, default=50)
    parser.add_argument("--linear-batch-size", type=int, default=256)
    parser.add_argument("--linear-lr", type=float, default=1e-3)
    parser.add_argument("--linear-weight-decay", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
