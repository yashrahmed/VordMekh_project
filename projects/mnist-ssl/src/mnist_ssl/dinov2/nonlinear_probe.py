"""Train and evaluate a small nonlinear probe on frozen DINOv2 features.

The experiment deliberately changes only the classification head.  The DINO
backbone and the existing linear-probe checkpoint remain frozen and are
fingerprinted throughout the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import MODELS_DIR, OUT_DIR

from .eval_frozen import (
    backbone_fingerprint,
    extract_features,
    load_backbone,
    make_loader,
    seed_everything,
)
from .train import pick_device


DEFAULT_BACKBONE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt"
)
DEFAULT_LINEAR_PROBE = (
    MODELS_DIR
    / "dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "dinov2_nonlinear_probe_50ep"


class SmallNonlinearProbe(nn.Module):
    """A compact nonlinear alternative to the frozen linear probe."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_cache(
    path: Path,
    *,
    checkpoint_sha256: str,
    source_split: str,
    pool: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load a cache only when it belongs to the requested backbone and split."""

    cached = torch.load(path, map_location="cpu", weights_only=False)
    signature = cached.get("signature", {})
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "pool": pool,
        "source_split": source_split,
    }
    if signature != expected:
        raise ValueError(
            f"feature cache signature mismatch for {path}: "
            f"expected {expected}, found {signature}"
        )
    features = cached["features"].float()
    labels = cached["labels"].long()
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(f"invalid feature cache tensors in {path}")
    return features, labels, cached.get("backbone", {})


def save_feature_cache(
    path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    checkpoint_sha256: str,
    source_split: str,
    pool: str,
    backbone: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": {
                "checkpoint_sha256": checkpoint_sha256,
                "pool": pool,
                "source_split": source_split,
            },
            "features": features.cpu(),
            "labels": labels.cpu(),
            "backbone": backbone,
        },
        path,
    )


def batched_logits(
    head: nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    head.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            outputs.append(head(features[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if include_mask is None:
        include_mask = torch.ones(len(labels), dtype=torch.bool)
    predictions = logits.argmax(dim=1)
    errors = include_mask & predictions.ne(labels)
    scored = int(include_mask.sum().item())
    error_count = int(errors.sum().item())
    top2 = logits.topk(2, dim=1).indices[:, 1]
    recoverable = errors & top2.eq(labels)
    return {
        "scored_examples": scored,
        "errors": error_count,
        "accuracy": 1.0 - error_count / scored,
        "top2_recoverable_errors": int(recoverable.sum().item()),
        "top2_oracle_accuracy": 1.0
        - (error_count - int(recoverable.sum().item())) / scored,
    }


def compare_predictions(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, int]:
    if include_mask is None:
        include_mask = torch.ones(len(labels), dtype=torch.bool)
    baseline_correct = baseline_logits.argmax(dim=1).eq(labels)
    candidate_correct = candidate_logits.argmax(dim=1).eq(labels)
    fixed = include_mask & ~baseline_correct & candidate_correct
    broken = include_mask & baseline_correct & ~candidate_correct
    both_wrong = include_mask & ~baseline_correct & ~candidate_correct
    changed = include_mask & baseline_logits.argmax(dim=1).ne(
        candidate_logits.argmax(dim=1)
    )
    return {
        "changed_predictions": int(changed.sum().item()),
        "fixed_errors": int(fixed.sum().item()),
        "new_errors": int(broken.sum().item()),
        "both_wrong": int(both_wrong.sum().item()),
        "net_error_reduction": int(fixed.sum().item() - broken.sum().item()),
    }


def train_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[SmallNonlinearProbe, list[dict[str, float]]]:
    head = SmallNonlinearProbe(
        train_features.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        metrics = {"epoch": epoch, "loss": loss_sum / seen}
        history.append(metrics)
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"nonlinear_probe_epoch={epoch}/{epochs} "
                f"loss={metrics['loss']:.6f}",
                flush=True,
            )
    return head, history


def load_linear_probe(path: Path, device: torch.device) -> nn.Linear:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    head = nn.Linear(checkpoint["in_dim"], checkpoint.get("n_classes", 10)).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head.eval()
    return head


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_output = args.output_dir / "nonlinear_probe.pt"
    summary_output = args.output_dir / "summary.json"
    predictions_output = args.output_dir / "predictions.pt"
    for output in (checkpoint_output, summary_output, predictions_output):
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing experiment artifact: {output}"
            )

    checkpoint_sha = file_sha256(args.backbone)
    model, transform, backbone_metadata = load_backbone(args.backbone, device)
    before = backbone_fingerprint(model.teacher_backbone)
    backbone_metadata = {
        **backbone_metadata,
        "checkpoint_sha256": checkpoint_sha,
        "backbone_sha256": before,
        "pool": args.pool,
        "frozen": True,
    }
    print(
        f"device={device} backbone={args.backbone.name} "
        f"backbone_frozen=true",
        flush=True,
    )

    if args.train_cache:
        train_features, train_labels, _ = load_feature_cache(
            args.train_cache,
            checkpoint_sha256=checkpoint_sha,
            source_split="MNIST train (canonical order)",
            pool=args.pool,
        )
        print(f"reused_train_cache={args.train_cache}", flush=True)
    else:
        train_loader = make_loader(
            True, transform, args.feature_batch_size, args.workers
        )
        train_features, train_labels = extract_features(
            model, train_loader, device, args.pool
        )
        save_feature_cache(
            args.output_dir / "train_features.pt",
            train_features,
            train_labels,
            checkpoint_sha256=checkpoint_sha,
            source_split="MNIST train (canonical order)",
            pool=args.pool,
            backbone=backbone_metadata,
        )

    test_cache = args.output_dir / "test_features.pt"
    if test_cache.exists():
        test_features, test_labels, _ = load_feature_cache(
            test_cache,
            checkpoint_sha256=checkpoint_sha,
            source_split="MNIST test (canonical order)",
            pool=args.pool,
        )
        print(f"reused_test_cache={test_cache}", flush=True)
    else:
        test_loader = make_loader(
            False, transform, args.feature_batch_size, args.workers
        )
        test_features, test_labels = extract_features(
            model, test_loader, device, args.pool
        )
        save_feature_cache(
            test_cache,
            test_features,
            test_labels,
            checkpoint_sha256=checkpoint_sha,
            source_split="MNIST test (canonical order)",
            pool=args.pool,
            backbone=backbone_metadata,
        )
    print(
        f"features train={tuple(train_features.shape)} "
        f"test={tuple(test_features.shape)}",
        flush=True,
    )

    linear_head = load_linear_probe(args.linear_probe, device)
    baseline_train_logits = batched_logits(linear_head, train_features, device)
    baseline_test_logits = batched_logits(linear_head, test_features, device)

    nonlinear_head, history = train_probe(
        train_features,
        train_labels,
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    nonlinear_train_logits = batched_logits(
        nonlinear_head, train_features, device
    )
    nonlinear_test_logits = batched_logits(nonlinear_head, test_features, device)

    after = backbone_fingerprint(model.teacher_backbone)
    if before != after:
        raise RuntimeError("DINO backbone changed during nonlinear-probe experiment")
    reviewed = apply_mnist_test_label_policy(test_labels)

    canonical_baseline = classification_metrics(
        baseline_test_logits, test_labels
    )
    canonical_nonlinear = classification_metrics(
        nonlinear_test_logits, test_labels
    )
    reviewed_baseline = classification_metrics(
        baseline_test_logits, reviewed.labels, reviewed.include_mask
    )
    reviewed_nonlinear = classification_metrics(
        nonlinear_test_logits, reviewed.labels, reviewed.include_mask
    )
    result = {
        "protocol": {
            "architecture_fixed_before_test": True,
            "backbone_frozen": True,
            "backbone_sha256_before": before,
            "backbone_sha256_after": after,
            "backbone": backbone_metadata,
            "linear_probe": str(args.linear_probe),
            "linear_probe_sha256": file_sha256(args.linear_probe),
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "architecture": {
            "type": "layernorm-mlp",
            "in_dim": train_features.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "n_classes": 10,
            "parameters": sum(
                parameter.numel() for parameter in nonlinear_head.parameters()
            ),
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "loss": "cross_entropy",
            "history": history,
        },
        "training_set": {
            "baseline": classification_metrics(
                baseline_train_logits, train_labels
            ),
            "nonlinear": classification_metrics(
                nonlinear_train_logits, train_labels
            ),
            "comparison": compare_predictions(
                baseline_train_logits, nonlinear_train_logits, train_labels
            ),
        },
        "canonical_test": {
            "baseline": canonical_baseline,
            "nonlinear": canonical_nonlinear,
            "comparison": compare_predictions(
                baseline_test_logits, nonlinear_test_logits, test_labels
            ),
        },
        "reviewed_test": {
            "policy": reviewed.metadata,
            "baseline": reviewed_baseline,
            "nonlinear": reviewed_nonlinear,
            "comparison": compare_predictions(
                baseline_test_logits,
                nonlinear_test_logits,
                reviewed.labels,
                reviewed.include_mask,
            ),
        },
    }
    torch.save(
        {
            "result": result,
            "head_type": "layernorm-mlp",
            "head_state_dict": nonlinear_head.state_dict(),
            "in_dim": train_features.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "n_classes": 10,
        },
        checkpoint_output,
    )
    torch.save(
        {
            "canonical_labels": test_labels,
            "reviewed_labels": reviewed.labels,
            "reviewed_include_mask": reviewed.include_mask,
            "baseline_logits": baseline_test_logits,
            "nonlinear_logits": nonlinear_test_logits,
        },
        predictions_output,
    )
    summary_output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "reviewed_test "
        f"baseline_errors={reviewed_baseline['errors']} "
        f"nonlinear_errors={reviewed_nonlinear['errors']} "
        f"net_error_reduction="
        f"{result['reviewed_test']['comparison']['net_error_reduction']}",
        flush=True,
    )
    print(f"summary={summary_output}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--linear-probe", type=Path, default=DEFAULT_LINEAR_PROBE)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pool", choices=("cls",), default="cls")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
