"""Evaluate a small nonlinear probe on the best frozen I-JEPA representation.

The recorded best individual I-JEPA member uses the 300-epoch 56x56 backbone,
48 target patches, and its flattened 64 x 128 token grid.  This experiment
replaces only its linear classification head with the same LayerNorm/GELU
architecture used by the DINO nonlinear-probe experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from mnist_ssl.baselines.mae import make_transform
from mnist_ssl.dinov2.eval_frozen import backbone_fingerprint, seed_everything
from mnist_ssl.dinov2.nonlinear_probe import (
    SmallNonlinearProbe,
    batched_logits,
    classification_metrics,
    compare_predictions,
    file_sha256,
)
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR

from . import custom_ijepa


DEFAULT_LINEAR_PROBE = (
    MODELS_DIR
    / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_"
    "base300ep_probe50ep.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "ijepa_nonlinear_probe_best300"
DEFAULT_MILESTONES = (50, 75, 100)


def parse_milestones(value: str) -> tuple[int, ...]:
    try:
        milestones = tuple(
            sorted({int(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "milestones must be comma-separated integers"
        ) from exc
    if not milestones or milestones[0] < 1:
        raise argparse.ArgumentTypeError("milestones must contain positive epochs")
    return milestones


def load_best_member(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Linear, dict[str, Any]]:
    """Rebuild the exact encoder and linear head stored in the best checkpoint."""

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("pool") != "flatten":
        raise ValueError("the best I-JEPA member must use flattened features")
    if checkpoint.get("mode") != "probe":
        raise ValueError("the I-JEPA baseline must be a frozen probe")
    if checkpoint.get("encoder", "custom_ijepa") != "custom_ijepa":
        raise ValueError("the I-JEPA baseline must use custom_ijepa")

    model = custom_ijepa.build_model(
        enc_dim=checkpoint.get("enc_dim", custom_ijepa.DEFAULT_ENC_DIM),
        n_targets=checkpoint.get("n_targets", custom_ijepa.N_TARGETS),
    ).to(device)
    model.load_state_dict(checkpoint["encoder_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    linear_head = nn.Linear(
        checkpoint["in_dim"], checkpoint.get("n_classes", 10)
    ).to(device)
    linear_head.load_state_dict(checkpoint["head_state_dict"])
    for parameter in linear_head.parameters():
        parameter.requires_grad_(False)
    linear_head.eval()

    metadata = {
        "family": checkpoint.get("family"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "encoder": "custom_ijepa",
        "pretraining_epochs": 300,
        "pool": "flatten",
        "preproc": checkpoint.get("preproc", True),
        "enc_dim": model.embed_dim,
        "n_patches": model.n_patches,
        "n_targets": model.n_targets,
        "feature_dim": checkpoint["in_dim"],
        "frozen": True,
    }
    return model, linear_head, metadata


@torch.no_grad()
def extract_split_features(
    model: nn.Module,
    *,
    train: bool,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract flattened features into one preallocated CPU tensor."""

    dataset = datasets.MNIST(
        str(DATASET_DIR),
        train=train,
        download=True,
        transform=make_transform(preproc=True),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    features: torch.Tensor | None = None
    labels = torch.empty(len(dataset), dtype=torch.long)
    offset = 0
    model.eval()
    for images, target in loader:
        batch_features = model.encode(
            images.to(device), pool="flatten"
        ).float().cpu()
        if features is None:
            features = torch.empty(
                len(dataset), batch_features.shape[1], dtype=torch.float32
            )
        end = offset + len(target)
        features[offset:end].copy_(batch_features)
        labels[offset:end].copy_(target)
        offset = end
    if features is None or offset != len(dataset):
        raise RuntimeError("failed to extract the complete MNIST split")
    return features, labels


def milestone_checkpoint_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"nonlinear_probe_{epoch}ep.pt"


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    milestones = tuple(args.milestones)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    predictions_path = args.output_dir / "predictions.pt"
    outputs = [
        summary_path,
        predictions_path,
        *(milestone_checkpoint_path(args.output_dir, epoch) for epoch in milestones),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing experiment artifacts: {existing}"
        )

    model, linear_head, backbone_metadata = load_best_member(
        args.linear_probe, device
    )
    before = backbone_fingerprint(model)
    print(
        f"device={device} backbone={args.linear_probe.name} "
        f"backbone_frozen=true pool=flatten",
        flush=True,
    )
    print("extracting_train_features=true", flush=True)
    train_features, train_labels = extract_split_features(
        model,
        train=True,
        device=device,
        batch_size=args.feature_batch_size,
        workers=args.workers,
    )
    print("extracting_test_features=true", flush=True)
    test_features, test_labels = extract_split_features(
        model,
        train=False,
        device=device,
        batch_size=args.feature_batch_size,
        workers=args.workers,
    )
    print(
        f"features train={tuple(train_features.shape)} "
        f"test={tuple(test_features.shape)}",
        flush=True,
    )

    baseline_train_logits = batched_logits(
        linear_head, train_features, device, args.eval_batch_size
    )
    baseline_test_logits = batched_logits(
        linear_head, test_features, device, args.eval_batch_size
    )
    reviewed = apply_mnist_test_label_policy(test_labels)

    nonlinear_head = SmallNonlinearProbe(
        train_features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        nonlinear_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    history: list[dict[str, float]] = []
    milestone_results: list[dict[str, Any]] = []
    saved_predictions: dict[int, torch.Tensor] = {}
    for epoch in range(1, max(milestones) + 1):
        nonlinear_head.train()
        loss_sum = 0.0
        seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(nonlinear_head(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        epoch_metrics = {"epoch": epoch, "loss": loss_sum / seen}
        history.append(epoch_metrics)
        if epoch % 10 == 0 or epoch in milestones:
            print(
                f"nonlinear_probe_epoch={epoch}/{max(milestones)} "
                f"loss={epoch_metrics['loss']:.6f}",
                flush=True,
            )
        if epoch not in milestones:
            continue

        train_logits = batched_logits(
            nonlinear_head, train_features, device, args.eval_batch_size
        )
        test_logits = batched_logits(
            nonlinear_head, test_features, device, args.eval_batch_size
        )
        canonical_metrics = classification_metrics(test_logits, test_labels)
        reviewed_metrics = classification_metrics(
            test_logits, reviewed.labels, reviewed.include_mask
        )
        milestone_result = {
            "epoch": epoch,
            "train": classification_metrics(train_logits, train_labels),
            "canonical_test": canonical_metrics,
            "reviewed_test": reviewed_metrics,
            "canonical_comparison": compare_predictions(
                baseline_test_logits, test_logits, test_labels
            ),
            "reviewed_comparison": compare_predictions(
                baseline_test_logits,
                test_logits,
                reviewed.labels,
                reviewed.include_mask,
            ),
        }
        milestone_results.append(milestone_result)
        saved_predictions[epoch] = test_logits
        torch.save(
            {
                "head_type": "layernorm-mlp",
                "head_state_dict": nonlinear_head.state_dict(),
                "in_dim": train_features.shape[1],
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "n_classes": 10,
                "epoch": epoch,
                "backbone": backbone_metadata,
                "metrics": milestone_result,
            },
            milestone_checkpoint_path(args.output_dir, epoch),
        )
        print(
            f"milestone={epoch} "
            f"canonical_errors={canonical_metrics['errors']} "
            f"reviewed_errors={reviewed_metrics['errors']}",
            flush=True,
        )

    after = backbone_fingerprint(model)
    if before != after:
        raise RuntimeError("I-JEPA backbone changed during nonlinear-probe training")

    result = {
        "protocol": {
            "architecture_fixed_before_test": True,
            "backbone_frozen": True,
            "backbone_sha256_before": before,
            "backbone_sha256_after": after,
            "backbone": backbone_metadata,
            "linear_probe": str(args.linear_probe),
            "milestones": list(milestones),
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
        "baseline": {
            "train": classification_metrics(
                baseline_train_logits, train_labels
            ),
            "canonical_test": classification_metrics(
                baseline_test_logits, test_labels
            ),
            "reviewed_test": classification_metrics(
                baseline_test_logits, reviewed.labels, reviewed.include_mask
            ),
        },
        "reviewed_label_policy": reviewed.metadata,
        "results": milestone_results,
    }
    torch.save(
        {
            "canonical_labels": test_labels,
            "reviewed_labels": reviewed.labels,
            "reviewed_include_mask": reviewed.include_mask,
            "baseline_logits": baseline_test_logits,
            "nonlinear_logits_by_epoch": saved_predictions,
        },
        predictions_path,
    )
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"summary={summary_path}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-probe", type=Path, default=DEFAULT_LINEAR_PROBE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--milestones",
        type=parse_milestones,
        default=DEFAULT_MILESTONES,
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
