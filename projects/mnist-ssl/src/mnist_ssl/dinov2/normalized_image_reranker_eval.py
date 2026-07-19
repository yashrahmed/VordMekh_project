"""Evaluate the frozen normalized-image reranker on MNIST test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR

from .nonlinear_probe import file_sha256, load_feature_cache
from .normalized_image_reranker import (
    DEFAULT_GATE_THRESHOLD,
    DEFAULT_LINEAR_PROBE,
    IndependentNormalizedReranker,
    gated_predictions,
    load_exact_linear_probe,
    normalized_top2_margin,
)
from .train import pick_device


DEFAULT_BACKBONE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt"
)
DEFAULT_RERANKER = (
    OUT_DIR
    / "dinov2_normalized_image_reranker_50ep"
    / "milestone_epoch45.pt"
)
DEFAULT_TEST_FEATURES = (
    OUT_DIR / "dinov2_nonlinear_probe_50ep" / "test_features.pt"
)
DEFAULT_TEST_IMAGES = (
    OUT_DIR / "dinov2_correction_addon_v2" / "test_image_views_uint8.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "dinov2_normalized_image_reranker_50ep"


def policy_metrics(
    predictions: torch.Tensor,
    *,
    labels: torch.Tensor,
    include_mask: torch.Tensor,
    base_top1: torch.Tensor,
    base_top2: torch.Tensor,
    normalized_margin: torch.Tensor,
    gate_threshold: float,
) -> dict[str, Any]:
    """Measure fixed-gate corrections for one label policy."""

    gate = include_mask & normalized_margin.le(gate_threshold)
    base_correct = base_top1.eq(labels)
    candidate_correct = predictions.eq(labels)
    changed = include_mask & predictions.ne(base_top1)
    fixes = include_mask & ~base_correct & candidate_correct
    breaks = include_mask & base_correct & ~candidate_correct
    wrong_to_wrong = changed & ~base_correct & ~candidate_correct
    recoverable = include_mask & ~base_correct & base_top2.eq(labels)
    base_errors = int((include_mask & ~base_correct).sum().item())
    reranked_errors = int((include_mask & ~candidate_correct).sum().item())
    scored = int(include_mask.sum().item())
    return {
        "scored_examples": scored,
        "gate_threshold": gate_threshold,
        "gate_eligible": int(gate.sum().item()),
        "gate_top1_correct": int((gate & base_correct).sum().item()),
        "gate_top2_recoverable": int((gate & recoverable).sum().item()),
        "gate_neither_candidate_correct": int(
            (gate & ~base_correct & ~base_top2.eq(labels)).sum().item()
        ),
        "changed_predictions": int(changed.sum().item()),
        "fixed_errors": int(fixes.sum().item()),
        "new_errors": int(breaks.sum().item()),
        "wrong_to_wrong": int(wrong_to_wrong.sum().item()),
        "net_error_reduction": base_errors - reranked_errors,
        "base_errors": base_errors,
        "reranked_errors": reranked_errors,
        "base_accuracy": 1.0 - base_errors / scored,
        "reranked_accuracy": 1.0 - reranked_errors / scored,
        "gate_oracle_errors": base_errors - int((gate & recoverable).sum().item()),
        "gate_oracle_accuracy": 1.0
        - (base_errors - int((gate & recoverable).sum().item())) / scored,
    }


def load_test_images(path: Path, labels: torch.Tensor) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    images = payload.get("bbox")
    if (
        images is None
        or images.dtype != torch.uint8
        or images.shape != (10_000, 1, 28, 28)
    ):
        raise ValueError(f"{path} is not the normalized MNIST test image cache")
    mnist = datasets.MNIST(str(DATASET_DIR), train=False, download=False)
    if not torch.equal(mnist.targets.long(), labels):
        raise ValueError("MNIST test order differs from the feature cache")
    return images


def load_reranker(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    if checkpoint.get("completed_epoch") != 45:
        raise ValueError(f"{path} is not the selected epoch-45 milestone")
    model = IndependentNormalizedReranker(
        channels=tuple(config["channels"]),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def score_images(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(
        TensorDataset(images),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    chunks = []
    for (batch,) in loader:
        chunks.append(model(batch.to(device).float().div_(255.0)).cpu())
    return torch.cat(chunks)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "test_evaluation_epoch45.json"
    predictions_path = args.output_dir / "test_predictions_epoch45.pt"
    if summary_path.exists() or predictions_path.exists():
        raise FileExistsError("refusing to overwrite completed test evaluation")

    device = torch.device(args.device) if args.device else pick_device()
    _, linear_weight, linear_bias = load_exact_linear_probe(args.linear_probe)
    test_features, test_labels, backbone = load_feature_cache(
        args.test_features,
        checkpoint_sha256=file_sha256(args.backbone),
        source_split="MNIST test (canonical order)",
        pool="cls",
    )
    if test_features.shape != (10_000, 128):
        raise ValueError("test feature cache has the wrong shape")
    test_images = load_test_images(args.test_images, test_labels)
    reranker, reranker_checkpoint = load_reranker(args.reranker, device)

    base_logits = test_features @ linear_weight.T + linear_bias
    top = base_logits.topk(2, dim=1)
    margin = normalized_top2_margin(base_logits)
    reranker_scores = score_images(
        reranker,
        test_images,
        device,
        args.batch_size,
    )
    predictions = gated_predictions(
        top.indices[:, 0],
        top.indices[:, 1],
        margin,
        reranker_scores,
        gate_threshold=args.gate_threshold,
    )
    canonical_include = torch.ones(len(test_labels), dtype=torch.bool)
    canonical = policy_metrics(
        predictions,
        labels=test_labels,
        include_mask=canonical_include,
        base_top1=top.indices[:, 0],
        base_top2=top.indices[:, 1],
        normalized_margin=margin,
        gate_threshold=args.gate_threshold,
    )
    reviewed = apply_mnist_test_label_policy(test_labels)
    reviewed_metrics = policy_metrics(
        predictions,
        labels=reviewed.labels,
        include_mask=reviewed.include_mask,
        base_top1=top.indices[:, 0],
        base_top2=top.indices[:, 1],
        normalized_margin=margin,
        gate_threshold=args.gate_threshold,
    )

    result = {
        "protocol": {
            "selection": "epoch 45 selected by gated training errors",
            "threshold_selection": "fixed from the full training-set grid search",
            "test_tuning": False,
            "linear_probe_type": "Linear(128,10)",
            "linear_probe_epochs": 50,
            "linear_probe": str(args.linear_probe),
            "linear_probe_sha256": file_sha256(args.linear_probe),
            "backbone": backbone,
            "backbone_checkpoint_sha256": file_sha256(args.backbone),
            "reranker": str(args.reranker),
            "reranker_sha256": file_sha256(args.reranker),
            "reranker_epoch": reranker_checkpoint["completed_epoch"],
            "reranker_inputs": "bbox-normalized MNIST pixels only, shape [1,28,28]",
            "reranker_receives_gate_or_logits": False,
            "gate_normalization": "top-two logit gap / per-sample std of ten logits",
            "gate_threshold": args.gate_threshold,
        },
        "canonical_test": canonical,
        "reviewed_test": {
            "policy": reviewed.metadata,
            **reviewed_metrics,
        },
    }
    torch.save(
        {
            "canonical_labels": test_labels,
            "reviewed_labels": reviewed.labels,
            "reviewed_include_mask": reviewed.include_mask,
            "base_logits": base_logits,
            "normalized_margin": margin,
            "reranker_scores": reranker_scores,
            "baseline_predictions": top.indices[:, 0],
            "runner_up_predictions": top.indices[:, 1],
            "reranked_predictions": predictions,
            "gate_threshold": args.gate_threshold,
        },
        predictions_path,
    )
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"canonical base_errors={canonical['base_errors']} "
        f"reranked_errors={canonical['reranked_errors']} "
        f"fixes={canonical['fixed_errors']} "
        f"breaks={canonical['new_errors']} "
        f"net={canonical['net_error_reduction']}",
        flush=True,
    )
    print(
        f"reviewed base_errors={reviewed_metrics['base_errors']} "
        f"reranked_errors={reviewed_metrics['reranked_errors']} "
        f"fixes={reviewed_metrics['fixed_errors']} "
        f"breaks={reviewed_metrics['new_errors']} "
        f"net={reviewed_metrics['net_error_reduction']}",
        flush=True,
    )
    print(f"summary={summary_path}", flush=True)
    print(f"predictions={predictions_path}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--linear-probe", type=Path, default=DEFAULT_LINEAR_PROBE)
    parser.add_argument("--reranker", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--test-features", type=Path, default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gate-threshold", type=float, default=DEFAULT_GATE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
