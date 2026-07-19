"""Evaluate the validation-selected normalized-image reranker once on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import OUT_DIR

from .nonlinear_probe import file_sha256, load_feature_cache
from .normalized_image_reranker import (
    DEFAULT_LINEAR_PROBE,
    IndependentNormalizedReranker,
    gated_predictions,
    load_exact_linear_probe,
    normalized_top2_margin,
)
from .normalized_image_reranker_eval import (
    DEFAULT_BACKBONE,
    DEFAULT_TEST_FEATURES,
    DEFAULT_TEST_IMAGES,
    load_test_images,
    policy_metrics,
    score_images,
)
from .normalized_image_reranker_split import DEFAULT_OUTPUT_DIR
from .train import pick_device


DEFAULT_RERANKER = DEFAULT_OUTPUT_DIR / "best_validation_reranker.pt"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "test_evaluation.json"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "test_predictions.pt"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("refusing to overwrite completed test evaluation")

    device = torch.device(args.device) if args.device else pick_device()
    _, linear_weight, linear_bias = load_exact_linear_probe(args.linear_probe)
    test_features, test_labels, backbone = load_feature_cache(
        args.test_features,
        checkpoint_sha256=file_sha256(args.backbone),
        source_split="MNIST test (canonical order)",
        pool="cls",
    )
    test_images = load_test_images(args.test_images, test_labels)

    checkpoint = torch.load(args.reranker, map_location=device, weights_only=False)
    selected_threshold = float(checkpoint["selected_threshold"])
    config = checkpoint["config"]
    model = IndependentNormalizedReranker(
        channels=tuple(config["channels"]),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    base_logits = test_features @ linear_weight.T + linear_bias
    top = base_logits.topk(2, dim=1)
    margin = normalized_top2_margin(base_logits)
    reranker_scores = score_images(
        model,
        test_images,
        device,
        args.batch_size,
    )
    predictions = gated_predictions(
        top.indices[:, 0],
        top.indices[:, 1],
        margin,
        reranker_scores,
        gate_threshold=selected_threshold,
    )
    canonical = policy_metrics(
        predictions,
        labels=test_labels,
        include_mask=torch.ones(len(test_labels), dtype=torch.bool),
        base_top1=top.indices[:, 0],
        base_top2=top.indices[:, 1],
        normalized_margin=margin,
        gate_threshold=selected_threshold,
    )
    reviewed = apply_mnist_test_label_policy(test_labels)
    reviewed_metrics = policy_metrics(
        predictions,
        labels=reviewed.labels,
        include_mask=reviewed.include_mask,
        base_top1=top.indices[:, 0],
        base_top2=top.indices[:, 1],
        normalized_margin=margin,
        gate_threshold=selected_threshold,
    )
    result = {
        "protocol": {
            "selection": "epoch and threshold selected only on correction validation",
            "test_tuning": False,
            "linear_probe_type": "Linear(128,10)",
            "linear_probe_epochs": 50,
            "linear_probe": str(args.linear_probe),
            "linear_probe_sha256": file_sha256(args.linear_probe),
            "backbone": backbone,
            "reranker": str(args.reranker),
            "reranker_sha256": file_sha256(args.reranker),
            "reranker_epoch": checkpoint["completed_epoch"],
            "validation_metrics_at_selection": checkpoint["validation_metrics"],
            "gate_threshold": selected_threshold,
            "gate_normalization": "top-two gap / per-sample std of ten logits",
            "reranker_inputs": "bbox-normalized MNIST pixels only",
        },
        "canonical_test": canonical,
        "reviewed_test": {
            "policy": reviewed.metadata,
            **reviewed_metrics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
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
            "gate_threshold": selected_threshold,
        },
        args.predictions,
    )
    print(
        f"selected_epoch={checkpoint['completed_epoch']} "
        f"threshold={selected_threshold:.9f}",
        flush=True,
    )
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
    print(f"summary={args.output}", flush=True)
    print(f"predictions={args.predictions}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--linear-probe", type=Path, default=DEFAULT_LINEAR_PROBE)
    parser.add_argument("--reranker", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--test-features", type=Path, default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args(argv)
