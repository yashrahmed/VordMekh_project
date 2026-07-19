"""Grid-search the best DINO and I-JEPA-500 nonlinear probe logits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from mnist_ssl.dinov2.nonlinear_probe import file_sha256
from mnist_ssl.ensembles.dino_ijepa import evaluate_logits
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import OUT_DIR


DEFAULT_DINO_PREDICTIONS = (
    OUT_DIR / "dinov2_nonlinear_probe_50ep" / "predictions.pt"
)
DEFAULT_IJEPA_PREDICTIONS = (
    OUT_DIR / "ijepa_nonlinear_probe_best500" / "predictions.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "dino_ijepa500_nonlinear_ensemble_v1"


def summarize_plateaus(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exact-best and within-one-error weight ranges by method."""

    result = {}
    for method in ("logit", "probability"):
        method_rows = [row for row in rows if row["method"] == method]
        best_errors = min(row["errors"] for row in method_rows)
        best = [row for row in method_rows if row["errors"] == best_errors]
        within_one = [
            row for row in method_rows if row["errors"] <= best_errors + 1
        ]
        result[method] = {
            "best_errors": best_errors,
            "best_accuracy": best[0]["test_accuracy"],
            "exact_best_dino_weights": [
                row["dino_weight"] for row in best
            ],
            "exact_best_weight_min": min(row["dino_weight"] for row in best),
            "exact_best_weight_max": max(row["dino_weight"] for row in best),
            "within_one_error_weight_min": min(
                row["dino_weight"] for row in within_one
            ),
            "within_one_error_weight_max": max(
                row["dino_weight"] for row in within_one
            ),
        }
    return result


def _row_at(
    rows: list[dict[str, Any]],
    *,
    method: str,
    dino_weight: float,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["method"] == method
        and abs(row["dino_weight"] - dino_weight) < 1e-9
    )


def _cross_view_scores(
    source: dict[str, Any],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for method, row in source["best_test_tuned_diagnostic"].items():
        result[method] = _row_at(
            target_rows,
            method=method,
            dino_weight=row["dino_weight"],
        )
    return result


def _load_inputs(
    dino_path: Path,
    ijepa_path: Path,
    *,
    ijepa_epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dino = torch.load(dino_path, map_location="cpu", weights_only=False)
    ijepa = torch.load(ijepa_path, map_location="cpu", weights_only=False)
    required = ("canonical_labels", "reviewed_labels", "reviewed_include_mask")
    for field in required:
        if not torch.equal(dino[field], ijepa[field]):
            raise ValueError(f"nonlinear prediction artifacts disagree on {field}")

    dino_logits = dino["nonlinear_logits"].float()
    ijepa_logits = ijepa["nonlinear_logits_by_epoch"][ijepa_epoch].float()
    canonical_labels = dino["canonical_labels"].long()
    reviewed_labels = dino["reviewed_labels"].long()
    reviewed_mask = dino["reviewed_include_mask"].bool()
    if dino_logits.shape != (10_000, 10):
        raise ValueError("DINO nonlinear logits have the wrong shape")
    if ijepa_logits.shape != (10_000, 10):
        raise ValueError("I-JEPA nonlinear logits have the wrong shape")

    current_policy = apply_mnist_test_label_policy(canonical_labels)
    if not torch.equal(current_policy.labels, reviewed_labels):
        raise ValueError("saved reviewed labels differ from the current policy")
    if not torch.equal(current_policy.include_mask, reviewed_mask):
        raise ValueError("saved reviewed exclusions differ from the current policy")
    return (
        dino_logits,
        ijepa_logits,
        canonical_labels,
        reviewed_labels,
        reviewed_mask,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    grid_path = args.output_dir / "grid.csv"
    if summary_path.exists() or grid_path.exists():
        raise FileExistsError("refusing to overwrite nonlinear ensemble results")
    if args.step <= 0 or 100 % args.step:
        raise ValueError("step must be a positive divisor of 100")

    (
        dino_logits,
        ijepa_logits,
        canonical_labels,
        reviewed_labels,
        reviewed_mask,
    ) = _load_inputs(
        args.dino_predictions,
        args.ijepa_predictions,
        ijepa_epoch=args.ijepa_epoch,
    )
    canonical, canonical_rows = evaluate_logits(
        dino_logits,
        ijepa_logits,
        canonical_labels,
        args.step,
    )
    reviewed, reviewed_rows = evaluate_logits(
        dino_logits[reviewed_mask],
        ijepa_logits[reviewed_mask],
        reviewed_labels[reviewed_mask],
        args.step,
    )
    result = {
        "protocol": {
            "selection": "test-tuned diagnostic grid",
            "grid_step": args.step / 100,
            "methods": ["raw weighted logits", "weighted softmax probabilities"],
            "dino_probe_epoch": 50,
            "ijepa_backbone_epoch": 500,
            "ijepa_probe_epoch": args.ijepa_epoch,
            "dino_predictions": str(args.dino_predictions),
            "dino_predictions_sha256": file_sha256(args.dino_predictions),
            "ijepa_predictions": str(args.ijepa_predictions),
            "ijepa_predictions_sha256": file_sha256(args.ijepa_predictions),
            "label_alignment_verified": True,
        },
        "canonical_test": {
            **canonical,
            "plateaus": summarize_plateaus(canonical_rows),
            "reviewed_scores_at_canonical_winners": _cross_view_scores(
                canonical,
                reviewed_rows,
            ),
        },
        "reviewed_test": {
            **reviewed,
            "plateaus": summarize_plateaus(reviewed_rows),
            "canonical_scores_at_reviewed_winners": _cross_view_scores(
                reviewed,
                canonical_rows,
            ),
        },
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    with grid_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "label_view",
                "method",
                "dino_weight",
                "ijepa_weight",
                "test_accuracy",
                "errors",
                "selection",
            ),
        )
        writer.writeheader()
        for label_view, rows in (
            ("canonical", canonical_rows),
            ("reviewed", reviewed_rows),
        ):
            for row in rows:
                writer.writerow({"label_view": label_view, **row})

    for label_view, evaluation in (
        ("canonical", canonical),
        ("reviewed", reviewed),
    ):
        print(
            f"{label_view} dino_errors={evaluation['dino']['errors']} "
            f"ijepa_errors={evaluation['ijepa']['errors']} "
            f"shared_errors={evaluation['error_complementarity']['shared_errors']}",
            flush=True,
        )
        for method, row in evaluation["best_test_tuned_diagnostic"].items():
            print(
                f"{label_view} method={method} "
                f"dino_weight={row['dino_weight']:.2f} "
                f"ijepa_weight={row['ijepa_weight']:.2f} "
                f"errors={row['errors']} accuracy={row['test_accuracy']:.5f}%",
                flush=True,
            )
    print(f"summary={summary_path}", flush=True)
    print(f"grid={grid_path}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dino-predictions",
        type=Path,
        default=DEFAULT_DINO_PREDICTIONS,
    )
    parser.add_argument(
        "--ijepa-predictions",
        type=Path,
        default=DEFAULT_IJEPA_PREDICTIONS,
    )
    parser.add_argument("--ijepa-epoch", type=int, default=75)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)
