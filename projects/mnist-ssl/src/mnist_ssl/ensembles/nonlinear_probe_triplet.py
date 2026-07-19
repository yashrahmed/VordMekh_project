"""Grid-search the best DINO and two I-JEPA nonlinear probe logits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from mnist_ssl.dinov2.nonlinear_probe import file_sha256
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import OUT_DIR


DEFAULT_DINO_PREDICTIONS = (
    OUT_DIR / "dinov2_nonlinear_probe_50ep" / "predictions.pt"
)
DEFAULT_IJEPA_300_PREDICTIONS = (
    OUT_DIR / "ijepa_nonlinear_probe_best300" / "predictions.pt"
)
DEFAULT_IJEPA_500_PREDICTIONS = (
    OUT_DIR / "ijepa_nonlinear_probe_best500" / "predictions.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "dino_ijepa300_500_nonlinear_ensemble_v1"


def simplex_weights(denominator: int) -> list[tuple[int, int, int]]:
    """Return every non-negative integer triplet summing to denominator."""

    if denominator < 1:
        raise ValueError("denominator must be positive")
    return [
        (dino, ijepa_300, denominator - dino - ijepa_300)
        for dino in range(denominator + 1)
        for ijepa_300 in range(denominator - dino + 1)
    ]


def refinement_weights(
    centers: Iterable[tuple[float, float, float]],
    *,
    denominator: int,
    radius: float,
) -> list[tuple[int, int, int]]:
    """Return a fine simplex grid around one or more coarse centers."""

    radius_units = round(radius * denominator)
    result: set[tuple[int, int, int]] = set()
    for center in centers:
        center_units = tuple(round(weight * denominator) for weight in center)
        for dino in range(
            max(0, center_units[0] - radius_units),
            min(denominator, center_units[0] + radius_units) + 1,
        ):
            for ijepa_300 in range(
                max(0, center_units[1] - radius_units),
                min(denominator - dino, center_units[1] + radius_units) + 1,
            ):
                ijepa_500 = denominator - dino - ijepa_300
                if abs(ijepa_500 - center_units[2]) <= radius_units:
                    result.add((dino, ijepa_300, ijepa_500))
    return sorted(result)


def _errors(predictions: torch.Tensor, labels: torch.Tensor) -> int:
    return int(predictions.ne(labels).sum().item())


def score_grid(
    scores: dict[str, torch.Tensor],
    *,
    canonical_labels: torch.Tensor,
    reviewed_labels: torch.Tensor,
    reviewed_mask: torch.Tensor,
    weights: Iterable[tuple[int, int, int]],
    denominator: int,
    method: str,
) -> list[dict[str, Any]]:
    """Score one simplex grid on canonical and reviewed labels together."""

    rows = []
    reviewed_count = int(reviewed_mask.sum().item())
    for dino_units, ijepa_300_units, ijepa_500_units in weights:
        combined = (
            dino_units * scores["dino"]
            + ijepa_300_units * scores["ijepa_300"]
            + ijepa_500_units * scores["ijepa_500"]
        )
        predictions = combined.argmax(dim=1)
        canonical_errors = _errors(predictions, canonical_labels)
        reviewed_errors = _errors(
            predictions[reviewed_mask],
            reviewed_labels[reviewed_mask],
        )
        rows.append(
            {
                "method": method,
                "dino_weight": dino_units / denominator,
                "ijepa_300_weight": ijepa_300_units / denominator,
                "ijepa_500_weight": ijepa_500_units / denominator,
                "canonical_errors": canonical_errors,
                "canonical_accuracy": 100.0
                * (1.0 - canonical_errors / len(canonical_labels)),
                "reviewed_errors": reviewed_errors,
                "reviewed_accuracy": 100.0
                * (1.0 - reviewed_errors / reviewed_count),
            }
        )
    return rows


def _best_row(rows: list[dict[str, Any]], error_field: str) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            row[error_field],
            -row["dino_weight"],
            -row["ijepa_300_weight"],
        ),
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize both label-view optima and their cross-view scores."""

    canonical_best = _best_row(rows, "canonical_errors")
    reviewed_best = _best_row(rows, "reviewed_errors")
    canonical_errors = canonical_best["canonical_errors"]
    reviewed_errors = reviewed_best["reviewed_errors"]
    canonical_ties = [
        row for row in rows if row["canonical_errors"] == canonical_errors
    ]
    reviewed_ties = [
        row for row in rows if row["reviewed_errors"] == reviewed_errors
    ]
    return {
        "canonical_winner": canonical_best,
        "reviewed_winner": reviewed_best,
        "canonical_exact_best_count": len(canonical_ties),
        "reviewed_exact_best_count": len(reviewed_ties),
        "canonical_exact_best_weights": [
            {
                "dino": row["dino_weight"],
                "ijepa_300": row["ijepa_300_weight"],
                "ijepa_500": row["ijepa_500_weight"],
            }
            for row in canonical_ties
        ],
        "reviewed_exact_best_weights": [
            {
                "dino": row["dino_weight"],
                "ijepa_300": row["ijepa_300_weight"],
                "ijepa_500": row["ijepa_500_weight"],
            }
            for row in reviewed_ties
        ],
    }


def _load_inputs(
    dino_path: Path,
    ijepa_300_path: Path,
    ijepa_500_path: Path,
    *,
    ijepa_300_epoch: int,
    ijepa_500_epoch: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    dino = torch.load(dino_path, map_location="cpu", weights_only=False)
    ijepa_300 = torch.load(
        ijepa_300_path,
        map_location="cpu",
        weights_only=False,
    )
    ijepa_500 = torch.load(
        ijepa_500_path,
        map_location="cpu",
        weights_only=False,
    )
    payloads = (dino, ijepa_300, ijepa_500)
    for field in ("canonical_labels", "reviewed_labels", "reviewed_include_mask"):
        if not all(torch.equal(payloads[0][field], item[field]) for item in payloads[1:]):
            raise ValueError(f"nonlinear prediction artifacts disagree on {field}")

    canonical_labels = dino["canonical_labels"].long()
    reviewed_labels = dino["reviewed_labels"].long()
    reviewed_mask = dino["reviewed_include_mask"].bool()
    current_policy = apply_mnist_test_label_policy(canonical_labels)
    if not torch.equal(current_policy.labels, reviewed_labels):
        raise ValueError("saved reviewed labels differ from the current policy")
    if not torch.equal(current_policy.include_mask, reviewed_mask):
        raise ValueError("saved reviewed exclusions differ from the current policy")

    logits = {
        "dino": dino["nonlinear_logits"].float(),
        "ijepa_300": ijepa_300["nonlinear_logits_by_epoch"][
            ijepa_300_epoch
        ].float(),
        "ijepa_500": ijepa_500["nonlinear_logits_by_epoch"][
            ijepa_500_epoch
        ].float(),
    }
    if any(member.shape != (10_000, 10) for member in logits.values()):
        raise ValueError("a nonlinear logit artifact has the wrong shape")
    return logits, canonical_labels, reviewed_labels, reviewed_mask


def _individual_metrics(
    logits: dict[str, torch.Tensor],
    canonical_labels: torch.Tensor,
    reviewed_labels: torch.Tensor,
    reviewed_mask: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = {}
    reviewed = {}
    wrong = {}
    for name, scores in logits.items():
        predictions = scores.argmax(dim=1)
        wrong[name] = predictions.ne(canonical_labels)
        canonical_errors = _errors(predictions, canonical_labels)
        reviewed_errors = _errors(
            predictions[reviewed_mask],
            reviewed_labels[reviewed_mask],
        )
        canonical[name] = {
            "errors": canonical_errors,
            "accuracy": 100.0 * (1.0 - canonical_errors / 10_000),
        }
        reviewed[name] = {
            "errors": reviewed_errors,
            "accuracy": 100.0
            * (1.0 - reviewed_errors / int(reviewed_mask.sum().item())),
        }
    canonical_shared = int(
        (wrong["dino"] & wrong["ijepa_300"] & wrong["ijepa_500"]).sum().item()
    )
    reviewed_wrong = {
        name: scores.argmax(dim=1)[reviewed_mask].ne(
            reviewed_labels[reviewed_mask]
        )
        for name, scores in logits.items()
    }
    reviewed_shared = int(
        (
            reviewed_wrong["dino"]
            & reviewed_wrong["ijepa_300"]
            & reviewed_wrong["ijepa_500"]
        )
        .sum()
        .item()
    )
    canonical["all_three_shared_errors"] = canonical_shared
    canonical["oracle_accuracy"] = 100.0 * (1.0 - canonical_shared / 10_000)
    reviewed_count = int(reviewed_mask.sum().item())
    reviewed["all_three_shared_errors"] = reviewed_shared
    reviewed["oracle_accuracy"] = 100.0 * (
        1.0 - reviewed_shared / reviewed_count
    )
    return canonical, reviewed


def _write_grid(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    coarse_path = args.output_dir / "coarse_grid.csv"
    refined_path = args.output_dir / "refined_grid.csv"
    if any(path.exists() for path in (summary_path, coarse_path, refined_path)):
        raise FileExistsError("refusing to overwrite nonlinear triplet results")

    logits, canonical_labels, reviewed_labels, reviewed_mask = _load_inputs(
        args.dino_predictions,
        args.ijepa_300_predictions,
        args.ijepa_500_predictions,
        ijepa_300_epoch=args.ijepa_300_epoch,
        ijepa_500_epoch=args.ijepa_500_epoch,
    )
    score_spaces = {
        "logit": logits,
        "probability": {
            name: F.softmax(scores, dim=1) for name, scores in logits.items()
        },
    }
    coarse_weights = simplex_weights(args.coarse_denominator)
    coarse_rows = []
    coarse_summaries = {}
    refinement_centers = {}
    for method, scores in score_spaces.items():
        rows = score_grid(
            scores,
            canonical_labels=canonical_labels,
            reviewed_labels=reviewed_labels,
            reviewed_mask=reviewed_mask,
            weights=coarse_weights,
            denominator=args.coarse_denominator,
            method=method,
        )
        coarse_rows.extend(rows)
        coarse_summaries[method] = summarize_rows(rows)
        minimum_errors = {
            error_field: min(row[error_field] for row in rows)
            for error_field in ("canonical_errors", "reviewed_errors")
        }
        refinement_centers[method] = {
            (
                row["dino_weight"],
                row["ijepa_300_weight"],
                row["ijepa_500_weight"],
            )
            for error_field in ("canonical_errors", "reviewed_errors")
            for row in rows
            if row[error_field] == minimum_errors[error_field]
        }

    refined_rows = []
    refined_summaries = {}
    for method, scores in score_spaces.items():
        weights = refinement_weights(
            refinement_centers[method],
            denominator=args.refined_denominator,
            radius=args.refinement_radius,
        )
        rows = score_grid(
            scores,
            canonical_labels=canonical_labels,
            reviewed_labels=reviewed_labels,
            reviewed_mask=reviewed_mask,
            weights=weights,
            denominator=args.refined_denominator,
            method=method,
        )
        refined_rows.extend(rows)
        refined_summaries[method] = summarize_rows(rows)

    canonical_individual, reviewed_individual = _individual_metrics(
        logits,
        canonical_labels,
        reviewed_labels,
        reviewed_mask,
    )
    result = {
        "protocol": {
            "selection": "test-tuned diagnostic grid",
            "methods": ["raw weighted logits", "weighted softmax probabilities"],
            "coarse_step": 1 / args.coarse_denominator,
            "refined_step": 1 / args.refined_denominator,
            "refinement_radius": args.refinement_radius,
            "dino_probe_epoch": 50,
            "ijepa_300_backbone_epoch": 300,
            "ijepa_300_probe_epoch": args.ijepa_300_epoch,
            "ijepa_500_backbone_epoch": 500,
            "ijepa_500_probe_epoch": args.ijepa_500_epoch,
            "label_alignment_verified": True,
            "input_artifacts": {
                "dino": {
                    "path": str(args.dino_predictions),
                    "sha256": file_sha256(args.dino_predictions),
                },
                "ijepa_300": {
                    "path": str(args.ijepa_300_predictions),
                    "sha256": file_sha256(args.ijepa_300_predictions),
                },
                "ijepa_500": {
                    "path": str(args.ijepa_500_predictions),
                    "sha256": file_sha256(args.ijepa_500_predictions),
                },
            },
        },
        "canonical_individual_and_oracle": canonical_individual,
        "reviewed_individual_and_oracle": reviewed_individual,
        "coarse_grid": coarse_summaries,
        "refined_grid": refined_summaries,
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    _write_grid(coarse_path, coarse_rows)
    _write_grid(refined_path, refined_rows)

    for method, summary in refined_summaries.items():
        canonical = summary["canonical_winner"]
        reviewed = summary["reviewed_winner"]
        print(
            f"method={method} canonical "
            f"weights={canonical['dino_weight']:.3f}/"
            f"{canonical['ijepa_300_weight']:.3f}/"
            f"{canonical['ijepa_500_weight']:.3f} "
            f"errors={canonical['canonical_errors']} "
            f"reviewed_at_winner={canonical['reviewed_errors']}",
            flush=True,
        )
        print(
            f"method={method} reviewed "
            f"weights={reviewed['dino_weight']:.3f}/"
            f"{reviewed['ijepa_300_weight']:.3f}/"
            f"{reviewed['ijepa_500_weight']:.3f} "
            f"errors={reviewed['reviewed_errors']} "
            f"canonical_at_winner={reviewed['canonical_errors']}",
            flush=True,
        )
    print(f"summary={summary_path}", flush=True)
    print(f"coarse_grid={coarse_path}", flush=True)
    print(f"refined_grid={refined_path}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dino-predictions",
        type=Path,
        default=DEFAULT_DINO_PREDICTIONS,
    )
    parser.add_argument(
        "--ijepa-300-predictions",
        type=Path,
        default=DEFAULT_IJEPA_300_PREDICTIONS,
    )
    parser.add_argument(
        "--ijepa-500-predictions",
        type=Path,
        default=DEFAULT_IJEPA_500_PREDICTIONS,
    )
    parser.add_argument("--ijepa-300-epoch", type=int, default=75)
    parser.add_argument("--ijepa-500-epoch", type=int, default=75)
    parser.add_argument("--coarse-denominator", type=int, default=100)
    parser.add_argument("--refined-denominator", type=int, default=1000)
    parser.add_argument("--refinement-radius", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)
