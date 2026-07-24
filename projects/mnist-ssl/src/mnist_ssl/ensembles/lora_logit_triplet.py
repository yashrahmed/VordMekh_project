"""Tune the complete three-family LoRA logit ensemble on MNIST train.

All three final nonlinear LoRA checkpoints from the audited backbone matrix are
included: DINOv2, I-JEPA-300, and I-JEPA-500. A coarse-to-fine raw-logit
simplex search uses only canonical MNIST training labels. The selected weights
are frozen before the test split is loaded and evaluated once.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from mnist_ssl.dinov2.nonlinear_probe import classification_metrics, file_sha256
from mnist_ssl.dinov2.train import pick_device
from mnist_ssl.ensembles.lora_logit_pair import (
    Candidate,
    DEFAULT_DINO_ADAPTER,
    DEFAULT_DINO_BACKBONE,
    DEFAULT_IJEPA_ADAPTER,
    DEFAULT_IJEPA_BACKBONE,
    LORA_ROOT,
    extract_candidate_logits,
)
from mnist_ssl.ensembles.nonlinear_probe_triplet import (
    refinement_weights,
    simplex_weights,
)
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.lora_probe import BackboneDefinition, load_split
from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR


MODEL_NAMES = ("dinov2", "ijepa_300", "ijepa_500")
DEFAULT_IJEPA_300_BACKBONE = (
    MODELS_DIR / "ijepa_mnist_custom_ijepa_p7_56_t48_300ep.pt"
)
DEFAULT_IJEPA_300_ADAPTER = (
    LORA_ROOT / "ijepa-300" / "nonlinear" / "epoch0150.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "lora_logit_triplet_2026-07-23"


def _candidates(args: argparse.Namespace) -> tuple[Candidate, ...]:
    return (
        Candidate(
            "dinov2",
            BackboneDefinition(
                "dinov2-best",
                "dinov2",
                args.dino_backbone,
                "cls",
                True,
            ),
            args.dino_adapter,
        ),
        Candidate(
            "ijepa_300",
            BackboneDefinition(
                "ijepa-300",
                "ijepa",
                args.ijepa_300_backbone,
                "flatten",
                False,
            ),
            args.ijepa_300_adapter,
        ),
        Candidate(
            "ijepa_500",
            BackboneDefinition(
                "ijepa-500",
                "ijepa",
                args.ijepa_500_backbone,
                "flatten",
                False,
            ),
            args.ijepa_500_adapter,
        ),
    )


def _source_signature(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "dino_backbone": args.dino_backbone,
        "dino_adapter": args.dino_adapter,
        "ijepa_300_backbone": args.ijepa_300_backbone,
        "ijepa_300_adapter": args.ijepa_300_adapter,
        "ijepa_500_backbone": args.ijepa_500_backbone,
        "ijepa_500_adapter": args.ijepa_500_adapter,
    }
    return {f"{name}_sha256": file_sha256(path) for name, path in paths.items()}


def _load_or_generate_training_logits(
    args: argparse.Namespace,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    signature = _source_signature(args)
    if path.exists():
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("signature") != signature:
            raise ValueError("training-logit artifact signature mismatch")
        print(f"reused_training_logits={path}", flush=True)
    else:
        print("loading_training_split=true", flush=True)
        images, labels = load_split(
            args.dataset_dir,
            train=True,
            batch_size=args.data_batch_size,
            workers=args.workers,
            subset=0,
        )
        outputs = []
        fingerprints = {}
        for candidate in _candidates(args):
            print(f"extracting_training_logits={candidate.name}", flush=True)
            logits, fingerprint = extract_candidate_logits(
                candidate,
                images,
                device=device,
                batch_size=args.eval_batch_size,
            )
            outputs.append(logits)
            fingerprints[candidate.name] = fingerprint
        artifact = {
            "signature": signature,
            "model_order": MODEL_NAMES,
            "labels": labels,
            "logits": torch.stack(outputs, dim=1),
            "base_fingerprints": fingerprints,
        }
        torch.save(artifact, path)
        print(f"training_logits={path}", flush=True)
    if tuple(artifact.get("model_order", ())) != MODEL_NAMES:
        raise ValueError("training-logit model order mismatch")
    if artifact["logits"].shape != (60_000, 3, 10):
        raise ValueError("training logits must have shape [60000,3,10]")
    if artifact["labels"].shape != (60_000,):
        raise ValueError("training labels must have shape [60000]")
    return artifact


def score_grid(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Iterable[tuple[int, int, int]],
    *,
    denominator: int,
    phase: str,
) -> list[dict[str, Any]]:
    predictions = logits.argmax(dim=-1)
    agreement = predictions.eq(predictions[:, :1]).all(dim=1)
    fixed_errors = int(predictions[agreement, 0].ne(labels[agreement]).sum())
    variable_logits = logits[~agreement]
    variable_labels = labels[~agreement]
    rows = []
    for dino, ijepa_300, ijepa_500 in weights:
        combined = (
            dino * variable_logits[:, 0]
            + ijepa_300 * variable_logits[:, 1]
            + ijepa_500 * variable_logits[:, 2]
        )
        errors = fixed_errors + int(combined.argmax(dim=1).ne(variable_labels).sum())
        rows.append(
            {
                "phase": phase,
                "dino_weight": dino / denominator,
                "ijepa_300_weight": ijepa_300 / denominator,
                "ijepa_500_weight": ijepa_500 / denominator,
                "train_errors": errors,
                "train_accuracy_percent": 100.0 * (1.0 - errors / len(labels)),
            }
        )
    return rows


def _distance_from_equal(row: dict[str, Any]) -> float:
    return sum(
        (row[field] - 1.0 / 3.0) ** 2
        for field in (
            "dino_weight",
            "ijepa_300_weight",
            "ijepa_500_weight",
        )
    )


def select_representative(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best_errors = min(row["train_errors"] for row in rows)
    exact_best = [row for row in rows if row["train_errors"] == best_errors]
    selected = min(
        exact_best,
        key=lambda row: (
            _distance_from_equal(row),
            -row["dino_weight"],
            -row["ijepa_300_weight"],
        ),
    )
    return selected, exact_best


def _grid_search(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coarse_rows = score_grid(
        logits,
        labels,
        simplex_weights(args.coarse_denominator),
        denominator=args.coarse_denominator,
        phase="coarse",
    )
    coarse_selected, coarse_best = select_representative(coarse_rows)
    if coarse_selected["train_errors"] == 0:
        centers = [
            (
                coarse_selected["dino_weight"],
                coarse_selected["ijepa_300_weight"],
                coarse_selected["ijepa_500_weight"],
            )
        ]
    else:
        centers = [
            (
                row["dino_weight"],
                row["ijepa_300_weight"],
                row["ijepa_500_weight"],
            )
            for row in coarse_best
        ]
    refined_rows = score_grid(
        logits,
        labels,
        refinement_weights(
            centers,
            denominator=args.refined_denominator,
            radius=args.refinement_radius,
        ),
        denominator=args.refined_denominator,
        phase="refined",
    )
    refined_selected, refined_best = select_representative(refined_rows)
    selection = {
        "coarse_weights_evaluated": len(coarse_rows),
        "coarse_best_errors": coarse_selected["train_errors"],
        "coarse_exact_best_count": len(coarse_best),
        "coarse_selected": coarse_selected,
        "refined_weights_evaluated": len(refined_rows),
        "refined_exact_best_count": len(refined_best),
        "selected": refined_selected,
        "selection_rule": (
            "minimum canonical-train errors; closest-to-equal weights resolve ties; "
            "higher DINOv2 then I-JEPA-300 weight resolves an exact distance tie"
        ),
    }
    return selection, coarse_rows + refined_rows


def _metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    return classification_metrics(logits, labels, include_mask)


def _evaluate_test(
    args: argparse.Namespace,
    selected: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print("loading_test_split_after_weight_selection=true", flush=True)
    images, labels = load_split(
        args.dataset_dir,
        train=False,
        batch_size=args.data_batch_size,
        workers=args.workers,
        subset=0,
    )
    outputs = []
    fingerprints = {}
    for candidate in _candidates(args):
        print(f"extracting_test_logits={candidate.name}", flush=True)
        logits, fingerprint = extract_candidate_logits(
            candidate,
            images,
            device=device,
            batch_size=args.eval_batch_size,
        )
        outputs.append(logits)
        fingerprints[candidate.name] = fingerprint
    logits = torch.stack(outputs, dim=1)
    weights = torch.tensor(
        [
            selected["dino_weight"],
            selected["ijepa_300_weight"],
            selected["ijepa_500_weight"],
        ],
        dtype=logits.dtype,
    )
    ensemble_logits = (logits * weights[None, :, None]).sum(dim=1)
    reviewed = apply_mnist_test_label_policy(labels)
    evaluation = {
        "members": {
            name: {
                "canonical_test": _metrics(logits[:, index], labels),
                "reviewed_test": _metrics(
                    logits[:, index],
                    reviewed.labels,
                    reviewed.include_mask,
                ),
            }
            for index, name in enumerate(MODEL_NAMES)
        },
        "ensemble": {
            "canonical_test": _metrics(ensemble_logits, labels),
            "reviewed_test": _metrics(
                ensemble_logits,
                reviewed.labels,
                reviewed.include_mask,
            ),
        },
    }
    return evaluation, {
        "reviewed_label_policy": reviewed.metadata,
        "test_base_fingerprints": fingerprints,
    }


def _write_grid(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.coarse_denominator < 1 or args.refined_denominator < 1:
        raise ValueError("grid denominators must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.output_dir / "training_logits.pt"
    grid_path = args.output_dir / "grid.csv"
    summary_path = args.output_dir / "summary.json"
    for output in (grid_path, summary_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    device = torch.device(args.device) if args.device else pick_device()
    training = _load_or_generate_training_logits(args, training_path, device)
    selection, rows = _grid_search(
        training["logits"].float(),
        training["labels"].long(),
        args,
    )
    selected = selection["selected"]
    print(
        f"selected_weights={selected['dino_weight']:.3f}/"
        f"{selected['ijepa_300_weight']:.3f}/"
        f"{selected['ijepa_500_weight']:.3f} "
        f"train_errors={selected['train_errors']}",
        flush=True,
    )

    evaluation, test_metadata = _evaluate_test(args, selected, device)
    _write_grid(grid_path, rows)
    result = {
        "schema_version": 1,
        "name": "train-selected-lora-logit-triplet",
        "protocol": {
            "candidate_selection": (
                "complete three-family set of final nonlinear LoRA checkpoints "
                "from the previously observed backbone matrix"
            ),
            "selection_split": "MNIST train, canonical labels",
            "score_space": "raw logits",
            "test_loaded_after_weight_selection": True,
            "test_labels_used_for_weight_selection": False,
            "model_order": MODEL_NAMES,
            "candidate_epochs": {name: 150 for name in MODEL_NAMES},
            "coarse_step": 1.0 / args.coarse_denominator,
            "refined_step": 1.0 / args.refined_denominator,
            "refinement_radius": args.refinement_radius,
            "source_sha256": training["signature"],
            "training_logits_sha256": file_sha256(training_path),
        },
        "selection": selection,
        "evaluation": evaluation,
        **test_metadata,
    }
    result["protocol"]["grid_sha256"] = file_sha256(grid_path)
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    ensemble = evaluation["ensemble"]
    print(
        f"canonical_test_errors={ensemble['canonical_test']['errors']} "
        f"canonical_test_accuracy={ensemble['canonical_test']['accuracy']:.5%}",
        flush=True,
    )
    print(
        f"reviewed_test_errors={ensemble['reviewed_test']['errors']} "
        f"reviewed_test_accuracy={ensemble['reviewed_test']['accuracy']:.5%}",
        flush=True,
    )
    print(f"summary={summary_path}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-backbone", type=Path, default=DEFAULT_DINO_BACKBONE)
    parser.add_argument("--dino-adapter", type=Path, default=DEFAULT_DINO_ADAPTER)
    parser.add_argument(
        "--ijepa-300-backbone",
        type=Path,
        default=DEFAULT_IJEPA_300_BACKBONE,
    )
    parser.add_argument(
        "--ijepa-300-adapter",
        type=Path,
        default=DEFAULT_IJEPA_300_ADAPTER,
    )
    parser.add_argument(
        "--ijepa-500-backbone",
        type=Path,
        default=DEFAULT_IJEPA_BACKBONE,
    )
    parser.add_argument(
        "--ijepa-500-adapter",
        type=Path,
        default=DEFAULT_IJEPA_ADAPTER,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coarse-denominator", type=int, default=100)
    parser.add_argument("--refined-denominator", type=int, default=1000)
    parser.add_argument("--refinement-radius", type=float, default=0.02)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--data-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args(argv)
