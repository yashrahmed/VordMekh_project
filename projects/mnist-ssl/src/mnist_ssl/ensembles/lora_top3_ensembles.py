"""Tune the top three observed LoRA classifiers as triplet and pair ensembles.

The candidate set is intentionally test-informed and exploratory: I-JEPA-500
nonlinear epoch 150, DINOv2 nonlinear epoch 50, and DINOv2 nonlinear epoch 150
are the three best individual measurements in the audited LoRA matrix. Raw
logit weights for the full triplet and all three pairs are selected using only
canonical MNIST training labels. Every rule is frozen before the test split is
loaded and evaluated once.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from mnist_ssl.dinov2.nonlinear_probe import classification_metrics, file_sha256
from mnist_ssl.dinov2.train import pick_device
from mnist_ssl.ensembles.nonlinear_probe_triplet import (
    refinement_weights as triplet_refinement_weights,
)
from mnist_ssl.ensembles.nonlinear_probe_triplet import simplex_weights
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.lora import (
    capture_base_tensors,
    load_adapter_state_dict,
    tensor_fingerprint,
)
from mnist_ssl.lora_probe import (
    BackboneDefinition,
    inject_transformer_lora,
    load_backbone,
    load_split,
    make_head,
    predict,
)
from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR


MODEL_NAMES = ("ijepa_500_e150", "dinov2_e50", "dinov2_e150")
PAIR_DEFINITIONS = (
    ("ijepa_500_e150__dinov2_e50", (0, 1)),
    ("ijepa_500_e150__dinov2_e150", (0, 2)),
    ("dinov2_e50__dinov2_e150", (1, 2)),
)
LORA_ROOT = OUT_DIR / "lora_backbone_probes_2026-07-20"
DEFAULT_DINO_BACKBONE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt"
)
DEFAULT_DINO_E50_ADAPTER = (
    LORA_ROOT / "dinov2-best" / "nonlinear" / "epoch0050.pt"
)
DEFAULT_DINO_E150_ADAPTER = (
    LORA_ROOT / "dinov2-best" / "nonlinear" / "epoch0150.pt"
)
DEFAULT_IJEPA_500_BACKBONE = (
    MODELS_DIR / "ijepa_mnist_custom_ijepa_p7_56_t48_500ep.pt"
)
DEFAULT_IJEPA_500_E150_ADAPTER = (
    LORA_ROOT / "ijepa-500" / "nonlinear" / "epoch0150.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "lora_top3_ensembles_2026-07-23"
WEIGHT_FIELDS = tuple(f"{name}_weight" for name in MODEL_NAMES)


@dataclass(frozen=True)
class Candidate:
    name: str
    epoch: int
    definition: BackboneDefinition
    adapter_checkpoint: Path


@dataclass
class LoadedCandidate:
    feature_extractor: nn.Module
    head: nn.Module
    normalize_input: bool
    base_fingerprint: str


def _candidates(args: argparse.Namespace) -> tuple[Candidate, ...]:
    return (
        Candidate(
            "ijepa_500_e150",
            150,
            BackboneDefinition(
                "ijepa-500",
                "ijepa",
                args.ijepa_500_backbone,
                "flatten",
                False,
            ),
            args.ijepa_500_e150_adapter,
        ),
        Candidate(
            "dinov2_e50",
            50,
            BackboneDefinition(
                "dinov2-best",
                "dinov2",
                args.dino_backbone,
                "cls",
                True,
            ),
            args.dino_e50_adapter,
        ),
        Candidate(
            "dinov2_e150",
            150,
            BackboneDefinition(
                "dinov2-best",
                "dinov2",
                args.dino_backbone,
                "cls",
                True,
            ),
            args.dino_e150_adapter,
        ),
    )


def _source_signature(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "dino_backbone": args.dino_backbone,
        "dino_e50_adapter": args.dino_e50_adapter,
        "dino_e150_adapter": args.dino_e150_adapter,
        "ijepa_500_backbone": args.ijepa_500_backbone,
        "ijepa_500_e150_adapter": args.ijepa_500_e150_adapter,
    }
    return {f"{name}_sha256": file_sha256(path) for name, path in paths.items()}


def _load_candidate(candidate: Candidate, device: torch.device) -> LoadedCandidate:
    checkpoint = torch.load(
        candidate.adapter_checkpoint,
        map_location=device,
        weights_only=False,
    )
    signature = checkpoint.get("signature", {})
    expected = {
        "backbone": candidate.definition.name,
        "probe_type": "nonlinear",
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "hidden_dim": 64,
        "dropout": 0.1,
    }
    for field, value in expected.items():
        if signature.get(field) != value:
            raise ValueError(
                f"{candidate.name} adapter has unexpected {field}: "
                f"{signature.get(field)!r} != {value!r}"
            )
    if checkpoint.get("epoch") != candidate.epoch:
        raise ValueError(
            f"{candidate.name} adapter epoch mismatch: "
            f"{checkpoint.get('epoch')} != {candidate.epoch}"
        )
    backbone_hash = file_sha256(candidate.definition.checkpoint)
    if signature.get("checkpoint_sha256") != backbone_hash:
        raise ValueError(f"{candidate.name} adapter/backbone hash mismatch")

    loaded = load_backbone(candidate.definition, device)
    base_tensors = capture_base_tensors(loaded.fingerprint_target)
    base_fingerprint = tensor_fingerprint(base_tensors)
    handles = inject_transformer_lora(
        loaded.adapter_target,
        candidate.definition.family,
        rank=signature["lora_rank"],
        alpha=signature["lora_alpha"],
    )
    load_adapter_state_dict(handles, checkpoint["adapter_state_dict"])
    head = make_head(
        signature["probe_type"],
        loaded.feature_dim,
        signature["hidden_dim"],
        signature["dropout"],
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    for parameter in loaded.feature_extractor.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    loaded.feature_extractor.eval()
    head.eval()
    if tensor_fingerprint(base_tensors) != base_fingerprint:
        raise RuntimeError(f"{candidate.name} base tensors changed while loading LoRA")
    return LoadedCandidate(
        feature_extractor=loaded.feature_extractor,
        head=head,
        normalize_input=candidate.definition.normalize_input,
        base_fingerprint=base_fingerprint,
    )


@torch.no_grad()
def _extract_candidate_logits(
    candidate: Candidate,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, str]:
    loaded = _load_candidate(candidate, device)
    logits = predict(
        loaded.feature_extractor,
        loaded.head,
        images,
        device=device,
        batch_size=batch_size,
        normalize_input=loaded.normalize_input,
    )
    return logits, loaded.base_fingerprint


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
            logits, fingerprint = _extract_candidate_logits(
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


def pair_weights(denominator: int) -> list[tuple[int, int]]:
    if denominator < 1:
        raise ValueError("denominator must be positive")
    return [(first, denominator - first) for first in range(denominator + 1)]


def pair_refinement_weights(
    centers: Iterable[float],
    *,
    denominator: int,
    radius: float,
) -> list[tuple[int, int]]:
    if denominator < 1 or radius < 0:
        raise ValueError("denominator must be positive and radius non-negative")
    radius_units = round(radius * denominator)
    first_weights = set()
    for center in centers:
        center_units = round(center * denominator)
        lower = max(0, center_units - radius_units)
        upper = min(denominator, center_units + radius_units)
        first_weights.update(range(lower, upper + 1))
    return [
        (first, denominator - first)
        for first in sorted(first_weights)
    ]


def _empty_weights() -> dict[str, float]:
    return {field: 0.0 for field in WEIGHT_FIELDS}


def score_pair_grid(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Iterable[tuple[int, int]],
    *,
    denominator: int,
    phase: str,
    pair_name: str,
    indices: tuple[int, int],
) -> list[dict[str, Any]]:
    pair_logits = logits[:, indices]
    predictions = pair_logits.argmax(dim=-1)
    agreement = predictions[:, 0].eq(predictions[:, 1])
    fixed_errors = int(predictions[agreement, 0].ne(labels[agreement]).sum())
    variable_logits = pair_logits[~agreement]
    variable_labels = labels[~agreement]
    rows = []
    for first, second in weights:
        combined = first * variable_logits[:, 0] + second * variable_logits[:, 1]
        errors = fixed_errors + int(combined.argmax(dim=1).ne(variable_labels).sum())
        row = {
            "ensemble": pair_name,
            "phase": phase,
            **_empty_weights(),
            "train_errors": errors,
            "train_accuracy_percent": 100.0 * (1.0 - errors / len(labels)),
        }
        row[WEIGHT_FIELDS[indices[0]]] = first / denominator
        row[WEIGHT_FIELDS[indices[1]]] = second / denominator
        rows.append(row)
    return rows


def score_triplet_grid(
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
    for first, second, third in weights:
        combined = (
            first * variable_logits[:, 0]
            + second * variable_logits[:, 1]
            + third * variable_logits[:, 2]
        )
        errors = fixed_errors + int(combined.argmax(dim=1).ne(variable_labels).sum())
        rows.append(
            {
                "ensemble": "top3_triplet",
                "phase": phase,
                WEIGHT_FIELDS[0]: first / denominator,
                WEIGHT_FIELDS[1]: second / denominator,
                WEIGHT_FIELDS[2]: third / denominator,
                "train_errors": errors,
                "train_accuracy_percent": 100.0 * (1.0 - errors / len(labels)),
            }
        )
    return rows


def _select_pair(
    rows: list[dict[str, Any]],
    indices: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best_errors = min(row["train_errors"] for row in rows)
    exact_best = [row for row in rows if row["train_errors"] == best_errors]
    first_field = WEIGHT_FIELDS[indices[0]]
    selected = min(
        exact_best,
        key=lambda row: (
            (row[first_field] - 0.5) ** 2,
            -row[first_field],
        ),
    )
    return selected, exact_best


def _triplet_distance_from_equal(row: dict[str, Any]) -> float:
    return sum((row[field] - 1.0 / 3.0) ** 2 for field in WEIGHT_FIELDS)


def _select_triplet(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best_errors = min(row["train_errors"] for row in rows)
    exact_best = [row for row in rows if row["train_errors"] == best_errors]
    selected = min(
        exact_best,
        key=lambda row: (
            _triplet_distance_from_equal(row),
            -row[WEIGHT_FIELDS[0]],
            -row[WEIGHT_FIELDS[1]],
        ),
    )
    return selected, exact_best


def _grid_pair(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pair_name: str,
    indices: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coarse_rows = score_pair_grid(
        logits,
        labels,
        pair_weights(args.coarse_denominator),
        denominator=args.coarse_denominator,
        phase="coarse",
        pair_name=pair_name,
        indices=indices,
    )
    coarse_selected, coarse_best = _select_pair(coarse_rows, indices)
    first_field = WEIGHT_FIELDS[indices[0]]
    centers = (
        [coarse_selected[first_field]]
        if coarse_selected["train_errors"] == 0
        else [row[first_field] for row in coarse_best]
    )
    refined_rows = score_pair_grid(
        logits,
        labels,
        pair_refinement_weights(
            centers,
            denominator=args.refined_denominator,
            radius=args.refinement_radius,
        ),
        denominator=args.refined_denominator,
        phase="refined",
        pair_name=pair_name,
        indices=indices,
    )
    refined_selected, refined_best = _select_pair(refined_rows, indices)
    return (
        {
            "members": [MODEL_NAMES[index] for index in indices],
            "coarse_weights_evaluated": len(coarse_rows),
            "coarse_best_errors": coarse_selected["train_errors"],
            "coarse_exact_best_count": len(coarse_best),
            "coarse_selected": coarse_selected,
            "refined_weights_evaluated": len(refined_rows),
            "refined_exact_best_count": len(refined_best),
            "selected": refined_selected,
            "selection_rule": (
                "minimum canonical-train errors; closest-to-equal weights resolve "
                "ties; higher first-member weight resolves an exact distance tie"
            ),
        },
        coarse_rows + refined_rows,
    )


def _grid_triplet(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coarse_rows = score_triplet_grid(
        logits,
        labels,
        simplex_weights(args.coarse_denominator),
        denominator=args.coarse_denominator,
        phase="coarse",
    )
    coarse_selected, coarse_best = _select_triplet(coarse_rows)
    centers = (
        [tuple(coarse_selected[field] for field in WEIGHT_FIELDS)]
        if coarse_selected["train_errors"] == 0
        else [tuple(row[field] for field in WEIGHT_FIELDS) for row in coarse_best]
    )
    refined_rows = score_triplet_grid(
        logits,
        labels,
        triplet_refinement_weights(
            centers,
            denominator=args.refined_denominator,
            radius=args.refinement_radius,
        ),
        denominator=args.refined_denominator,
        phase="refined",
    )
    refined_selected, refined_best = _select_triplet(refined_rows)
    return (
        {
            "members": list(MODEL_NAMES),
            "coarse_weights_evaluated": len(coarse_rows),
            "coarse_best_errors": coarse_selected["train_errors"],
            "coarse_exact_best_count": len(coarse_best),
            "coarse_selected": coarse_selected,
            "refined_weights_evaluated": len(refined_rows),
            "refined_exact_best_count": len(refined_best),
            "selected": refined_selected,
            "selection_rule": (
                "minimum canonical-train errors; closest-to-equal weights resolve "
                "ties; ranked model order resolves an exact distance tie"
            ),
        },
        coarse_rows + refined_rows,
    )


def _combined_logits(logits: torch.Tensor, row: dict[str, Any]) -> torch.Tensor:
    weights = torch.tensor(
        [row[field] for field in WEIGHT_FIELDS],
        dtype=logits.dtype,
    )
    return (logits * weights[None, :, None]).sum(dim=1)


def _metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    return classification_metrics(logits, labels, include_mask)


def _evaluate_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    reviewed: Any,
) -> dict[str, Any]:
    return {
        "canonical_test": _metrics(scores, labels),
        "reviewed_test": _metrics(
            scores,
            reviewed.labels,
            reviewed.include_mask,
        ),
    }


def _evaluate_test(
    args: argparse.Namespace,
    selections: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print("loading_test_split_after_all_weight_selection=true", flush=True)
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
        logits, fingerprint = _extract_candidate_logits(
            candidate,
            images,
            device=device,
            batch_size=args.eval_batch_size,
        )
        outputs.append(logits)
        fingerprints[candidate.name] = fingerprint
    logits = torch.stack(outputs, dim=1)
    reviewed = apply_mnist_test_label_policy(labels)
    evaluation = {
        "members": {
            name: _evaluate_scores(logits[:, index], labels, reviewed)
            for index, name in enumerate(MODEL_NAMES)
        },
        "triplet": _evaluate_scores(
            _combined_logits(logits, selections["triplet"]["selected"]),
            labels,
            reviewed,
        ),
        "pairs": {
            pair_name: _evaluate_scores(
                _combined_logits(logits, selections["pairs"][pair_name]["selected"]),
                labels,
                reviewed,
            )
            for pair_name, _ in PAIR_DEFINITIONS
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
    logits = training["logits"].float()
    labels = training["labels"].long()
    triplet_selection, triplet_rows = _grid_triplet(logits, labels, args)
    pair_selections = {}
    grid_rows = list(triplet_rows)
    for pair_name, indices in PAIR_DEFINITIONS:
        pair_selections[pair_name], pair_rows = _grid_pair(
            logits,
            labels,
            pair_name=pair_name,
            indices=indices,
            args=args,
        )
        grid_rows.extend(pair_rows)
    selections = {
        "individual_training_errors": {
            name: int(logits[:, index].argmax(dim=1).ne(labels).sum())
            for index, name in enumerate(MODEL_NAMES)
        },
        "triplet": triplet_selection,
        "pairs": pair_selections,
    }
    for ensemble_name, selection in (
        ("top3_triplet", triplet_selection),
        *pair_selections.items(),
    ):
        selected = selection["selected"]
        weights = "/".join(f"{selected[field]:.3f}" for field in WEIGHT_FIELDS)
        print(
            f"ensemble={ensemble_name} selected_weights={weights} "
            f"train_errors={selected['train_errors']}",
            flush=True,
        )

    evaluation, test_metadata = _evaluate_test(args, selections, device)
    _write_grid(grid_path, grid_rows)
    result = {
        "schema_version": 1,
        "name": "train-selected-top3-lora-logit-ensembles",
        "protocol": {
            "candidate_selection": (
                "test-informed exploratory top three individual canonical-test "
                "measurements from the audited LoRA matrix"
            ),
            "selection_split": "MNIST train, canonical labels",
            "score_space": "raw logits",
            "test_loaded_after_all_weight_selection": True,
            "test_labels_used_for_weight_selection": False,
            "model_order": MODEL_NAMES,
            "candidate_epochs": {
                "ijepa_500_e150": 150,
                "dinov2_e50": 50,
                "dinov2_e150": 150,
            },
            "coarse_step": 1.0 / args.coarse_denominator,
            "refined_step": 1.0 / args.refined_denominator,
            "refinement_radius": args.refinement_radius,
            "source_sha256": training["signature"],
            "training_logits_sha256": file_sha256(training_path),
        },
        "selection": selections,
        "evaluation": evaluation,
        **test_metadata,
    }
    result["protocol"]["grid_sha256"] = file_sha256(grid_path)
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    for ensemble_name, metrics in (
        ("top3_triplet", evaluation["triplet"]),
        *evaluation["pairs"].items(),
    ):
        print(
            f"ensemble={ensemble_name} "
            f"canonical_test_errors={metrics['canonical_test']['errors']} "
            f"reviewed_test_errors={metrics['reviewed_test']['errors']}",
            flush=True,
        )
    print(f"summary={summary_path}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-backbone", type=Path, default=DEFAULT_DINO_BACKBONE)
    parser.add_argument(
        "--dino-e50-adapter",
        type=Path,
        default=DEFAULT_DINO_E50_ADAPTER,
    )
    parser.add_argument(
        "--dino-e150-adapter",
        type=Path,
        default=DEFAULT_DINO_E150_ADAPTER,
    )
    parser.add_argument(
        "--ijepa-500-backbone",
        type=Path,
        default=DEFAULT_IJEPA_500_BACKBONE,
    )
    parser.add_argument(
        "--ijepa-500-e150-adapter",
        type=Path,
        default=DEFAULT_IJEPA_500_E150_ADAPTER,
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
