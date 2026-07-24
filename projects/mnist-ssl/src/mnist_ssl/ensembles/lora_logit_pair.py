"""Tune a two-member LoRA logit ensemble on MNIST train, then evaluate test.

The two members are the final nonlinear checkpoints from the strongest
previously observed LoRA-adapted I-JEPA and DINOv2 families.  Candidate
selection is therefore explicitly test-informed and exploratory.  The scalar
logit weight, however, is selected using only canonical MNIST training labels;
test images and labels are loaded only after that weight is frozen.
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


MODEL_NAMES = ("dinov2", "ijepa_500")
LORA_ROOT = OUT_DIR / "lora_backbone_probes_2026-07-20"
DEFAULT_DINO_BACKBONE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt"
)
DEFAULT_DINO_ADAPTER = LORA_ROOT / "dinov2-best" / "nonlinear" / "epoch0150.pt"
DEFAULT_IJEPA_BACKBONE = (
    MODELS_DIR / "ijepa_mnist_custom_ijepa_p7_56_t48_500ep.pt"
)
DEFAULT_IJEPA_ADAPTER = LORA_ROOT / "ijepa-500" / "nonlinear" / "epoch0150.pt"
DEFAULT_OUTPUT_DIR = OUT_DIR / "lora_logit_pair_2026-07-23"


@dataclass(frozen=True)
class Candidate:
    name: str
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
            "ijepa_500",
            BackboneDefinition(
                "ijepa-500",
                "ijepa",
                args.ijepa_backbone,
                "flatten",
                False,
            ),
            args.ijepa_adapter,
        ),
    )


def _source_signature(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "dino_backbone": args.dino_backbone,
        "dino_adapter": args.dino_adapter,
        "ijepa_backbone": args.ijepa_backbone,
        "ijepa_adapter": args.ijepa_adapter,
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
    if checkpoint.get("epoch") != 150:
        raise ValueError(f"{candidate.name} adapter is not the final epoch-150 checkpoint")
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
def _extract_logits(
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
            logits, fingerprint = _extract_logits(
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
    if artifact["logits"].shape != (60_000, 2, 10):
        raise ValueError("training logits must have shape [60000,2,10]")
    if artifact["labels"].shape != (60_000,):
        raise ValueError("training labels must have shape [60000]")
    return artifact


def pair_weights(denominator: int) -> list[tuple[int, int]]:
    if denominator < 1:
        raise ValueError("denominator must be positive")
    return [(dino, denominator - dino) for dino in range(denominator + 1)]


def refinement_weights(
    centers: Iterable[float],
    *,
    denominator: int,
    radius: float,
) -> list[tuple[int, int]]:
    if denominator < 1 or radius < 0:
        raise ValueError("denominator must be positive and radius non-negative")
    integer_radius = round(radius * denominator)
    dino_weights = set()
    for center in centers:
        center_int = round(center * denominator)
        lower = max(0, center_int - integer_radius)
        upper = min(denominator, center_int + integer_radius)
        dino_weights.update(range(lower, upper + 1))
    return [(dino, denominator - dino) for dino in sorted(dino_weights)]


def score_grid(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Iterable[tuple[int, int]],
    *,
    denominator: int,
    phase: str,
) -> list[dict[str, Any]]:
    predictions = logits.argmax(dim=-1)
    agreement = predictions[:, 0].eq(predictions[:, 1])
    fixed_errors = int(predictions[agreement, 0].ne(labels[agreement]).sum())
    variable_logits = logits[~agreement]
    variable_labels = labels[~agreement]
    rows = []
    for dino, ijepa in weights:
        combined = dino * variable_logits[:, 0] + ijepa * variable_logits[:, 1]
        errors = fixed_errors + int(combined.argmax(dim=1).ne(variable_labels).sum())
        rows.append(
            {
                "phase": phase,
                "dino_weight": dino / denominator,
                "ijepa_500_weight": ijepa / denominator,
                "train_errors": errors,
                "train_accuracy_percent": 100.0 * (1.0 - errors / len(labels)),
            }
        )
    return rows


def select_representative(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best_errors = min(row["train_errors"] for row in rows)
    exact_best = [row for row in rows if row["train_errors"] == best_errors]
    selected = min(
        exact_best,
        key=lambda row: (
            (row["dino_weight"] - 0.5) ** 2,
            -row["dino_weight"],
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
        pair_weights(args.coarse_denominator),
        denominator=args.coarse_denominator,
        phase="coarse",
    )
    coarse_selected, coarse_best = select_representative(coarse_rows)
    centers = [row["dino_weight"] for row in coarse_best]
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
        "refined_weights_evaluated": len(refined_rows),
        "refined_exact_best_count": len(refined_best),
        "selected": refined_selected,
        "selection_rule": (
            "minimum canonical-train errors; closest-to-equal weights resolve ties; "
            "higher DINO weight resolves an exact distance tie"
        ),
        "refined_exact_best_dino_weight_min": min(
            row["dino_weight"] for row in refined_best
        ),
        "refined_exact_best_dino_weight_max": max(
            row["dino_weight"] for row in refined_best
        ),
    }
    return selection, coarse_rows + refined_rows


def _metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    if include_mask is None:
        include_mask = torch.ones(len(labels), dtype=torch.bool)
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
        logits, fingerprint = _extract_logits(
            candidate,
            images,
            device=device,
            batch_size=args.eval_batch_size,
        )
        outputs.append(logits)
        fingerprints[candidate.name] = fingerprint
    logits = torch.stack(outputs, dim=1)
    weights = torch.tensor(
        [selected["dino_weight"], selected["ijepa_500_weight"]],
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
        f"{selected['ijepa_500_weight']:.3f} "
        f"train_errors={selected['train_errors']}",
        flush=True,
    )

    evaluation, test_metadata = _evaluate_test(args, selected, device)
    _write_grid(grid_path, rows)
    result = {
        "schema_version": 1,
        "name": "train-selected-lora-logit-pair",
        "protocol": {
            "candidate_selection": (
                "test-informed exploratory choice: strongest prior observed final "
                "nonlinear LoRA checkpoint from each of the DINOv2 and I-JEPA families"
            ),
            "candidate_independence_rule": (
                "at most one checkpoint per backbone family; excludes correlated "
                "milestones from the same fixed training trajectory"
            ),
            "selection_split": "MNIST train, canonical labels",
            "score_space": "raw logits",
            "test_loaded_after_weight_selection": True,
            "test_labels_used_for_weight_selection": False,
            "model_order": MODEL_NAMES,
            "candidate_epochs": {"dinov2": 150, "ijepa_500": 150},
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
    parser.add_argument("--ijepa-backbone", type=Path, default=DEFAULT_IJEPA_BACKBONE)
    parser.add_argument("--ijepa-adapter", type=Path, default=DEFAULT_IJEPA_ADAPTER)
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
