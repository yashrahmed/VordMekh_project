"""Select linear- and nonlinear-probe triplet weights on MNIST train.

Both three-model groups search raw-logit and probability mixtures without
loading test prediction artifacts.  The chosen weights are then frozen and
evaluated once on canonical and reviewed test labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from mnist_ssl.baselines.mae import make_transform
from mnist_ssl.dinov2.nonlinear_probe import (
    batched_logits,
    file_sha256,
    load_feature_cache,
    load_linear_probe,
)
from mnist_ssl.dinov2.train import pick_device
from mnist_ssl.ensembles.nonlinear_probe_triplet import (
    refinement_weights,
    simplex_weights,
)
from mnist_ssl.ensembles.temperature_diagonal import (
    DEFAULT_DINO_BACKBONE,
    DEFAULT_DINO_HEAD,
    DEFAULT_DINO_TEST,
    DEFAULT_DINO_TRAIN_CACHE,
    DEFAULT_IJEPA_300_BASE,
    DEFAULT_IJEPA_300_HEAD,
    DEFAULT_IJEPA_300_TEST,
    DEFAULT_IJEPA_500_BASE,
    DEFAULT_IJEPA_500_HEAD,
    DEFAULT_IJEPA_500_TEST,
    MODEL_NAMES,
    _load_nonlinear_head,
)
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.ijepa.nonlinear_probe import load_best_member
from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR


DEFAULT_DINO_LINEAR = (
    MODELS_DIR
    / "dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "training_selected_probe_triplets_v1"
METHODS = ("logit", "probability")
GROUPS = ("linear", "nonlinear")


def _source_signature(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "dino_backbone": args.dino_backbone,
        "dino_linear": args.dino_linear,
        "dino_nonlinear": args.dino_nonlinear,
        "ijepa_300_base": args.ijepa_300_base,
        "ijepa_300_nonlinear": args.ijepa_300_nonlinear,
        "ijepa_500_base": args.ijepa_500_base,
        "ijepa_500_nonlinear": args.ijepa_500_nonlinear,
    }
    return {f"{name}_sha256": file_sha256(path) for name, path in paths.items()}


@torch.no_grad()
def _extract_ijepa_groups(
    members: dict[str, tuple[nn.Module, nn.Module, nn.Module]],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
    dataset = datasets.MNIST(
        str(DATASET_DIR),
        train=True,
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
    outputs = {
        group: {
            name: torch.empty(len(dataset), 10, dtype=torch.float32)
            for name in members
        }
        for group in GROUPS
    }
    labels = torch.empty(len(dataset), dtype=torch.long)
    offset = 0
    for images, targets in loader:
        images = images.to(device)
        end = offset + len(targets)
        labels[offset:end].copy_(targets)
        for name, (model, linear_head, nonlinear_head) in members.items():
            features = model.encode(images, pool="flatten").float()
            outputs["linear"][name][offset:end].copy_(
                linear_head(features).cpu()
            )
            outputs["nonlinear"][name][offset:end].copy_(
                nonlinear_head(features).cpu()
            )
        offset = end
    if offset != len(dataset):
        raise RuntimeError("failed to extract the complete MNIST training split")
    return outputs, labels


def _generate_training_logits(
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    signature = _source_signature(args)
    dino_features, dino_labels, _ = load_feature_cache(
        args.dino_train_cache,
        checkpoint_sha256=signature["dino_backbone_sha256"],
        source_split="MNIST train (canonical order)",
        pool="cls",
    )
    dino_linear = load_linear_probe(args.dino_linear, device)
    dino_nonlinear = _load_nonlinear_head(args.dino_nonlinear, device)
    dino_logits = {
        "linear": batched_logits(
            dino_linear,
            dino_features,
            device,
            args.eval_batch_size,
        ),
        "nonlinear": batched_logits(
            dino_nonlinear,
            dino_features,
            device,
            args.eval_batch_size,
        ),
    }
    del dino_features, dino_linear, dino_nonlinear

    ijepa_300, ijepa_300_linear, _ = load_best_member(
        args.ijepa_300_base,
        device,
        pretraining_epochs=300,
    )
    ijepa_500, ijepa_500_linear, _ = load_best_member(
        args.ijepa_500_base,
        device,
        pretraining_epochs=500,
    )
    ijepa_groups, ijepa_labels = _extract_ijepa_groups(
        {
            "ijepa_300": (
                ijepa_300,
                ijepa_300_linear,
                _load_nonlinear_head(args.ijepa_300_nonlinear, device),
            ),
            "ijepa_500": (
                ijepa_500,
                ijepa_500_linear,
                _load_nonlinear_head(args.ijepa_500_nonlinear, device),
            ),
        },
        device=device,
        batch_size=args.feature_batch_size,
        workers=args.workers,
    )
    if not torch.equal(dino_labels, ijepa_labels):
        raise ValueError("DINO and I-JEPA training labels are not aligned")

    artifact = {
        "signature": signature,
        "model_order": MODEL_NAMES,
        "labels": dino_labels,
        "groups": {
            group: torch.stack(
                [
                    dino_logits[group],
                    ijepa_groups[group]["ijepa_300"],
                    ijepa_groups[group]["ijepa_500"],
                ],
                dim=1,
            )
            for group in GROUPS
        },
    }
    torch.save(artifact, output)
    return artifact


def _load_or_generate_training_logits(
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    expected = _source_signature(args)
    if output.exists():
        artifact = torch.load(output, map_location="cpu", weights_only=False)
        if artifact.get("signature") != expected:
            raise ValueError("training-logit artifact signature mismatch")
        print(f"reused_training_logits={output}", flush=True)
    else:
        print("generating_training_logits=true", flush=True)
        artifact = _generate_training_logits(args, output, device)
        print(f"training_logits={output}", flush=True)
    if tuple(artifact.get("model_order", ())) != MODEL_NAMES:
        raise ValueError("training model order mismatch")
    for group in GROUPS:
        if artifact["groups"][group].shape != (60_000, 3, 10):
            raise ValueError(f"{group} training logits have the wrong shape")
    if artifact["labels"].shape != (60_000,):
        raise ValueError("training labels have the wrong shape")
    return artifact


def _score_grid(
    scores: torch.Tensor,
    labels: torch.Tensor,
    weights: Iterable[tuple[int, int, int]],
    *,
    denominator: int,
    group: str,
    method: str,
    phase: str,
) -> list[dict[str, Any]]:
    predictions = scores.argmax(dim=-1)
    agreement = predictions.eq(predictions[:, :1]).all(dim=1)
    fixed_errors = int(predictions[agreement, 0].ne(labels[agreement]).sum())
    variable_scores = scores[~agreement]
    variable_labels = labels[~agreement]
    rows = []
    for dino, ijepa_300, ijepa_500 in weights:
        combined = (
            dino * variable_scores[:, 0]
            + ijepa_300 * variable_scores[:, 1]
            + ijepa_500 * variable_scores[:, 2]
        )
        errors = fixed_errors + int(
            combined.argmax(dim=1).ne(variable_labels).sum()
        )
        rows.append(
            {
                "group": group,
                "method": method,
                "phase": phase,
                "dino_weight": dino / denominator,
                "ijepa_300_weight": ijepa_300 / denominator,
                "ijepa_500_weight": ijepa_500 / denominator,
                "train_errors": errors,
                "train_accuracy_percent": 100.0
                * (1.0 - errors / len(labels)),
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


def _select_representative(
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


def _grid_group(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    group: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transformed = {
        "logit": logits,
        "probability": F.softmax(logits, dim=-1),
    }
    result = {}
    all_rows = []
    for method, scores in transformed.items():
        coarse_rows = _score_grid(
            scores,
            labels,
            simplex_weights(args.coarse_denominator),
            denominator=args.coarse_denominator,
            group=group,
            method=method,
            phase="coarse",
        )
        coarse_selected, coarse_best = _select_representative(coarse_rows)
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
        refined_integer_weights = refinement_weights(
            centers,
            denominator=args.refined_denominator,
            radius=args.refinement_radius,
        )
        refined_rows = _score_grid(
            scores,
            labels,
            refined_integer_weights,
            denominator=args.refined_denominator,
            group=group,
            method=method,
            phase="refined",
        )
        refined_selected, refined_best = _select_representative(refined_rows)
        result[method] = {
            "coarse_best_errors": coarse_selected["train_errors"],
            "coarse_exact_best_count": len(coarse_best),
            "refined_weights_evaluated": len(refined_rows),
            "refined_exact_best_count": len(refined_best),
            "selected": refined_selected,
            "tie_breaker": "minimum squared distance from equal weights",
        }
        all_rows.extend(coarse_rows)
        all_rows.extend(refined_rows)
    selected_method = min(
        METHODS,
        key=lambda method: (
            result[method]["selected"]["train_errors"],
            0 if method == "probability" else 1,
            _distance_from_equal(result[method]["selected"]),
        ),
    )
    result["selected_method"] = selected_method
    result["selection_rule"] = (
        "minimum train errors; probability wins a method tie; "
        "closest-to-equal weights resolve within-method ties"
    )
    return result, all_rows


def _load_test_logits(
    args: argparse.Namespace,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, str],
]:
    paths = {
        "dino": args.dino_test,
        "ijepa_300": args.ijepa_300_test,
        "ijepa_500": args.ijepa_500_test,
    }
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    reference = payloads["dino"]
    for name, payload in payloads.items():
        for field in (
            "canonical_labels",
            "reviewed_labels",
            "reviewed_include_mask",
        ):
            if not torch.equal(reference[field], payload[field]):
                raise ValueError(f"{name} disagrees on {field}")
    groups = {
        "linear": torch.stack(
            [payloads[name]["baseline_logits"].float() for name in MODEL_NAMES],
            dim=1,
        ),
        "nonlinear": torch.stack(
            [
                payloads["dino"]["nonlinear_logits"].float(),
                payloads["ijepa_300"]["nonlinear_logits_by_epoch"][75].float(),
                payloads["ijepa_500"]["nonlinear_logits_by_epoch"][75].float(),
            ],
            dim=1,
        ),
    }
    return (
        groups,
        reference["canonical_labels"].long(),
        reference["reviewed_labels"].long(),
        reference["reviewed_include_mask"].bool(),
        {name: file_sha256(path) for name, path in paths.items()},
    )


def _evaluate_selected(
    logits: torch.Tensor,
    row: dict[str, Any],
    *,
    method: str,
    canonical_labels: torch.Tensor,
    reviewed_labels: torch.Tensor,
    reviewed_mask: torch.Tensor,
) -> dict[str, Any]:
    scores = F.softmax(logits, dim=-1) if method == "probability" else logits
    weights = torch.tensor(
        [
            row["dino_weight"],
            row["ijepa_300_weight"],
            row["ijepa_500_weight"],
        ]
    )
    predictions = (scores * weights[None, :, None]).sum(dim=1).argmax(dim=1)
    canonical_errors = int(predictions.ne(canonical_labels).sum())
    reviewed_errors = int(
        predictions[reviewed_mask].ne(reviewed_labels[reviewed_mask]).sum()
    )
    reviewed_count = int(reviewed_mask.sum())
    return {
        "canonical_test": {
            "errors": canonical_errors,
            "accuracy_percent": 100.0 * (1.0 - canonical_errors / 10_000),
        },
        "reviewed_test": {
            "scored_examples": reviewed_count,
            "errors": reviewed_errors,
            "accuracy_percent": 100.0
            * (1.0 - reviewed_errors / reviewed_count),
        },
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
    training = _load_or_generate_training_logits(
        args,
        training_path,
        device,
    )
    selections = {}
    grid_rows = []
    for group in GROUPS:
        selections[group], rows = _grid_group(
            training["groups"][group].float(),
            training["labels"].long(),
            group=group,
            args=args,
        )
        grid_rows.extend(rows)
        method = selections[group]["selected_method"]
        selected = selections[group][method]["selected"]
        print(
            f"group={group} method={method} "
            f"weights={selected['dino_weight']:.3f}/"
            f"{selected['ijepa_300_weight']:.3f}/"
            f"{selected['ijepa_500_weight']:.3f} "
            f"train_errors={selected['train_errors']}",
            flush=True,
        )

    # Test prediction artifacts are deliberately loaded only after selection.
    (
        test_groups,
        canonical_labels,
        reviewed_labels,
        reviewed_mask,
        test_hashes,
    ) = _load_test_logits(args)
    policy = apply_mnist_test_label_policy(canonical_labels)
    if not torch.equal(policy.labels, reviewed_labels):
        raise ValueError("reviewed labels differ from the current policy")
    if not torch.equal(policy.include_mask, reviewed_mask):
        raise ValueError("reviewed mask differs from the current policy")

    evaluations = {}
    for group in GROUPS:
        evaluations[group] = {}
        for method in METHODS:
            evaluations[group][method] = _evaluate_selected(
                test_groups[group],
                selections[group][method]["selected"],
                method=method,
                canonical_labels=canonical_labels,
                reviewed_labels=reviewed_labels,
                reviewed_mask=reviewed_mask,
            )
        selected_method = selections[group]["selected_method"]
        evaluations[group]["selected_method"] = selected_method

    _write_grid(grid_path, grid_rows)
    result = {
        "protocol": {
            "selection_split": "MNIST train, canonical labels",
            "test_loaded_after_all_weight_selection": True,
            "test_labels_used_for_selection": False,
            "model_order": MODEL_NAMES,
            "groups": {
                "linear": "50-epoch linear probes",
                "nonlinear": "DINO-50/I-JEPA-300-75/I-JEPA-500-75 nonlinear probes",
            },
            "methods": {
                "logit": "non-negative weighted raw logits",
                "probability": "non-negative weighted softmax probabilities",
            },
            "coarse_step": 1.0 / args.coarse_denominator,
            "refined_step": 1.0 / args.refined_denominator,
            "refinement_radius": args.refinement_radius,
            "training_logits_sha256": file_sha256(training_path),
            "test_predictions_sha256": test_hashes,
        },
        "selection": selections,
        "evaluation": evaluations,
        "reviewed_label_policy": policy.metadata,
    }
    result["protocol"]["grid_sha256"] = file_sha256(grid_path)
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    for group in GROUPS:
        method = evaluations[group]["selected_method"]
        evaluation = evaluations[group][method]
        print(
            f"group={group} selected_method={method} "
            f"canonical_errors={evaluation['canonical_test']['errors']} "
            f"reviewed_errors={evaluation['reviewed_test']['errors']}",
            flush=True,
        )
    print(f"summary={summary_path}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-backbone", type=Path, default=DEFAULT_DINO_BACKBONE)
    parser.add_argument(
        "--dino-train-cache",
        type=Path,
        default=DEFAULT_DINO_TRAIN_CACHE,
    )
    parser.add_argument("--dino-linear", type=Path, default=DEFAULT_DINO_LINEAR)
    parser.add_argument(
        "--dino-nonlinear",
        type=Path,
        default=DEFAULT_DINO_HEAD,
    )
    parser.add_argument(
        "--ijepa-300-base",
        type=Path,
        default=DEFAULT_IJEPA_300_BASE,
    )
    parser.add_argument(
        "--ijepa-300-nonlinear",
        type=Path,
        default=DEFAULT_IJEPA_300_HEAD,
    )
    parser.add_argument(
        "--ijepa-500-base",
        type=Path,
        default=DEFAULT_IJEPA_500_BASE,
    )
    parser.add_argument(
        "--ijepa-500-nonlinear",
        type=Path,
        default=DEFAULT_IJEPA_500_HEAD,
    )
    parser.add_argument("--dino-test", type=Path, default=DEFAULT_DINO_TEST)
    parser.add_argument(
        "--ijepa-300-test",
        type=Path,
        default=DEFAULT_IJEPA_300_TEST,
    )
    parser.add_argument(
        "--ijepa-500-test",
        type=Path,
        default=DEFAULT_IJEPA_500_TEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coarse-denominator", type=int, default=100)
    parser.add_argument("--refined-denominator", type=int, default=1000)
    parser.add_argument("--refinement-radius", type=float, default=0.02)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args(argv)
