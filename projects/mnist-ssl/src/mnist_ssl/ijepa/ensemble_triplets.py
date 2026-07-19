"""Select the legacy three-I-JEPA linear-probe mixture on MNIST train.

The ensemble combines the historical 28x28 epoch-500 probe with the 56x56
epoch-300 and epoch-500 probes.  Non-negative raw-logit weights are selected
without loading test data, then frozen and evaluated once on the canonical and
reviewed test-label views.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from mnist_ssl.baselines.mae import DATASET_DIR, MODELS_DIR, pick_device
from mnist_ssl.dinov2.nonlinear_probe import file_sha256
from mnist_ssl.ensembles.nonlinear_probe_triplet import (
    refinement_weights,
    simplex_weights,
)
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import OUT_DIR

from .ensemble_probes import (
    DEFAULT_OLD_PROBE,
    load_probe,
    make_dual_transform,
)


PROBES = (
    {
        "name": "old28_500_flatten",
        "path": DEFAULT_OLD_PROBE,
        "legacy_28": True,
        "view": "old",
    },
    {
        "name": "new56_300_flatten",
        "path": (
            MODELS_DIR
            / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base300ep_probe50ep.pt"
        ),
        "legacy_28": False,
        "view": "new",
    },
    {
        "name": "new56_500_flatten",
        "path": (
            MODELS_DIR
            / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base500ep_probe50ep.pt"
        ),
        "legacy_28": False,
        "view": "new",
    },
)
MODEL_NAMES = tuple(str(cfg["name"]) for cfg in PROBES)
DEFAULT_OUTPUT_DIR = OUT_DIR / "ijepa_train_selected_triplet_v1"


def _source_signature() -> dict[str, str]:
    return {
        f"{cfg['name']}_sha256": file_sha256(Path(cfg["path"]))
        for cfg in PROBES
    }


@torch.no_grad()
def collect_all_logits(
    device: torch.device,
    batch_size: int,
    *,
    train: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collect aligned logits for the three frozen probes on one MNIST split."""

    loaded = []
    for cfg in PROBES:
        path = Path(cfg["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        model, head, pool = load_probe(
            path,
            device,
            legacy_28=bool(cfg["legacy_28"]),
        )
        loaded.append((cfg, model, head, pool))

    dataset = datasets.MNIST(
        root=str(DATASET_DIR),
        train=train,
        download=True,
        transform=make_dual_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    logits_by_name = {name: [] for name in MODEL_NAMES}
    labels = []
    for (images_28, images_56), targets in loader:
        images_28 = images_28.to(device)
        images_56 = images_56.to(device)
        for cfg, model, head, pool in loaded:
            images = images_28 if cfg["view"] == "old" else images_56
            logits_by_name[str(cfg["name"])].append(
                head(model.encode(images, pool=pool)).cpu()
            )
        labels.append(targets)

    return (
        {name: torch.cat(chunks) for name, chunks in logits_by_name.items()},
        torch.cat(labels),
    )


def _stack_logits(logits_by_name: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([logits_by_name[name].float() for name in MODEL_NAMES], dim=1)


def _load_or_generate_training_logits(
    path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    expected_signature = _source_signature()
    if path.exists():
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("signature") != expected_signature:
            raise ValueError("training-logit artifact signature mismatch")
        print(f"reused_training_logits={path}", flush=True)
    else:
        logits_by_name, labels = collect_all_logits(
            device,
            batch_size,
            train=True,
        )
        artifact = {
            "signature": expected_signature,
            "model_order": MODEL_NAMES,
            "labels": labels,
            "logits": _stack_logits(logits_by_name),
        }
        torch.save(artifact, path)
        print(f"training_logits={path}", flush=True)

    if tuple(artifact.get("model_order", ())) != MODEL_NAMES:
        raise ValueError("training model order mismatch")
    if artifact["logits"].shape != (60_000, 3, 10):
        raise ValueError("training logits have the wrong shape")
    if artifact["labels"].shape != (60_000,):
        raise ValueError("training labels have the wrong shape")
    return artifact


def _score_grid(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Iterable[tuple[int, int, int]],
    *,
    denominator: int,
    phase: str,
) -> list[dict[str, Any]]:
    """Score weights efficiently by removing unanimous predictions."""

    member_predictions = logits.argmax(dim=-1)
    agreement = member_predictions.eq(member_predictions[:, :1]).all(dim=1)
    fixed_errors = int(
        member_predictions[agreement, 0].ne(labels[agreement]).sum()
    )
    variable_logits = logits[~agreement]
    variable_labels = labels[~agreement]

    rows = []
    for old_28, new_300, new_500 in weights:
        combined = (
            old_28 * variable_logits[:, 0]
            + new_300 * variable_logits[:, 1]
            + new_500 * variable_logits[:, 2]
        )
        errors = fixed_errors + int(
            combined.argmax(dim=1).ne(variable_labels).sum()
        )
        rows.append(
            {
                "phase": phase,
                "old28_500_weight": old_28 / denominator,
                "new56_300_weight": new_300 / denominator,
                "new56_500_weight": new_500 / denominator,
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
            "old28_500_weight",
            "new56_300_weight",
            "new56_500_weight",
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
            -row["old28_500_weight"],
            -row["new56_300_weight"],
        ),
    )
    return selected, exact_best


def _weight_tuple(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        row["old28_500_weight"],
        row["new56_300_weight"],
        row["new56_500_weight"],
    )


def _select_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    coarse_denominator: int,
    refined_denominator: int,
    refinement_radius: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coarse_rows = _score_grid(
        logits,
        labels,
        simplex_weights(coarse_denominator),
        denominator=coarse_denominator,
        phase="coarse",
    )
    coarse_selected, coarse_best = _select_representative(coarse_rows)
    centers = [_weight_tuple(row) for row in coarse_best]
    refined_rows = _score_grid(
        logits,
        labels,
        refinement_weights(
            centers,
            denominator=refined_denominator,
            radius=refinement_radius,
        ),
        denominator=refined_denominator,
        phase="refined",
    )
    refined_selected, refined_best = _select_representative(refined_rows)
    return (
        {
            "coarse_best_errors": coarse_selected["train_errors"],
            "coarse_exact_best_count": len(coarse_best),
            "refined_weights_evaluated": len(refined_rows),
            "refined_exact_best_count": len(refined_best),
            "selected": refined_selected,
            "tie_breaker": "minimum squared distance from equal weights",
        },
        coarse_rows + refined_rows,
    )


def _evaluate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: tuple[float, float, float],
    *,
    include_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    predictions = (
        logits
        * torch.tensor(weights, dtype=logits.dtype)[None, :, None]
    ).sum(dim=1).argmax(dim=1)
    if include_mask is not None:
        predictions = predictions[include_mask]
        labels = labels[include_mask]
    errors = int(predictions.ne(labels).sum())
    return {
        "scored_examples": len(labels),
        "errors": errors,
        "accuracy_percent": 100.0 * (1.0 - errors / len(labels)),
    }


def _individual_errors(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int]:
    return {
        name: int(logits[:, index].argmax(dim=1).ne(labels).sum())
        for index, name in enumerate(MODEL_NAMES)
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
    print(f"device={device}", flush=True)
    training = _load_or_generate_training_logits(
        training_path,
        device=device,
        batch_size=args.batch_size,
    )
    training_logits = training["logits"].float()
    training_labels = training["labels"].long()
    selection, rows = _select_weights(
        training_logits,
        training_labels,
        coarse_denominator=args.coarse_denominator,
        refined_denominator=args.refined_denominator,
        refinement_radius=args.refinement_radius,
    )
    selected = selection["selected"]
    selected_weights = _weight_tuple(selected)
    print(
        "selected_weights="
        f"{selected_weights[0]:.3f}/{selected_weights[1]:.3f}/"
        f"{selected_weights[2]:.3f} "
        f"train_errors={selected['train_errors']}",
        flush=True,
    )

    # Deliberately load the test split only after every weight is frozen.
    test_logits_by_name, canonical_labels = collect_all_logits(
        device,
        args.batch_size,
        train=False,
    )
    test_logits = _stack_logits(test_logits_by_name)
    reviewed = apply_mnist_test_label_policy(canonical_labels)
    canonical_test = _evaluate(
        test_logits,
        canonical_labels,
        selected_weights,
    )
    reviewed_test = _evaluate(
        test_logits,
        reviewed.labels,
        selected_weights,
        include_mask=reviewed.include_mask,
    )

    member_predictions = training_logits.argmax(dim=-1)
    all_three_shared = member_predictions.ne(
        training_labels[:, None]
    ).all(dim=1)
    oracle_errors = int(
        member_predictions.ne(training_labels[:, None]).all(dim=1).sum()
    )

    _write_grid(grid_path, rows)
    result = {
        "protocol": {
            "selection_split": "MNIST train, canonical labels",
            "selection_examples": len(training_labels),
            "test_loaded_after_weight_selection": True,
            "test_labels_used_for_selection": False,
            "score_space": "raw logits",
            "model_order": MODEL_NAMES,
            "coarse_step": 1.0 / args.coarse_denominator,
            "refined_step": 1.0 / args.refined_denominator,
            "refinement_radius": args.refinement_radius,
            "tie_breaker": selection["tie_breaker"],
        },
        "sources": _source_signature(),
        "training": {
            "individual_errors": _individual_errors(
                training_logits,
                training_labels,
            ),
            "all_three_shared_errors": int(all_three_shared.sum()),
            "oracle_errors": oracle_errors,
            "oracle_accuracy_percent": 100.0
            * (1.0 - oracle_errors / len(training_labels)),
        },
        "selection": selection,
        "evaluation": {
            "canonical_test": canonical_test,
            "reviewed_test": reviewed_test,
            "individual_canonical_errors": _individual_errors(
                test_logits,
                canonical_labels,
            ),
        },
        "reviewed_label_policy": reviewed.metadata,
    }
    result["artifacts"] = {
        "training_logits_sha256": file_sha256(training_path),
        "grid_sha256": file_sha256(grid_path),
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"canonical_errors={canonical_test['errors']} "
        f"reviewed_errors={reviewed_test['errors']}",
        flush=True,
    )
    print(f"summary={summary_path}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coarse-denominator", type=int, default=100)
    parser.add_argument("--refined-denominator", type=int, default=1000)
    parser.add_argument("--refinement-radius", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
