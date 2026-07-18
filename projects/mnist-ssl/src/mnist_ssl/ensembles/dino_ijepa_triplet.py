"""Grid-search DINOv2 plus the two 56x56 I-JEPA flattened probes.

This mirrors the existing I-JEPA triplet's weighted-logit search: non-negative
weights sum to one and are swept on a one-percent grid by default. Because the
grid is scored directly on MNIST test labels, its best row is a test-tuned
diagnostic rather than clean held-out model selection.

Run from ``projects/mnist-ssl``::

    uv run python scripts/reproduce/best_ensemble.py --apply-known-corrections
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets


from mnist_ssl.baselines.mae import pick_device
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.ijepa.ensemble_probes import load_probe
from mnist_ssl.paths import DATASET_DIR, PROJECT_ROOT
from mnist_ssl.provenance import artifact_paths, verify_artifacts

from .dino_ijepa import (
    DualTransform,
    accuracy,
    errors,
    fingerprint,
    load_dino,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "best" / "dino_ijepa_triplet.json"


def project_path(path: Path) -> Path:
    """Resolve command-line paths consistently from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"unsupported reproduction config schema in {path}")
    members = config.get("members", {})
    required = {
        "dino": ("backbone_artifact_id", "probe_artifact_id"),
        "ijepa_300": ("probe_artifact_id",),
        "ijepa_500": ("probe_artifact_id",),
    }
    for member, fields in required.items():
        if member not in members:
            raise ValueError(f"missing member {member!r} in {path}")
        for field in fields:
            if field not in members[member]:
                raise ValueError(f"missing members.{member}.{field} in {path}")
    if "label_policy" not in config:
        raise ValueError(f"missing label_policy in {path}")
    return config


def configured_artifact_ids(config: dict) -> dict[str, str]:
    members = config["members"]
    return {
        "dino_backbone": members["dino"]["backbone_artifact_id"],
        "dino_probe": members["dino"]["probe_artifact_id"],
        "ijepa_300_probe": members["ijepa_300"]["probe_artifact_id"],
        "ijepa_500_probe": members["ijepa_500"]["probe_artifact_id"],
    }


def assert_expected_result(result: dict, expected: dict) -> None:
    """Fail loudly if a canonical reproduction drifts from its recorded result."""

    for member, expected_errors in expected["individual_errors"].items():
        actual_errors = result["individual"][member]["errors"]
        if actual_errors != expected_errors:
            raise RuntimeError(
                f"{member} drifted: expected {expected_errors} errors, got {actual_errors}"
            )

    best = result["best_test_tuned_grid_row"]
    expected_best = expected["best"]
    for field in ("dino_weight", "ijepa_300_weight", "ijepa_500_weight"):
        if not math.isclose(best[field], expected_best[field], abs_tol=1e-12):
            raise RuntimeError(
                f"best {field} drifted: expected {expected_best[field]}, got {best[field]}"
            )
    if best["errors"] != expected_best["errors"]:
        raise RuntimeError(
            "best ensemble drifted: "
            f"expected {expected_best['errors']} errors, got {best['errors']}"
        )
    if not math.isclose(
        best["test_accuracy"],
        expected_best["test_accuracy_percent"],
        abs_tol=1e-12,
    ):
        raise RuntimeError("best ensemble accuracy drifted")
    if result["all_three_shared_errors"] != expected["all_three_shared_errors"]:
        raise RuntimeError("shared-error count drifted")
    if not math.isclose(
        result["oracle_accuracy"],
        expected["oracle_accuracy_percent"],
        abs_tol=1e-12,
    ):
        raise RuntimeError("oracle accuracy drifted")

    if "reviewed_label_evaluation" not in result:
        return
    reviewed = result["reviewed_label_evaluation"]
    expected_reviewed = expected["reviewed_label_evaluation"]
    if reviewed["label_policy"]["policy_sha256"] != expected_reviewed["policy_sha256"]:
        raise RuntimeError("reviewed-label policy drifted")
    if (
        reviewed["label_policy"]["decision_counts"]
        != expected_reviewed["decision_counts"]
    ):
        raise RuntimeError("reviewed-label decision counts drifted")
    if reviewed["scored_examples"] != expected_reviewed["scored_examples"]:
        raise RuntimeError("reviewed-label denominator drifted")
    for member, expected_errors in expected_reviewed["individual_errors"].items():
        if reviewed["individual"][member]["errors"] != expected_errors:
            raise RuntimeError(f"{member} reviewed-label errors drifted")

    reviewed_best = reviewed["best_test_tuned_grid_row"]
    expected_reviewed_best = expected_reviewed["best"]
    for field in ("dino_weight", "ijepa_300_weight", "ijepa_500_weight"):
        if not math.isclose(
            reviewed_best[field], expected_reviewed_best[field], abs_tol=1e-12
        ):
            raise RuntimeError(f"best reviewed-label {field} drifted")
    if reviewed_best["errors"] != expected_reviewed_best["errors"]:
        raise RuntimeError("best reviewed-label ensemble drifted")
    if not math.isclose(
        reviewed_best["test_accuracy"],
        expected_reviewed_best["test_accuracy_percent"],
        abs_tol=1e-12,
    ):
        raise RuntimeError("best reviewed-label ensemble accuracy drifted")
    if (
        reviewed["all_three_shared_errors"]
        != expected_reviewed["all_three_shared_errors"]
    ):
        raise RuntimeError("reviewed-label shared-error count drifted")
    if not math.isclose(
        reviewed["oracle_accuracy"],
        expected_reviewed["oracle_accuracy_percent"],
        abs_tol=1e-12,
    ):
        raise RuntimeError("reviewed-label oracle accuracy drifted")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def freeze(model: nn.Module, head: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    model.eval()
    head.eval()


@torch.no_grad()
def collect_logits(
    dino_model: nn.Module,
    dino_head: nn.Module,
    dino_pool: str,
    ijepa_300_model: nn.Module,
    ijepa_300_head: nn.Module,
    ijepa_300_pool: str,
    ijepa_500_model: nn.Module,
    ijepa_500_head: nn.Module,
    ijepa_500_pool: str,
    transform: object,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    dataset = datasets.MNIST(
        root=str(DATASET_DIR),
        train=False,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    chunks: dict[str, list[torch.Tensor]] = {
        "dino": [],
        "ijepa_300": [],
        "ijepa_500": [],
    }
    labels = []
    for (dino_images, ijepa_images), target in loader:
        dino_images = dino_images.to(device)
        ijepa_images = ijepa_images.to(device)
        chunks["dino"].append(
            dino_head(dino_model.encode(dino_images, pool=dino_pool)).float().cpu()
        )
        chunks["ijepa_300"].append(
            ijepa_300_head(
                ijepa_300_model.encode(ijepa_images, pool=ijepa_300_pool)
            )
            .float()
            .cpu()
        )
        chunks["ijepa_500"].append(
            ijepa_500_head(
                ijepa_500_model.encode(ijepa_images, pool=ijepa_500_pool)
            )
            .float()
            .cpu()
        )
        labels.append(target)
    return {name: torch.cat(parts) for name, parts in chunks.items()}, torch.cat(labels)


def grid_search(
    logits: dict[str, torch.Tensor], labels: torch.Tensor, step: int
) -> list[dict]:
    rows = []
    for dino_pct in range(0, 101, step):
        for ijepa_300_pct in range(0, 101 - dino_pct, step):
            ijepa_500_pct = 100 - dino_pct - ijepa_300_pct
            combined = (
                (dino_pct / 100.0) * logits["dino"]
                + (ijepa_300_pct / 100.0) * logits["ijepa_300"]
                + (ijepa_500_pct / 100.0) * logits["ijepa_500"]
            )
            error_count = errors(combined, labels)
            rows.append(
                {
                    "dino_weight": dino_pct / 100.0,
                    "ijepa_300_weight": ijepa_300_pct / 100.0,
                    "ijepa_500_weight": ijepa_500_pct / 100.0,
                    "test_accuracy": accuracy(error_count, len(labels)),
                    "errors": error_count,
                }
            )
    rows.sort(
        key=lambda row: (
            row["errors"],
            -row["dino_weight"],
            -row["ijepa_300_weight"],
        )
    )
    return rows


def evaluate_logits(
    logits: dict[str, torch.Tensor],
    labels: torch.Tensor,
    step: int,
) -> tuple[dict, list[dict]]:
    """Score members, the weight grid, and the label oracle for one label view."""

    rows = grid_search(logits, labels, step)
    individual = {}
    wrong = {}
    for name, member_logits in logits.items():
        error_count = errors(member_logits, labels)
        individual[name] = {
            "test_accuracy": accuracy(error_count, len(labels)),
            "errors": error_count,
        }
        wrong[name] = member_logits.argmax(dim=1) != labels
    all_shared_errors = int(
        (wrong["dino"] & wrong["ijepa_300"] & wrong["ijepa_500"]).sum().item()
    )
    return (
        {
            "scored_examples": len(labels),
            "individual": individual,
            "best_test_tuned_grid_row": rows[0],
            "all_three_shared_errors": all_shared_errors,
            "oracle_accuracy": accuracy(all_shared_errors, len(labels)),
        },
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dino-backbone", type=Path)
    parser.add_argument("--dino-probe", type=Path)
    parser.add_argument("--ijepa-300-probe", type=Path)
    parser.add_argument("--ijepa-500-probe", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--apply-known-corrections",
        action="store_true",
        help="also score the manually reviewed MNIST labels and exclusions",
    )
    parser.add_argument(
        "--label-policy",
        type=Path,
        help="override the reviewed-label policy used with --apply-known-corrections",
    )
    parser.add_argument(
        "--verify-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify manifest-pinned checkpoint hashes before evaluation",
    )
    parser.add_argument(
        "--check-expected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail unless canonical metrics match the config's expected result",
    )
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = load_config(config_path)
    artifact_ids = configured_artifact_ids(config)
    resolver = verify_artifacts if args.verify_artifacts else artifact_paths
    resolved = resolver(artifact_ids.values())
    overrides = {
        "dino_backbone": args.dino_backbone,
        "dino_probe": args.dino_probe,
        "ijepa_300_probe": args.ijepa_300_probe,
        "ijepa_500_probe": args.ijepa_500_probe,
    }
    paths = {}
    for name, artifact_id in artifact_ids.items():
        override = overrides[name]
        paths[name] = project_path(override) if override else resolved[artifact_id]
    step = args.step if args.step is not None else config["grid_step_percent"]
    output = project_path(args.output or Path(config["output"]))
    label_policy_path = project_path(
        args.label_policy or Path(config["label_policy"])
    )
    if args.label_policy is not None and not args.apply_known_corrections:
        parser.error("--label-policy requires --apply-known-corrections")
    canonical_run = (
        not any(overrides.values())
        and args.label_policy is None
        and step == config["grid_step_percent"]
    )
    if args.check_expected and not canonical_run:
        raise ValueError(
            "--check-expected requires the config artifacts and grid step; "
            "pass --no-check-expected for an exploratory override"
        )
    if step <= 0 or 100 % step:
        raise ValueError("--step must be a positive divisor of 100")
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    integrity = "verified" if args.verify_artifacts else "not checked"
    print(
        f"config={display_path(config_path)} artifacts={integrity}",
        flush=True,
    )

    device = pick_device()
    print(f"device={device}", flush=True)
    dino_model, dino_head, dino_pool, dino_transform = load_dino(
        paths["dino_backbone"], paths["dino_probe"], device
    )
    ijepa_300_model, ijepa_300_head, ijepa_300_pool = load_probe(
        paths["ijepa_300_probe"], device, legacy_28=False
    )
    ijepa_500_model, ijepa_500_head, ijepa_500_pool = load_probe(
        paths["ijepa_500_probe"], device, legacy_28=False
    )
    if ijepa_300_pool != "flatten" or ijepa_500_pool != "flatten":
        raise ValueError("both I-JEPA members must use flattened probes")
    freeze(ijepa_300_model, ijepa_300_head)
    freeze(ijepa_500_model, ijepa_500_head)

    before = {
        "dino": fingerprint(dino_model.teacher_backbone),
        "ijepa_300": fingerprint(ijepa_300_model.target),
        "ijepa_500": fingerprint(ijepa_500_model.target),
    }
    logits, labels = collect_logits(
        dino_model,
        dino_head,
        dino_pool,
        ijepa_300_model,
        ijepa_300_head,
        ijepa_300_pool,
        ijepa_500_model,
        ijepa_500_head,
        ijepa_500_pool,
        DualTransform(dino_transform),
        device,
        args.batch_size,
        args.workers,
    )
    after = {
        "dino": fingerprint(dino_model.teacher_backbone),
        "ijepa_300": fingerprint(ijepa_300_model.target),
        "ijepa_500": fingerprint(ijepa_500_model.target),
    }
    if before != after:
        raise RuntimeError("a frozen backbone changed during triplet evaluation")

    original, rows = evaluate_logits(logits, labels, step)
    best = original["best_test_tuned_grid_row"]

    result = {
        "reproduction_config": display_path(config_path),
        "reproduction_name": config["name"],
        "canonical_config_run": canonical_run,
        "checkpoint_file_hashes_verified": args.verify_artifacts,
        "artifact_ids": artifact_ids,
        "evaluation_split": "MNIST test (10,000 examples, canonical order)",
        "selection": config["selection"],
        "backbones_frozen": True,
        "known_corrections_applied": args.apply_known_corrections,
        "scored_examples": original["scored_examples"],
        "individual": original["individual"],
        "best_test_tuned_grid_row": best,
        "best_rows": rows[:20],
        "all_three_shared_errors": original["all_three_shared_errors"],
        "oracle_accuracy": original["oracle_accuracy"],
        "backbone_sha256_before": before,
        "backbone_sha256_after": after,
        "checkpoints": {name: display_path(path) for name, path in paths.items()},
        "caveat": config["caveat"],
    }
    reviewed_rows = None
    if args.apply_known_corrections:
        applied_policy = apply_mnist_test_label_policy(labels, label_policy_path)
        reviewed_logits = {
            name: member_logits[applied_policy.include_mask]
            for name, member_logits in logits.items()
        }
        reviewed_labels = applied_policy.labels[applied_policy.include_mask]
        reviewed, reviewed_rows = evaluate_logits(
            reviewed_logits, reviewed_labels, step
        )
        reviewed_original_winner = next(
            row
            for row in reviewed_rows
            if all(
                math.isclose(row[field], best[field], abs_tol=1e-12)
                for field in ("dino_weight", "ijepa_300_weight", "ijepa_500_weight")
            )
        )
        result["reviewed_label_evaluation"] = {
            "label_policy": {
                **applied_policy.metadata,
                "path": display_path(label_policy_path),
            },
            **reviewed,
            "original_label_winner_rescored": reviewed_original_winner,
            "best_rows": reviewed_rows[:20],
        }
    if args.check_expected:
        assert_expected_result(result, config["expected"])
        result["expected_result_verified"] = True
    else:
        result["expected_result_verified"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reviewed_csv_path = None
    if reviewed_rows is not None:
        reviewed_csv_path = output.with_name(f"{output.stem}_reviewed_labels.csv")
        with reviewed_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reviewed_rows[0]))
            writer.writeheader()
            writer.writerows(reviewed_rows)

    for name, metrics in original["individual"].items():
        print(
            f"{name}: {metrics['test_accuracy']:.2f}% ({metrics['errors']} errors)",
            flush=True,
        )
    print(
        "best: "
        f"{best['test_accuracy']:.2f}% ({best['errors']} errors), "
        f"DINO={best['dino_weight']:.2f}, "
        f"I-JEPA-300={best['ijepa_300_weight']:.2f}, "
        f"I-JEPA-500={best['ijepa_500_weight']:.2f}",
        flush=True,
    )
    print(
        f"all_three_shared_errors={original['all_three_shared_errors']} "
        f"oracle={result['oracle_accuracy']:.2f}%",
        flush=True,
    )
    if args.apply_known_corrections:
        reviewed = result["reviewed_label_evaluation"]
        reviewed_best = reviewed["best_test_tuned_grid_row"]
        print(
            "reviewed_labels: "
            f"{reviewed['scored_examples']} scored, "
            f"best={reviewed_best['test_accuracy']:.2f}% "
            f"({reviewed_best['errors']} errors), "
            f"DINO={reviewed_best['dino_weight']:.2f}, "
            f"I-JEPA-300={reviewed_best['ijepa_300_weight']:.2f}, "
            f"I-JEPA-500={reviewed_best['ijepa_500_weight']:.2f}; "
            f"all_three_shared_errors={reviewed['all_three_shared_errors']} "
            f"oracle={reviewed['oracle_accuracy']:.2f}%",
            flush=True,
        )
    expected = "verified" if result["expected_result_verified"] else "not checked"
    print(f"expected_result={expected}", flush=True)
    written = f"wrote={output} grid={csv_path}"
    if reviewed_csv_path is not None:
        written += f" reviewed_grid={reviewed_csv_path}"
    print(written, flush=True)


if __name__ == "__main__":
    main()
