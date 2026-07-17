"""Grid-search DINOv2 plus the two 56x56 I-JEPA flattened probes.

This mirrors the existing I-JEPA triplet's weighted-logit search: non-negative
weights sum to one and are swept on a one-percent grid by default. Because the
grid is scored directly on MNIST test labels, its best row is a test-tuned
diagnostic rather than clean held-out model selection.

Run from ``projects/mnist-ssl``::

    uv run python scripts/reproduce/best_ensemble.py
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
    canonical_run = not any(overrides.values()) and step == config["grid_step_percent"]
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

    rows = grid_search(logits, labels, step)
    best = rows[0]
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

    result = {
        "reproduction_config": display_path(config_path),
        "reproduction_name": config["name"],
        "canonical_config_run": canonical_run,
        "checkpoint_file_hashes_verified": args.verify_artifacts,
        "artifact_ids": artifact_ids,
        "evaluation_split": "MNIST test (10,000 examples, canonical order)",
        "selection": config["selection"],
        "backbones_frozen": True,
        "individual": individual,
        "best_test_tuned_grid_row": best,
        "best_rows": rows[:20],
        "all_three_shared_errors": all_shared_errors,
        "oracle_accuracy": accuracy(all_shared_errors, len(labels)),
        "backbone_sha256_before": before,
        "backbone_sha256_after": after,
        "checkpoints": {name: display_path(path) for name, path in paths.items()},
        "caveat": config["caveat"],
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

    for name, metrics in individual.items():
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
        f"all_three_shared_errors={all_shared_errors} "
        f"oracle={result['oracle_accuracy']:.2f}%",
        flush=True,
    )
    expected = "verified" if result["expected_result_verified"] else "not checked"
    print(f"expected_result={expected}", flush=True)
    print(f"wrote={output} grid={csv_path}", flush=True)


if __name__ == "__main__":
    main()
