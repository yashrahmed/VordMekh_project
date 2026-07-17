"""Grid-search DINOv2 plus the two 56x56 I-JEPA flattened probes.

This mirrors the existing I-JEPA triplet's weighted-logit search: non-negative
weights sum to one and are swept on a one-percent grid by default. Because the
grid is scored directly on MNIST test labels, its best row is a test-tuned
diagnostic rather than clean held-out model selection.

Run from ``minst-experiments``::

    uv run python dino-trials/ensemble_ijepa_triplet.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ensemble_ijepa import (  # noqa: E402
    DEFAULT_DINO_BACKBONE,
    DEFAULT_DINO_PROBE,
    DualTransform,
    accuracy,
    errors,
    fingerprint,
    load_dino,
)
from ijepa_trials.ensemble_probes import load_probe  # noqa: E402
from trials.mae import DATASET_DIR, MODELS_DIR, pick_device  # noqa: E402


DEFAULT_IJEPA_300_PROBE = (
    MODELS_DIR
    / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base300ep_probe50ep.pt"
)
DEFAULT_IJEPA_500_PROBE = (
    MODELS_DIR
    / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base500ep_probe50ep.pt"
)


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
    parser.add_argument("--dino-backbone", type=Path, default=DEFAULT_DINO_BACKBONE)
    parser.add_argument("--dino-probe", type=Path, default=DEFAULT_DINO_PROBE)
    parser.add_argument("--ijepa-300-probe", type=Path, default=DEFAULT_IJEPA_300_PROBE)
    parser.add_argument("--ijepa-500-probe", type=Path, default=DEFAULT_IJEPA_500_PROBE)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/dino_ijepa_triplet_grid_results.json"),
    )
    args = parser.parse_args()
    if args.step <= 0 or 100 % args.step:
        raise ValueError("--step must be a positive divisor of 100")
    for path in (
        args.dino_backbone,
        args.dino_probe,
        args.ijepa_300_probe,
        args.ijepa_500_probe,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    device = pick_device()
    print(f"device={device}", flush=True)
    dino_model, dino_head, dino_pool, dino_transform = load_dino(
        args.dino_backbone, args.dino_probe, device
    )
    ijepa_300_model, ijepa_300_head, ijepa_300_pool = load_probe(
        args.ijepa_300_probe, device, legacy_28=False
    )
    ijepa_500_model, ijepa_500_head, ijepa_500_pool = load_probe(
        args.ijepa_500_probe, device, legacy_28=False
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

    rows = grid_search(logits, labels, args.step)
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
        "evaluation_split": "MNIST test (10,000 examples, canonical order)",
        "selection": "one-percent weighted-logit grid scored on test labels",
        "backbones_frozen": True,
        "individual": individual,
        "best_test_tuned_grid_row": best,
        "best_rows": rows[:20],
        "all_three_shared_errors": all_shared_errors,
        "oracle_accuracy": accuracy(all_shared_errors, len(labels)),
        "backbone_sha256_before": before,
        "backbone_sha256_after": after,
        "checkpoints": {
            "dino_backbone": str(args.dino_backbone),
            "dino_probe": str(args.dino_probe),
            "ijepa_300_probe": str(args.ijepa_300_probe),
            "ijepa_500_probe": str(args.ijepa_500_probe),
        },
        "caveat": (
            "The best weights were chosen on MNIST test labels and therefore are "
            "an exploratory diagnostic, not held-out model selection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
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
    print(f"wrote={args.output} grid={csv_path}", flush=True)


if __name__ == "__main__":
    main()
