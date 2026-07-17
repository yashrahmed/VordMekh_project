"""Ensemble the best saved DINOv2 and single-model I-JEPA linear probes.

The two backbones remain frozen and retain their native deterministic input
pipelines: normalized 56x56 upscale-bbox input for DINOv2 and unnormalized
56x56 upscale-bbox input for I-JEPA. The script reports fixed equal-weight
averages as the honest, prespecified comparison. It also sweeps pair weights on
the test set as an explicitly test-tuned diagnostic; that sweep is not a clean
held-out model-selection result.

Run from ``minst-experiments``::

    uv run python dino-trials/ensemble_ijepa.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import EvaluationTransform  # noqa: E402
from eval_knn import build_teacher  # noqa: E402
from ijepa_trials import custom_ijepa  # noqa: E402
from ijepa_trials.ensemble_probes import load_probe  # noqa: E402
from trials.mae import DATASET_DIR, MODELS_DIR, pick_device  # noqa: E402


DEFAULT_DINO_BACKBONE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt"
)
DEFAULT_DINO_PROBE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt"
)
DEFAULT_IJEPA_PROBE = (
    MODELS_DIR
    / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base300ep_probe50ep.pt"
)


def fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_dino(
    backbone_path: Path,
    probe_path: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, str, object]:
    checkpoint = torch.load(backbone_path, map_location=device, weights_only=False)
    probe = torch.load(probe_path, map_location=device, weights_only=False)
    result = probe.get("result", {})
    pool = result.get("pool", "cls")
    if pool != "cls":
        raise ValueError(f"expected the best DINO probe to use CLS, got {pool!r}")

    model = build_teacher(checkpoint, device)
    head = nn.Linear(probe["in_dim"], probe.get("n_classes", 10)).to(device)
    head.load_state_dict(probe["head_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    model.eval()
    head.eval()

    config = checkpoint["config"]
    transform = EvaluationTransform(
        config.get("global_size", 56), config.get("preprocess", True)
    )
    return model, head, pool, transform


class DualTransform:
    """Apply each backbone's deterministic preprocessing to the same image."""

    def __init__(self, dino_transform: object):
        self.dino_transform = dino_transform
        self.ijepa_transform = custom_ijepa.make_transform(preproc=True)

    def __call__(self, image):
        return self.dino_transform(image), self.ijepa_transform(image)


@torch.no_grad()
def collect_logits(
    dino_model: nn.Module,
    dino_head: nn.Module,
    dino_pool: str,
    ijepa_model: nn.Module,
    ijepa_head: nn.Module,
    ijepa_pool: str,
    transform: object,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    dino_logits, ijepa_logits, labels = [], [], []
    for (dino_images, ijepa_images), target in loader:
        dino_features = dino_model.encode(dino_images.to(device), pool=dino_pool)
        ijepa_features = ijepa_model.encode(ijepa_images.to(device), pool=ijepa_pool)
        dino_logits.append(dino_head(dino_features).float().cpu())
        ijepa_logits.append(ijepa_head(ijepa_features).float().cpu())
        labels.append(target)
    return torch.cat(dino_logits), torch.cat(ijepa_logits), torch.cat(labels)


def errors(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) != labels).sum().item())


def accuracy(error_count: int, total: int) -> float:
    return 100.0 * (1.0 - error_count / total)


def ensemble_rows(
    dino_logits: torch.Tensor,
    ijepa_logits: torch.Tensor,
    labels: torch.Tensor,
    step: int,
) -> list[dict]:
    rows = []
    for method in ("logit", "probability"):
        dino_scores = dino_logits
        ijepa_scores = ijepa_logits
        if method == "probability":
            dino_scores = F.softmax(dino_scores, dim=1)
            ijepa_scores = F.softmax(ijepa_scores, dim=1)
        for dino_weight_pct in range(0, 101, step):
            dino_weight = dino_weight_pct / 100.0
            combined = (
                dino_weight * dino_scores + (1.0 - dino_weight) * ijepa_scores
            )
            error_count = errors(combined, labels)
            rows.append(
                {
                    "method": method,
                    "dino_weight": dino_weight,
                    "ijepa_weight": 1.0 - dino_weight,
                    "test_accuracy": accuracy(error_count, len(labels)),
                    "errors": error_count,
                    "selection": "test-tuned diagnostic",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-backbone", type=Path, default=DEFAULT_DINO_BACKBONE)
    parser.add_argument("--dino-probe", type=Path, default=DEFAULT_DINO_PROBE)
    parser.add_argument("--ijepa-probe", type=Path, default=DEFAULT_IJEPA_PROBE)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("out/dino_ijepa_ensemble_results.json")
    )
    args = parser.parse_args()
    if args.step <= 0 or 100 % args.step:
        raise ValueError("--step must be a positive divisor of 100")
    for path in (args.dino_backbone, args.dino_probe, args.ijepa_probe):
        if not path.exists():
            raise FileNotFoundError(path)

    device = pick_device()
    print(f"device={device}", flush=True)
    dino_model, dino_head, dino_pool, dino_transform = load_dino(
        args.dino_backbone, args.dino_probe, device
    )
    ijepa_model, ijepa_head, ijepa_pool = load_probe(
        args.ijepa_probe, device, legacy_28=False
    )
    for parameter in ijepa_model.parameters():
        parameter.requires_grad_(False)
    for parameter in ijepa_head.parameters():
        parameter.requires_grad_(False)
    ijepa_model.eval()
    ijepa_head.eval()

    dino_before = fingerprint(dino_model.teacher_backbone)
    ijepa_before = fingerprint(ijepa_model.target)
    dino_logits, ijepa_logits, labels = collect_logits(
        dino_model,
        dino_head,
        dino_pool,
        ijepa_model,
        ijepa_head,
        ijepa_pool,
        DualTransform(dino_transform),
        device,
        args.batch_size,
        args.workers,
    )
    dino_after = fingerprint(dino_model.teacher_backbone)
    ijepa_after = fingerprint(ijepa_model.target)
    if dino_before != dino_after or ijepa_before != ijepa_after:
        raise RuntimeError("a frozen backbone changed during ensemble evaluation")

    total = len(labels)
    dino_errors = errors(dino_logits, labels)
    ijepa_errors = errors(ijepa_logits, labels)
    dino_wrong = dino_logits.argmax(dim=1) != labels
    ijepa_wrong = ijepa_logits.argmax(dim=1) != labels
    shared_errors = int((dino_wrong & ijepa_wrong).sum().item())
    disagreements = int(
        (dino_logits.argmax(dim=1) != ijepa_logits.argmax(dim=1)).sum().item()
    )

    rows = ensemble_rows(dino_logits, ijepa_logits, labels, args.step)
    equal_rows = [row for row in rows if row["dino_weight"] == 0.5]
    for row in equal_rows:
        row["selection"] = "prespecified equal weight"
    best_by_method = {
        method: min(
            (row for row in rows if row["method"] == method),
            key=lambda row: (row["errors"], abs(row["dino_weight"] - 0.5)),
        )
        for method in ("logit", "probability")
    }

    result = {
        "evaluation_split": "MNIST test (10,000 examples, canonical order)",
        "backbones_frozen": True,
        "dino": {
            "backbone": str(args.dino_backbone),
            "probe": str(args.dino_probe),
            "pool": dino_pool,
            "test_accuracy": accuracy(dino_errors, total),
            "errors": dino_errors,
            "backbone_sha256_before": dino_before,
            "backbone_sha256_after": dino_after,
        },
        "ijepa": {
            "probe": str(args.ijepa_probe),
            "pool": ijepa_pool,
            "test_accuracy": accuracy(ijepa_errors, total),
            "errors": ijepa_errors,
            "backbone_sha256_before": ijepa_before,
            "backbone_sha256_after": ijepa_after,
        },
        "error_complementarity": {
            "shared_errors": shared_errors,
            "dino_wrong_ijepa_right": int((dino_wrong & ~ijepa_wrong).sum().item()),
            "ijepa_wrong_dino_right": int((ijepa_wrong & ~dino_wrong).sum().item()),
            "prediction_disagreements": disagreements,
            "oracle_accuracy": accuracy(shared_errors, total),
        },
        "equal_weight": {row["method"]: row for row in equal_rows},
        "best_test_tuned_diagnostic": best_by_method,
        "caveat": (
            "Equal weights were fixed before scoring. Best swept weights use test labels "
            "and are exploratory upper-bound diagnostics, not held-out model selection."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"DINO: {result['dino']['test_accuracy']:.2f}% ({dino_errors} errors)\n"
        f"I-JEPA: {result['ijepa']['test_accuracy']:.2f}% ({ijepa_errors} errors)\n"
        f"Shared errors: {shared_errors}; oracle: "
        f"{result['error_complementarity']['oracle_accuracy']:.2f}%",
        flush=True,
    )
    for row in equal_rows:
        print(
            f"Equal {row['method']}: {row['test_accuracy']:.2f}% "
            f"({row['errors']} errors)",
            flush=True,
        )
    for method, row in best_by_method.items():
        print(
            f"Best test-tuned {method}: {row['test_accuracy']:.2f}% "
            f"({row['errors']} errors), DINO weight={row['dino_weight']:.2f}",
            flush=True,
        )
    print(f"wrote={args.output} sweep={csv_path}", flush=True)


if __name__ == "__main__":
    main()
