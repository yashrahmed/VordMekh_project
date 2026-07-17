"""Weighted logit ensembles over triplets of available I-JEPA probes."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from mnist_ssl.baselines.mae import DATASET_DIR, MODELS_DIR, pick_device

from .ensemble_probes import (
    DEFAULT_OLD_PROBE,
    load_probe,
    make_dual_transform,
)


PROBES = [
    {
        "name": "old28_500_flatten",
        "path": DEFAULT_OLD_PROBE,
        "legacy_28": True,
        "view": "old",
    },
    {
        "name": "new56_300_flatten",
        "path": MODELS_DIR / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base300ep_probe50ep.pt",
        "legacy_28": False,
        "view": "new",
    },
    {
        "name": "new56_500_flatten",
        "path": MODELS_DIR / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base500ep_probe50ep.pt",
        "legacy_28": False,
        "view": "new",
    },
    {
        "name": "new56_300_mean",
        "path": MODELS_DIR / "ijepa_clf_custom_ijepa_upscale_bbox_p7_mean_t48_base300ep_probe50ep.pt",
        "legacy_28": False,
        "view": "new",
    },
    {
        "name": "new56_500_mean",
        "path": MODELS_DIR / "ijepa_clf_custom_ijepa_upscale_bbox_p7_mean_t48_base500ep_probe50ep.pt",
        "legacy_28": False,
        "view": "new",
    },
]


@torch.no_grad()
def collect_all_logits(device: torch.device, batch_size: int):
    loaded = []
    for cfg in PROBES:
        path = Path(cfg["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        model, head, pool = load_probe(path, device, legacy_28=bool(cfg["legacy_28"]))
        loaded.append((cfg, model, head, pool))

    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=False,
        download=True,
        transform=make_dual_transform(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logits_by_name = {cfg["name"]: [] for cfg, *_ in loaded}
    labels = []
    for (imgs28, imgs56), y in loader:
        imgs28 = imgs28.to(device)
        imgs56 = imgs56.to(device)
        for cfg, model, head, pool in loaded:
            imgs = imgs28 if cfg["view"] == "old" else imgs56
            logits_by_name[cfg["name"]].append(head(model.encode(imgs, pool=pool)).cpu())
        labels.append(y)

    logits_by_name = {name: torch.cat(chunks) for name, chunks in logits_by_name.items()}
    labels = torch.cat(labels)
    return logits_by_name, labels


def errors_for(logits: torch.Tensor, y: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) != y).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--step", type=int, default=1, help="Weight-grid step in percent.")
    parser.add_argument("--out", type=Path, default=Path("out/ensemble_triplet_results.csv"))
    args = parser.parse_args()

    if 100 % args.step != 0:
        raise ValueError("--step must divide 100")

    device = pick_device()
    print(f"Device: {device}")
    logits_by_name, y = collect_all_logits(device, args.batch_size)

    print("\nIndividual probes:")
    individual = {}
    for cfg in PROBES:
        name = cfg["name"]
        errors = errors_for(logits_by_name[name], y)
        acc = 100.0 * (1.0 - errors / len(y))
        individual[name] = {"errors": errors, "acc": acc}
        print(f"  {name}: {acc:.2f}% ({errors} errors)")

    rows = []
    for names in itertools.combinations([cfg["name"] for cfg in PROBES], 3):
        a, b, c = names
        best = None
        equal_logits = (logits_by_name[a] + logits_by_name[b] + logits_by_name[c]) / 3.0
        equal_errors = errors_for(equal_logits, y)
        equal_acc = 100.0 * (1.0 - equal_errors / len(y))

        for wa_pct in range(0, 101, args.step):
            for wb_pct in range(0, 101 - wa_pct, args.step):
                wc_pct = 100 - wa_pct - wb_pct
                weights = (wa_pct / 100.0, wb_pct / 100.0, wc_pct / 100.0)
                combined = (
                    weights[0] * logits_by_name[a]
                    + weights[1] * logits_by_name[b]
                    + weights[2] * logits_by_name[c]
                )
                errors = errors_for(combined, y)
                acc = 100.0 * (1.0 - errors / len(y))
                if best is None or errors < best["errors"]:
                    best = {
                        "models": "+".join(names),
                        "a": a,
                        "b": b,
                        "c": c,
                        "wa": weights[0],
                        "wb": weights[1],
                        "wc": weights[2],
                        "errors": errors,
                        "test_acc": acc,
                        "equal_errors": equal_errors,
                        "equal_acc": equal_acc,
                    }
        assert best is not None
        rows.append(best)

    rows.sort(key=lambda r: (r["errors"], -r["test_acc"]))
    print("\nBest triplets:")
    for row in rows[:10]:
        print(
            f"  {row['test_acc']:.2f}% ({row['errors']} errors) "
            f"{row['a']}:{row['wa']:.2f} {row['b']}:{row['wb']:.2f} {row['c']}:{row['wc']:.2f} "
            f"[equal {row['equal_acc']:.2f}% / {row['equal_errors']} errors]"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = [
            "models",
            "a",
            "b",
            "c",
            "wa",
            "wb",
            "wc",
            "test_acc",
            "errors",
            "equal_acc",
            "equal_errors",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote results -> {args.out}")


if __name__ == "__main__":
    main()
