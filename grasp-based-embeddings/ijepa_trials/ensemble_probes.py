"""Ensemble the best 28x28 and 56x56 custom I-JEPA linear probes.

The current ``custom_ijepa`` module builds the 56x56 / 64-token architecture.
This script includes a small compatibility module for the older 28x28 / 16-token
custom-I-JEPA checkpoint so the old best probe can still be evaluated and
ensembled with the new upscaled-bbox probes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from ijepa_trials import custom_ijepa
from trials.mae import DATASET_DIR, MODELS_DIR, TransformerEncoder, ViTTower, pick_device


N_CLASSES = 10
PATCH = 7
OLD_IMG_SIZE = 28
OLD_N_PATCHES = 16
OLD_PATCH_DIM = PATCH * PATCH
OLD_N_TARGETS = 10
DEFAULT_OLD_PROBE = MODELS_DIR / "ijepa_clf_custom_ijepa_t10_probe_flatten_base500ep_probe50ep_rerender.pt"
DEFAULT_UPSCALED_PROBES = (
    MODELS_DIR / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base300ep_probe50ep.pt",
    MODELS_DIR / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base500ep_probe50ep.pt",
    MODELS_DIR / "ijepa_clf_custom_ijepa_upscale_bbox_p7_mean_t48_base300ep_probe50ep.pt",
    MODELS_DIR / "ijepa_clf_custom_ijepa_upscale_bbox_p7_mean_t48_base500ep_probe50ep.pt",
)


def bbox_rescale_28(img: torch.Tensor) -> torch.Tensor:
    """Old bbox preprocessing: crop the 28x28 digit bbox and stretch to 28x28."""
    fg = img[0] > 0
    if not fg.any():
        return img
    rows = torch.where(fg.any(dim=1))[0]
    cols = torch.where(fg.any(dim=0))[0]
    crop = img[:, rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return TF.resize(crop, [OLD_IMG_SIZE, OLD_IMG_SIZE], antialias=True)


def make_dual_transform():
    to_tensor = transforms.ToTensor()

    def transform(img):
        x = to_tensor(img)
        return bbox_rescale_28(x), custom_ijepa.make_transform(preproc=True)(img)

    return transform


class LegacyCustomIJEPA(nn.Module):
    """28x28 / 4x4-token custom I-JEPA compatibility model for old checkpoints."""

    def __init__(
        self,
        enc_dim: int = 128,
        pred_dim: int = 64,
        enc_depth: int = 4,
        enc_heads: int = 4,
        pred_depth: int = 2,
        pred_heads: int = 4,
        n_targets: int = OLD_N_TARGETS,
    ):
        super().__init__()
        self.n_patches = OLD_N_PATCHES
        self.patch_dim = OLD_PATCH_DIM
        self.embed_dim = enc_dim
        self.pred_dim = pred_dim
        self.n_targets = n_targets
        self.context = ViTTower(OLD_PATCH_DIM, OLD_N_PATCHES, enc_dim, enc_depth, enc_heads)
        self.target = ViTTower(OLD_PATCH_DIM, OLD_N_PATCHES, enc_dim, enc_depth, enc_heads)
        self.target.load_state_dict(self.context.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.pred_embed = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pred_pos = nn.Parameter(torch.zeros(1, OLD_N_PATCHES, pred_dim))
        self.predictor = TransformerEncoder(pred_dim, pred_depth, pred_heads)
        self.pred_proj = nn.Linear(pred_dim, enc_dim)

    def encode(self, imgs: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        feats = self.target.tokens(imgs)
        return feats.flatten(1) if pool == "flatten" else feats.mean(dim=1)


def load_probe(path: Path, device: torch.device, legacy_28: bool) -> tuple[nn.Module, nn.Module, str]:
    ckpt = torch.load(path, map_location=device)
    pool = ckpt.get("pool", "flatten")
    head = nn.Linear(ckpt["in_dim"], ckpt.get("n_classes", N_CLASSES)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    if legacy_28:
        model = LegacyCustomIJEPA(
            enc_dim=ckpt.get("enc_dim", 128),
            pred_dim=ckpt.get("enc_dim", 128) // 2,
            n_targets=ckpt.get("n_targets", OLD_N_TARGETS),
        ).to(device)
    else:
        model = custom_ijepa.build_model(
            enc_dim=ckpt.get("enc_dim", 128),
            n_targets=ckpt.get("n_targets", custom_ijepa.N_TARGETS),
        ).to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.eval()
    head.eval()
    return model, head, pool


@torch.no_grad()
def collect_logits(old_path: Path, new_path: Path, device: torch.device, batch_size: int):
    old_model, old_head, old_pool = load_probe(old_path, device, legacy_28=True)
    new_model, new_head, new_pool = load_probe(new_path, device, legacy_28=False)

    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=False,
        download=True,
        transform=make_dual_transform(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    old_logits, new_logits, labels = [], [], []
    for (imgs28, imgs56), y in loader:
        imgs28 = imgs28.to(device)
        imgs56 = imgs56.to(device)
        old_logits.append(old_head(old_model.encode(imgs28, pool=old_pool)).cpu())
        new_logits.append(new_head(new_model.encode(imgs56, pool=new_pool)).cpu())
        labels.append(y)
    return torch.cat(old_logits), torch.cat(new_logits), torch.cat(labels)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0


def summarize_pair(old_logits: torch.Tensor, new_logits: torch.Tensor, y: torch.Tensor):
    old_pred = old_logits.argmax(dim=1)
    new_pred = new_logits.argmax(dim=1)
    old_wrong = old_pred != y
    new_wrong = new_pred != y
    return {
        "old_acc": accuracy(old_logits, y),
        "new_acc": accuracy(new_logits, y),
        "old_errors": int(old_wrong.sum().item()),
        "new_errors": int(new_wrong.sum().item()),
        "overlap_errors": int((old_wrong & new_wrong).sum().item()),
        "old_wrong_new_right": int((old_wrong & ~new_wrong).sum().item()),
        "new_wrong_old_right": int((new_wrong & ~old_wrong).sum().item()),
        "disagreements": int((old_pred != new_pred).sum().item()),
        "new_right_on_disagree": int(((old_pred != new_pred) & (new_pred == y)).sum().item()),
        "old_right_on_disagree": int(((old_pred != new_pred) & (old_pred == y)).sum().item()),
    }


def sweep_ensembles(old_logits: torch.Tensor, new_logits: torch.Tensor, y: torch.Tensor):
    rows = []
    for method in ("logit", "prob"):
        left = old_logits if method == "logit" else F.softmax(old_logits, dim=1)
        right = new_logits if method == "logit" else F.softmax(new_logits, dim=1)
        for i in range(101):
            w_new = i / 100.0
            combined = (1.0 - w_new) * left + w_new * right
            pred = combined.argmax(dim=1)
            errors = int((pred != y).sum().item())
            rows.append(
                {
                    "method": method,
                    "w_new": w_new,
                    "test_acc": 100.0 * (1.0 - errors / len(y)),
                    "errors": errors,
                }
            )
    rows.sort(key=lambda r: (-r["test_acc"], r["method"], r["w_new"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-probe", type=Path, default=DEFAULT_OLD_PROBE)
    parser.add_argument("--new-probe", type=Path, action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("out/ensemble_probe_results.csv"))
    args = parser.parse_args()

    new_paths = args.new_probe or [p for p in DEFAULT_UPSCALED_PROBES if p.exists()]
    if not args.old_probe.exists():
        raise FileNotFoundError(args.old_probe)
    if not new_paths:
        raise FileNotFoundError("No upscaled probe checkpoints found")

    device = pick_device()
    print(f"Device: {device}")
    all_rows = []
    for new_path in new_paths:
        if not new_path.exists():
            raise FileNotFoundError(new_path)
        print(f"\nOld: {args.old_probe.name}")
        print(f"New: {new_path.name}")
        old_logits, new_logits, y = collect_logits(args.old_probe, new_path, device, args.batch_size)
        summary = summarize_pair(old_logits, new_logits, y)
        print(
            "Individual: "
            f"old {summary['old_acc']:.2f}% ({summary['old_errors']} errors), "
            f"new {summary['new_acc']:.2f}% ({summary['new_errors']} errors)"
        )
        print(
            "Error overlap: "
            f"{summary['overlap_errors']} shared; "
            f"old wrong/new right {summary['old_wrong_new_right']}; "
            f"new wrong/old right {summary['new_wrong_old_right']}; "
            f"disagreements {summary['disagreements']}"
        )
        rows = sweep_ensembles(old_logits, new_logits, y)
        best = rows[0]
        print(
            "Best ensemble: "
            f"{best['method']} average, w_new={best['w_new']:.2f}, "
            f"acc {best['test_acc']:.2f}% ({best['errors']} errors)"
        )
        for row in rows:
            row = dict(row)
            row["old_probe"] = args.old_probe.name
            row["new_probe"] = new_path.name
            row.update(summary)
            all_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = [
            "old_probe",
            "new_probe",
            "method",
            "w_new",
            "test_acc",
            "errors",
            "old_acc",
            "new_acc",
            "old_errors",
            "new_errors",
            "overlap_errors",
            "old_wrong_new_right",
            "new_wrong_old_right",
            "disagreements",
            "new_right_on_disagree",
            "old_right_on_disagree",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote sweep -> {args.out}")


if __name__ == "__main__":
    main()
