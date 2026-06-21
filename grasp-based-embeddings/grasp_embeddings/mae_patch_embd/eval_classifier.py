"""Evaluate a saved MNIST classifier and print its train/test accuracy.

Loads a classifier checkpoint written by
:mod:`grasp_embeddings.mae_patch_embd.train_classifier`, rebuilds the head
(plus its encoder or BRIEF descriptor) entirely from the file -- nothing else is
needed -- and reports the misclassification rate on the MNIST train and test
splits.

    python -m grasp_embeddings.mae_patch_embd.eval_classifier \
        --model models/clf_mnist_vit_probe_mean.pt

This module also holds the shared classifier primitives (data loading, encoder
loading, feature extraction, error scoring, the checkpoint schema) imported by
the trainer, so the two scripts agree on exactly one representation of a saved
classifier.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from grasp_embeddings.mae_patch_embd import brief
from grasp_embeddings.mae_patch_embd.mae import (
    DATASET_DIR,
    build_model,
    find_checkpoint,
    pick_device,
)

N_CLASSES = 10

# Checkpoint schema (see train_classifier.save_classifier). A saved classifier is
# one of two families:
#   "encoder" -- a linear head over a learned encoder (vit/cnn/jepa). The encoder
#                weights live in the file too (the frozen pretrained encoder for a
#                probe, the fine-tuned one for --unfreeze), so eval is standalone.
#   "brief"   -- a linear head over a handcrafted BRIEF bit vector. The descriptor
#                is regenerated deterministically from ``brief_cfg``.
ENCODER_FAMILY = "encoder"
BRIEF_FAMILY = "brief"


# --------------------------------------------------------------------------- #
# Data + encoder helpers (shared with the trainer)
# --------------------------------------------------------------------------- #
def load_encoder(
    device: torch.device, arch: str, random_init: bool, epochs: int | None = None
) -> nn.Module:
    """Build ``arch`` and (unless ``random_init``) load its pretrained weights."""
    model = build_model(arch).to(device)
    if not random_init:
        ckpt_path = find_checkpoint(arch, epochs)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {ckpt_path.name}")
    return model


def mnist_loader(train: bool, batch_size: int, shuffle: bool) -> DataLoader:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def encode(model: nn.Module, imgs: torch.Tensor, pool: str = "mean") -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, D) image embeddings.

    ``pool`` selects how the patch grid is collapsed: ``"mean"`` (D = embed_dim)
    or ``"flatten"`` (concatenate the tokens, keeping the per-patch layout).
    Differentiable -- gradients flow into the encoder when it is unfrozen.
    """
    return model.encode(imgs, pool=pool)


@torch.no_grad()
def extract_features(
    model: nn.Module, train: bool, device: torch.device, pool: str = "mean"
):
    """Run the frozen encoder over a split once, return (features, labels)."""
    model.eval()
    feats, labels = [], []
    for imgs, y in mnist_loader(train, batch_size=512, shuffle=False):
        feats.append(encode(model, imgs.to(device), pool=pool).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def brief_features(arch: str, patch: int, n: int, brief_seed: int, grid: int):
    """BRIEF-describe MNIST as cached descriptor features for the linear head.

    Returns ``(Xtr, ytr, Xte, yte, n_bits)`` with the bit vectors as float
    tensors -- a fixed, zero-learning "encoder" the head probes. Deterministic in
    ``(arch, patch, n, brief_seed, grid)``, so the trainer and eval regenerate
    byte-identical features from the saved config.
    """
    features, extent, label = brief.make_features(
        arch, patch=patch, n=n, seed=brief_seed, grid=grid
    )
    print(f"Describing MNIST with {label}...")
    tr_desc, tr_lab = brief.describe_split(features, extent, train=True)
    te_desc, te_lab = brief.describe_split(features, extent, train=False)
    Xtr = torch.from_numpy(tr_desc.astype(np.float32))
    Xte = torch.from_numpy(te_desc.astype(np.float32))
    ytr = torch.from_numpy(tr_lab.astype(np.int64))
    yte = torch.from_numpy(te_lab.astype(np.int64))
    return Xtr, ytr, Xte, yte, features.shape[0]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def error_features(head: nn.Module, X, y, device) -> float:
    """Misclassification rate over cached features (probe / BRIEF path)."""
    head.eval()
    correct = 0
    for i in range(0, len(X), 1024):
        pred = head(X[i : i + 1024].to(device)).argmax(dim=1).cpu()
        correct += (pred == y[i : i + 1024]).sum().item()
    return 1.0 - correct / len(X)


@torch.no_grad()
def error_images(
    model: nn.Module, head: nn.Module, train: bool, device, pool: str = "mean"
) -> float:
    """Misclassification rate, encoding images on the fly with ``pool``."""
    model.eval()
    head.eval()
    correct, total = 0, 0
    for imgs, y in mnist_loader(train, batch_size=512, shuffle=False):
        pred = head(encode(model, imgs.to(device), pool=pool)).argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return 1.0 - correct / total


def _report(mode: str, train_err: float, test_err: float) -> None:
    print(f"\n--- {mode} ---")
    print(f"Train error: {train_err:.2%}  (acc {1 - train_err:.2%})")
    print(f"Test  error: {test_err:.2%}  (acc {1 - test_err:.2%})")


# --------------------------------------------------------------------------- #
# Load + evaluate a saved classifier
# --------------------------------------------------------------------------- #
def evaluate(ckpt: dict, device: torch.device) -> tuple[float, float]:
    """Rebuild the classifier from ``ckpt`` and return (train_err, test_err)."""
    head = nn.Linear(ckpt["in_dim"], ckpt.get("n_classes", N_CLASSES)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    if ckpt["family"] == BRIEF_FAMILY:
        cfg = ckpt["brief_cfg"]
        Xtr, ytr, Xte, yte, _ = brief_features(
            ckpt["arch"],
            patch=cfg["patch"],
            n=cfg["n"],
            brief_seed=cfg["brief_seed"],
            grid=cfg["grid"],
        )
        return (
            error_features(head, Xtr, ytr, device),
            error_features(head, Xte, yte, device),
        )

    # Encoder family: rebuild the (probe-frozen or fine-tuned) encoder from the
    # weights stored alongside the head and score images on the fly.
    pool = ckpt.get("pool", "mean")
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.eval()
    return (
        error_images(model, head, True, device, pool=pool),
        error_images(model, head, False, device, pool=pool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a classifier checkpoint saved by train_classifier.",
    )
    args = parser.parse_args()

    device = pick_device()
    ckpt = torch.load(args.model, map_location=device)
    print(f"Device: {device}  model: {args.model}")
    print(f"  family: {ckpt['family']}  arch: {ckpt['arch']}  mode: {ckpt['mode']}")

    train_err, test_err = evaluate(ckpt, device)
    _report(ckpt.get("mode_label", ckpt["mode"]), train_err, test_err)


if __name__ == "__main__":
    main()
