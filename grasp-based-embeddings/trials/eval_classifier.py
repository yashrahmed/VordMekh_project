"""Evaluate a saved MNIST classifier and print its test accuracy.

Loads a classifier checkpoint written by
:mod:`trials.train_classifier`, rebuilds the head
(plus its encoder or BRIEF descriptor) entirely from the file -- nothing else is
needed -- and reports the misclassification rate on the held-out MNIST **test**
split. (Train accuracy is reported by the trainer at the end of training; eval
only ever touches the test set.)

    python -m trials.eval_classifier \
        --model models/clf_mnist_vit_probe_mean.pt

It can also evaluate the training-free geodesic descriptor directly, with no
checkpoint -- ``--arch geodesic`` builds the geodesic D2 histograms and classifies
by nearest class-centroid (a closed-form class mean, no head, no training):

    python -m trials.eval_classifier --arch geodesic

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from trials import brief
from trials.geodesic import (
    ALPHA,
    DEFAULT_CONNECTIVITY,
    GEODESIC_ARCH,
    GEODESIC_BINS,
    geodesic_features,
)
from trials.mae import (
    DATASET_DIR,
    build_model,
    ckpt_tag,
    find_checkpoint,
    make_transform,
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
    device: torch.device,
    arch: str,
    random_init: bool,
    epochs: int | None = None,
    preproc: bool = False,
) -> nn.Module:
    """Build ``arch`` and (unless ``random_init``) load its pretrained weights.

    ``preproc`` only selects which checkpoint to load (the '<arch>-preproc' tag);
    the model architecture itself is unchanged by bbox preprocessing.
    """
    model = build_model(arch).to(device)
    if not random_init:
        ckpt_path = find_checkpoint(ckpt_tag(arch, preproc), epochs)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {ckpt_path.name}")
    return model


def mnist_loader(
    train: bool, batch_size: int, shuffle: bool, preproc: bool = False
) -> DataLoader:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=train,
        download=True,
        transform=make_transform(preproc),
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
    model: nn.Module,
    train: bool,
    device: torch.device,
    pool: str = "mean",
    preproc: bool = False,
):
    """Run the frozen encoder over a split once, return (features, labels)."""
    model.eval()
    feats, labels = [], []
    for imgs, y in mnist_loader(train, batch_size=512, shuffle=False, preproc=preproc):
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
    model: nn.Module,
    head: nn.Module,
    train: bool,
    device,
    pool: str = "mean",
    preproc: bool = False,
) -> float:
    """Misclassification rate, encoding images on the fly with ``pool``."""
    model.eval()
    head.eval()
    correct, total = 0, 0
    for imgs, y in mnist_loader(train, batch_size=512, shuffle=False, preproc=preproc):
        pred = head(encode(model, imgs.to(device), pool=pool)).argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return 1.0 - correct / total


def _report(mode: str, test_err: float) -> None:
    print(f"\n--- {mode} ---")
    print(f"Test error: {test_err:.2%}  (acc {1 - test_err:.2%})")


# --------------------------------------------------------------------------- #
# Load + evaluate a saved classifier
# --------------------------------------------------------------------------- #
def evaluate(ckpt: dict, device: torch.device) -> float:
    """Rebuild the classifier from ``ckpt`` and return the held-out test error.

    Eval scores the **test** split only; train accuracy is reported by
    ``train_classifier`` at the end of training, where the train set lives.
    """
    head = nn.Linear(ckpt["in_dim"], ckpt.get("n_classes", N_CLASSES)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    if ckpt["family"] == BRIEF_FAMILY:
        cfg = ckpt["brief_cfg"]
        _, _, Xte, yte, _ = brief_features(
            ckpt["arch"],
            patch=cfg["patch"],
            n=cfg["n"],
            brief_seed=cfg["brief_seed"],
            grid=cfg["grid"],
        )
        return error_features(head, Xte, yte, device)

    # Encoder family: rebuild the (probe-frozen or fine-tuned) encoder from the
    # weights stored alongside the head and score the test images on the fly. The
    # test split must get the same bbox preprocessing the head was trained under.
    pool = ckpt.get("pool", "mean")
    preproc = ckpt.get("preproc", False)
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.eval()
    return error_images(model, head, False, device, pool=pool, preproc=preproc)


def evaluate_geodesic(device: torch.device, bins: int) -> float:
    """Training-free eval of the geodesic D2 descriptor by nearest class-centroid.

    Builds the geodesic histogram for every image (see
    :func:`trials.geodesic.geodesic_features`), forms one
    centroid per class from the L2-normalised 60K train features, and labels each
    10K test image by its nearest class centroid in cosine space. No head and no
    gradient training -- the centroids are a closed-form class mean. Returns the
    held-out **test** error only: there is no meaningful train error for a
    non-parametric matcher (it would be resubstitution on a zero-capacity
    classifier), just as ``knn.py`` reports test accuracy alone.

    This is the linear, parameter-free counterpart to ``knn.py``'s cosine k-NN on
    the same descriptor (knn ~44.2% test); both read the fixed geodesic features
    without any learning.
    """
    print(f"Describing MNIST train split with geodesic histograms ({bins} bins)...")
    Xtr, ytr = geodesic_features(mnist_loader(train=True, batch_size=512, shuffle=False), bins)
    print(f"Describing MNIST test split with geodesic histograms ({bins} bins)...")
    Xte, yte = geodesic_features(mnist_loader(train=False, batch_size=512, shuffle=False), bins)
    print(f"  train: {tuple(Xtr.shape)}   test: {tuple(Xte.shape)}")

    # Class centroids in cosine space (features are already L2-normalised).
    centroids = torch.stack([Xtr[ytr == c].mean(0) for c in range(N_CLASSES)])
    centroids = F.normalize(centroids, dim=-1).to(device)

    pred = (Xte.to(device) @ centroids.T).argmax(dim=1).cpu()
    return 1.0 - (pred == yte).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Path to a classifier checkpoint saved by train_classifier.",
    )
    parser.add_argument(
        "--arch",
        choices=(GEODESIC_ARCH,),
        default=None,
        help="Evaluate a training-free descriptor instead of a saved --model. "
        "Only 'geodesic' (geodesic D2 histogram, nearest class-centroid).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=GEODESIC_BINS,
        help="[--arch geodesic] number of distance-histogram bins.",
    )
    args = parser.parse_args()

    device = pick_device()

    if args.arch == GEODESIC_ARCH:
        if args.model:
            parser.error("--arch geodesic needs no checkpoint; do not also pass --model.")
        print(
            f"Device: {device}  geodesic D2 ({args.bins} bins, "
            f"{DEFAULT_CONNECTIVITY}-conn, alpha {ALPHA})  nearest class-centroid"
        )
        test_err = evaluate_geodesic(device, args.bins)
        _report(
            f"geodesic D2 histogram, nearest class-centroid ({args.bins} bins)", test_err
        )
        return

    if not args.model:
        parser.error("provide --model (a saved classifier) or --arch geodesic.")

    ckpt = torch.load(args.model, map_location=device)
    print(f"Device: {device}  model: {args.model}")
    print(f"  family: {ckpt['family']}  arch: {ckpt['arch']}  mode: {ckpt['mode']}")

    test_err = evaluate(ckpt, device)
    _report(ckpt.get("mode_label", ckpt["mode"]), test_err)


if __name__ == "__main__":
    main()
