"""k-NN classification on the frozen MAE encoder embeddings.

Embeds the full MNIST train and test splits with the (unmasked) encoder, then
classifies each test image by a majority vote over its ``k`` nearest training
neighbours in cosine space. This is a non-parametric read of how
class-discriminative the *self-supervised* representation is -- no head is
trained, so it complements the linear probe in ``train_classifier.py``.

    python -m grasp_embeddings.mae_patch_embd.knn --arch vit
    python -m grasp_embeddings.mae_patch_embd.knn --arch cnn --k 5
    python -m grasp_embeddings.mae_patch_embd.knn --arch no-enc  # raw-pixel baseline
    python -m grasp_embeddings.mae_patch_embd.knn --arch brief      # random BRIEF
    python -m grasp_embeddings.mae_patch_embd.knn --arch brief-mod  # structured BRIEF
    python -m grasp_embeddings.mae_patch_embd.knn --arch geodesic   # geodesic histogram
    python -m grasp_embeddings.mae_patch_embd.knn --no-model-init  # baseline

``--arch {vit,cnn,jepa,ijepa-canonical}`` selects which pretrained encoder to load
(``ijepa-canonical`` is I-JEPA trained with canonical block masking). ``--arch
no-enc`` skips the encoder entirely and runs k-NN on the raw flattened pixels
(Euclidean image difference) -- the floor that any learned embedding should
beat. ``--no-model-init`` uses a random, untrained encoder as a baseline.

``--arch brief`` / ``--arch brief-mod`` skip the encoder too and run k-NN on a
handcrafted, zero-learning BRIEF descriptor instead -- random comparison pairs
(``--patch`` / ``--n`` / ``--brief-seed``) or a structured lattice -- compared by
**Hamming** distance over the bit vectors (see :mod:`brief`). ``brief-mod``'s
grid is locked to ``brief.BRIEF_MOD_GRID`` so the result matches the linear
probe. ``--viz-only`` shows/saves the comparison pattern instead of evaluating.

``--arch geodesic`` skips the encoder and runs cosine k-NN on a geodesic
shape-distribution histogram (see :mod:`geodesic`): each image is 2x
average-downsampled, an intensity-weighted grid graph is built, and the
upper-triangle of its all-pairs geodesic distance matrix is histogrammed into
``--bins`` bins (an Osada-style D2 descriptor). This is the compact, scalable
stand-in for retrieve.py's flattened distance matrix, which is far too large to
k-NN over the full 70k splits.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from grasp_embeddings.mae_patch_embd import brief
from grasp_embeddings.mae_patch_embd.geodesic import (
    ALPHA,
    DEFAULT_CONNECTIVITY,
    GEODESIC_ARCH,
    geodesic_features,
)
from grasp_embeddings.mae_patch_embd.mae import (
    ARCHES,
    DATASET_DIR,
    build_model,
    ckpt_tag,
    find_checkpoint,
    make_transform,
    pick_device,
)

N_CLASSES = 10
NO_ENC = "no-enc"
BRIEF_ARCHES = brief.BRIEF_ARCHES


def load_encoder(
    device: torch.device,
    arch: str,
    random_init: bool,
    epochs: int | None = None,
    preproc: bool = False,
) -> nn.Module:
    model = build_model(arch).to(device)
    if not random_init:
        ckpt_path = find_checkpoint(ckpt_tag(arch, preproc), epochs)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {ckpt_path.name}")
    model.eval()
    return model


def mnist_loader(train: bool, batch_size: int = 512, preproc: bool = False):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=train,
        download=True,
        transform=make_transform(preproc),
    )
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def embed_split(
    model: nn.Module | None,
    train: bool,
    device: torch.device,
    pool: str = "mean",
    preproc: bool = False,
):
    """Return (N, D) features and (N,) labels.

    With ``model`` set, features are L2-normalised encoder embeddings (compared
    by cosine); ``pool`` selects how the patch grid is collapsed -- ``"mean"``
    (D = embed_dim) or ``"flatten"`` (D = N * embed_dim, per-patch layout kept).
    With ``model is None`` (the ``no-enc`` baseline), D = 784 raw flattened
    pixels, compared by direct Euclidean distance.
    """
    feats, labels = [], []
    for imgs, y in mnist_loader(train, preproc=preproc):
        if model is None:
            emb = imgs.flatten(1)  # (B, 784) raw pixels
        else:
            emb = F.normalize(model.encode(imgs.to(device), pool=pool), dim=-1).cpu()
        feats.append(emb)
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_predict(
    test_emb: torch.Tensor,
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    k: int,
    device: torch.device,
    metric: str = "cosine",
    chunk: int = 1024,
) -> torch.Tensor:
    """Majority-vote k-NN.

    ``metric="cosine"`` ranks by largest dot product (embeddings are unit-norm);
    ``metric="euclidean"`` ranks by smallest pixel distance (raw-pixel baseline).
    """
    train_emb = train_emb.to(device)
    train_labels = train_labels.to(device)
    preds = []
    for i in range(0, len(test_emb), chunk):
        q = test_emb[i : i + chunk].to(device)  # (c, D)
        if metric == "cosine":
            idx = (q @ train_emb.T).topk(k, dim=1).indices  # (c, k)
        else:  # euclidean: nearest = smallest distance
            dist = torch.cdist(q, train_emb)  # (c, N)
            idx = dist.topk(k, dim=1, largest=False).indices  # (c, k)
        votes = train_labels[idx]  # (c, k)
        onehot = F.one_hot(votes, N_CLASSES).sum(dim=1)  # (c, n_classes)
        preds.append(onehot.argmax(dim=1).cpu())
    return torch.cat(preds)


def run_brief(arch: str, args) -> None:
    """k-NN with a handcrafted BRIEF descriptor (no encoder, Hamming distance).

    ``arch`` is ``"brief"`` (random pairs) or ``"brief-mod"`` (structured
    lattice). ``brief-mod``'s grid is locked to :data:`brief.BRIEF_MOD_GRID` --
    the value benchmarked here and by the linear probe -- so results compare.
    With ``--viz-only`` the comparison pattern is shown/saved instead of evaluated.
    """
    features, extent, label = brief.make_features(
        arch,
        patch=args.patch,
        n=args.n,
        seed=args.brief_seed,
        grid=brief.BRIEF_MOD_GRID,
    )
    print(f"Features: {label} (no encoder).")

    if args.viz_only:
        if arch == "brief":
            brief.visualize_random(features, extent, args.save)
        else:
            brief.visualize_structured(features, brief.BRIEF_MOD_GRID, args.save)
        return

    print("Describing MNIST train split with BRIEF...")
    train_desc, train_labels = brief.describe_split(features, extent, train=True)
    print("Describing MNIST test split with BRIEF...")
    test_desc, test_labels = brief.describe_split(features, extent, train=False)
    print(f"  train: {train_desc.shape}   test: {test_desc.shape}")

    pred = brief.knn_predict(test_desc, train_desc, train_labels, args.k)
    acc = (pred == test_labels).mean()

    print(f"\n--- {args.k}-NN on {features.shape[0]}-bit {arch} descriptors (Hamming) ---")
    print(f"Test accuracy: {acc:.2%}  (error {1 - acc:.2%})")


def run_geodesic(args, device: torch.device) -> None:
    """k-NN over geodesic shape-distribution histograms (no encoder, cosine).

    The descriptor lives in :mod:`geodesic` (shared with eval_classifier); here it
    is scored by cosine k-NN.
    """
    print(
        f"Device: {device}  features: geodesic histogram "
        f"({args.bins} bins, {DEFAULT_CONNECTIVITY}-conn, alpha {ALPHA})  k: {args.k}"
    )
    print("Describing MNIST train split with geodesic histograms...")
    Xtr, ytr = geodesic_features(mnist_loader(train=True), bins=args.bins)
    print("Describing MNIST test split with geodesic histograms...")
    Xte, yte = geodesic_features(mnist_loader(train=False), bins=args.bins)
    print(f"  train: {tuple(Xtr.shape)}   test: {tuple(Xte.shape)}")

    pred = knn_predict(Xte, Xtr, ytr, args.k, device, metric="cosine")
    acc = (pred == yte).float().mean().item()

    print(f"\n--- {args.k}-NN on {args.bins}-bin geodesic shape-distribution histograms ---")
    print(f"Test accuracy: {acc:.2%}  (error {1 - acc:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", choices=(*ARCHES, NO_ENC, *BRIEF_ARCHES, GEODESIC_ARCH), default="vit"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--no-model-init",
        action="store_true",
        help="Use a random, untrained encoder as a baseline.",
    )
    parser.add_argument(
        "--ckpt-epochs",
        type=int,
        default=None,
        help="Pretraining length of the checkpoint to load "
        "(default: the most-trained one on disk).",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Concatenate the patch tokens instead of mean-pooling them "
        "(keeps per-patch layout; ignored for --arch no-enc).",
    )
    parser.add_argument(
        "--preproc",
        action="store_true",
        help="Use the bbox crop-and-rescale preprocessing (loads the "
        "'<arch>-preproc' encoder; applies the same transform to both splits).",
    )
    parser.add_argument(
        "--patch", type=int, default=4, help="[--arch brief] frame side."
    )
    parser.add_argument(
        "--n", type=int, default=64, help="[--arch brief] feature count."
    )
    parser.add_argument(
        "--brief-seed", type=int, default=0, help="[--arch brief] seed."
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=64,
        help="[--arch geodesic] number of distance-histogram bins.",
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="[--arch brief/brief-mod] show/save the comparison pattern; skip k-NN.",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="[--viz-only] path to save the figure."
    )
    args = parser.parse_args()

    if args.arch in BRIEF_ARCHES:
        if args.flatten:
            parser.error("--flatten does not apply to --arch brief/brief-mod "
                         "(BRIEF has no patch tokens).")
        print(f"Device: (cpu, numpy)  arch: {args.arch}  k: {args.k}")
        run_brief(args.arch, args)
        return

    if args.viz_only or args.save:
        parser.error("--viz-only/--save only apply to --arch brief/brief-mod.")

    device = pick_device()

    if args.arch == GEODESIC_ARCH:
        if args.flatten:
            parser.error("--flatten does not apply to --arch geodesic.")
        run_geodesic(args, device)
        return

    pool = "flatten" if args.flatten else "mean"
    print(f"Device: {device}  arch: {args.arch}  k: {args.k}  pool: {pool}")

    no_enc = args.arch == NO_ENC
    if no_enc:
        print("No encoder: k-NN on raw flattened pixels (Euclidean).")
        model = None
        metric = "euclidean"
    else:
        if args.no_model_init:
            print("Using an UNINITIALIZED (untrained) encoder.")
        model = load_encoder(
            device,
            args.arch,
            random_init=args.no_model_init,
            epochs=args.ckpt_epochs,
            preproc=args.preproc,
        )
        metric = "cosine"

    print("Embedding train split...")
    train_emb, train_labels = embed_split(
        model, train=True, device=device, pool=pool, preproc=args.preproc
    )
    print("Embedding test split...")
    test_emb, test_labels = embed_split(
        model, train=False, device=device, pool=pool, preproc=args.preproc
    )
    print(f"  train: {tuple(train_emb.shape)}   test: {tuple(test_emb.shape)}")

    pred = knn_predict(
        test_emb, train_emb, train_labels, args.k, device, metric=metric
    )
    acc = (pred == test_labels).float().mean().item()

    src = "raw pixels" if no_enc else "frozen encoder embeddings"
    print(f"\n--- {args.k}-NN on {src} ---")
    print(f"Test accuracy: {acc:.2%}  (error {1 - acc:.2%})")


if __name__ == "__main__":
    main()
