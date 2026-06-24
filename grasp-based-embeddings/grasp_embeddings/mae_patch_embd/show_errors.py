"""Show test digits a saved linear-probe classifier gets wrong.

Loads a classifier checkpoint written by
:mod:`grasp_embeddings.mae_patch_embd.train_classifier`, rebuilds the encoder +
head straight from the file (exactly as ``eval_classifier`` does), scores the
held-out MNIST **test** split, and renders a grid of the first ``--n``
misclassified digits annotated with their expected (true) and predicted labels.

    python -m grasp_embeddings.mae_patch_embd.show_errors \
        --model models/clf_mnist_jepa-preproc_probe_flatten.pt --n 30

Each tile shows the *raw* MNIST digit (what a human reads) titled ``T:<true>
P:<pred>``; the model itself saw the bbox-preprocessed version when the
checkpoint was trained with ``--preproc`` (noted in the figure subtitle). Writes
a PNG to ``out/`` and prints the indices so they can be inspected further.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from grasp_embeddings.mae_patch_embd.eval_classifier import N_CLASSES, encode
from grasp_embeddings.mae_patch_embd.mae import (
    DATASET_DIR,
    PROJECT_ROOT,
    build_model,
    make_transform,
    pick_device,
)

OUT_DIR = PROJECT_ROOT / "out"


@torch.no_grad()
def find_errors(model: nn.Module, head: nn.Module, device, pool: str, preproc: bool):
    """Return (indices, true, pred) for every misclassified test image, in order.

    Iterates the test split once with ``shuffle=False`` so the running counter is
    the dataset index -- the same index that addresses the raw-image dataset used
    for display.
    """
    model.eval()
    head.eval()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR), train=False, download=True,
        transform=make_transform(preproc),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)

    idx, true, pred = [], [], []
    base = 0
    for imgs, y in loader:
        p = head(encode(model, imgs.to(device), pool=pool)).argmax(dim=1).cpu()
        wrong = (p != y).nonzero(as_tuple=True)[0]
        for w in wrong.tolist():
            idx.append(base + w)
            true.append(int(y[w]))
            pred.append(int(p[w]))
        base += len(y)
    return idx, true, pred


def plot_errors(idx, true, pred, n: int, preproc: bool, out_path) -> None:
    """Render the first ``n`` errors as a grid of raw digits with T/P labels."""
    # Raw, unpreprocessed images for human-readable display (index-aligned).
    raw = datasets.MNIST(
        root=str(DATASET_DIR), train=False, download=True,
        transform=transforms.ToTensor(),
    )
    n = min(n, len(idx))
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for k in range(n):
        i = idx[k]
        img = raw[i][0].squeeze(0).numpy()
        ax = axes[k]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{true[k]}  P:{pred[k]}", fontsize=11)
    saw = "bbox-preprocessed" if preproc else "raw"
    fig.suptitle(
        f"{n} misclassified MNIST test digits  (model saw the {saw} input)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/clf_mnist_jepa-preproc_probe_flatten.pt",
        help="Path to a classifier checkpoint saved by train_classifier.",
    )
    parser.add_argument("--n", type=int, default=30, help="how many errors to show")
    parser.add_argument(
        "--out", default=None, help="output PNG path (default out/<model>_errors.png)"
    )
    args = parser.parse_args()

    device = pick_device()
    ckpt = torch.load(args.model, map_location=device)
    if ckpt.get("family") != "encoder":
        parser.error("show_errors only supports encoder-family probe checkpoints.")
    pool = ckpt.get("pool", "mean")
    preproc = ckpt.get("preproc", False)

    head = nn.Linear(ckpt["in_dim"], ckpt.get("n_classes", N_CLASSES)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])

    print(f"Device: {device}  model: {args.model}")
    print(f"  arch: {ckpt['arch']}  pool: {pool}  preproc: {preproc}")

    idx, true, pred = find_errors(model, head, device, pool, preproc)
    err = len(idx)
    print(f"Test errors: {err}/10000  ({err / 100:.2f}%)")
    print(f"First {min(args.n, err)} error indices: {idx[: args.n]}")

    out_path = (
        OUT_DIR / f"{Path(args.model).stem}_errors.png"
        if args.out is None
        else Path(args.out)
    )
    plot_errors(idx, true, pred, args.n, preproc, out_path)


if __name__ == "__main__":
    main()
