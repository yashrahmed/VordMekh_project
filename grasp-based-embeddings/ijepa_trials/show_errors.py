"""Show test digits a saved I-JEPA flatten probe gets wrong.

The ``ijepa_trials`` counterpart to :mod:`trials.show_errors`. Loads a probe
checkpoint written by :mod:`ijepa_trials.train_probe`, rebuilds the head plus its
encoder via the ``ENCODERS`` registry (exactly as :mod:`ijepa_trials.eval_probe`
does), scores the held-out MNIST **test** split, and renders a grid of the first
``--n`` misclassified digits annotated with their true / predicted labels. The
plotting is reused verbatim from ``trials.show_errors`` so the figure matches.

    python -m ijepa_trials.show_errors \
        --model models/ijepa_clf_custom_ijepa_probe_flatten_50ep.pt --n 30

Each tile shows the *raw* MNIST digit titled ``T:<true> P:<pred>``; the encoder
itself saw the bbox-preprocessed input (always on here). Writes a PNG to ``out/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from ijepa_trials.train_probe import ENCODERS, N_CLASSES, POOL
from trials.mae import DATASET_DIR, PROJECT_ROOT, make_transform, pick_device

OUT_DIR = PROJECT_ROOT / "out"


@torch.no_grad()
def find_errors(model: nn.Module, head: nn.Module, device, pool: str, preproc: bool):
    """Return (indices, true, pred) for every misclassified test image, in order.

    Mirrors :func:`trials.show_errors.find_errors` but drives the I-JEPA encoder
    through its ``.encode(imgs, pool)`` method instead of the ``trials`` free
    function. ``shuffle=False`` keeps the running counter equal to the dataset
    index used for display.
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
        p = head(model.encode(imgs.to(device), pool=pool)).argmax(dim=1).cpu()
        wrong = (p != y).nonzero(as_tuple=True)[0]
        for w in wrong.tolist():
            idx.append(base + w)
            true.append(int(y[w]))
            pred.append(int(p[w]))
        base += len(y)
    return idx, true, pred


def plot_errors(idx, true, pred, n: int, preproc: bool, out_path) -> None:
    """Render the first ``n`` errors as a grid of the *preprocessed* inputs.

    Unlike :func:`trials.show_errors.plot_errors` (which shows the raw,
    human-readable digit), this displays the exact bbox-cropped-and-stretched
    image the encoder was fed -- index-aligned to the test split.
    """
    ds = datasets.MNIST(
        root=str(DATASET_DIR), train=False, download=True,
        transform=make_transform(preproc),
    )
    n = min(n, len(idx))
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for k in range(n):
        img = ds[idx[k]][0].squeeze(0).numpy()
        ax = axes[k]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{true[k]}  P:{pred[k]}", fontsize=11)
    saw = "bbox-preprocessed" if preproc else "raw"
    fig.suptitle(
        f"{n} misclassified MNIST test digits  (as the network sees them: {saw})",
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
        default="models/ijepa_clf_custom_ijepa_probe_flatten_50ep.pt",
        help="Path to a probe checkpoint saved by ijepa_trials.train_probe.",
    )
    parser.add_argument("--n", type=int, default=30, help="how many errors to show")
    parser.add_argument(
        "--out", default=None, help="output PNG path (default out/<model>_errors.png)"
    )
    args = parser.parse_args()

    device = pick_device()
    ckpt = torch.load(args.model, map_location=device)
    if ckpt.get("family") != "ijepa-flatten-probe":
        parser.error("ijepa_trials.show_errors only supports ijepa-flatten-probe checkpoints.")
    pool = ckpt.get("pool", POOL)
    preproc = ckpt.get("preproc", True)

    head = nn.Linear(ckpt["in_dim"], ckpt.get("n_classes", N_CLASSES)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])

    encoder = ckpt.get("encoder", "custom_ijepa")
    encoder = {"canonical": "custom_ijepa", "cnn-stem": "cnn_stem_ijepa"}.get(encoder, encoder)
    enc_dim = ckpt.get("enc_dim")
    n_targets = ckpt.get("n_targets")
    mod = ENCODERS[encoder]
    kwargs = {}
    if enc_dim:
        kwargs["enc_dim"] = enc_dim
    if n_targets:
        kwargs["n_targets"] = n_targets
    model = mod.build_model(**kwargs).to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])

    print(f"Device: {device}  model: {args.model}")
    print(f"  encoder: {encoder}  pool: {pool}  preproc: {preproc}")

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
