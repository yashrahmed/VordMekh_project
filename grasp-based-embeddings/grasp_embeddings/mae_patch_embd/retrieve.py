"""Nearest-neighbor image retrieval with the trained MAE encoder.

Loads the checkpoint written by :mod:`grasp_embeddings.mae_patch_embd.mae`,
embeds images with
the (unmasked) encoder, then runs a simple cosine nearest-neighbor search:

    1. Build a gallery of 5 random images from each MNIST class (0-9).
    2. Pick one random query image.
    3. Embed everything with the encoder and find the query's nearest gallery
       neighbours.
    4. Show the query alongside its top-3 matches.

The window has a **Refresh** button that re-runs steps 1-4 on a fresh random
draw. Passing ``--save`` instead writes a single static figure (no button).

Usage:

    python -m grasp_embeddings.mae_patch_embd.retrieve              # show figure
    python -m grasp_embeddings.mae_patch_embd.retrieve --arch cnn --seed 0
    python -m grasp_embeddings.mae_patch_embd.retrieve --no-model-init  # baseline
    python -m grasp_embeddings.mae_patch_embd.retrieve --arch brief      # random BRIEF
    python -m grasp_embeddings.mae_patch_embd.retrieve --arch brief-mod  # structured

``--arch {vit,cnn,jepa,ijepa-canonical}`` selects which pretrained encoder to load
(``ijepa-canonical`` is the I-JEPA target encoder trained with canonical block
masking; see :mod:`grasp_embeddings.mae_patch_embd.mae`).
``--no-model-init`` skips the checkpoint and embeds with a random, untrained
encoder -- a baseline showing how much the training actually buys you.
``--arch brief`` / ``--arch brief-mod`` skip the encoder entirely and retrieve on
a handcrafted BRIEF descriptor (random pairs / structured lattice; see
:mod:`brief`) -- cosine NN over the bit vectors, zero learning. ``brief-mod``'s
grid is locked to the value benchmarked by ``knn`` and the linear probe
(``BRIEF_MOD_GRID``) so the retrieval demo matches the recorded results.

``--arch geodesic`` also skips the encoder and retrieves on an intensity-geodesic
signature (see :mod:`geodesic`): each image is 2x average-downsampled, a grid
graph is built with edges weighted by ``|dI| + ALPHA``, and the flattened
all-pairs geodesic distance matrix is the embedding -- cosine NN over those, zero
learning and fully deterministic (``--factor`` / ``--connectivity`` shape the
graph; the per-step cost ALPHA is locked).

The embedding for an image is the encoder's pooled representation with no
masking applied (the full image is seen): mean-pooled tokens for the ViT,
global-avg-pooled feature map for the CNN, mean-pooled target-encoder tokens
for I-JEPA.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from grasp_embeddings.mae_patch_embd import brief
from grasp_embeddings.mae_patch_embd.brief import BRIEF_ARCHES, BRIEF_MOD_GRID
from grasp_embeddings.mae_patch_embd.geodesic import (
    ALPHA,
    GEODESIC_ARCH,
    downsample_avg,
    geodesic_matrix,
)
from grasp_embeddings.mae_patch_embd.mae import (
    ARCHES,
    DATASET_DIR,
    build_model,
    find_checkpoint,
    pick_device,
)


def load_model(device: torch.device, arch: str, epochs: int | None = None) -> nn.Module:
    ckpt_path = find_checkpoint(arch, epochs)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(arch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path.name}")
    return model


@torch.no_grad()
def embed(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, embed_dim) L2-normalised embeddings.

    Pools the encoder's representation over the full image (no masking).
    """
    return F.normalize(model.encode(imgs), dim=-1)


def make_brief_embedder(kind: str, args):
    """Build an ``embed_fn`` that turns images into L2-normalised BRIEF bit vectors.

    A drop-in for the encoder's :func:`embed`: no learning, the descriptor *is*
    the embedding, and cosine similarity over the (normalised) bit vectors drives
    the same nearest-neighbour retrieval.
    """
    features, extent, label = brief.make_features(
        kind, patch=args.patch, n=args.n, seed=args.brief_seed, grid=BRIEF_MOD_GRID
    )
    print(f"Embedding with {label} (no encoder).")

    def _embed(imgs: torch.Tensor) -> torch.Tensor:
        arr = imgs.detach().squeeze(1).cpu().numpy().astype(np.float64)  # (B, 28, 28)
        bits = brief.evaluate_batch(arr, features, extent).astype(np.float32)
        return F.normalize(torch.from_numpy(bits), dim=-1)

    return _embed


def make_geodesic_embedder(args):
    """Build an ``embed_fn`` that turns images into L2-normalised geodesic signatures.

    Each image is 2x average-downsampled, a grid graph is built with edges
    weighted by intensity difference, and the flattened all-pairs geodesic
    distance matrix is the embedding (see :mod:`geodesic`). Cosine similarity over
    these signatures drives the same nearest-neighbour retrieval. No learning and
    no randomness -- the signature is fully determined by the image and the graph
    settings, so there is no seed to set.
    """
    print(
        f"Embedding with geodesic distance matrices "
        f"(factor {args.factor}, {args.connectivity}-conn, alpha {ALPHA}); no encoder."
    )

    def _embed(imgs: torch.Tensor) -> torch.Tensor:
        arr = imgs.detach().squeeze(1).cpu().numpy().astype(np.float32)  # (B, 28, 28)
        vecs = [
            geodesic_matrix(
                downsample_avg(img, args.factor),
                connectivity=args.connectivity,
            ).reshape(-1)
            for img in arr
        ]
        feats = torch.from_numpy(np.stack(vecs).astype(np.float32))
        return F.normalize(feats, dim=-1)

    return _embed


def build_gallery(per_class: int, generator: torch.Generator):
    """Sample ``per_class`` images from each MNIST class from the test set."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    targets = ds.targets

    imgs, labels = [], []
    for cls in range(10):
        idx = (targets == cls).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(idx), generator=generator)[:per_class]
        for i in idx[perm]:
            img, label = ds[int(i)]
            imgs.append(img)
            labels.append(label)
    return torch.stack(imgs), torch.tensor(labels)


def run_round(embed_fn, device, per_class, topk, generator):
    """Sample a fresh gallery + query, embed, and return the top-k matches.

    ``embed_fn`` maps a batch of images on ``device`` to L2-normalised
    embeddings (a learned encoder or a handcrafted BRIEF descriptor). Returns
    ``(query_img, query_label, match_imgs, match_labels, scores)``. Advancing
    ``generator`` between calls yields a new random draw each time.
    """
    gallery_imgs, gallery_labels = build_gallery(per_class, generator)

    # Pick a single random query image from the gallery's categories.
    q = int(torch.randint(len(gallery_imgs), (1,), generator=generator))
    query_img = gallery_imgs[q]
    query_label = int(gallery_labels[q])

    # Exclude the query itself from the gallery so it can't match itself.
    keep = torch.ones(len(gallery_imgs), dtype=torch.bool)
    keep[q] = False
    gallery_imgs = gallery_imgs[keep]
    gallery_labels = gallery_labels[keep]

    query_emb = embed_fn(query_img.unsqueeze(0).to(device))  # (1, D)
    gallery_emb = embed_fn(gallery_imgs.to(device))  # (G, D)

    sims = (query_emb @ gallery_emb.T).squeeze(0)  # cosine similarity
    k = min(topk, len(gallery_imgs))
    scores, idx = sims.topk(k)
    idx, scores = idx.cpu(), scores.cpu()

    print(f"Query label: {query_label}")
    for rank, (i, s) in enumerate(zip(idx.tolist(), scores.tolist()), start=1):
        print(f"  match {rank}: label {int(gallery_labels[i])}  cos={s:.3f}")

    return query_img, query_label, gallery_imgs[idx], gallery_labels[idx], scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        choices=(*ARCHES, *BRIEF_ARCHES, GEODESIC_ARCH),
        default="vit",
        help="Learned encoder (vit/cnn/jepa/ijepa-canonical), a handcrafted BRIEF "
        "descriptor (brief = random pairs, brief-mod = structured lattice), or "
        "geodesic (intensity-geodesic distance matrix); brief/geodesic use no "
        "encoder.",
    )
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save the figure (PNG)."
    )
    parser.add_argument(
        "--no-model-init",
        action="store_true",
        help="Skip the checkpoint and use a random, uninitialized encoder.",
    )
    parser.add_argument(
        "--ckpt-epochs",
        type=int,
        default=None,
        help="Pretraining length of the checkpoint to load "
        "(default: the most-trained one on disk).",
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
        "--factor", type=int, default=2, help="[--arch geodesic] downsample factor."
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="[--arch geodesic] grid neighbourhood.",
    )
    args = parser.parse_args()

    device = pick_device()
    brief_kind = args.arch if args.arch in BRIEF_ARCHES else None

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    if brief_kind is not None:
        print(f"Device: {device}  features: {brief_kind} (no encoder)")
        embed_fn = make_brief_embedder(brief_kind, args)
    elif args.arch == GEODESIC_ARCH:
        print(f"Device: {device}  features: geodesic (no encoder)")
        embed_fn = make_geodesic_embedder(args)
    elif args.no_model_init:
        print(f"Device: {device}  arch: {args.arch}")
        print("Using an UNINITIALIZED (untrained) encoder.")
        model = build_model(args.arch).to(device)
        model.eval()
        embed_fn = lambda imgs: embed(model, imgs)  # noqa: E731
    else:
        print(f"Device: {device}  arch: {args.arch}")
        model = load_model(device, args.arch, epochs=args.ckpt_epochs)
        embed_fn = lambda imgs: embed(model, imgs)  # noqa: E731

    result = run_round(embed_fn, device, args.per_class, args.topk, generator)

    if args.save:
        _save(result, args.save)
    else:
        _show_interactive(
            result,
            redraw=lambda: run_round(
                embed_fn, device, args.per_class, args.topk, generator
            ),
        )


def _draw(axes, result):
    """Render one retrieval result into a pre-built row of axes."""
    query_img, query_label, match_imgs, match_labels, scores = result

    axes[0].clear()
    axes[0].imshow(query_img.squeeze(0).cpu(), cmap="gray")
    axes[0].set_title(f"query: {query_label}", fontweight="bold")
    axes[0].axis("off")

    for k in range(len(match_imgs)):
        ax = axes[k + 1]
        ax.clear()
        ax.imshow(match_imgs[k].squeeze(0).cpu(), cmap="gray")
        ax.set_title(f"#{k + 1}: {int(match_labels[k])}\ncos={scores[k]:.3f}")
        ax.axis("off")


def _save(result, path):
    import matplotlib.pyplot as plt

    n = len(result[2])
    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
    _draw(axes, result)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Saved figure -> {path}")


def _show_interactive(result, redraw):
    """Show the result with a Refresh button that re-samples on click."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    n = len(result[2])
    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.8))
    fig.subplots_adjust(bottom=0.18)  # leave room for the button

    _draw(axes, result)

    btn_ax = fig.add_axes([0.42, 0.04, 0.16, 0.08])
    button = Button(btn_ax, "Refresh")

    def on_click(_event):
        _draw(axes, redraw())
        fig.canvas.draw_idle()

    button.on_clicked(on_click)
    fig._refresh_button = button  # keep a reference so it isn't GC'd
    plt.show()


if __name__ == "__main__":
    main()
