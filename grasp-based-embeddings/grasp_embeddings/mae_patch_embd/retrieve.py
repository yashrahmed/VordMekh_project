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
    python -m grasp_embeddings.mae_patch_embd.retrieve --save out.png --seed 0
    python -m grasp_embeddings.mae_patch_embd.retrieve --no-model-init  # baseline

``--no-model-init`` skips the checkpoint and embeds with a random, untrained
encoder -- a baseline showing how much the training actually buys you.

The embedding for an image is the mean over the encoder's patch tokens with no
masking applied (the full image is seen).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from grasp_embeddings.mae_patch_embd.mae import (
    DATASET_DIR,
    MODELS_DIR,
    MAE,
    patchify,
    pick_device,
)


def load_model(device: torch.device) -> MAE:
    ckpt_path = MODELS_DIR / "mae_mnist.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train one first with "
            "`python -m mae_patch_embd.mae`."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MAE().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def embed(model: MAE, imgs: torch.Tensor) -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, enc_dim) L2-normalised embeddings.

    Runs the encoder on every patch (no masking) and mean-pools the tokens.
    """
    tokens = model.patch_embed(patchify(imgs)) + model.enc_pos
    feats = model.encoder(tokens)  # (B, N, enc_dim)
    emb = feats.mean(dim=1)  # (B, enc_dim)
    return F.normalize(emb, dim=-1)


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


def run_round(model, device, per_class, topk, generator):
    """Sample a fresh gallery + query, embed, and return the top-k matches.

    Returns ``(query_img, query_label, match_imgs, match_labels, scores)``.
    Advancing ``generator`` between calls yields a new random draw each time.
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

    query_emb = embed(model, query_img.unsqueeze(0).to(device))  # (1, D)
    gallery_emb = embed(model, gallery_imgs.to(device))  # (G, D)

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
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    if args.no_model_init:
        print("Using an UNINITIALIZED (untrained) encoder.")
        model = MAE().to(device)
        model.eval()
    else:
        model = load_model(device)
    result = run_round(model, device, args.per_class, args.topk, generator)

    if args.save:
        _save(result, args.save)
    else:
        _show_interactive(
            result,
            redraw=lambda: run_round(
                model, device, args.per_class, args.topk, generator
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
