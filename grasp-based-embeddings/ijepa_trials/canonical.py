"""Canonical I-JEPA on MNIST with multi-block targets (Assran et al., 2023).

A faithful (small-scale) port of canonical I-JEPA, in contrast to the single-2x2
``ijepa.py``:

* **Grid** -- 7x7-px patches on the 28x28 image => a 4x4 = 16-patch grid (same as
  ``ijepa.py`` and the scatter baseline), so this isolates the *masking scheme*
  against scatter at matched resolution.
* **Targets** -- the canonical multi-block sampling: ``N_TARGET_BLOCKS`` blocks,
  each a rectangle of area ``TARGET_SCALE`` of the grid with aspect ratio in
  ``TARGET_ASPECT``, at a random position, **resampled every step** and allowed
  to overlap each other (canonical's ``allow_overlap`` governs context-vs-target
  only). Each block is predicted in its own predictor pass: **intra-block
  attention on** (a block's patches attend to each other), **inter-block
  attention off** (blocks are predicted independently).
* **Context** -- **fixed to the entire grid** with the target union removed (no
  context-block sampling): the context encoder always sees every non-target
  patch. This departs from the official collator (which samples a 0.85-1.0
  context block first) to maximise visible context.
* **Encoder** -- ``enc_dim`` 128, predictor 64. Same EMA target encoder +
  latent-space MSE as every I-JEPA here.

Images are **always** bbox-preprocessed; downstream :meth:`encode` flattens the
16 tokens (16 * 128 = 2048-d). Both match the rest of ``ijepa_trials``.

    python -m ijepa_trials.canonical --epochs 50 --seed 0

Writes <project>/models/ijepa_mnist_canonical_<epochs>ep.pt (gitignored, disjoint
from ``ijepa_mnist_scatter*`` and ``trials``' names). Checkpointed every 50
epochs and resumable, exactly like ``ijepa.py``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from ijepa_trials._ckpt import (
    CKPT_INTERVAL,
    MODELS_DIR,
    clear_partials,
    final_path,
    find_latest_partial,
    partial_path,
    set_seed,
)
from trials.mae import (
    DATASET_DIR,
    IMG_SIZE,
    TransformerEncoder,
    ViTTower,
    make_transform,
    patchify,
    pick_device,
)

PATCH = 7  # 7x7-px patches
GRID = IMG_SIZE // PATCH  # 4
N_PATCHES = GRID * GRID  # 16
PATCH_DIM = PATCH * PATCH  # 49

# Canonical multi-block sampling (Assran 2023; values from the official
# in1k_vith14 config). Targets may overlap each other. The context is **fixed to
# the entire grid** with the target union removed (no context-block sampling) --
# the encoder always sees every non-target patch.
N_TARGET_BLOCKS = 4
TARGET_SCALE = (0.15, 0.20)
TARGET_ASPECT = (0.75, 1.5)
MIN_KEEP_CTX = 4  # resample if carving leaves fewer context patches
MIN_KEEP_TGT = 1

ARCH_TAG = "canonical"
CKPT_STEM = f"ijepa_mnist_{ARCH_TAG}"
DEFAULT_ENC_DIM = 128  # predictor = enc_dim // 2


# --------------------------------------------------------------------------- #
# Block sampling
# --------------------------------------------------------------------------- #
def _uniform(lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand(1).item()


def _sample_rect(grid: int, scale_range, aspect_range) -> list[int]:
    """One rectangular block's flattened patch indices (canonical sampling).

    Area = scale * grid^2, sides from the aspect ratio, random valid position.
    """
    n = grid * grid
    area = _uniform(*scale_range) * n
    aspect = _uniform(*aspect_range)
    h = max(1, min(grid, int(round(math.sqrt(area * aspect)))))
    w = max(1, min(grid, int(round(math.sqrt(area / aspect)))))
    top = torch.randint(0, grid - h + 1, (1,)).item()
    left = torch.randint(0, grid - w + 1, (1,)).item()
    return [(top + i) * grid + (left + j) for i in range(h) for j in range(w)]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Tower(ViTTower):
    """ViTTower whose full-image :meth:`tokens` patchifies at PATCH (=7)."""

    def tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(patchify(imgs, patch=PATCH)) + self.pos
        return self.encoder(x)


class CanonicalIJEPA(nn.Module):
    """Canonical multi-block I-JEPA on the 4x4 grid (see module docstring)."""

    def __init__(
        self,
        patch_dim: int = PATCH_DIM,
        n_patches: int = N_PATCHES,
        enc_dim: int = 32,
        enc_depth: int = 4,
        enc_heads: int = 4,
        pred_dim: int = 16,
        pred_depth: int = 2,
        pred_heads: int = 4,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.patch_dim = patch_dim
        self.embed_dim = enc_dim
        self.pred_dim = pred_dim
        self.momentum = momentum
        self.grid = int(round(n_patches**0.5))

        self.context = Tower(patch_dim, n_patches, enc_dim, enc_depth, enc_heads)
        self.target = Tower(patch_dim, n_patches, enc_dim, enc_depth, enc_heads)
        self.target.load_state_dict(self.context.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)  # EMA-updated, never by the optimizer

        self.pred_embed = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pred_pos = nn.Parameter(torch.zeros(1, n_patches, pred_dim))
        self.predictor = TransformerEncoder(pred_dim, pred_depth, pred_heads)
        self.pred_proj = nn.Linear(pred_dim, enc_dim)

        nn.init.trunc_normal_(self.pred_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @torch.no_grad()
    def ema_update(self) -> None:
        m = self.momentum
        for tp, cp in zip(self.target.parameters(), self.context.parameters()):
            tp.mul_(m).add_(cp.detach(), alpha=1.0 - m)

    def encode(self, imgs: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, D): target-encoder tokens, then pool the grid."""
        feats = self.target.tokens(imgs)  # (B, 16, embed_dim)
        return feats.flatten(1) if pool == "flatten" else feats.mean(dim=1)

    def _sample_masks(self, device: torch.device):
        """One canonical mask layout (batch-shared, K=1), resampled each call.

        Returns ``(ctx_ids, target_blocks)``: ``ctx_ids`` the context patch indices
        (**the entire grid** minus the target union); ``target_blocks`` a list of
        :data:`N_TARGET_BLOCKS` index tensors (targets may overlap each other).
        Bounded retries keep at least :data:`MIN_KEEP_CTX` context patches.
        """
        g = self.grid
        all_patches = set(range(g * g))
        ctx, blocks = [], []
        for _ in range(50):
            blocks = [_sample_rect(g, TARGET_SCALE, TARGET_ASPECT) for _ in range(N_TARGET_BLOCKS)]
            tgt_union = {p for blk in blocks for p in blk}
            ctx = sorted(all_patches - tgt_union)
            if len(ctx) >= MIN_KEEP_CTX and all(len(b) >= MIN_KEEP_TGT for b in blocks):
                break
        ctx_ids = torch.tensor(ctx, dtype=torch.long, device=device)
        target_blocks = [
            torch.tensor(sorted(set(b)), dtype=torch.long, device=device) for b in blocks
        ]
        return ctx_ids, target_blocks

    def forward(self, imgs: torch.Tensor):
        b = imgs.size(0)
        patches = patchify(imgs, patch=PATCH)  # (B, 16, patch_dim)

        # --- targets: full-image target encoder, stop-gradient, LayerNorm'd ---
        with torch.no_grad():
            target_tokens = self.target.tokens(imgs)
            target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))

        ctx_ids, target_blocks = self._sample_masks(imgs.device)
        n_ctx = ctx_ids.numel()

        # --- encode the carved context with the context encoder ---
        ctx = self.context.tokens_from(patches, ctx_ids.unsqueeze(0).expand(b, -1))
        pred_pos = self.pred_pos[0]  # (N, pred_dim)
        ctx_tok = self.pred_embed(ctx) + pred_pos[ctx_ids].unsqueeze(0)

        # --- predict each target block in its OWN pass ---
        # Intra-block attention: a block's mask tokens share one sequence, so they
        # attend to each other (and the context). Inter-block: separate passes, so
        # different blocks never attend to one another. Loss averaged over blocks.
        loss = imgs.new_zeros(())
        for blk in target_blocks:
            mask_tok = (self.mask_token + pred_pos[blk].unsqueeze(0)).expand(b, -1, -1)
            seq = self.predictor(torch.cat([ctx_tok, mask_tok], dim=1))
            pred = self.pred_proj(seq[:, n_ctx:])  # (B, |blk|, enc_dim)
            loss = loss + F.mse_loss(pred, target_tokens[:, blk])
        loss = loss / len(target_blocks)

        mask = torch.zeros(b, self.n_patches, device=imgs.device)
        mask[:, torch.cat(target_blocks).unique()] = 1.0
        return loss, None, mask


def build_model(enc_dim: int = DEFAULT_ENC_DIM) -> CanonicalIJEPA:
    return CanonicalIJEPA(enc_dim=enc_dim, pred_dim=enc_dim // 2)


# --------------------------------------------------------------------------- #
# Checkpoints (mirrors ijepa.py, with the canonical stem)
# --------------------------------------------------------------------------- #
def model_path(epochs: int) -> Path:
    return final_path(CKPT_STEM, epochs)


def find_checkpoint(epochs: int | None = None) -> Path:
    if epochs is not None:
        path = model_path(epochs)
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {path}. Train one with "
                f"`python -m ijepa_trials.canonical --epochs {epochs}`."
            )
        return path
    candidates = sorted(
        MODELS_DIR.glob(f"{CKPT_STEM}_*ep.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1].removesuffix("ep")),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for {CKPT_STEM!r} in {MODELS_DIR}. Train one with "
            f"`python -m ijepa_trials.canonical`."
        )
    return candidates[-1]


def _ckpt_dict(model, seed, *, opt=None, sched=None, epoch=None) -> dict:
    ckpt: dict = {
        "state_dict": model.state_dict(),
        "config": {
            "arch": ARCH_TAG,
            "patch_size": PATCH,
            "grid": GRID,
            "n_patches": N_PATCHES,
            "enc_dim": model.embed_dim,
            "pred_dim": model.pred_dim,
            "n_target_blocks": N_TARGET_BLOCKS,
            "target_scale": TARGET_SCALE,
            "target_aspect": TARGET_ASPECT,
            "context": "full-grid-minus-targets",
            "preproc": True,
            "seed": seed,
        },
    }
    if opt is not None:
        ckpt["optim_state_dict"] = opt.state_dict()
    if sched is not None:
        ckpt["sched_state_dict"] = sched.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    return ckpt


# --------------------------------------------------------------------------- #
# Data + training
# --------------------------------------------------------------------------- #
def make_loader(batch_size: int) -> DataLoader:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train = datasets.MNIST(
        root=str(DATASET_DIR), train=True, download=True, transform=make_transform(preproc=True)
    )
    return DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )


def train(
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    seed: int = 0,
    enc_dim: int = DEFAULT_ENC_DIM,
    device: torch.device | None = None,
) -> Path:
    set_seed(seed)
    device = device or pick_device()
    print(
        f"Device: {device}  arch: {ARCH_TAG} (4x4 grid, multi-block, full-grid "
        f"context, preproc)  enc_dim: {enc_dim}  seed: {seed}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = model_path(epochs)
    if out.exists():
        print(f"Final checkpoint already exists -> {out} (nothing to do)")
        return out

    loader = make_loader(batch_size)
    model = build_model(enc_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loader)
    )

    start_epoch = 1
    resume = find_latest_partial(CKPT_STEM, epochs)
    if resume is not None:
        path, done = resume
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optim_state_dict"])
        sched.load_state_dict(ckpt["sched_state_dict"])
        start_epoch = done + 1
        print(f"Resuming from {path} -> epoch {start_epoch}/{epochs}")

    model.train()
    for epoch in range(start_epoch, epochs + 1):
        running, seen = 0.0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            loss, _, _ = model(imgs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            model.ema_update()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        print(f"epoch {epoch:3d}/{epochs}  latent_mse {running / seen:.5f}")

        if epoch % CKPT_INTERVAL == 0 and epoch < epochs:
            part = partial_path(CKPT_STEM, epoch, epochs)
            torch.save(_ckpt_dict(model, seed, opt=opt, sched=sched, epoch=epoch), part)
            clear_partials(CKPT_STEM, epochs, keep=epoch)
            print(f"  checkpoint -> {part}")

    torch.save(_ckpt_dict(model, seed), out)
    clear_partials(CKPT_STEM, epochs)
    print(f"Saved model -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument(
        "--enc-dim", type=int, default=DEFAULT_ENC_DIM,
        help="Per-patch embedding dim (predictor = enc_dim // 2).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()
    train(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, enc_dim=args.enc_dim,
    )


if __name__ == "__main__":
    main()
