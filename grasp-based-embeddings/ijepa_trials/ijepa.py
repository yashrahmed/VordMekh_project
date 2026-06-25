"""I-JEPA on MNIST with a single random 2x2 target block (Assran et al., 2023).

A pared-down I-JEPA, canonical in spirit:

* **target encoder** -- an EMA copy of the context encoder (stop-gradient) that
  sees the *full* image and produces patch-level prediction targets.
* a single contiguous **2x2 target block** (4 patches) is sampled at a random
  position on the 4x4 patch grid, batch-shared per step (K=1, as in I-JEPA's
  multi-block collator -- different layouts come across steps, not within a batch).
* **context encoder** -- the trainable tower; it sees every patch *outside* the
  target block.
* **predictor** -- a slim Transformer that, given the encoded context tokens plus
  a mask token at one target position, predicts the target encoder's latent
  representation there. The loss is the MSE in representation space.

Each of the block's 4 target patches is predicted **independently** -- one
predictor pass per patch with a single mask token -- so the target patches never
attend to one another (they attend only to the context). Only one block is
sampled per step, so there is likewise no cross-block attention.

Images are **always** bbox-preprocessed (each digit cropped to its bounding box
and stretched to fill the frame) and the patch/embed dimensions match ``trials``'
I-JEPA exactly, so this is directly comparable.

    python -m ijepa_trials.ijepa --epochs 50
    python -m ijepa_trials.ijepa --epochs 300 --seed 0

Downloads MNIST into <project>/dataset and writes weights to
<project>/models/ijepa_mnist_scatter_<epochs>ep.pt (both gitignored). Training is
checkpointed every 50 epochs to a resumable ``..._e<epoch>of<total>.partial.pt``;
re-running the same ``--epochs`` resumes from the latest partial (model +
optimizer + LR-schedule state), and the partials are deleted once the final
checkpoint is written.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
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
)

# Reuse the shared building blocks from trials' I-JEPA so the patch grid, embed
# dimensions, preprocessing and patchifier are identical (the spec's "same as
# before"). Only the masking -- a single 2x2 target block -- is new here.
from trials.mae import (
    DATASET_DIR,
    IMG_SIZE,
    PATCH_SIZE,
    TransformerEncoder,
    ViTTower,
    make_transform,
    patchify,
    pick_device,
)

N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 16 on the 4x4 grid
PATCH_DIM = PATCH_SIZE * PATCH_SIZE  # 49
GRID = IMG_SIZE // PATCH_SIZE  # 4
TARGET_BLOCK = 2  # a contiguous 2x2 block (4 patches)

ARCH_TAG = "scatter"  # single random 2x2 block; bbox-preproc always on
CKPT_STEM = f"ijepa_mnist_{ARCH_TAG}"


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/Torch so a run is reproducible.

    Covers the encoder/predictor weight init, the block-position sampling and the
    training-loader shuffle (all draw from torch's global RNG).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ScatterIJEPA(nn.Module):
    """I-JEPA predicting a single random 2x2 target block's latent reps.

    Shares I-JEPA's twin-tower + predictor structure (see :class:`trials.mae.JEPA`):
    a trainable context encoder, an EMA target encoder (stop-gradient, the source
    of the prediction targets, which prevents collapse), and a predictor that maps
    context tokens + a mask token to one target rep. The only difference from the
    ``trials`` versions is the masking: exactly one contiguous 2x2 block, at a
    random grid position, is the target; everything else is context. Each of the
    block's 4 target patches is predicted independently (a separate predictor pass
    per patch), so target patches never attend to one another.

    Downstream :meth:`encode` uses the *target* encoder over the full image (the
    tower that has always seen whole images).
    """

    def __init__(
        self,
        patch_dim: int = PATCH_DIM,
        n_patches: int = N_PATCHES,
        enc_dim: int = 128,
        enc_depth: int = 4,
        enc_heads: int = 4,
        pred_dim: int = 64,
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
        if self.grid * self.grid != n_patches:
            raise ValueError(f"block masking needs a square grid; got {n_patches}.")
        if self.grid < TARGET_BLOCK:
            raise ValueError(f"grid {self.grid} too small for a {TARGET_BLOCK}x{TARGET_BLOCK} block.")

        self.context = ViTTower(patch_dim, n_patches, enc_dim, enc_depth, enc_heads)
        self.target = ViTTower(patch_dim, n_patches, enc_dim, enc_depth, enc_heads)
        self.target.load_state_dict(self.context.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)  # EMA-updated, never by the optimizer

        # Predictor: context tokens (+ positions) + mask tokens -> target reps.
        self.pred_embed = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pred_pos = nn.Parameter(torch.zeros(1, n_patches, pred_dim))
        self.predictor = TransformerEncoder(pred_dim, pred_depth, pred_heads)
        self.pred_proj = nn.Linear(pred_dim, enc_dim)

        nn.init.trunc_normal_(self.pred_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @torch.no_grad()
    def ema_update(self) -> None:
        """Nudge the target encoder toward the context encoder (called per step)."""
        m = self.momentum
        for tp, cp in zip(self.target.parameters(), self.context.parameters()):
            tp.mul_(m).add_(cp.detach(), alpha=1.0 - m)

    def encode(self, imgs: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, D): target-encoder tokens, then pool the grid.

        ``pool="mean"`` averages the N tokens (D = embed_dim); ``pool="flatten"``
        concatenates them (D = N * embed_dim), keeping the per-patch layout.
        Differentiable, so gradients flow into the target encoder when it is
        unfrozen downstream.
        """
        feats = self.target.tokens(imgs)  # (B, N, embed_dim)
        return feats.flatten(1) if pool == "flatten" else feats.mean(dim=1)

    def _sample_block(self, device: torch.device) -> torch.Tensor:
        """Sample one random 2x2 block's patch indices (batch-shared, K=1).

        Picks a top-left corner ``(r, c)`` with ``r, c in [0, grid - 2]`` so the
        2x2 block fits, and returns its 4 flattened patch indices.
        """
        g, blk = self.grid, TARGET_BLOCK
        r = torch.randint(0, g - blk + 1, (1,)).item()
        c = torch.randint(0, g - blk + 1, (1,)).item()
        ids = [(r + i) * g + (c + j) for i in range(blk) for j in range(blk)]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, imgs: torch.Tensor):
        b = imgs.size(0)
        patches = patchify(imgs)  # (B, N, patch_dim)

        # --- targets: full-image target encoder, stop-gradient, LayerNorm'd ---
        # LayerNorm over the feature dim before the loss (I-JEPA's loss_fn, Assran
        # 2023) keeps the target on a stable scale as the EMA encoder drifts.
        with torch.no_grad():
            target_tokens = self.target.tokens(imgs)  # (B, N, enc_dim)
            target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))

        # --- pick the single 2x2 target block; context = every other patch ---
        block_ids = self._sample_block(imgs.device)  # (4,)
        block_set = set(block_ids.tolist())
        ctx_ids = torch.tensor(
            [i for i in range(self.n_patches) if i not in block_set],
            dtype=torch.long,
            device=imgs.device,
        )
        n_ctx = ctx_ids.numel()

        # --- encode the context patches with the context encoder ---
        ctx = self.context.tokens_from(patches, ctx_ids.unsqueeze(0).expand(b, -1))
        pred_pos = self.pred_pos[0]  # (N, pred_dim)
        ctx_tok = self.pred_embed(ctx) + pred_pos[ctx_ids].unsqueeze(0)

        # --- predict each target patch INDEPENDENTLY ---
        # One predictor pass per target patch, each with a single mask token, so
        # the target patches never share a sequence and thus never attend to one
        # another -- they attend only to the context. The per-patch latent-space
        # MSEs are averaged over the block.
        loss = imgs.new_zeros(())
        for pid in block_ids.tolist():
            mask_tok = (self.mask_token + pred_pos[pid].view(1, 1, -1)).expand(b, -1, -1)
            seq = self.predictor(torch.cat([ctx_tok, mask_tok], dim=1))
            pred = self.pred_proj(seq[:, n_ctx:])  # (B, 1, enc_dim)
            loss = loss + F.mse_loss(pred[:, 0], target_tokens[:, pid])
        loss = loss / block_ids.numel()

        # 0/1 mask (1 = predicted) in original patch order, for interface parity.
        mask = torch.zeros(b, self.n_patches, device=imgs.device)
        mask[:, block_ids] = 1.0
        return loss, None, mask


def build_model() -> ScatterIJEPA:
    return ScatterIJEPA()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def model_path(epochs: int) -> Path:
    """Final-checkpoint path for an ``epochs``-epoch run."""
    return final_path(CKPT_STEM, epochs)


def find_checkpoint(epochs: int | None = None) -> Path:
    """Resolve a trained encoder checkpoint.

    With ``epochs`` given, returns that exact path. Otherwise returns the
    most-trained checkpoint on disk (largest epoch count). Raises
    ``FileNotFoundError`` if none exists.
    """
    if epochs is not None:
        path = model_path(epochs)
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {path}. Train one with "
                f"`python -m ijepa_trials.ijepa --epochs {epochs}`."
            )
        return path

    candidates = sorted(
        MODELS_DIR.glob(f"{CKPT_STEM}_*ep.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1].removesuffix("ep")),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for {CKPT_STEM!r} in {MODELS_DIR}. Train one with "
            f"`python -m ijepa_trials.ijepa`."
        )
    return candidates[-1]


def _ckpt_dict(
    model: nn.Module,
    seed: int,
    *,
    opt: torch.optim.Optimizer | None = None,
    sched: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int | None = None,
) -> dict:
    """Checkpoint payload. The final checkpoint carries only ``state_dict`` +
    ``config``; partials additionally stash optimizer/scheduler/epoch to resume."""
    ckpt: dict = {
        "state_dict": model.state_dict(),
        "config": {
            "arch": ARCH_TAG,
            "patch_size": PATCH_SIZE,
            "img_size": IMG_SIZE,
            "target_block": TARGET_BLOCK,
            "preproc": True,  # always on
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
    """MNIST train loader with bbox preprocessing always applied."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    tfm = make_transform(preproc=True)
    train = datasets.MNIST(
        root=str(DATASET_DIR), train=True, download=True, transform=tfm
    )
    return DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )


def train(
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    seed: int = 0,
    device: torch.device | None = None,
) -> Path:
    set_seed(seed)
    device = device or pick_device()
    print(f"Device: {device}  arch: {ARCH_TAG} (single 2x2 block, preproc)  seed: {seed}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = model_path(epochs)
    if out.exists():  # already finished this exact run; nothing to resume
        print(f"Final checkpoint already exists -> {out} (nothing to do)")
        return out

    loader = make_loader(batch_size)
    model = build_model().to(device)
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
            model.ema_update()  # nudge the target encoder
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        print(f"epoch {epoch:3d}/{epochs}  latent_mse {running / seen:.5f}")

        if epoch % CKPT_INTERVAL == 0 and epoch < epochs:
            part = partial_path(CKPT_STEM, epoch, epochs)
            torch.save(
                _ckpt_dict(model, seed, opt=opt, sched=sched, epoch=epoch), part
            )
            clear_partials(CKPT_STEM, epochs, keep=epoch)  # write-then-prune
            print(f"  checkpoint -> {part}")

    torch.save(_ckpt_dict(model, seed), out)
    clear_partials(CKPT_STEM, epochs)  # final reached: drop all intermediates
    print(f"Saved model -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed (reproducible runs)."
    )
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
