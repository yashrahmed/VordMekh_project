"""I-JEPA on MNIST with a dense ConvNet stem before feature-space patching.

This is a separate experiment from :mod:`ijepa_trials.custom_ijepa`. Images are
still **always** bbox-preprocessed, but the token grid is built in ConvNet feature
space instead of directly from raw 7x7 image patches:

* ``1x28x28 -> Conv3 s2 p1 -> 32x14x14``
* ``32x14x14 -> Conv2 s2 p1 -> 64x8x8``
* group the ``8x8`` feature map into a ``4x4`` grid of ``2x2x64`` feature patches
* linearly project each 256-d feature patch to ``enc_dim``

The JEPA split matches the current best recipe: 10 single-patch joint targets and
6 context patches by default, resampled every step. Because masking happens after
the dense ConvNet stem, this is explicitly a **feature-space JEPA** variant:
neighboring context/target tokens can have overlapping raw-image receptive fields.

    python -m ijepa_trials.cnn_stem_ijepa --epochs 50 --seed 0

Writes <project>/models/ijepa_mnist_cnn_stem_ijepa_<epochs>ep.pt (gitignored,
disjoint from the custom I-JEPA checkpoints).
"""

from __future__ import annotations

import argparse
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
    make_transform,
    pick_device,
)

STEM_CHANNELS = 64
FEATURE_GRID = 8
TOKEN_FEATURE_PATCH = 2
GRID = FEATURE_GRID // TOKEN_FEATURE_PATCH  # 4
N_PATCHES = GRID * GRID  # 16
PATCH_DIM = STEM_CHANNELS * TOKEN_FEATURE_PATCH * TOKEN_FEATURE_PATCH  # 256
N_TARGETS = 10

ARCH_TAG = "cnn_stem_ijepa"
CKPT_STEM = f"ijepa_mnist_{ARCH_TAG}"
DEFAULT_ENC_DIM = 128


class ConvFeaturePatchEmbed(nn.Module):
    """Dense CNN stem followed by 2x2 feature-space patch grouping."""

    def __init__(self, dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.GELU(),
            nn.Conv2d(32, STEM_CHANNELS, kernel_size=2, stride=2, padding=1),  # 14 -> 8
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(PATCH_DIM, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, 16, dim)."""
        feats = self.stem(imgs)
        if feats.shape[-2:] != (FEATURE_GRID, FEATURE_GRID):
            raise RuntimeError(f"expected {FEATURE_GRID}x{FEATURE_GRID}, got {feats.shape[-2:]}")

        b, c, h, w = feats.shape
        p = TOKEN_FEATURE_PATCH
        patches = (
            feats.reshape(b, c, h // p, p, w // p, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, N_PATCHES, p * p * c)
        )
        return self.proj(patches)


class Tower(nn.Module):
    """Conv feature patch embed + positional embedding + Transformer."""

    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.patch_embed = ConvFeaturePatchEmbed(dim)
        self.pos = nn.Parameter(torch.zeros(1, N_PATCHES, dim))
        self.encoder = TransformerEncoder(dim, depth, heads)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(imgs) + self.pos
        return self.encoder(x)

    def tokens_from(self, imgs: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """Encode only selected feature-space patches after the dense stem.

        The ConvNet stem sees the full image; masking is applied to the resulting
        4x4 feature-token grid before the Transformer encoder.
        """
        b, n_keep = ids_keep.shape
        d = self.pos.size(-1)
        patches = self.patch_embed(imgs)
        kept = torch.gather(patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        pos = torch.gather(
            self.pos.expand(b, -1, -1), 1, ids_keep.unsqueeze(-1).expand(-1, -1, d)
        )
        return self.encoder(kept + pos)


class CNNStemIJEPA(nn.Module):
    """Feature-space I-JEPA on a 4x4 grid built from an 8x8 ConvNet feature map."""

    def __init__(
        self,
        enc_dim: int = DEFAULT_ENC_DIM,
        enc_depth: int = 4,
        enc_heads: int = 4,
        pred_dim: int = DEFAULT_ENC_DIM // 2,
        pred_depth: int = 2,
        pred_heads: int = 4,
        momentum: float = 0.996,
        n_targets: int = N_TARGETS,
    ):
        super().__init__()
        if not 0 < n_targets < N_PATCHES:
            raise ValueError(f"n_targets must be in (0, {N_PATCHES}); got {n_targets}")

        self.n_patches = N_PATCHES
        self.patch_dim = PATCH_DIM
        self.embed_dim = enc_dim
        self.pred_dim = pred_dim
        self.momentum = momentum
        self.n_targets = n_targets
        self.grid = GRID

        self.context = Tower(enc_dim, enc_depth, enc_heads)
        self.target = Tower(enc_dim, enc_depth, enc_heads)
        self.target.load_state_dict(self.context.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.pred_embed = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pred_pos = nn.Parameter(torch.zeros(1, N_PATCHES, pred_dim))
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
        feats = self.target.tokens(imgs)
        return feats.flatten(1) if pool == "flatten" else feats.mean(dim=1)

    def _sample_masks(self, device: torch.device):
        perm = torch.randperm(N_PATCHES, device=device)
        target_ids = perm[: self.n_targets].sort().values
        ctx_ids = perm[self.n_targets :].sort().values
        return ctx_ids, target_ids

    def forward(self, imgs: torch.Tensor):
        b = imgs.size(0)
        with torch.no_grad():
            target_tokens = self.target.tokens(imgs)
            target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))

        ctx_ids, target_ids = self._sample_masks(imgs.device)
        n_ctx = ctx_ids.numel()

        ctx = self.context.tokens_from(imgs, ctx_ids.unsqueeze(0).expand(b, -1))
        pred_pos = self.pred_pos[0]
        ctx_tok = self.pred_embed(ctx) + pred_pos[ctx_ids].unsqueeze(0)
        mask_tok = (self.mask_token + pred_pos[target_ids].unsqueeze(0)).expand(b, -1, -1)
        seq = self.predictor(torch.cat([ctx_tok, mask_tok], dim=1))
        pred = self.pred_proj(seq[:, n_ctx:])
        loss = F.mse_loss(pred, target_tokens[:, target_ids])

        mask = torch.zeros(b, N_PATCHES, device=imgs.device)
        mask[:, target_ids] = 1.0
        return loss, None, mask


def build_model(enc_dim: int = DEFAULT_ENC_DIM, n_targets: int = N_TARGETS) -> CNNStemIJEPA:
    return CNNStemIJEPA(enc_dim=enc_dim, pred_dim=enc_dim // 2, n_targets=n_targets)


def stem_for(n_targets: int = N_TARGETS) -> str:
    return CKPT_STEM if n_targets == N_TARGETS else f"{CKPT_STEM}_t{n_targets}"


def model_path(epochs: int, n_targets: int = N_TARGETS) -> Path:
    return final_path(stem_for(n_targets), epochs)


def find_checkpoint(epochs: int | None = None, n_targets: int = N_TARGETS) -> Path:
    stem = stem_for(n_targets)
    if epochs is not None:
        path = model_path(epochs, n_targets)
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {path}. Train one with "
                f"`python -m ijepa_trials.cnn_stem_ijepa --epochs {epochs} --n-targets {n_targets}`."
            )
        return path
    candidates = sorted(
        (
            p
            for p in MODELS_DIR.glob(f"{stem}_*ep.pt")
            if p.stem.rsplit("_", 1)[0] == stem
        ),
        key=lambda p: int(p.stem.rsplit("_", 1)[1].removesuffix("ep")),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for {stem!r} in {MODELS_DIR}. Train one with "
            f"`python -m ijepa_trials.cnn_stem_ijepa`."
        )
    return candidates[-1]


def _ckpt_dict(model, seed, *, opt=None, sched=None, epoch=None) -> dict:
    ckpt: dict = {
        "state_dict": model.state_dict(),
        "config": {
            "arch": ARCH_TAG,
            "stem": "dense-conv-feature-space",
            "input_size": IMG_SIZE,
            "feature_grid": FEATURE_GRID,
            "feature_channels": STEM_CHANNELS,
            "token_feature_patch": TOKEN_FEATURE_PATCH,
            "grid": GRID,
            "n_patches": N_PATCHES,
            "patch_dim": PATCH_DIM,
            "enc_dim": model.embed_dim,
            "pred_dim": model.pred_dim,
            "n_targets": model.n_targets,
            "targets": "single-patch-joint",
            "context": "remaining-feature-patches",
            "pos_embed": "learned-absolute",
            "preproc": True,
            "note": "feature-space JEPA; dense conv receptive fields can overlap across context/target tokens",
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
    n_targets: int = N_TARGETS,
    device: torch.device | None = None,
) -> Path:
    set_seed(seed)
    device = device or pick_device()
    stem = stem_for(n_targets)
    print(
        f"Device: {device}  arch: {ARCH_TAG} (dense conv 28->14->8, "
        f"4x4 feature patches, {n_targets} targets / {N_PATCHES - n_targets} context, "
        f"preproc)  enc_dim: {enc_dim}  seed: {seed}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = model_path(epochs, n_targets)
    if out.exists():
        print(f"Final checkpoint already exists -> {out} (nothing to do)")
        return out

    loader = make_loader(batch_size)
    model = build_model(enc_dim, n_targets=n_targets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loader)
    )

    start_epoch = 1
    resume = find_latest_partial(stem, epochs)
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
            part = partial_path(stem, epoch, epochs)
            torch.save(_ckpt_dict(model, seed, opt=opt, sched=sched, epoch=epoch), part)
            clear_partials(stem, epochs, keep=epoch)
            print(f"  checkpoint -> {part}")

    torch.save(_ckpt_dict(model, seed), out)
    clear_partials(stem, epochs)
    print(f"Saved model -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument(
        "--enc-dim",
        type=int,
        default=DEFAULT_ENC_DIM,
        help="Per-token embedding dim (predictor = enc_dim // 2).",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=N_TARGETS,
        help=f"Number of feature-space target tokens per step. Default {N_TARGETS} "
        f"({N_TARGETS}-{N_PATCHES - N_TARGETS} split).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        enc_dim=args.enc_dim,
        n_targets=args.n_targets,
    )


if __name__ == "__main__":
    main()
