"""A small modern Masked Autoencoder (MAE) trained on MNIST.

Two encoder architectures, selected with ``--arch``:

* ``vit`` -- a ViT-style MAE in the spirit of He et al., 2021 ("Masked
  Autoencoders Are Scalable Vision Learners"): patchify the image, randomly
  *drop* a large fraction of the patches, encode only the visible ones with a
  Transformer, then decode from the visible tokens plus learned mask tokens.
* ``cnn`` -- a masked conv autoencoder (Context-Encoder style, Pathak 2016).
  Convolutions need a dense grid, so masked patches are *zeroed* in the input
  rather than dropped; a conv encoder -> conv decoder reconstructs the image.

Either way the loss is the MSE on the masked patches only.

    python -m grasp_embeddings.mae_patch_embd.mae                   # vit
    python -m grasp_embeddings.mae_patch_embd.mae --arch cnn --epochs 50

Downloads MNIST into <project>/dataset and writes the trained weights to
<project>/models/mae_mnist_<arch>.pt (both are gitignored). Here <project> is
the grasp-based-embeddings root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"

IMG_SIZE = 28
PATCH_SIZE = 7  # -> 4x4 = 16 patches of 49 pixels each


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Patch helpers
# --------------------------------------------------------------------------- #
def patchify(imgs: torch.Tensor, patch: int = PATCH_SIZE) -> torch.Tensor:
    """(B, 1, H, W) -> (B, N, patch*patch) flattened non-overlapping patches."""
    b, c, h, w = imgs.shape
    gh, gw = h // patch, w // patch
    x = imgs.reshape(b, c, gh, patch, gw, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, gh, gw, patch, patch, c)
    return x.reshape(b, gh * gw, patch * patch * c)


def unpatchify(patches: torch.Tensor, patch: int = PATCH_SIZE) -> torch.Tensor:
    """(B, N, patch*patch) -> (B, 1, H, W). Inverse of :func:`patchify`."""
    b, n, _ = patches.shape
    g = int(n**0.5)
    x = patches.reshape(b, g, g, patch, patch, 1)
    x = x.permute(0, 5, 1, 3, 2, 4)  # (B, c, gh, patch, gw, patch)
    return x.reshape(b, 1, g * patch, g * patch)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TransformerEncoder(nn.Module):
    """A thin stack of pre-norm Transformer encoder blocks."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(x))


class MAE(nn.Module):
    def __init__(
        self,
        patch_dim: int = PATCH_SIZE * PATCH_SIZE,
        n_patches: int = (IMG_SIZE // PATCH_SIZE) ** 2,
        enc_dim: int = 128,
        enc_depth: int = 4,
        enc_heads: int = 4,
        dec_dim: int = 64,
        dec_depth: int = 2,
        dec_heads: int = 4,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.patch_dim = patch_dim
        self.embed_dim = enc_dim

        # Encoder: visible patches -> tokens.
        self.patch_embed = nn.Linear(patch_dim, enc_dim)
        self.enc_pos = nn.Parameter(torch.zeros(1, n_patches, enc_dim))
        self.encoder = TransformerEncoder(enc_dim, enc_depth, enc_heads)

        # Decoder: full token grid (encoded visible + mask tokens) -> pixels.
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, n_patches, dec_dim))
        self.decoder = TransformerEncoder(dec_dim, dec_depth, dec_heads)
        self.dec_pred = nn.Linear(dec_dim, patch_dim)

        nn.init.trunc_normal_(self.enc_pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """Per-sample shuffle then keep the first ``(1 - mask_ratio)`` tokens.

        Returns the kept tokens, a 0/1 mask (1 = masked) and the restore index
        used to scatter tokens back into their original order.
        """
        b, n, d = x.shape
        keep = max(1, int(round(n * (1 - mask_ratio))))
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :keep]
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))

        mask = torch.ones(b, n, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_kept, mask, ids_restore

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, embed_dim): encode every patch, mean-pool.

        No masking is applied (the full image is seen). Differentiable, so
        gradients flow into the encoder when it is unfrozen downstream.
        """
        tokens = self.patch_embed(patchify(imgs)) + self.enc_pos
        feats = self.encoder(tokens)  # (B, N, embed_dim)
        return feats.mean(dim=1)

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        target = patchify(imgs)  # (B, N, patch_dim)

        # --- encode visible patches ---
        tokens = self.patch_embed(target) + self.enc_pos
        vis, mask, ids_restore = self.random_masking(tokens, mask_ratio)
        vis = self.encoder(vis)

        # --- assemble full grid and decode ---
        dec = self.enc_to_dec(vis)
        b, kept, d = dec.shape
        n_mask = self.n_patches - kept
        mask_tokens = self.mask_token.expand(b, n_mask, -1)
        full = torch.cat([dec, mask_tokens], dim=1)
        full = torch.gather(
            full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d)
        )  # unshuffle to original patch order
        full = full + self.dec_pos
        pred = self.dec_pred(self.decoder(full))  # (B, N, patch_dim)

        # --- reconstruction loss on masked patches only ---
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, pred, mask


class ConvMAE(nn.Module):
    """Masked conv autoencoder (Context-Encoder style, Pathak 2016).

    Unlike the ViT MAE, convolutions slide over a dense grid, so masked patches
    cannot simply be dropped -- they are *zeroed* in the input image. A conv
    encoder downsamples 28 -> 14 -> 7, and a conv decoder upsamples 7 -> 14 ->
    28 to reconstruct. The loss is the MSE on the masked patch regions only.
    The image-level embedding is the global average pool of the final encoder
    feature map, mirroring the ViT's mean-pool over tokens.
    """

    def __init__(self, embed_dim: int = 128, patch: int = PATCH_SIZE):
        super().__init__()
        self.patch = patch
        self.grid = IMG_SIZE // patch  # 4
        self.n_patches = self.grid**2
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28 -> 14
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.GELU(),
            nn.Conv2d(64, embed_dim, 3, padding=1),  # 7 -> 7
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 4, stride=2, padding=1),  # 7 ->14
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14 -> 28
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),  # 28 -> 28
        )

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, embed_dim): conv-encode, global-avg-pool.

        No masking (the full image is seen). Differentiable for fine-tuning.
        """
        feat = self.encoder(imgs)  # (B, embed_dim, 7, 7)
        return feat.mean(dim=(2, 3))

    def _patch_mask(self, b: int, device, mask_ratio: float) -> torch.Tensor:
        """Per-sample 0/1 patch mask (1 = masked) of shape (B, n_patches)."""
        n = self.n_patches
        keep = max(1, int(round(n * (1 - mask_ratio))))
        noise = torch.rand(b, n, device=device)
        ids_keep = noise.argsort(dim=1)[:, :keep]
        mask = torch.ones(b, n, device=device)
        mask.scatter_(1, ids_keep, 0.0)
        return mask

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        b = imgs.size(0)
        mask = self._patch_mask(b, imgs.device, mask_ratio)  # (B, n_patches)

        # Upsample the patch mask to a pixel mask and zero the masked patches.
        pix = mask.reshape(b, 1, self.grid, self.grid)
        pix = pix.repeat_interleave(self.patch, dim=2).repeat_interleave(
            self.patch, dim=3
        )  # (B, 1, 28, 28)
        corrupted = imgs * (1.0 - pix)

        pred = self.decoder(self.encoder(corrupted))  # (B, 1, 28, 28)

        # Reconstruction loss on the masked pixels only.
        loss = ((pred - imgs) ** 2 * pix).sum() / pix.sum().clamp_min(1.0)
        return loss, pred, mask


# --------------------------------------------------------------------------- #
# Architecture registry
# --------------------------------------------------------------------------- #
ARCHES = ("vit", "cnn")


def build_model(arch: str) -> nn.Module:
    """Construct an MAE for the given architecture name."""
    if arch == "vit":
        return MAE()
    if arch == "cnn":
        return ConvMAE()
    raise ValueError(f"Unknown arch {arch!r}; choose from {ARCHES}.")


def model_path(arch: str) -> Path:
    """Checkpoint path for the given architecture."""
    return MODELS_DIR / f"mae_mnist_{arch}.pt"


# --------------------------------------------------------------------------- #
# Data + training
# --------------------------------------------------------------------------- #
def make_loader(batch_size: int) -> DataLoader:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()  # [0, 1], shape (1, 28, 28)
    train = datasets.MNIST(
        root=str(DATASET_DIR), train=True, download=True, transform=tfm
    )
    return DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )


def train(
    arch: str = "vit",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    mask_ratio: float = 0.75,
    device: torch.device | None = None,
) -> Path:
    device = device or pick_device()
    print(f"Device: {device}  arch: {arch}")

    loader = make_loader(batch_size)
    model = build_model(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loader)
    )

    model.train()
    for epoch in range(1, epochs + 1):
        running, seen = 0.0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            loss, _, _ = model(imgs, mask_ratio)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        print(f"epoch {epoch:3d}/{epochs}  recon_mse {running / seen:.5f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = model_path(arch)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "arch": arch,
                "patch_size": PATCH_SIZE,
                "img_size": IMG_SIZE,
                "mask_ratio": mask_ratio,
            },
        },
        out,
    )
    print(f"Saved model -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHES, default="vit")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    args = parser.parse_args()
    train(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
    )


if __name__ == "__main__":
    main()
