"""A small modern Masked Autoencoder (MAE) trained on MNIST.

Three encoder architectures, selected with ``--arch``:

* ``vit`` -- a ViT-style MAE in the spirit of He et al., 2021 ("Masked
  Autoencoders Are Scalable Vision Learners"): patchify the image, randomly
  *drop* a large fraction of the patches, encode only the visible ones with a
  Transformer, then decode from the visible tokens plus learned mask tokens.
* ``cnn`` -- a masked conv autoencoder (Context-Encoder style, Pathak 2016).
  Convolutions need a dense grid, so masked patches are *zeroed* in the input
  rather than dropped; a conv encoder -> conv decoder reconstructs the image.
* ``jepa`` -- an I-JEPA (Assran et al., 2023, "Self-Supervised Learning from
  Images with a Joint-Embedding Predictive Architecture"): a trainable context
  encoder encodes the visible patches, an EMA *target* encoder encodes the full
  image into patch-level targets, and a predictor predicts the target's latent
  representations at the masked positions.

For ``vit``/``cnn`` the loss is the MSE on the masked *pixels*; for ``jepa`` it
is the MSE on the masked patches' *latent representations* -- prediction happens
in representation space, not pixel space.

    python -m grasp_embeddings.mae_patch_embd.mae                   # vit
    python -m grasp_embeddings.mae_patch_embd.mae --arch cnn --epochs 50
    python -m grasp_embeddings.mae_patch_embd.mae --arch jepa --epochs 50

Downloads MNIST into <project>/dataset and writes the trained weights to
<project>/models/mae_mnist_<arch>_<epochs>ep.pt (both are gitignored). Here <project> is
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


class ViTTower(nn.Module):
    """patch-embed + positional embedding + Transformer -> per-patch tokens.

    The shared building block for I-JEPA's twin encoders. Unlike :class:`MAE`,
    it carries no decoder -- it just maps patches to a grid of token vectors.
    """

    def __init__(
        self,
        patch_dim: int,
        n_patches: int,
        dim: int,
        depth: int,
        heads: int,
    ):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, dim))
        self.encoder = TransformerEncoder(dim, depth, heads)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, N, dim): encode every patch (no masking)."""
        x = self.patch_embed(patchify(imgs)) + self.pos
        return self.encoder(x)

    def tokens_from(
        self, patches: torch.Tensor, ids_keep: torch.Tensor
    ) -> torch.Tensor:
        """Encode only the ``ids_keep`` patches, with their position embeddings.

        ``patches`` is (B, N, patch_dim); ``ids_keep`` is (B, n_keep). Returns
        (B, n_keep, dim).
        """
        b, n_keep = ids_keep.shape
        d = self.pos.size(-1)
        kept = torch.gather(
            patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        )
        pos = torch.gather(
            self.pos.expand(b, -1, -1), 1, ids_keep.unsqueeze(-1).expand(-1, -1, d)
        )
        return self.encoder(self.patch_embed(kept) + pos)


class JEPA(nn.Module):
    """I-JEPA: predict masked patches' *latent* representations (Assran 2023).

    Three parts:

    * **context encoder** (trainable :class:`ViTTower`) -- sees only the visible
      (context) patches.
    * **target encoder** (an EMA copy of the context encoder, no gradients) --
      sees the *full* image and produces the patch-level prediction targets.
      Being an EMA with stop-gradient is what prevents representation collapse.
    * **predictor** (a slim Transformer) -- given the encoded context tokens plus
      learned mask tokens at the target positions, predicts the target encoder's
      representations there.

    The loss is the MSE between the predictor's output and the (stop-gradient)
    target representations at the masked positions -- entirely in latent space,
    with no pixel decoder. Downstream :meth:`encode` uses the *target* encoder
    over the full image (it is the tower that has always seen whole images).
    """

    def __init__(
        self,
        patch_dim: int = PATCH_SIZE * PATCH_SIZE,
        n_patches: int = (IMG_SIZE // PATCH_SIZE) ** 2,
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

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, embed_dim): target encoder, mean-pooled tokens.

        Differentiable, so gradients flow into the target encoder when it is
        unfrozen for fine-tuning downstream.
        """
        return self.target.tokens(imgs).mean(dim=1)

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        b = imgs.size(0)
        patches = patchify(imgs)  # (B, N, patch_dim)
        n, d = self.n_patches, self.embed_dim

        # --- targets: full-image target encoder, stop-gradient ---
        # LayerNorm the targets over the feature dim before the loss, matching
        # I-JEPA's loss_fn (Assran 2023): keeps the prediction target on a
        # stable scale as the EMA encoder drifts.
        with torch.no_grad():
            target_tokens = self.target.tokens(imgs)  # (B, N, enc_dim)
            target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))

        # --- split patches into context (visible) and predicted (masked) ---
        keep = max(1, int(round(n * (1 - mask_ratio))))
        ids_shuffle = torch.rand(b, n, device=imgs.device).argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]  # context positions
        ids_pred = ids_shuffle[:, keep:]  # masked positions to predict
        n_pred = n - keep

        # --- encode context with the context encoder ---
        ctx = self.context.tokens_from(patches, ids_keep)  # (B, keep, enc_dim)

        # --- predictor: context tokens + mask tokens at target positions ---
        pred_pos = self.pred_pos.expand(b, -1, -1)
        ctx_tok = self.pred_embed(ctx) + torch.gather(
            pred_pos, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.pred_dim)
        )
        mask_tok = self.mask_token.expand(b, n_pred, -1) + torch.gather(
            pred_pos, 1, ids_pred.unsqueeze(-1).expand(-1, -1, self.pred_dim)
        )
        seq = self.predictor(torch.cat([ctx_tok, mask_tok], dim=1))
        pred = self.pred_proj(seq[:, keep:])  # (B, n_pred, enc_dim)

        # --- latent-space MSE on the masked positions only ---
        tgt = torch.gather(
            target_tokens, 1, ids_pred.unsqueeze(-1).expand(-1, -1, d)
        )
        loss = F.mse_loss(pred, tgt)

        # 0/1 mask (1 = predicted) in original patch order, for interface parity.
        mask = torch.ones(b, n, device=imgs.device)
        mask.scatter_(1, ids_keep, 0.0)
        return loss, pred, mask


# --------------------------------------------------------------------------- #
# Architecture registry
# --------------------------------------------------------------------------- #
ARCHES = ("vit", "cnn", "jepa")


def build_model(arch: str) -> nn.Module:
    """Construct an MAE for the given architecture name."""
    if arch == "vit":
        return MAE()
    if arch == "cnn":
        return ConvMAE()
    if arch == "jepa":
        return JEPA()
    raise ValueError(f"Unknown arch {arch!r}; choose from {ARCHES}.")


def model_path(arch: str, epochs: int) -> Path:
    """Checkpoint path for an architecture trained for ``epochs`` epochs.

    The epoch count is in the filename so runs of different length don't clobber
    each other (e.g. ``mae_mnist_vit_50ep.pt`` vs ``mae_mnist_vit_300ep.pt``).
    """
    return MODELS_DIR / f"mae_mnist_{arch}_{epochs}ep.pt"


def find_checkpoint(arch: str, epochs: int | None = None) -> Path:
    """Resolve a trained checkpoint for ``arch``.

    With ``epochs`` given, returns that exact path. Otherwise returns the
    most-trained checkpoint on disk (largest epoch count), so evals pick up the
    longest run by default. Raises ``FileNotFoundError`` if none exists.
    """
    if epochs is not None:
        path = model_path(arch, epochs)
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {path}. Train one with "
                f"`python -m grasp_embeddings.mae_patch_embd.mae --arch {arch} "
                f"--epochs {epochs}`."
            )
        return path

    candidates = sorted(
        MODELS_DIR.glob(f"mae_mnist_{arch}_*ep.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1].removesuffix("ep")),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for arch {arch!r} in {MODELS_DIR} "
            f"(looked for mae_mnist_{arch}_*ep.pt). Train one with "
            f"`python -m grasp_embeddings.mae_patch_embd.mae --arch {arch}`."
        )
    return candidates[-1]


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
            if hasattr(model, "ema_update"):
                model.ema_update()  # I-JEPA: nudge the target encoder
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        print(f"epoch {epoch:3d}/{epochs}  recon_mse {running / seen:.5f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = model_path(arch, epochs)
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
