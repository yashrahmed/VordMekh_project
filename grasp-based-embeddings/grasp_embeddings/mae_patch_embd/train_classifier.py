"""Train an MNIST classifier on top of the MAE encoder and save it.

Trains a linear classification head to predict MNIST digits, reports its **train
accuracy** (on the 60K train set) at the end, and writes the result to
``models/`` so :mod:`grasp_embeddings.mae_patch_embd.eval_classifier` can score
it on the held-out **test** set. Train accuracy is reported here, where the train
set lives; the eval reports test accuracy only. Three training modes:

* **Frozen (default)** -- a linear *probe*. The encoder never updates; its
  embeddings are computed once and cached, and only the head trains. Measures how
  linearly separable the encoder's features already are.
* **``--unfreeze``** -- fine-tune end-to-end. The encoder is trainable and is
  optimized together with the head on the labels (embeddings recomputed every
  step), so the saved checkpoint stores the *fine-tuned* encoder.
* **``--arch brief`` / ``--arch brief-mod``** -- probe a fixed, zero-learning
  handcrafted BRIEF descriptor (random pairs / structured lattice; see
  :mod:`brief`). Always frozen -- ``--unfreeze`` does not apply.

    python -m grasp_embeddings.mae_patch_embd.train_classifier
    python -m grasp_embeddings.mae_patch_embd.train_classifier --arch cnn --unfreeze
    python -m grasp_embeddings.mae_patch_embd.train_classifier --no-model-init
    python -m grasp_embeddings.mae_patch_embd.train_classifier --arch brief-mod

``--arch {vit,cnn,jepa,ijepa-canonical}`` selects which pretrained encoder to load
(and must match an architecture trained by ``mae.py``); for ``jepa`` /
``ijepa-canonical`` the encoder is the I-JEPA target encoder (the latter trained
with canonical block masking). The checkpoint is named after the choices that define it,
e.g. ``clf_mnist_vit_probe_mean.pt`` / ``clf_mnist_jepa_finetune.pt`` /
``clf_mnist_brief-mod_probe.pt``; pass ``--out`` to override. The saved file is
self-contained -- it carries the head plus its encoder weights (or BRIEF config)
-- so the eval needs nothing but the file.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from grasp_embeddings.mae_patch_embd import brief
from grasp_embeddings.mae_patch_embd.eval_classifier import (
    BRIEF_FAMILY,
    ENCODER_FAMILY,
    N_CLASSES,
    brief_features,
    encode,
    error_features,
    error_images,
    extract_features,
    load_encoder,
    mnist_loader,
)
from grasp_embeddings.mae_patch_embd.mae import ARCHES, MODELS_DIR, pick_device


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/Torch so a probe run is reproducible.

    Covers the two randomness sources on the probe path -- the linear head's
    weight init and the training-loader shuffle (both draw from torch's global
    RNG) -- plus NumPy for the BRIEF descriptor path.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_linear_probe(Xtr, ytr, in_dim: int, args, device) -> nn.Module:
    """Train a linear head on cached features and return the trained head."""
    head = nn.Linear(in_dim, N_CLASSES).to(device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True
    )

    for epoch in range(1, args.epochs + 1):
        head.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(head(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs}  loss {running / seen:.4f}")

    return head


def run_frozen(model: nn.Module, args, device, pool: str = "mean"):
    """Linear probe: freeze the encoder, train only the head on cached feats.

    Returns ``(head, in_dim)``.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    print("Extracting embeddings from the frozen encoder...")
    Xtr, ytr = extract_features(model, train=True, device=device, pool=pool)
    print(f"  train: {tuple(Xtr.shape)}")
    in_dim = Xtr.shape[1]
    return train_linear_probe(Xtr, ytr, in_dim, args, device), in_dim


def run_unfrozen(model: nn.Module, enc_dim: int, args, device) -> nn.Module:
    """Fine-tune the encoder + head end-to-end on the labels; return the head.

    The encoder is mutated in place, so the caller saves ``model`` afterward.
    """
    # Make the whole encoder trainable. Matters for the I-JEPA target encoder,
    # which ships frozen (requires_grad=False) from EMA pretraining; a no-op for
    # the MAE/ConvMAE encoders, whose parameters are already trainable.
    for p in model.parameters():
        p.requires_grad_(True)

    head = nn.Linear(enc_dim, N_CLASSES).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    crit = nn.CrossEntropyLoss()
    loader = mnist_loader(train=True, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        running, seen = 0.0, 0
        for imgs, y in loader:
            imgs, y = imgs.to(device), y.to(device)
            loss = crit(head(encode(model, imgs)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs}  loss {running / seen:.4f}")

    return head


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def default_path(arch: str, mode: str, pool: str | None) -> Path:
    """Name the checkpoint after the choices that define it.

    ``clf_mnist_<arch>_<mode>[_<pool>].pt`` -- the pool suffix is kept only for
    the encoder probe (the one place it varies) so probe runs of different pooling
    don't clobber each other.
    """
    parts = ["clf_mnist", arch, mode]
    if mode == "probe" and pool is not None:
        parts.append(pool)
    return MODELS_DIR / ("_".join(parts) + ".pt")


def save_classifier(ckpt: dict, out: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved classifier -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        choices=(*ARCHES, *brief.BRIEF_ARCHES),
        default="vit",
        help="Learned encoder (vit/cnn/jepa/ijepa-canonical) or a handcrafted BRIEF "
        "descriptor (brief = random pairs, brief-mod = structured lattice, no "
        "encoder).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Fine-tune the encoder end-to-end instead of a frozen probe.",
    )
    parser.add_argument(
        "--no-model-init",
        action="store_true",
        help="Start from a random, untrained encoder as a baseline.",
    )
    parser.add_argument(
        "--ckpt-epochs",
        type=int,
        default=None,
        help="Pretraining length of the encoder checkpoint to load "
        "(default: the most-trained one on disk).",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Probe the concatenated patch tokens instead of the mean-pooled "
        "embedding (frozen only; keeps per-patch layout).",
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
        "--out",
        type=str,
        default=None,
        help="Checkpoint path (default: models/clf_mnist_<arch>_<mode>[_<pool>].pt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the head init and loader shuffle (reproducible runs).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"Seed: {args.seed}")
    brief_kind = args.arch if args.arch in brief.BRIEF_ARCHES else None

    # --- handcrafted BRIEF: probe the fixed bit vector, no encoder ---
    if brief_kind is not None:
        if args.unfreeze:
            parser.error("--unfreeze has no effect with --arch brief/brief-mod "
                         "(there is no encoder to fine-tune).")
        if args.flatten:
            parser.error("--flatten does not apply to --arch brief/brief-mod "
                         "(BRIEF has no patch tokens).")
        print(f"Device: {device}  features: {brief_kind} (no encoder)")
        grid = brief.BRIEF_MOD_GRID
        Xtr, ytr, _, _, in_dim = brief_features(
            brief_kind, args.patch, args.n, args.brief_seed, grid
        )
        print(f"  train: {tuple(Xtr.shape)}")
        head = train_linear_probe(Xtr, ytr, in_dim, args, device)

        train_err = error_features(head, Xtr, ytr, device)
        print(f"Train accuracy: {1 - train_err:.2%}  (error {train_err:.2%})")

        out = Path(args.out) if args.out else default_path(brief_kind, "probe", None)
        save_classifier(
            {
                "family": BRIEF_FAMILY,
                "arch": brief_kind,
                "mode": "probe",
                "mode_label": f"{brief_kind} linear probe",
                "pool": None,
                "in_dim": in_dim,
                "n_classes": N_CLASSES,
                "head_state_dict": head.state_dict(),
                "encoder_state_dict": None,
                "brief_cfg": {
                    "patch": args.patch,
                    "n": args.n,
                    "brief_seed": args.brief_seed,
                    "grid": grid,
                },
            },
            out,
        )
        return

    # --- learned encoder: linear probe or fine-tune ---
    print(f"Device: {device}  arch: {args.arch}")
    if args.no_model_init:
        print("Starting from an UNINITIALIZED (untrained) encoder.")

    if args.flatten and args.unfreeze:
        parser.error("--flatten is frozen-probe only (use it without --unfreeze).")

    model = load_encoder(
        device, args.arch, random_init=args.no_model_init, epochs=args.ckpt_epochs
    )
    enc_dim = model.embed_dim

    if args.unfreeze:
        print("Fine-tuning: encoder UNFROZEN, training encoder + head.")
        head = run_unfrozen(model, enc_dim, args, device)
        mode, pool, in_dim = "finetune", "mean", enc_dim
        mode_label = "fine-tuned encoder + linear head"
    else:
        pool = "flatten" if args.flatten else "mean"
        print(f"Pooling: {pool}")
        head, in_dim = run_frozen(model, args, device, pool=pool)
        mode = "probe"
        mode_label = f"frozen-encoder linear probe ({pool}-pooled)"

    train_err = error_images(model, head, train=True, device=device, pool=pool)
    print(f"Train accuracy: {1 - train_err:.2%}  (error {train_err:.2%})")

    out = Path(args.out) if args.out else default_path(args.arch, mode, pool)
    save_classifier(
        {
            "family": ENCODER_FAMILY,
            "arch": args.arch,
            "mode": mode,
            "mode_label": mode_label,
            "pool": pool,
            "in_dim": in_dim,
            "n_classes": N_CLASSES,
            "head_state_dict": head.state_dict(),
            "encoder_state_dict": model.state_dict(),
            "brief_cfg": None,
        },
        out,
    )


if __name__ == "__main__":
    main()
