"""Classification on top of the MAE encoder.

Loads the trained MAE encoder and trains a linear classification head on top to
predict MNIST digits, then reports the final train and test error.

Two modes:

* **Frozen (default)** -- a linear *probe*. The encoder never updates; its
  embeddings are computed once and cached, and only the head trains. The
  accuracy measures how linearly separable the encoder's features already are.
* **``--unfreeze``** -- fine-tune end-to-end. The encoder is trainable and is
  optimized together with the head on the labels, so embeddings are recomputed
  every step (no caching). This typically closes most of the gap to a
  supervised model.

    python -m grasp_embeddings.mae_patch_embd.classify
    python -m grasp_embeddings.mae_patch_embd.classify --arch cnn --unfreeze
    python -m grasp_embeddings.mae_patch_embd.classify --no-model-init  # baseline
    python -m grasp_embeddings.mae_patch_embd.classify --brief          # random BRIEF
    python -m grasp_embeddings.mae_patch_embd.classify --brief-mod --grid 8

``--arch {vit,cnn,jepa}`` selects which pretrained encoder to load (and must
match an architecture trained by ``mae.py``). For ``jepa`` the encoder is the
I-JEPA target encoder. This model is evaluation-only, never saved.

``--brief`` / ``--brief-mod`` skip the encoder entirely and probe a fixed,
zero-learning handcrafted descriptor instead (random pairs / structured lattice;
see :mod:`brief`): the BRIEF bit vector is the "embedding" and only the linear
head trains. Always frozen -- ``--unfreeze`` does not apply.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from grasp_embeddings.mae_patch_embd import brief
from grasp_embeddings.mae_patch_embd.mae import (
    ARCHES,
    DATASET_DIR,
    build_model,
    find_checkpoint,
    pick_device,
)

N_CLASSES = 10


def load_encoder(
    device: torch.device, arch: str, random_init: bool, epochs: int | None = None
) -> nn.Module:
    model = build_model(arch).to(device)
    if not random_init:
        ckpt_path = find_checkpoint(arch, epochs)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {ckpt_path.name}")
    return model


def encode(model: nn.Module, imgs: torch.Tensor, pool: str = "mean") -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, D) image embeddings.

    ``pool`` selects how the patch grid is collapsed: ``"mean"`` (D = embed_dim)
    or ``"flatten"`` (concatenate the tokens, keeping the per-patch layout).
    Differentiable -- gradients flow into the encoder when it is unfrozen.
    """
    return model.encode(imgs, pool=pool)


def mnist_loader(train: bool, batch_size: int, shuffle: bool) -> DataLoader:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)


@torch.no_grad()
def extract_features(
    model: nn.Module, train: bool, device: torch.device, pool: str = "mean"
):
    """Run the frozen encoder over a split once, return (features, labels)."""
    model.eval()
    feats, labels = [], []
    for imgs, y in mnist_loader(train, batch_size=512, shuffle=False):
        feats.append(encode(model, imgs.to(device), pool=pool).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def error_features(head: nn.Module, X, y, device) -> float:
    """Misclassification rate over cached features."""
    head.eval()
    correct = 0
    for i in range(0, len(X), 1024):
        pred = head(X[i : i + 1024].to(device)).argmax(dim=1).cpu()
        correct += (pred == y[i : i + 1024]).sum().item()
    return 1.0 - correct / len(X)


@torch.no_grad()
def error_images(model: nn.Module, head: nn.Module, train: bool, device) -> float:
    """Misclassification rate, encoding images on the fly."""
    model.eval()
    head.eval()
    correct, total = 0, 0
    for imgs, y in mnist_loader(train, batch_size=512, shuffle=False):
        pred = head(encode(model, imgs.to(device))).argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return 1.0 - correct / total


def train_linear_probe(Xtr, ytr, Xte, yte, in_dim: int, args, device):
    """Train a linear head on cached features and return (train_err, test_err)."""
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

    return error_features(head, Xtr, ytr, device), error_features(
        head, Xte, yte, device
    )


def run_frozen(model: nn.Module, args, device, pool: str = "mean"):
    """Linear probe: freeze the encoder, train only the head on cached feats."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    print("Extracting embeddings from the frozen encoder...")
    Xtr, ytr = extract_features(model, train=True, device=device, pool=pool)
    Xte, yte = extract_features(model, train=False, device=device, pool=pool)
    print(f"  train: {tuple(Xtr.shape)}   test: {tuple(Xte.shape)}")

    return train_linear_probe(Xtr, ytr, Xte, yte, Xtr.shape[1], args, device)


def extract_brief(kind: str, args, device):
    """BRIEF-describe MNIST as cached descriptor features for the linear probe.

    Returns ``(Xtr, ytr, Xte, yte, n_bits)`` with the bit vectors as float
    tensors -- a fixed, zero-learning "encoder" the linear head probes, exactly
    like the frozen-encoder path.
    """
    features, extent, label = brief.make_features(
        kind, patch=args.patch, n=args.n, seed=args.brief_seed, grid=args.grid
    )
    print(f"Describing MNIST with {label}...")
    tr_desc, tr_lab = brief.describe_split(features, extent, train=True)
    te_desc, te_lab = brief.describe_split(features, extent, train=False)
    Xtr = torch.from_numpy(tr_desc.astype(np.float32))
    Xte = torch.from_numpy(te_desc.astype(np.float32))
    ytr = torch.from_numpy(tr_lab.astype(np.int64))
    yte = torch.from_numpy(te_lab.astype(np.int64))
    return Xtr, ytr, Xte, yte, features.shape[0]


def run_unfrozen(model: nn.Module, enc_dim: int, args, device):
    """Fine-tune the encoder + head end-to-end on the labels."""
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

    return error_images(model, head, True, device), error_images(
        model, head, False, device
    )


def _report(mode: str, train_err: float, test_err: float) -> None:
    print(f"\n--- {mode} ---")
    print(f"Train error: {train_err:.2%}  (acc {1 - train_err:.2%})")
    print(f"Test  error: {test_err:.2%}  (acc {1 - test_err:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHES, default="vit")
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
    brief_group = parser.add_mutually_exclusive_group()
    brief_group.add_argument(
        "--brief",
        action="store_true",
        help="Probe random handcrafted BRIEF descriptors instead of an encoder.",
    )
    brief_group.add_argument(
        "--brief-mod",
        action="store_true",
        help="Probe structured BRIEF descriptors instead of an encoder.",
    )
    parser.add_argument("--patch", type=int, default=4, help="[--brief] frame side.")
    parser.add_argument("--n", type=int, default=64, help="[--brief] feature count.")
    parser.add_argument("--brief-seed", type=int, default=0, help="[--brief] seed.")
    parser.add_argument("--grid", type=int, default=8, help="[--brief-mod] GxG lattice.")
    args = parser.parse_args()

    device = pick_device()
    brief_kind = "brief-mod" if args.brief_mod else "brief" if args.brief else None

    if brief_kind is not None:
        if args.unfreeze:
            parser.error("--unfreeze has no effect with --brief/--brief-mod "
                         "(there is no encoder to fine-tune).")
        print(f"Device: {device}  features: {brief_kind} (no encoder)")
        Xtr, ytr, Xte, yte, in_dim = extract_brief(brief_kind, args, device)
        print(f"  train: {tuple(Xtr.shape)}   test: {tuple(Xte.shape)}")
        train_err, test_err = train_linear_probe(
            Xtr, ytr, Xte, yte, in_dim, args, device
        )
        mode = f"{brief_kind} linear probe"
        _report(mode, train_err, test_err)
        return

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
        train_err, test_err = run_unfrozen(model, enc_dim, args, device)
        mode = "fine-tuned encoder + linear head"
    else:
        pool = "flatten" if args.flatten else "mean"
        print(f"Pooling: {pool}")
        train_err, test_err = run_frozen(model, args, device, pool=pool)
        mode = f"frozen-encoder linear probe ({pool}-pooled)"

    _report(mode, train_err, test_err)


if __name__ == "__main__":
    main()
