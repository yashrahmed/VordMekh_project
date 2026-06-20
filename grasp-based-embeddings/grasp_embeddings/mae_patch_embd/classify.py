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

``--arch {vit,cnn}`` selects which pretrained encoder to load (and must match an
architecture trained by ``mae.py``). This model is evaluation-only, never saved.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from grasp_embeddings.mae_patch_embd.mae import (
    ARCHES,
    DATASET_DIR,
    build_model,
    model_path,
    pick_device,
)

N_CLASSES = 10


def load_encoder(device: torch.device, arch: str, random_init: bool) -> nn.Module:
    model = build_model(arch).to(device)
    if not random_init:
        ckpt_path = model_path(arch)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Train one first with "
                f"`python -m grasp_embeddings.mae_patch_embd.mae --arch {arch}`."
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    return model


def encode(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """(B, 1, 28, 28) -> (B, embed_dim) image embeddings.

    Differentiable -- gradients flow into the encoder when it is unfrozen.
    """
    return model.encode(imgs)


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
def extract_features(model: nn.Module, train: bool, device: torch.device):
    """Run the frozen encoder over a split once, return (features, labels)."""
    model.eval()
    feats, labels = [], []
    for imgs, y in mnist_loader(train, batch_size=512, shuffle=False):
        feats.append(encode(model, imgs.to(device)).cpu())
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


def run_frozen(model: nn.Module, enc_dim: int, args, device):
    """Linear probe: freeze the encoder, train only the head on cached feats."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    print("Extracting embeddings from the frozen encoder...")
    Xtr, ytr = extract_features(model, train=True, device=device)
    Xte, yte = extract_features(model, train=False, device=device)
    print(f"  train: {tuple(Xtr.shape)}   test: {tuple(Xte.shape)}")

    head = nn.Linear(enc_dim, N_CLASSES).to(device)
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


def run_unfrozen(model: nn.Module, enc_dim: int, args, device):
    """Fine-tune the encoder + head end-to-end on the labels."""
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
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}  arch: {args.arch}")
    if args.no_model_init:
        print("Starting from an UNINITIALIZED (untrained) encoder.")

    model = load_encoder(device, args.arch, random_init=args.no_model_init)
    enc_dim = model.embed_dim

    if args.unfreeze:
        print("Fine-tuning: encoder UNFROZEN, training encoder + head.")
        train_err, test_err = run_unfrozen(model, enc_dim, args, device)
        mode = "fine-tuned encoder + linear head"
    else:
        train_err, test_err = run_frozen(model, enc_dim, args, device)
        mode = "frozen-encoder linear probe"

    print(f"\n--- {mode} ---")
    print(f"Train error: {train_err:.2%}  (acc {1 - train_err:.2%})")
    print(f"Test  error: {test_err:.2%}  (acc {1 - test_err:.2%})")


if __name__ == "__main__":
    main()
