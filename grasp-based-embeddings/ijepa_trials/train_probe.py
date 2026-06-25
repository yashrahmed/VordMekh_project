"""Train a linear probe on the flattened I-JEPA output (or fine-tune end-to-end).

Puts a linear classification head on the **flattened** encoder output -- the N
patch tokens concatenated (D = n_patches * embed_dim), keeping the per-patch
layout rather than mean-pooling. Reports train accuracy on the 60K train set and
writes a self-contained checkpoint (head + encoder weights + config) that
:mod:`ijepa_trials.eval_probe` scores on the held-out test set.

Two modes, chosen with ``--unfreeze`` (default: frozen):

* **frozen (default)** -- a linear *probe*. The encoder never updates; its
  flattened embeddings are computed once, cached, and only the head trains.
* **``--unfreeze``** -- fine-tune the encoder + head end-to-end on the labels
  (embeddings recomputed every step); the saved checkpoint stores the fine-tuned
  encoder.

Images are **always** bbox-preprocessed and the seed defaults to 0. Training is
checkpointed every 50 epochs to a resumable partial; re-running the same
``--epochs``/mode resumes from the latest, and the partials are pruned once the
final checkpoint is written.

    python -m ijepa_trials.train_probe
    python -m ijepa_trials.train_probe --unfreeze --epochs 50

Writes <project>/models/ijepa_clf_<mode>_flatten_<epochs>ep.pt (gitignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ijepa_trials._ckpt import (
    CKPT_INTERVAL,
    MODELS_DIR,
    clear_partials,
    final_path,
    find_latest_partial,
    partial_path,
)
from ijepa_trials.ijepa import N_PATCHES, build_model, find_checkpoint, set_seed
from trials.eval_classifier import mnist_loader
from trials.mae import pick_device

N_CLASSES = 10
POOL = "flatten"  # the spec: always probe the flattened output
FAMILY = "ijepa-flatten-probe"


def ckpt_stem(mode: str) -> str:
    """Partial/final filename stem for a given mode (``probe`` / ``finetune``)."""
    return f"ijepa_clf_{mode}_{POOL}"


# --------------------------------------------------------------------------- #
# Encoder loading + feature extraction
# --------------------------------------------------------------------------- #
def load_encoder(device: torch.device, epochs: int | None) -> nn.Module:
    """Build the I-JEPA and load its pretrained weights (most-trained by default)."""
    model = build_model().to(device)
    ckpt_path = find_checkpoint(epochs)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded encoder checkpoint: {ckpt_path.name}")
    return model


@torch.no_grad()
def extract_flatten_features(model: nn.Module, device: torch.device):
    """Run the frozen encoder over the (preprocessed) train split once.

    Returns ``(X, y)`` with X the flattened per-patch embeddings.
    """
    model.eval()
    feats, labels = [], []
    for imgs, y in mnist_loader(train=True, batch_size=512, shuffle=False, preproc=True):
        feats.append(model.encode(imgs.to(device), pool=POOL).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# --------------------------------------------------------------------------- #
# Resumable-checkpoint helpers (partials carry head + encoder + optimizer)
# --------------------------------------------------------------------------- #
def _save_partial(stem, epoch, total, head, model, opt) -> None:
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "encoder_state_dict": model.state_dict(),
            "optim_state_dict": opt.state_dict(),
            "epoch": epoch,
        },
        partial_path(stem, epoch, total),
    )
    clear_partials(stem, total, keep=epoch)  # write-then-prune: never 0 partials
    print(f"  checkpoint -> {partial_path(stem, epoch, total)}")


def _maybe_resume(stem, total, head, model, opt, device) -> int:
    """Reload the latest partial for this run; return the epoch to start from."""
    resume = find_latest_partial(stem, total)
    if resume is None:
        return 1
    path, done = resume
    ckpt = torch.load(path, map_location=device)
    head.load_state_dict(ckpt["head_state_dict"])
    model.load_state_dict(ckpt["encoder_state_dict"])
    opt.load_state_dict(ckpt["optim_state_dict"])
    print(f"Resuming from {path} -> epoch {done + 1}/{total}")
    return done + 1


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_frozen(model: nn.Module, args, device):
    """Linear probe: freeze the encoder, train only the head on cached feats."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    print("Extracting flattened embeddings from the frozen encoder...")
    Xtr, ytr = extract_flatten_features(model, device)
    print(f"  train: {tuple(Xtr.shape)}")
    in_dim = Xtr.shape[1]

    head = nn.Linear(in_dim, N_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)

    stem = ckpt_stem("probe")
    start = _maybe_resume(stem, args.epochs, head, model, opt, device)
    for epoch in range(start, args.epochs + 1):
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
        if epoch % CKPT_INTERVAL == 0 and epoch < args.epochs:
            _save_partial(stem, epoch, args.epochs, head, model, opt)
    clear_partials(stem, args.epochs)
    return head, in_dim


def run_unfrozen(model: nn.Module, args, device):
    """Fine-tune the encoder + head end-to-end on the labels (flatten-pooled)."""
    # Make the whole encoder trainable -- the I-JEPA target encoder ships frozen
    # (requires_grad=False) from EMA pretraining.
    for p in model.parameters():
        p.requires_grad_(True)

    in_dim = N_PATCHES * model.embed_dim
    head = nn.Linear(in_dim, N_CLASSES).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    crit = nn.CrossEntropyLoss()
    loader = mnist_loader(train=True, batch_size=args.batch_size, shuffle=True, preproc=True)

    stem = ckpt_stem("finetune")
    start = _maybe_resume(stem, args.epochs, head, model, opt, device)
    for epoch in range(start, args.epochs + 1):
        model.train()
        head.train()
        running, seen = 0.0, 0
        for imgs, y in loader:
            imgs, y = imgs.to(device), y.to(device)
            loss = crit(head(model.encode(imgs, pool=POOL)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs}  loss {running / seen:.4f}")
        if epoch % CKPT_INTERVAL == 0 and epoch < args.epochs:
            _save_partial(stem, epoch, args.epochs, head, model, opt)
    clear_partials(stem, args.epochs)
    return head, in_dim


# --------------------------------------------------------------------------- #
# Scoring + saving
# --------------------------------------------------------------------------- #
@torch.no_grad()
def train_error(model: nn.Module, head: nn.Module, device) -> float:
    """Misclassification rate on the (preprocessed) train split, flatten-pooled."""
    model.eval()
    head.eval()
    correct, total = 0, 0
    for imgs, y in mnist_loader(train=True, batch_size=512, shuffle=False, preproc=True):
        pred = head(model.encode(imgs.to(device), pool=POOL)).argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return 1.0 - correct / total


def save_classifier(ckpt: dict, out: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved classifier -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--ckpt-epochs",
        type=int,
        default=None,
        help="Pretraining length of the encoder checkpoint to load "
        "(default: the most-trained one on disk).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Checkpoint path (default: models/ijepa_clf_<mode>_flatten_<epochs>ep.pt).",
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
    print(f"Seed: {args.seed}  Device: {device}  pool: {POOL} (preproc)")

    model = load_encoder(device, args.ckpt_epochs)

    if args.unfreeze:
        print("Fine-tuning: encoder UNFROZEN, training encoder + head.")
        head, in_dim = run_unfrozen(model, args, device)
        mode, mode_label = "finetune", "fine-tuned encoder + linear head (flatten)"
    else:
        print("Frozen linear probe (flatten-pooled).")
        head, in_dim = run_frozen(model, args, device)
        mode, mode_label = "probe", "frozen-encoder linear probe (flatten-pooled)"

    err = train_error(model, head, device)
    print(f"Train accuracy: {1 - err:.2%}  (error {err:.2%})")

    out = Path(args.out) if args.out else final_path(ckpt_stem(mode), args.epochs)
    save_classifier(
        {
            "family": FAMILY,
            "arch": "scatter",
            "mode": mode,
            "mode_label": mode_label,
            "pool": POOL,
            "preproc": True,
            "in_dim": in_dim,
            "n_classes": N_CLASSES,
            "head_state_dict": head.state_dict(),
            "encoder_state_dict": model.state_dict(),
        },
        out,
    )


if __name__ == "__main__":
    main()
