"""Evaluate a saved I-JEPA flatten probe and print its test accuracy.

Loads a classifier checkpoint written by :mod:`ijepa_trials.train_probe`,
rebuilds the head plus its encoder entirely from the file -- nothing else is
needed -- and reports the misclassification rate on the held-out MNIST **test**
split. The test images get the same bbox preprocessing the probe was trained
under (always on). Train accuracy is reported by the trainer; eval only ever
touches the test set.

    python -m ijepa_trials.eval_probe \
        --model models/ijepa_clf_probe_flatten_50ep.pt
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from ijepa_trials.train_probe import N_CLASSES, POOL
from ijepa_trials.ijepa import build_model
from trials.eval_classifier import mnist_loader
from trials.mae import pick_device


@torch.no_grad()
def test_error(model: nn.Module, head: nn.Module, device) -> float:
    """Misclassification rate on the preprocessed MNIST test split (flatten)."""
    model.eval()
    head.eval()
    correct, total = 0, 0
    for imgs, y in mnist_loader(train=False, batch_size=512, shuffle=False, preproc=True):
        pred = head(model.encode(imgs.to(device), pool=POOL)).argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return 1.0 - correct / total


def evaluate(ckpt: dict, device: torch.device) -> float:
    """Rebuild the (frozen-probe or fine-tuned) encoder + head and score the test set."""
    head = nn.Linear(ckpt["in_dim"], ckpt.get("n_classes", N_CLASSES)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])

    model = build_model().to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])
    return test_error(model, head, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a probe checkpoint saved by ijepa_trials.train_probe.",
    )
    args = parser.parse_args()

    device = pick_device()
    ckpt = torch.load(args.model, map_location=device)
    print(f"Device: {device}  model: {args.model}")
    print(f"  family: {ckpt['family']}  mode: {ckpt['mode']}  pool: {ckpt.get('pool', POOL)}")

    err = evaluate(ckpt, device)
    print(f"\n--- {ckpt.get('mode_label', ckpt['mode'])} ---")
    print(f"Test error: {err:.2%}  (acc {1 - err:.2%})")


if __name__ == "__main__":
    main()
