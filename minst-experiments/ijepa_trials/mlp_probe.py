"""Train a two-layer MLP on the best frozen custom I-JEPA backbone.

The default backbone is the best configuration currently recorded in this
project: 56x56 bbox-normalized inputs, 7px patches (64 tokens), 48 target / 16
context tokens, encoder dimension 128, and 300 pretraining epochs. The target
encoder is frozen, its flattened embeddings are cached, and only this head is
trained::

    Linear(8192, 256) -> GELU -> Dropout(0.1) -> Linear(256, 10)

One continuous, seed-0 run is evaluated at head epochs 50, 75, and 100. This is
important: the three measurements differ only by training duration, rather than
independent initialization or data-order noise.

    uv run python -m ijepa_trials.mlp_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ijepa_trials import custom_ijepa
from ijepa_trials._ckpt import MODELS_DIR, set_seed
from ijepa_trials.train_probe import N_CLASSES, TwoLayerMLP, extract_features
from trials.eval_classifier import error_features
from trials.mae import pick_device

FAMILY = "ijepa-probe"
HEAD_TYPE = "mlp-2layer"
DEFAULT_BACKBONE_EPOCHS = 300
DEFAULT_N_TARGETS = 48
DEFAULT_MILESTONES = (50, 75, 100)


def parse_milestones(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, strictly positive epoch list."""
    try:
        epochs = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from exc
    if not epochs or epochs[0] < 1:
        raise argparse.ArgumentTypeError("milestones must contain positive epochs")
    return epochs


def load_best_backbone(device: torch.device, epochs: int, n_targets: int):
    """Load the requested custom I-JEPA target encoder and freeze all weights."""
    path = custom_ijepa.find_checkpoint(epochs, n_targets=n_targets)
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {})
    model = custom_ijepa.build_model(
        enc_dim=config.get("enc_dim", custom_ijepa.DEFAULT_ENC_DIM),
        n_targets=config.get("n_targets", n_targets),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, path, config


def checkpoint_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"ijepa_clf_custom_ijepa_mlp2_flatten_{epoch}ep.pt"


def save_milestone(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    in_dim: int,
    hidden_dim: int,
    dropout: float,
    backbone_path: Path,
    backbone_config: dict,
    seed: int,
    metrics: dict,
) -> Path:
    """Write a self-contained checkpoint compatible with ``eval_probe``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(output_dir, epoch)
    torch.save(
        {
            "family": FAMILY,
            "encoder": "custom_ijepa",
            "arch": "custom_ijepa",
            "mode": "probe",
            "mode_label": "frozen custom I-JEPA + two-layer MLP (flattened tokens)",
            "head_type": HEAD_TYPE,
            "pool": "flatten",
            "preproc": True,
            "enc_dim": model.embed_dim,
            "n_targets": model.n_targets,
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "n_classes": N_CLASSES,
            "head_epoch": epoch,
            "seed": seed,
            "backbone_checkpoint": backbone_path.name,
            "backbone_config": backbone_config,
            "metrics": metrics,
            "head_state_dict": head.state_dict(),
            "encoder_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return path


def run(args: argparse.Namespace) -> list[dict]:
    set_seed(args.seed)
    device = pick_device()
    print(f"Seed: {args.seed}  Device: {device}", flush=True)
    model, backbone_path, backbone_config = load_best_backbone(
        device, args.backbone_epochs, args.n_targets
    )
    print(f"Frozen backbone: {backbone_path.name}", flush=True)
    print("Extracting flattened train/test embeddings once...", flush=True)
    Xtr, ytr = extract_features(model, device, "flatten")
    # Use the shared loader logic via a small local equivalent, since
    # train_probe.extract_features intentionally targets only the train split.
    from trials.eval_classifier import extract_features as extract_split_features

    Xte, yte = extract_split_features(model, False, device, pool="flatten", preproc=True)
    print(f"Features: train={tuple(Xtr.shape)} test={tuple(Xte.shape)}", flush=True)

    head = TwoLayerMLP(Xtr.shape[1], args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    milestones = set(args.milestones)
    results: list[dict] = []
    for epoch in range(1, max(milestones) + 1):
        head.train()
        loss_sum = 0.0
        seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(head(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * xb.size(0)
            seen += xb.size(0)

        if epoch % 10 == 0 or epoch in milestones:
            print(f"epoch {epoch:3d}/{max(milestones)} loss={loss_sum / seen:.6f}", flush=True)
        if epoch not in milestones:
            continue

        train_acc = 1.0 - error_features(head, Xtr, ytr, device)
        test_acc = 1.0 - error_features(head, Xte, yte, device)
        metrics = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        }
        results.append(metrics)
        saved = save_milestone(
            args.output_dir,
            epoch,
            model,
            head,
            optimizer,
            in_dim=Xtr.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            backbone_path=backbone_path,
            backbone_config=backbone_config,
            seed=args.seed,
            metrics=metrics,
        )
        print(
            f"MILESTONE epoch={epoch} train={train_acc:.2%} test={test_acc:.2%} "
            f"checkpoint={saved.name}",
            flush=True,
        )

    summary = {
        "seed": args.seed,
        "backbone_checkpoint": backbone_path.name,
        "backbone_config": backbone_config,
        "backbone_frozen": True,
        "pool": "flatten",
        "head": {
            "type": HEAD_TYPE,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"ijepa_mlp2_seed{args.seed}_results.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Results -> {summary_path}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-epochs", type=int, default=DEFAULT_BACKBONE_EPOCHS)
    parser.add_argument("--n-targets", type=int, default=DEFAULT_N_TARGETS)
    parser.add_argument("--milestones", type=parse_milestones, default=DEFAULT_MILESTONES)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
