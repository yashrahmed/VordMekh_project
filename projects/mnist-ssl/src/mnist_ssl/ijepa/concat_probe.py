"""Train linear probes on concatenated frozen I-JEPA embeddings.

This tests whether feature-level fusion beats logit-level ensembling. The old
28x28 custom-I-JEPA checkpoint is loaded through the compatibility model from
``ensemble_probes``; current 56x56 checkpoints are loaded through
``custom_ijepa``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from mnist_ssl.baselines.mae import DATASET_DIR, make_transform, pick_device

from . import custom_ijepa
from .ensemble_probes import (
    DEFAULT_OLD_PROBE,
    LegacyCustomIJEPA,
    bbox_rescale_28,
)
from ._ckpt import MODELS_DIR, set_seed


N_CLASSES = 10
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.05


def old_transform(img):
    return bbox_rescale_28(transforms.ToTensor()(img))


def loader_for_transform(train: bool, batch_size: int, transform) -> DataLoader:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def load_old_model(device: torch.device):
    ckpt = torch.load(DEFAULT_OLD_PROBE, map_location=device)
    model = LegacyCustomIJEPA(
        enc_dim=ckpt.get("enc_dim", 128),
        pred_dim=ckpt.get("enc_dim", 128) // 2,
        n_targets=ckpt.get("n_targets", 10),
    ).to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.eval()
    return model


def load_new_model(device: torch.device, epochs: int, n_targets: int = 48):
    ckpt_path = custom_ijepa.find_checkpoint(epochs=epochs, n_targets=n_targets)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    model = custom_ijepa.build_model(
        enc_dim=config.get("enc_dim", 128),
        n_targets=config.get("n_targets", n_targets),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def extract_features(model: nn.Module, device: torch.device, train: bool, pool: str, transform):
    feats, labels = [], []
    for imgs, y in loader_for_transform(train, batch_size=512, transform=transform):
        feats.append(model.encode(imgs.to(device), pool=pool).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_head(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
):
    head = nn.Linear(Xtr.shape[1], N_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
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
        if epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs}  loss {running / seen:.4f}", flush=True)

    head.eval()
    with torch.no_grad():
        train_pred = []
        for xb, _ in DataLoader(TensorDataset(Xtr, ytr), batch_size=1024):
            train_pred.append(head(xb.to(device)).argmax(dim=1).cpu())
        test_pred = []
        for xb, _ in DataLoader(TensorDataset(Xte, yte), batch_size=1024):
            test_pred.append(head(xb.to(device)).argmax(dim=1).cpu())
    train_pred = torch.cat(train_pred)
    test_pred = torch.cat(test_pred)
    train_acc = (train_pred == ytr).float().mean().item() * 100.0
    test_acc = (test_pred == yte).float().mean().item() * 100.0
    return head, train_acc, test_acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out/concat_probe_results.csv"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}  seed: {args.seed}", flush=True)

    print("Loading/extracting old 28x28 500ep flatten features...", flush=True)
    old_model = load_old_model(device)
    Xold_tr, ytr = extract_features(old_model, device, True, "flatten", old_transform)
    Xold_te, yte = extract_features(old_model, device, False, "flatten", old_transform)
    print(f"  old train: {tuple(Xold_tr.shape)}  old test: {tuple(Xold_te.shape)}", flush=True)
    del old_model

    rows: list[dict[str, object]] = []
    for new_epochs, new_pool in ((300, "flatten"), (500, "flatten"), (300, "mean"), (500, "mean")):
        print(f"\nLoading/extracting 56x56 t48 {new_epochs}ep {new_pool} features...", flush=True)
        new_model = load_new_model(device, epochs=new_epochs, n_targets=48)
        Xnew_tr, ytr2 = extract_features(new_model, device, True, new_pool, make_transform(preproc=True))
        Xnew_te, yte2 = extract_features(new_model, device, False, new_pool, make_transform(preproc=True))
        if not torch.equal(ytr, ytr2) or not torch.equal(yte, yte2):
            raise RuntimeError("Label ordering mismatch between feature extractors")
        del new_model

        Xtr = torch.cat([Xold_tr, Xnew_tr], dim=1)
        Xte = torch.cat([Xold_te, Xnew_te], dim=1)
        print(f"  concat train: {tuple(Xtr.shape)}  concat test: {tuple(Xte.shape)}", flush=True)
        head, train_acc, test_acc = train_head(
            Xtr, ytr, Xte, yte, device,
            args.epochs, args.batch_size, args.lr, args.weight_decay,
        )
        out_probe = (
            MODELS_DIR
            / f"ijepa_clf_concat_old28_flatten_new56_t48_{new_epochs}ep_{new_pool}_probe{args.epochs}ep.pt"
        )
        torch.save(
            {
                "family": "ijepa-concat-probe",
                "old": "custom_ijepa_28x28_t10_500ep_flatten",
                "new": f"custom_ijepa_56x56_t48_{new_epochs}ep_{new_pool}",
                "probe_epochs": args.epochs,
                "in_dim": Xtr.shape[1],
                "n_classes": N_CLASSES,
                "head_state_dict": head.state_dict(),
            },
            out_probe,
        )
        print(f"Result: train {train_acc:.2f}%  test {test_acc:.2f}%  -> {out_probe}", flush=True)
        rows.append(
            {
                "old": "28x28_t10_500ep_flatten",
                "new_epochs": new_epochs,
                "new_pool": new_pool,
                "probe_epochs": args.epochs,
                "in_dim": Xtr.shape[1],
                "train_acc": round(train_acc, 2),
                "test_acc": round(test_acc, 2),
                "probe_path": str(out_probe),
            }
        )
        del Xnew_tr, Xnew_te, Xtr, Xte, head

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "old",
                "new_epochs",
                "new_pool",
                "probe_epochs",
                "in_dim",
                "train_acc",
                "test_acc",
                "probe_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
