"""k-NN classification on the frozen MAE encoder embeddings.

Embeds the full MNIST train and test splits with the (unmasked) encoder, then
classifies each test image by a majority vote over its ``k`` nearest training
neighbours in cosine space. This is a non-parametric read of how
class-discriminative the *self-supervised* representation is -- no head is
trained, so it complements the linear probe in ``classify.py``.

    python -m grasp_embeddings.mae_patch_embd.knn --arch vit
    python -m grasp_embeddings.mae_patch_embd.knn --arch cnn --k 5
    python -m grasp_embeddings.mae_patch_embd.knn --arch no-enc  # raw-pixel baseline
    python -m grasp_embeddings.mae_patch_embd.knn --no-model-init  # baseline

``--arch {vit,cnn,jepa}`` selects which pretrained encoder to load. ``--arch
no-enc`` skips the encoder entirely and runs k-NN on the raw flattened pixels
(Euclidean image difference) -- the floor that any learned embedding should
beat. ``--no-model-init`` uses a random, untrained encoder as a baseline.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from grasp_embeddings.mae_patch_embd.mae import (
    ARCHES,
    DATASET_DIR,
    build_model,
    find_checkpoint,
    pick_device,
)

N_CLASSES = 10
NO_ENC = "no-enc"


def load_encoder(
    device: torch.device, arch: str, random_init: bool, epochs: int | None = None
) -> nn.Module:
    model = build_model(arch).to(device)
    if not random_init:
        ckpt_path = find_checkpoint(arch, epochs)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint: {ckpt_path.name}")
    model.eval()
    return model


def mnist_loader(train: bool, batch_size: int = 512):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def embed_split(model: nn.Module | None, train: bool, device: torch.device):
    """Return (N, D) features and (N,) labels.

    With ``model`` set, D = embed_dim and features are L2-normalised encoder
    embeddings (compared by cosine). With ``model is None`` (the ``no-enc``
    baseline), D = 784 raw flattened pixels, compared by direct Euclidean
    distance.
    """
    feats, labels = [], []
    for imgs, y in mnist_loader(train):
        if model is None:
            emb = imgs.flatten(1)  # (B, 784) raw pixels
        else:
            emb = F.normalize(model.encode(imgs.to(device)), dim=-1).cpu()
        feats.append(emb)
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_predict(
    test_emb: torch.Tensor,
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    k: int,
    device: torch.device,
    metric: str = "cosine",
    chunk: int = 1024,
) -> torch.Tensor:
    """Majority-vote k-NN.

    ``metric="cosine"`` ranks by largest dot product (embeddings are unit-norm);
    ``metric="euclidean"`` ranks by smallest pixel distance (raw-pixel baseline).
    """
    train_emb = train_emb.to(device)
    train_labels = train_labels.to(device)
    preds = []
    for i in range(0, len(test_emb), chunk):
        q = test_emb[i : i + chunk].to(device)  # (c, D)
        if metric == "cosine":
            idx = (q @ train_emb.T).topk(k, dim=1).indices  # (c, k)
        else:  # euclidean: nearest = smallest distance
            dist = torch.cdist(q, train_emb)  # (c, N)
            idx = dist.topk(k, dim=1, largest=False).indices  # (c, k)
        votes = train_labels[idx]  # (c, k)
        onehot = F.one_hot(votes, N_CLASSES).sum(dim=1)  # (c, n_classes)
        preds.append(onehot.argmax(dim=1).cpu())
    return torch.cat(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=(*ARCHES, NO_ENC), default="vit")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--no-model-init",
        action="store_true",
        help="Use a random, untrained encoder as a baseline.",
    )
    parser.add_argument(
        "--ckpt-epochs",
        type=int,
        default=None,
        help="Pretraining length of the checkpoint to load "
        "(default: the most-trained one on disk).",
    )
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}  arch: {args.arch}  k: {args.k}")

    no_enc = args.arch == NO_ENC
    if no_enc:
        print("No encoder: k-NN on raw flattened pixels (Euclidean).")
        model = None
        metric = "euclidean"
    else:
        if args.no_model_init:
            print("Using an UNINITIALIZED (untrained) encoder.")
        model = load_encoder(
            device, args.arch, random_init=args.no_model_init, epochs=args.ckpt_epochs
        )
        metric = "cosine"

    print("Embedding train split...")
    train_emb, train_labels = embed_split(model, train=True, device=device)
    print("Embedding test split...")
    test_emb, test_labels = embed_split(model, train=False, device=device)
    print(f"  train: {tuple(train_emb.shape)}   test: {tuple(test_emb.shape)}")

    pred = knn_predict(
        test_emb, train_emb, train_labels, args.k, device, metric=metric
    )
    acc = (pred == test_labels).float().mean().item()

    src = "raw pixels" if no_enc else "frozen encoder embeddings"
    print(f"\n--- {args.k}-NN on {src} ---")
    print(f"Test accuracy: {acc:.2%}  (error {1 - acc:.2%})")


if __name__ == "__main__":
    main()
