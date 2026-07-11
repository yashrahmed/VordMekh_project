"""Evaluate frozen DINOv2 teacher features with weighted k-nearest neighbors.

The deterministic input pipeline follows the checkpoint's ``preprocess`` flag.
New and legacy checkpoints default to the custom-I-JEPA upscale/bbox pipeline;
``--no-preprocess`` is available only for an explicit ablation/override.

Example:
    uv run python dino-trials/eval_knn.py \
        --model models/dinov2_mnist_preproc.pt --k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from data import EvaluationTransform
from model import StudentTeacher
from train import DATASET_DIR, MODELS_DIR, pick_device


def build_teacher(checkpoint: dict, device: torch.device) -> StudentTeacher:
    """Rebuild the exact checkpoint architecture and load its EMA teacher."""
    config = checkpoint["config"]
    model = StudentTeacher(
        image_size=config.get("global_size", 56),
        patch_size=config.get("patch_size", 7),
        dim=config.get("dim", 192),
        depth=config.get("depth", 6),
        heads=config.get("heads", 6),
        prototypes=config.get("prototypes", 1024),
        head_hidden_dim=config.get("head_hidden_dim", 512),
        bottleneck_dim=config.get("bottleneck_dim", 128),
        drop_path_rate=config.get("drop_path_rate", 0.1),
    ).to(device)
    model.teacher_backbone.load_state_dict(checkpoint["teacher_backbone"])
    # k-NN uses only backbone features. Load heads when present so this helper
    # remains useful for inspecting both new untied-head and legacy checkpoints.
    if "teacher_dino_head" in checkpoint:
        model.teacher_dino_head.load_state_dict(checkpoint["teacher_dino_head"])
        model.teacher_ibot_head.load_state_dict(checkpoint["teacher_ibot_head"])
    elif "teacher_head" in checkpoint:
        model.teacher_dino_head.load_state_dict(checkpoint["teacher_head"])
        model.teacher_ibot_head.load_state_dict(checkpoint["teacher_head"])
    model.eval()
    return model


def make_loader(
    train: bool,
    image_size: int,
    preprocess: bool,
    batch_size: int,
    workers: int,
    subset: int = 0,
) -> DataLoader:
    dataset = datasets.MNIST(
        str(DATASET_DIR),
        train=train,
        download=True,
        transform=EvaluationTransform(image_size, preprocess),
    )
    if subset:
        dataset = Subset(dataset, range(min(subset, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def extract_features(
    model: StudentTeacher,
    loader: DataLoader,
    device: torch.device,
    pool: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for images, target in loader:
        encoded = model.encode(images.to(device), pool=pool)
        features.append(F.normalize(encoded.float(), dim=-1).cpu())
        labels.append(target)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def weighted_knn_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 5,
    temperature: float = 0.07,
    query_batch_size: int = 256,
) -> float:
    """Cosine k-NN with temperature-weighted class votes."""
    if not 0 < k <= len(train_features):
        raise ValueError(f"k must be in [1, {len(train_features)}], got {k}")
    correct = 0
    classes = int(train_labels.max().item()) + 1
    for start in range(0, len(test_features), query_batch_size):
        query = test_features[start : start + query_batch_size]
        similarities = query @ train_features.T
        values, indices = similarities.topk(k, dim=1)
        neighbor_labels = train_labels[indices]
        votes = torch.zeros(len(query), classes)
        votes.scatter_add_(1, neighbor_labels, (values / temperature).exp())
        predictions = votes.argmax(dim=1)
        correct += (predictions == test_labels[start : start + len(query)]).sum().item()
    return correct / len(test_labels)


def evaluate(
    checkpoint_path: Path,
    k: int = 5,
    pool: str = "cls",
    preprocess: bool | None = None,
    batch_size: int = 512,
    workers: int = 2,
    train_subset: int = 0,
    test_subset: int = 0,
    device: torch.device | None = None,
) -> float:
    device = device or pick_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    checkpoint_preprocess = config.get("preprocess", True)
    preprocess = checkpoint_preprocess if preprocess is None else preprocess
    if preprocess != checkpoint_preprocess:
        print(
            f"warning: evaluation preprocess={preprocess} overrides checkpoint "
            f"preprocess={checkpoint_preprocess}",
            flush=True,
        )
    image_size = config.get("global_size", 56)
    model = build_teacher(checkpoint, device)
    train_loader = make_loader(
        True, image_size, preprocess, batch_size, workers, train_subset
    )
    test_loader = make_loader(
        False, image_size, preprocess, batch_size, workers, test_subset
    )
    print(
        f"device={device} preprocess={preprocess} pool={pool} k={k} "
        f"reference={len(train_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )
    train_features, train_labels = extract_features(model, train_loader, device, pool)
    test_features, test_labels = extract_features(model, test_loader, device, pool)
    accuracy = weighted_knn_accuracy(
        train_features, train_labels, test_features, test_labels, k=k
    )
    print(f"test_accuracy={accuracy:.2%}", flush=True)
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=MODELS_DIR / "dinov2_mnist_preproc.pt"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool", choices=("cls", "mean", "concat"), default="cls")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint setting; by default evaluation matches it.",
    )
    args = parser.parse_args()
    evaluate(
        args.model,
        k=args.k,
        pool=args.pool,
        preprocess=args.preprocess,
        batch_size=args.batch_size,
        workers=args.workers,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        device=torch.device(args.device) if args.device else None,
    )


if __name__ == "__main__":
    main()
