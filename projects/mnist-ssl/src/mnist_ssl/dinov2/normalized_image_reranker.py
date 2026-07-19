"""Train an independent normalized-image reranker above a frozen linear probe.

The frozen DINOv2 linear probe supplies only the candidate classes and a fixed
normalized-margin gate.  The reranker never receives DINO features, probe
logits, margins, or gate status.  It learns from bbox-normalized MNIST pixels
using one true-class-versus-hardest-probe-negative pair per training example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR

from .data import upscale_bbox
from .nonlinear_probe import file_sha256
from .train import pick_device


DEFAULT_LINEAR_PROBE = (
    MODELS_DIR
    / "dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt"
)
DEFAULT_FEATURE_CACHE = (
    OUT_DIR
    / "dinov2_reranker_oof_10fold_ep50_75_seed0"
    / "train_features.pt"
)
DEFAULT_IMAGE_CACHE = (
    OUT_DIR / "dinov2_correction_addon_v2" / "image_views_uint8.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "dinov2_normalized_image_reranker_50ep"
DEFAULT_GATE_THRESHOLD = 0.0367


@dataclass(frozen=True)
class RerankerConfig:
    channels: tuple[int, int, int] = (16, 32, 64)
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    evaluate_every: int = 5
    seed: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def normalized_top2_margin(logits: torch.Tensor) -> torch.Tensor:
    """Return a shift- and positive-scale-invariant top-two logit margin."""

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [samples, classes>=2]")
    top_values = logits.topk(2, dim=1).values
    scale = logits.std(dim=1).clamp_min(1e-6)
    return (top_values[:, 0] - top_values[:, 1]) / scale


def derive_pair_metadata(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gate_threshold: float,
) -> dict[str, torch.Tensor]:
    """Generate one true-class-versus-hardest-wrong pair per example."""

    if logits.ndim != 2 or labels.shape != (len(logits),):
        raise ValueError("logits and labels have incompatible shapes")
    rows = torch.arange(len(labels))
    top = logits.topk(2, dim=1)
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    hardest_wrong = masked.argmax(dim=1)
    if hardest_wrong.eq(labels).any():
        raise RuntimeError("hard-negative generation returned a true class")
    margin = normalized_top2_margin(logits)
    return {
        "sample_index": rows,
        "label": labels.long(),
        "hardest_wrong": hardest_wrong.long(),
        "linear_top1": top.indices[:, 0].long(),
        "linear_top2": top.indices[:, 1].long(),
        "normalized_margin": margin.float(),
        "gate_eligible": margin.le(gate_threshold),
    }


class NormalizedImagePairDataset(Dataset):
    """Pair metadata joined to deterministic uint8 normalized MNIST images."""

    def __init__(
        self,
        normalized_images: torch.Tensor,
        labels: torch.Tensor,
        hardest_wrong: torch.Tensor,
    ) -> None:
        if normalized_images.shape != (len(labels), 1, 28, 28):
            raise ValueError("normalized image cache must have shape [N,1,28,28]")
        if hardest_wrong.shape != labels.shape:
            raise ValueError("hard-negative labels must match true labels")
        self.normalized_images = normalized_images
        self.labels = labels.long()
        self.hardest_wrong = hardest_wrong.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.normalized_images[index],
            self.labels[index],
            self.hardest_wrong[index],
            torch.tensor(index, dtype=torch.long),
        )


class IndependentNormalizedReranker(nn.Module):
    """A compact image-only ConvNet that emits ten independent class scores."""

    def __init__(
        self,
        channels: tuple[int, int, int] = (16, 32, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        first, second, third = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, first, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(4, first)), first),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(first, second, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, second)), second),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(second, third, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, third)), third),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(third, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images).flatten(1)
        return self.classifier(self.dropout(features))


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    hardest_wrong: torch.Tensor,
) -> torch.Tensor:
    """Logistic ranking loss for the true class against one hard negative."""

    rows = torch.arange(len(labels), device=scores.device)
    true_scores = scores[rows, labels]
    wrong_scores = scores[rows, hardest_wrong]
    return F.softplus(wrong_scores - true_scores).mean()


def gated_predictions(
    base_top1: torch.Tensor,
    base_top2: torch.Tensor,
    normalized_margin: torch.Tensor,
    reranker_scores: torch.Tensor,
    *,
    gate_threshold: float,
) -> torch.Tensor:
    """Use reranker scores only within the fixed base-margin gate."""

    if reranker_scores.shape[0] != len(base_top1):
        raise ValueError("reranker scores and base predictions have different sizes")
    rows = torch.arange(len(base_top1))
    candidate_scores = reranker_scores[
        rows[:, None], torch.stack((base_top1, base_top2), dim=1)
    ]
    reranked = torch.stack((base_top1, base_top2), dim=1)[
        rows, candidate_scores.argmax(dim=1)
    ]
    return torch.where(
        normalized_margin.le(gate_threshold),
        reranked,
        base_top1,
    )


def reranking_metrics(
    predictions: torch.Tensor,
    *,
    labels: torch.Tensor,
    base_top1: torch.Tensor,
    base_top2: torch.Tensor,
    normalized_margin: torch.Tensor,
    gate_threshold: float,
) -> dict[str, Any]:
    base_correct = base_top1.eq(labels)
    candidate_correct = predictions.eq(labels)
    gate = normalized_margin.le(gate_threshold)
    fixes = ~base_correct & candidate_correct
    breaks = base_correct & ~candidate_correct
    wrong_to_wrong = (
        ~base_correct
        & predictions.ne(base_top1)
        & ~candidate_correct
    )
    base_errors = int((~base_correct).sum().item())
    errors = int((~candidate_correct).sum().item())
    recoverable = ~base_correct & base_top2.eq(labels)
    return {
        "samples": len(labels),
        "gate_threshold": gate_threshold,
        "gate_eligible": int(gate.sum().item()),
        "gate_top1_correct": int((gate & base_correct).sum().item()),
        "gate_top2_recoverable": int((gate & recoverable).sum().item()),
        "gate_neither_candidate_correct": int(
            (gate & ~base_correct & ~base_top2.eq(labels)).sum().item()
        ),
        "changed_predictions": int(predictions.ne(base_top1).sum().item()),
        "fixed_errors": int(fixes.sum().item()),
        "new_errors": int(breaks.sum().item()),
        "wrong_to_wrong": int(wrong_to_wrong.sum().item()),
        "net_error_reduction": base_errors - errors,
        "base_errors": base_errors,
        "reranked_errors": errors,
        "base_accuracy": 1.0 - base_errors / len(labels),
        "reranked_accuracy": 1.0 - errors / len(labels),
    }


def load_exact_linear_probe(
    path: Path,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("head_state_dict", {})
    weight = state.get("weight")
    bias = state.get("bias")
    if weight is None or bias is None or weight.shape != (10, 128):
        raise ValueError(f"{path} is not the expected Linear(128,10) probe")
    epochs = checkpoint.get("result", {}).get("linear_probe", {}).get("epochs")
    if epochs != 50:
        raise ValueError(f"{path} is not the expected 50-epoch linear probe")
    return checkpoint, weight.float(), bias.float()


def load_feature_cache(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload.get("features")
    labels = payload.get("labels")
    if features is None or labels is None:
        raise ValueError(f"{path} does not contain frozen features and labels")
    if features.shape != (60_000, 128) or labels.shape != (60_000,):
        raise ValueError(f"{path} is not the complete MNIST training cache")
    signature = payload.get("signature", {})
    if signature.get("source_split") != "MNIST train (canonical order)":
        raise ValueError(f"{path} belongs to the wrong dataset split")
    return features.float(), labels.long(), payload.get("backbone", {})


def build_or_load_normalized_images(
    path: Path,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Load an existing bbox cache or create a compact normalized-only cache."""

    mnist = datasets.MNIST(str(DATASET_DIR), train=True, download=False)
    if not torch.equal(mnist.targets.long(), labels):
        raise ValueError("MNIST dataset order differs from the feature cache")
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        images = payload.get("bbox")
        if (
            images is not None
            and images.dtype == torch.uint8
            and images.shape == (60_000, 1, 28, 28)
        ):
            print(f"normalized_images=reused path={path}", flush=True)
            return images, True
        images = payload.get("normalized_images")
        if (
            images is not None
            and images.dtype == torch.uint8
            and images.shape == (60_000, 1, 28, 28)
        ):
            print(f"normalized_images=reused path={path}", flush=True)
            return images, True
        raise ValueError(f"{path} is not a valid normalized image cache")

    images = torch.empty((60_000, 1, 28, 28), dtype=torch.uint8)
    for index, raw in enumerate(mnist.data):
        normalized = upscale_bbox(
            raw.unsqueeze(0).float().div(255.0),
            size=28,
        )
        images[index] = (
            normalized.mul(255.0).round().clamp(0, 255).to(torch.uint8)
        )
        if (index + 1) % 10_000 == 0:
            print(f"normalized_images_generated={index + 1}/60000", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "storage": "uint8",
            "split": "train",
            "normalized_images": images,
            "recipe": "upscale_bbox(raw MNIST, size=28)",
        },
        path,
    )
    print(f"normalized_images=created path={path}", flush=True)
    return images, False


@torch.no_grad()
def score_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    model.eval()
    scores = torch.empty((len(loader.dataset), 10), dtype=torch.float32)
    pair_correct = 0
    for images, labels, hardest_wrong, sample_index in loader:
        output = model(images.to(device).float().div_(255.0)).cpu()
        scores[sample_index] = output
        rows = torch.arange(len(labels))
        pair_correct += int(
            (
                output[rows, labels]
                > output[rows, hardest_wrong]
            )
            .sum()
            .item()
        )
    return scores, pair_correct / len(loader.dataset)


def build_pair_metadata(
    *,
    output_path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    linear_probe_path: Path,
    gate_threshold: float,
    backbone: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    signature = {
        "source_split": "MNIST train (60,000 examples, canonical order)",
        "linear_probe_sha256": file_sha256(linear_probe_path),
        "linear_probe_type": "Linear(128,10)",
        "linear_probe_epochs": 50,
        "labels_sha256": tensor_sha256(labels),
        "gate_normalization": "top-two gap / per-sample std of ten logits",
        "gate_threshold": gate_threshold,
    }
    if output_path.exists():
        saved = torch.load(output_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise ValueError(f"{output_path} belongs to a different experiment")
        print(f"pair_metadata=reused path={output_path}", flush=True)
        return saved, True

    logits = features @ linear_weight.T + linear_bias
    fields = derive_pair_metadata(
        logits,
        labels,
        gate_threshold=gate_threshold,
    )
    payload = {
        "signature": signature,
        "backbone": backbone,
        **fields,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"pair_metadata=created samples={len(labels)} "
        f"gate_eligible={int(fields['gate_eligible'].sum())} "
        f"path={output_path}",
        flush=True,
    )
    return payload, False


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.output_dir / "training_pairs.pt"
    checkpoint_path = args.output_dir / "normalized_image_reranker.pt"
    summary_path = args.output_dir / "summary.json"
    resume_path = args.output_dir / "normalized_image_reranker_resume.pt"
    if checkpoint_path.exists() or summary_path.exists():
        raise FileExistsError(
            "refusing to overwrite completed normalized-image reranker artifacts"
        )

    config = RerankerConfig(
        channels=tuple(args.channels),
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluate_every=args.evaluate_every,
        seed=args.seed,
    )
    seed_everything(config.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(
        f"device={device} epochs={config.epochs} "
        f"gate_threshold={args.gate_threshold:.4f} "
        "reranker_inputs=normalized_pixels_only",
        flush=True,
    )

    linear_checkpoint, linear_weight, linear_bias = load_exact_linear_probe(
        args.linear_probe
    )
    features, labels, backbone = load_feature_cache(args.feature_cache)
    pairs, pairs_reused = build_pair_metadata(
        output_path=pair_path,
        features=features,
        labels=labels,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
        linear_probe_path=args.linear_probe,
        gate_threshold=args.gate_threshold,
        backbone=backbone,
    )
    normalized_images, image_cache_reused = build_or_load_normalized_images(
        args.image_cache,
        labels,
    )
    dataset = NormalizedImagePairDataset(
        normalized_images,
        pairs["label"],
        pairs["hardest_wrong"],
    )
    train_generator = torch.Generator().manual_seed(config.seed)
    evaluation_generator = torch.Generator().manual_seed(config.seed)
    train_loader = _loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    evaluation_loader = _loader(
        dataset,
        batch_size=1024,
        shuffle=False,
        generator=evaluation_generator,
    )

    model = IndependentNormalizedReranker(
        channels=config.channels,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved.get("config") != asdict(config):
            raise ValueError(f"{resume_path} belongs to a different configuration")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        train_generator.set_state(saved["train_generator_state"])
        history = saved["history"]
        start_epoch = int(saved["completed_epoch"]) + 1
        print(f"training=resumed start_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, batch_labels, hardest_wrong, _ in train_loader:
            images = images.to(device).float().div_(255.0)
            batch_labels = batch_labels.to(device)
            hardest_wrong = hardest_wrong.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = pairwise_ranking_loss(
                scores,
                batch_labels,
                hardest_wrong,
            )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_labels)
            seen += len(batch_labels)

        entry: dict[str, Any] = {
            "epoch": epoch,
            "pairwise_loss": loss_sum / seen,
        }
        if (
            epoch == 1
            or epoch % config.evaluate_every == 0
            or epoch == config.epochs
        ):
            training_scores, pair_accuracy = score_dataset(
                model,
                evaluation_loader,
                device,
            )
            predictions = gated_predictions(
                pairs["linear_top1"],
                pairs["linear_top2"],
                pairs["normalized_margin"],
                training_scores,
                gate_threshold=args.gate_threshold,
            )
            entry["pair_accuracy"] = pair_accuracy
            entry["gated_training"] = reranking_metrics(
                predictions,
                labels=labels,
                base_top1=pairs["linear_top1"],
                base_top2=pairs["linear_top2"],
                normalized_margin=pairs["normalized_margin"],
                gate_threshold=args.gate_threshold,
            )
            print(
                f"epoch={epoch}/{config.epochs} "
                f"loss={entry['pairwise_loss']:.6f} "
                f"pair_accuracy={pair_accuracy:.6f} "
                f"fixes={entry['gated_training']['fixed_errors']} "
                f"breaks={entry['gated_training']['new_errors']} "
                f"net={entry['gated_training']['net_error_reduction']} "
                f"errors={entry['gated_training']['reranked_errors']}",
                flush=True,
            )
        elif epoch % 2 == 0:
            print(
                f"epoch={epoch}/{config.epochs} "
                f"loss={entry['pairwise_loss']:.6f}",
                flush=True,
            )
        history.append(entry)
        if epoch % 5 == 0 and epoch < config.epochs:
            torch.save(
                {
                    "config": asdict(config),
                    "completed_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_generator_state": train_generator.get_state(),
                    "history": history,
                },
                resume_path,
            )

    final_scores, final_pair_accuracy = score_dataset(
        model,
        evaluation_loader,
        device,
    )
    final_predictions = gated_predictions(
        pairs["linear_top1"],
        pairs["linear_top2"],
        pairs["normalized_margin"],
        final_scores,
        gate_threshold=args.gate_threshold,
    )
    final_metrics = reranking_metrics(
        final_predictions,
        labels=labels,
        base_top1=pairs["linear_top1"],
        base_top2=pairs["linear_top2"],
        normalized_margin=pairs["normalized_margin"],
        gate_threshold=args.gate_threshold,
    )
    changed = final_predictions.ne(pairs["linear_top1"])
    torch.save(
        {
            "model_type": "independent-normalized-image-pairwise-reranker",
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "gate_threshold": args.gate_threshold,
            "linear_probe": str(args.linear_probe),
            "linear_probe_sha256": file_sha256(args.linear_probe),
            "pair_metadata": str(pair_path),
            "final_pair_accuracy": final_pair_accuracy,
            "training_metrics": final_metrics,
            "changed_sample_indices": torch.where(changed)[0],
            "training_scores": final_scores,
        },
        checkpoint_path,
    )
    resume_path.unlink(missing_ok=True)

    summary = {
        "protocol": {
            "source_split": "MNIST train (60,000 examples, canonical order)",
            "test_set_loaded": False,
            "folds_used": False,
            "linear_probe_frozen": True,
            "linear_probe_type": "Linear(128,10)",
            "linear_probe_epochs": 50,
            "linear_probe": str(args.linear_probe),
            "linear_probe_sha256": file_sha256(args.linear_probe),
            "linear_probe_result": linear_checkpoint["result"]["linear_probe"],
            "reranker_inputs": "bbox-normalized MNIST pixels only, shape [1,28,28]",
            "reranker_receives_gate_or_logits": False,
            "training_pair": "true class versus highest-logit incorrect linear-probe class",
            "gate_normalization": "top-two logit gap / per-sample std of ten logits",
            "gate_threshold": args.gate_threshold,
        },
        "artifacts": {
            "training_pairs": str(pair_path),
            "normalized_image_cache": str(args.image_cache),
            "checkpoint": str(checkpoint_path),
        },
        "cache_reuse": {
            "training_pairs": pairs_reused,
            "normalized_images": image_cache_reused,
        },
        "architecture": {
            "channels": list(config.channels),
            "dropout": config.dropout,
            "class_scores": 10,
            "parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "optimization": {
            **asdict(config),
            "optimizer": "AdamW",
            "loss": "softplus(hardest_wrong_score - true_class_score)",
        },
        "training_pair_accuracy": final_pair_accuracy,
        "gated_training": final_metrics,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"training_complete base_errors={final_metrics['base_errors']} "
        f"reranked_errors={final_metrics['reranked_errors']} "
        f"fixes={final_metrics['fixed_errors']} "
        f"breaks={final_metrics['new_errors']} "
        f"net={final_metrics['net_error_reduction']}",
        flush=True,
    )
    print(f"checkpoint={checkpoint_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-probe", type=Path, default=DEFAULT_LINEAR_PROBE)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gate-threshold", type=float, default=DEFAULT_GATE_THRESHOLD)
    parser.add_argument("--channels", type=int, nargs=3, default=(16, 32, 64))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
