"""Supervised LoRA adaptation of the best I-JEPA and DINOv2 backbones.

The protocol trains only rank-constrained adapters in every transformer block
and a classification head. The pretrained tensors remain frozen, are omitted
from result checkpoints, and are fingerprinted before and after every run.

Each head follows one fixed trajectory through the prespecified 50, 75, 100,
and 150 epoch milestones. Test metrics are observed only at those milestones;
they do not alter training or select a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from mnist_ssl.baselines.mae import make_transform
from mnist_ssl.dinov2.eval_knn import build_teacher
from mnist_ssl.dinov2.nonlinear_probe import (
    SmallNonlinearProbe,
    classification_metrics,
    file_sha256,
)
from mnist_ssl.evaluation_labels import AppliedLabelPolicy, apply_mnist_test_label_policy
from mnist_ssl.ijepa import custom_ijepa
from mnist_ssl.lora import (
    LoRAHandle,
    adapter_parameters,
    adapter_state_dict,
    add_lora,
    capture_base_tensors,
    load_adapter_state_dict,
    tensor_fingerprint,
)
from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR


DEFAULT_MILESTONES = (50, 75, 100, 150)
BACKBONE_NAMES = ("ijepa-300", "ijepa-500", "dinov2-best")
PROBE_TYPES = ("linear", "nonlinear")
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass(frozen=True)
class BackboneDefinition:
    name: str
    family: str
    checkpoint: Path
    pool: str
    normalize_input: bool


@dataclass
class LoadedBackbone:
    feature_extractor: nn.Module
    adapter_target: nn.Module
    fingerprint_target: nn.Module
    feature_dim: int
    metadata: dict[str, Any]


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, family: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.family = family

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.family == "dinov2":
            cls, _ = self.backbone.forward_features(images)
            return cls
        return self.backbone.tokens(images).flatten(1)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or parsed[0] < 1:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def parse_choices(value: str, allowed: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated subset of {tuple(allowed)}; unknown={unknown}"
        )
    return tuple(dict.fromkeys(selected))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_backbone(definition: BackboneDefinition, device: torch.device) -> LoadedBackbone:
    checkpoint = torch.load(definition.checkpoint, map_location=device, weights_only=False)
    checkpoint_sha = file_sha256(definition.checkpoint)
    if definition.family == "dinov2":
        config = checkpoint["config"]
        if not config.get("preprocess", True):
            raise ValueError("the selected DINOv2 checkpoint does not use bbox preprocessing")
        owner = build_teacher(checkpoint, device)
        backbone = owner.teacher_backbone
        metadata = {
            "family": "dinov2",
            "checkpoint": str(definition.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": checkpoint.get("completed_epoch", config.get("epochs")),
            "checkpoint_config": config,
            "pool": "cls",
        }
        feature_dim = backbone.dim
        fingerprint_target = backbone
    elif definition.family == "ijepa":
        config = checkpoint.get("config", {})
        if not config.get("preproc", True):
            raise ValueError("the selected I-JEPA checkpoint does not use bbox preprocessing")
        owner = custom_ijepa.build_model(
            enc_dim=config.get("enc_dim", custom_ijepa.DEFAULT_ENC_DIM),
            n_targets=config.get("n_targets", custom_ijepa.N_TARGETS),
        ).to(device)
        owner.load_state_dict(checkpoint["state_dict"])
        backbone = owner.target
        metadata = {
            "family": "ijepa",
            "checkpoint": str(definition.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_config": config,
            "pool": "flatten",
        }
        feature_dim = owner.n_patches * owner.embed_dim
        # Match the established I-JEPA backbone fingerprint: the complete loaded
        # pretrained model is frozen even though only its target tower is read.
        fingerprint_target = owner
    else:
        raise ValueError(f"unknown backbone family: {definition.family}")

    for parameter in owner.parameters():
        parameter.requires_grad_(False)
    owner.eval()
    return LoadedBackbone(
        feature_extractor=FeatureExtractor(backbone, definition.family),
        adapter_target=backbone,
        fingerprint_target=fingerprint_target,
        feature_dim=feature_dim,
        metadata=metadata,
    )


def inject_transformer_lora(
    backbone: nn.Module,
    family: str,
    *,
    rank: int,
    alpha: float,
) -> list[LoRAHandle]:
    """Attach LoRA to every attention and MLP matrix in every encoder block."""

    handles: list[LoRAHandle] = []
    for name, module in backbone.named_modules():
        if family == "dinov2":
            if name.startswith("blocks.") and isinstance(module, nn.Linear):
                handles.append(
                    add_lora(
                        module,
                        "weight",
                        logical_name=f"{name}.weight",
                        rank=rank,
                        alpha=alpha,
                    )
                )
        elif family == "ijepa":
            if not name.startswith("encoder.blocks.layers."):
                continue
            if isinstance(module, nn.MultiheadAttention):
                handles.append(
                    add_lora(
                        module,
                        "in_proj_weight",
                        logical_name=f"{name}.in_proj_weight",
                        rank=rank,
                        alpha=alpha,
                    )
                )
            elif isinstance(module, nn.Linear):
                handles.append(
                    add_lora(
                        module,
                        "weight",
                        logical_name=f"{name}.weight",
                        rank=rank,
                        alpha=alpha,
                    )
                )
        else:
            raise ValueError(f"unknown backbone family: {family}")
    if not handles:
        raise RuntimeError(f"no LoRA targets found for {family}")
    return handles


def load_split(
    dataset_dir: Path,
    *,
    train: bool,
    batch_size: int,
    workers: int,
    subset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.MNIST(
        str(dataset_dir), train=train, download=True, transform=make_transform(preproc=True)
    )
    if subset:
        dataset = Subset(dataset, range(min(subset, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    images = torch.empty(len(dataset), 1, 56, 56, dtype=torch.float32)
    labels = torch.empty(len(dataset), dtype=torch.long)
    offset = 0
    for batch_images, batch_labels in loader:
        end = offset + len(batch_labels)
        images[offset:end].copy_(batch_images)
        labels[offset:end].copy_(batch_labels)
        offset = end
    if offset != len(dataset):
        raise RuntimeError("failed to load the complete MNIST split")
    return images, labels


def normalized_batch(images: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return images
    return (images - MNIST_MEAN) / MNIST_STD


def reviewed_labels(labels: torch.Tensor) -> AppliedLabelPolicy:
    """Apply the reviewed policy to a full test split; keep smoke subsets canonical."""

    if len(labels) == 10_000:
        return apply_mnist_test_label_policy(labels)
    return AppliedLabelPolicy(
        labels=labels.clone(),
        include_mask=torch.ones(len(labels), dtype=torch.bool),
        metadata={
            "name": "canonical-labels-only-subset",
            "original_test_examples": len(labels),
            "scored_test_examples": len(labels),
        },
    )


@torch.no_grad()
def predict(
    feature_extractor: nn.Module,
    head: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    normalize_input: bool,
) -> torch.Tensor:
    feature_extractor.eval()
    head.eval()
    logits = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        batch = normalized_batch(batch, normalize_input)
        logits.append(head(feature_extractor(batch)).float().cpu())
    return torch.cat(logits)


def make_head(probe_type: str, feature_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if probe_type == "linear":
        return nn.Linear(feature_dim, 10)
    if probe_type == "nonlinear":
        return SmallNonlinearProbe(feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"unknown probe type: {probe_type}")


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
        "loader": generator.get_state(),
    }
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            state["mps"] = torch.mps.get_rng_state()
        except RuntimeError:
            pass
    return state


def _restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    torch.set_rng_state(state["cpu"])
    generator.set_state(state["loader"])
    if "mps" in state and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state["mps"])


def _run_signature(
    definition: BackboneDefinition, probe_type: str, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backbone": definition.name,
        "checkpoint_sha256": file_sha256(definition.checkpoint),
        "probe_type": probe_type,
        "milestones": list(args.milestones),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_learning_rate": args.lora_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim if probe_type == "nonlinear" else None,
        "dropout": args.dropout if probe_type == "nonlinear" else None,
        "seed": args.seed,
        "train_examples": len(args.train_images),
        "test_examples": len(args.test_images),
    }


def run_one(
    definition: BackboneDefinition,
    probe_type: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = args.output_dir / definition.name / probe_type
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    resume_path = output_dir / "resume.pt"
    signature = _run_signature(definition, probe_type, args)
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if result.get("protocol", {}).get("signature") != signature:
            raise ValueError(f"completed run signature mismatch in {summary_path}")
        print(f"completed_run_reused={summary_path}", flush=True)
        return result

    seed_everything(args.seed)
    loaded = load_backbone(definition, args.device)
    base_tensors = capture_base_tensors(loaded.fingerprint_target)
    fingerprint_before = tensor_fingerprint(base_tensors)
    handles = inject_transformer_lora(
        loaded.adapter_target,
        definition.family,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    head = make_head(probe_type, loaded.feature_dim, args.hidden_dim, args.dropout).to(
        args.device
    )
    lora_params = adapter_parameters(handles)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_learning_rate},
            {"params": list(head.parameters()), "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict[str, float | int]] = []
    milestone_results: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=args.device, weights_only=False)
        if saved.get("signature") != signature:
            raise ValueError(f"resume signature mismatch in {resume_path}")
        load_adapter_state_dict(handles, saved["adapter_state_dict"])
        head.load_state_dict(saved["head_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history = saved["history"]
        milestone_results = saved["results"]
        start_epoch = saved["completed_epoch"] + 1
        _restore_rng_state(saved["rng_state"], generator)
        print(f"resumed={resume_path} start_epoch={start_epoch}", flush=True)

    adapter_count = sum(parameter.numel() for parameter in lora_params)
    head_count = sum(parameter.numel() for parameter in head.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in list(loaded.feature_extractor.parameters()) + list(head.parameters())
        if parameter.requires_grad
    )
    if trainable_count != adapter_count + head_count:
        raise RuntimeError(
            f"unexpected trainable parameter count: {trainable_count} != "
            f"{adapter_count} + {head_count}"
        )
    print(
        f"run={definition.name}/{probe_type} device={args.device} "
        f"base_frozen=true lora_targets={len(handles)} "
        f"adapter_parameters={adapter_count} head_parameters={head_count}",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss()
    reviewed = reviewed_labels(args.test_labels)
    max_epoch = max(args.milestones)
    for epoch in range(start_epoch, max_epoch + 1):
        loaded.feature_extractor.train()
        head.train()
        order = torch.randperm(len(args.train_images), generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            images = args.train_images[indices].to(args.device)
            images = normalized_batch(images, definition.normalize_input)
            labels = args.train_labels[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(loaded.feature_extractor(images)), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            seen += len(labels)
        history.append({"epoch": epoch, "loss": loss_sum / seen})

        if epoch in args.milestones:
            train_logits = predict(
                loaded.feature_extractor,
                head,
                args.train_images,
                device=args.device,
                batch_size=args.eval_batch_size,
                normalize_input=definition.normalize_input,
            )
            test_logits = predict(
                loaded.feature_extractor,
                head,
                args.test_images,
                device=args.device,
                batch_size=args.eval_batch_size,
                normalize_input=definition.normalize_input,
            )
            result = {
                "epoch": epoch,
                "train": classification_metrics(train_logits, args.train_labels),
                "canonical_test": classification_metrics(test_logits, args.test_labels),
                "reviewed_test": classification_metrics(
                    test_logits, reviewed.labels, reviewed.include_mask
                ),
            }
            milestone_results.append(result)
            torch.save(
                {
                    "signature": signature,
                    "epoch": epoch,
                    "adapter_state_dict": adapter_state_dict(handles),
                    "head_state_dict": head.state_dict(),
                    "test_logits": test_logits,
                    "metrics": result,
                    "base_backbone_included": False,
                },
                output_dir / f"epoch{epoch:04d}.pt",
            )
            print(
                f"milestone={epoch} train_errors={result['train']['errors']} "
                f"test_errors={result['canonical_test']['errors']} "
                f"test_accuracy={result['canonical_test']['accuracy']:.4%}",
                flush=True,
            )
        elif epoch % 10 == 0 or epoch == max_epoch:
            print(
                f"epoch={epoch}/{max_epoch} loss={history[-1]['loss']:.6f}", flush=True
            )

        if epoch % args.checkpoint_every == 0 and epoch < max_epoch:
            torch.save(
                {
                    "signature": signature,
                    "completed_epoch": epoch,
                    "adapter_state_dict": adapter_state_dict(handles),
                    "head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "results": milestone_results,
                    "rng_state": _rng_state(generator),
                    "base_backbone_included": False,
                },
                resume_path,
            )

    fingerprint_after = tensor_fingerprint(base_tensors)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError("a frozen pretrained backbone tensor changed during LoRA training")
    result = {
        "protocol": {
            "signature": signature,
            "trajectory": "one fixed training trajectory evaluated at prespecified milestones",
            "test_used_for_selection": False,
            "base_backbone_frozen": True,
            "base_backbone_included_in_checkpoints": False,
            "base_backbone_sha256_before": fingerprint_before,
            "base_backbone_sha256_after": fingerprint_after,
            "backbone": loaded.metadata,
        },
        "architecture": {
            "probe_type": probe_type,
            "feature_dim": loaded.feature_dim,
            "head_hidden_dim": args.hidden_dim if probe_type == "nonlinear" else None,
            "head_dropout": args.dropout if probe_type == "nonlinear" else None,
            "head_parameters": head_count,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_targets": [handle.name for handle in handles],
            "lora_parameters": adapter_count,
            "total_trainable_parameters": trainable_count,
        },
        "optimization": {
            "optimizer": "AdamW",
            "lora_learning_rate": args.lora_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "loss": "cross_entropy",
            "history": history,
        },
        "reviewed_label_policy": reviewed.metadata,
        "results": milestone_results,
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    resume_path.unlink(missing_ok=True)
    print(f"completed={summary_path}", flush=True)
    return result


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("loading_preprocessed_mnist=true", flush=True)
    args.train_images, args.train_labels = load_split(
        args.dataset_dir,
        train=True,
        batch_size=args.data_batch_size,
        workers=args.workers,
        subset=args.train_subset,
    )
    args.test_images, args.test_labels = load_split(
        args.dataset_dir,
        train=False,
        batch_size=args.data_batch_size,
        workers=args.workers,
        subset=args.test_subset,
    )
    print(
        f"dataset train={tuple(args.train_images.shape)} test={tuple(args.test_images.shape)}",
        flush=True,
    )
    definitions = {
        "ijepa-300": BackboneDefinition(
            "ijepa-300", "ijepa", args.ijepa_300, "flatten", False
        ),
        "ijepa-500": BackboneDefinition(
            "ijepa-500", "ijepa", args.ijepa_500, "flatten", False
        ),
        "dinov2-best": BackboneDefinition(
            "dinov2-best", "dinov2", args.dinov2_best, "cls", True
        ),
    }
    runs = []
    for backbone_name in args.backbones:
        definition = definitions[backbone_name]
        if not definition.checkpoint.is_file():
            raise FileNotFoundError(definition.checkpoint)
        for probe_type in args.probe_types:
            runs.append(run_one(definition, probe_type, args))

    flattened = []
    for run in runs:
        signature = run["protocol"]["signature"]
        for result in run["results"]:
            flattened.append(
                {
                    "backbone": signature["backbone"],
                    "probe_type": signature["probe_type"],
                    "epochs": result["epoch"],
                    "train_accuracy": result["train"]["accuracy"],
                    "canonical_test_accuracy": result["canonical_test"]["accuracy"],
                    "canonical_test_errors": result["canonical_test"]["errors"],
                    "reviewed_test_accuracy": result["reviewed_test"]["accuracy"],
                    "reviewed_test_errors": result["reviewed_test"]["errors"],
                }
            )
    summary = {
        "schema_version": 1,
        "protocol": {
            "backbones": list(args.backbones),
            "probe_types": list(args.probe_types),
            "milestones": list(args.milestones),
            "seed": args.seed,
            "test_used_for_selection": False,
        },
        "results": flattened,
        "runs": runs,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"matrix_summary={summary_path}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbones",
        type=lambda value: parse_choices(value, BACKBONE_NAMES),
        default=BACKBONE_NAMES,
    )
    parser.add_argument(
        "--probe-types",
        type=lambda value: parse_choices(value, PROBE_TYPES),
        default=PROBE_TYPES,
    )
    parser.add_argument("--milestones", type=parse_positive_ints, default=DEFAULT_MILESTONES)
    parser.add_argument(
        "--ijepa-300",
        type=Path,
        default=MODELS_DIR / "ijepa_mnist_custom_ijepa_p7_56_t48_300ep.pt",
    )
    parser.add_argument(
        "--ijepa-500",
        type=Path,
        default=MODELS_DIR / "ijepa_mnist_custom_ijepa_p7_56_t48_500ep.pt",
    )
    parser.add_argument(
        "--dinov2-best",
        type=Path,
        default=MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR / "lora_backbone_probes")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--data-batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    args = parser.parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    return args


if __name__ == "__main__":
    run_matrix(parse_args())
