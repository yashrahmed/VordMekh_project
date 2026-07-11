"""Train a from-scratch, MNIST-scaled DINOv2 model.

Example smoke run:
    uv run python dino-trials/train.py --epochs 2 --subset 512 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from data import MultiCropMNIST, make_masks
from losses import CenteredTeacher, dino_loss, ibot_loss, koleo_loss
from model import StudentTeacher


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"


@dataclass
class Config:
    # MNIST-scaled architecture: 56/7 gives the same 8x8 token grid used by the
    # project's other SSL experiments, while retaining DINOv2's ViT structure.
    global_size: int = 56
    local_size: int = 28
    patch_size: int = 7
    dim: int = 192
    depth: int = 6
    heads: int = 6
    prototypes: int = 1024
    head_hidden_dim: int = 512
    bottleneck_dim: int = 128
    drop_path_rate: float = 0.1
    local_crops: int = 4
    # Match the best custom-I-JEPA input pipeline by default.
    preprocess: bool = True
    # Loss and teacher settings follow the DINOv2 recipe.
    dino_weight: float = 1.0
    ibot_weight: float = 1.0
    koleo_weight: float = 0.1
    student_temperature: float = 0.1
    teacher_temperature_start: float = 0.04
    teacher_temperature: float = 0.07
    center_momentum: float = 0.9
    teacher_momentum: float = 0.994
    final_teacher_momentum: float = 1.0
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.5
    mask_probability: float = 0.5
    # Optimization is shortened/scaled for MNIST but keeps AdamW + cosine paths.
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    warmup_epochs: int = 10
    teacher_warmup_epochs: int = 5
    freeze_last_layer_epochs: int = 1
    gradient_clip: float = 3.0
    subset: int = 0
    workers: int = 2
    seed: int = 0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine(start: float, end: float, progress: float) -> float:
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def scheduled_value(
    start: float,
    end: float,
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_start: float = 0.0,
) -> float:
    if warmup_steps and step < warmup_steps:
        return warmup_start + (start - warmup_start) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return cosine(start, end, min(1.0, max(0.0, progress)))


def parameter_groups(model: StudentTeacher, weight_decay: float):
    decay, no_decay = [], []
    named_parameters = (
        [(f"backbone.{name}", parameter) for name, parameter in model.student_backbone.named_parameters()]
        + [(f"dino_head.{name}", parameter) for name, parameter in model.student_dino_head.named_parameters()]
        + [(f"ibot_head.{name}", parameter) for name, parameter in model.student_ibot_head.named_parameters()]
    )
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or any(
            token in name for token in ("pos_embed", "cls_token", "mask_token", "scale")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay, "apply_weight_decay": True},
        {"params": no_decay, "weight_decay": 0.0, "apply_weight_decay": False},
    ]


def make_loader(config: Config) -> DataLoader:
    transform = MultiCropMNIST(
        global_size=config.global_size,
        local_size=config.local_size,
        local_crops=config.local_crops,
        preprocess=config.preprocess,
    )
    dataset = datasets.MNIST(str(DATASET_DIR), train=True, download=True, transform=transform)
    if config.subset:
        count = min(config.subset, len(dataset))
        generator = torch.Generator().manual_seed(config.seed)
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
    )


def train(config: Config, output: Path, device: torch.device | None = None) -> dict:
    seed_everything(config.seed)
    device = device or pick_device()
    loader = make_loader(config)
    if not len(loader):
        raise ValueError("subset must contain at least one full batch")
    model = StudentTeacher(
        image_size=config.global_size,
        patch_size=config.patch_size,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        prototypes=config.prototypes,
        head_hidden_dim=config.head_hidden_dim,
        bottleneck_dim=config.bottleneck_dim,
        drop_path_rate=config.drop_path_rate,
    ).to(device)
    class_center = CenteredTeacher(config.prototypes, config.center_momentum).to(device)
    patch_center = CenteredTeacher(config.prototypes, config.center_momentum).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
    )
    total_steps = config.epochs * len(loader)
    warmup_steps = min(config.warmup_epochs, config.epochs // 2) * len(loader)
    teacher_warmup_steps = min(config.teacher_warmup_epochs, config.epochs // 2) * len(loader)
    history = []
    global_step = 0
    print(
        f"device={device} samples={len(loader.dataset)} batches={len(loader)} "
        f"model=ViT-{config.depth}x{config.dim} patches={config.global_size // config.patch_size}x"
        f"{config.global_size // config.patch_size} preprocess={config.preprocess}",
        flush=True,
    )

    model.train()
    for epoch in range(config.epochs):
        totals = {"loss": 0.0, "dino": 0.0, "ibot": 0.0, "koleo": 0.0}
        for views, _ in loader:
            global_crops = [crop.to(device) for crop in views["global"]]
            local_crops = [crop.to(device) for crop in views["local"]]
            batch_size = global_crops[0].size(0)
            masks = [
                make_masks(
                    batch_size,
                    config.global_size // config.patch_size,
                    device,
                    (config.mask_ratio_min, config.mask_ratio_max),
                    config.mask_probability,
                )
                for _ in global_crops
            ]
            lr = scheduled_value(
                config.learning_rate,
                config.min_learning_rate,
                global_step,
                total_steps,
                warmup_steps,
                warmup_start=config.min_learning_rate,
            )
            wd = scheduled_value(
                config.weight_decay, config.final_weight_decay, global_step, total_steps
            )
            momentum = scheduled_value(
                config.teacher_momentum,
                config.final_teacher_momentum,
                global_step,
                total_steps,
            )
            teacher_temp = scheduled_value(
                config.teacher_temperature,
                config.teacher_temperature,
                global_step,
                total_steps,
                teacher_warmup_steps,
                warmup_start=config.teacher_temperature_start,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
                if group["apply_weight_decay"]:
                    group["weight_decay"] = wd

            with torch.no_grad():
                teacher_features = [
                    model.teacher_backbone.forward_features(crop) for crop in global_crops
                ]
                teacher_cls_logits = [
                    model.teacher_dino_head(cls) for cls, _ in teacher_features
                ]
                teacher_patch_logits = [
                    model.teacher_ibot_head(patches) for _, patches in teacher_features
                ]
                teacher_cls_probs = [
                    class_center.probabilities(logits, teacher_temp)
                    for logits in teacher_cls_logits
                ]
                teacher_patch_probs = [
                    patch_center.probabilities(logits, teacher_temp)
                    for logits in teacher_patch_logits
                ]

            student_global_features = [
                model.student_backbone.forward_features(crop, mask)
                for crop, mask in zip(global_crops, masks)
            ]
            student_local_features = [
                model.student_backbone.forward_features(crop) for crop in local_crops
            ]
            student_global_cls = [cls for cls, _ in student_global_features]
            student_global_cls_logits = [
                model.student_dino_head(cls) for cls in student_global_cls
            ]
            student_local_cls_logits = [
                model.student_dino_head(cls) for cls, _ in student_local_features
            ]
            student_patch_logits = [
                model.student_ibot_head(patches) for _, patches in student_global_features
            ]

            loss_dino = dino_loss(
                student_global_cls_logits,
                student_local_cls_logits,
                teacher_cls_probs,
                config.student_temperature,
            )
            loss_ibot = ibot_loss(
                student_patch_logits,
                teacher_patch_probs,
                masks,
                config.student_temperature,
            )
            loss_koleo = torch.stack([koleo_loss(cls) for cls in student_global_cls]).mean()
            loss = (
                config.dino_weight * loss_dino
                + config.ibot_weight * loss_ibot
                + config.koleo_weight * loss_koleo
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch < config.freeze_last_layer_epochs:
                model.student_dino_head.prototype_weight.grad = None
                model.student_ibot_head.prototype_weight.grad = None
            clip_grad_norm_(list(model.student_parameters()), config.gradient_clip)
            optimizer.step()
            model.update_teacher(momentum)
            class_center.update(torch.cat(teacher_cls_logits, dim=0))
            patch_center.update(torch.cat(teacher_patch_logits, dim=0))

            for key, value in (
                ("loss", loss), ("dino", loss_dino), ("ibot", loss_ibot), ("koleo", loss_koleo)
            ):
                totals[key] += value.item()
            global_step += 1

        metrics = {key: value / len(loader) for key, value in totals.items()}
        metrics.update({"epoch": epoch + 1, "lr": lr, "weight_decay": wd, "momentum": momentum})
        history.append(metrics)
        print(
            f"epoch {epoch + 1:3d}/{config.epochs} loss={metrics['loss']:.4f} "
            f"dino={metrics['dino']:.4f} ibot={metrics['ibot']:.4f} "
            f"koleo={metrics['koleo']:.4f}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": asdict(config),
        "history": history,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.student_parameters()),
    }
    torch.save(
        {
            **result,
            "teacher_backbone": model.teacher_backbone.state_dict(),
            "teacher_dino_head": model.teacher_dino_head.state_dict(),
            "teacher_ibot_head": model.teacher_ibot_head.state_dict(),
            "student_backbone": model.student_backbone.state_dict(),
            "student_dino_head": model.student_dino_head.state_dict(),
            "student_ibot_head": model.student_ibot_head.state_dict(),
        },
        output,
    )
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved={output} metrics={metrics_path}", flush=True)
    return result


def parse_args() -> tuple[Config, Path, torch.device | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset", type=int, default=0, help="0 uses all 60,000 samples")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--prototypes", type=int, default=1024)
    parser.add_argument("--local-crops", type=int, default=4)
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Upscale to 56x56, bbox-crop the digit, and stretch it back before "
            "multi-crop augmentation (default: enabled)."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--output", type=Path, default=MODELS_DIR / "dinov2_mnist_preproc.pt"
    )
    args = parser.parse_args()
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset=args.subset,
        workers=args.workers,
        seed=args.seed,
        learning_rate=args.lr,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        prototypes=args.prototypes,
        local_crops=args.local_crops,
        preprocess=args.preprocess,
    )
    return config, args.output, torch.device(args.device) if args.device else None


if __name__ == "__main__":
    train(*parse_args())
