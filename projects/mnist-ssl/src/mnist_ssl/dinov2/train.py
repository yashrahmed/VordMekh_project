"""Train a from-scratch, MNIST-scaled DINOv2 model.

Example smoke run:
    uv run python scripts/train/dinov2.py --epochs 2 --subset 512 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from mnist_ssl.paths import DATASET_DIR, MODELS_DIR

from .data import MultiCropMNIST, make_masks
from .losses import CenteredTeacher, dino_loss, ibot_loss, koleo_loss
from .model import StudentTeacher


@dataclass
class Config:
    # MNIST-scaled architecture: 56/7 gives the same 8x8 token grid used by the
    # project's other SSL experiments, while retaining DINOv2's ViT structure.
    global_size: int = 56
    local_size: int = 28
    patch_size: int = 7
    dim: int = 128
    depth: int = 4
    heads: int = 4
    prototypes: int = 1024
    head_hidden_dim: int = 512
    bottleneck_dim: int = 128
    drop_path_rate: float = 0.1
    local_crops: int = 4
    # Match the best custom-I-JEPA input pipeline by default.
    preprocess: bool = True
    # Crop-only ablation: brightness/contrast, blur, and solarization are off.
    photometric_augmentations: bool = False
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


CHECKPOINT_VERSION = 3
MIN_RESUMABLE_CHECKPOINT_VERSION = 2
SCHEDULE_VERSION = 1


def milestone_path(output: Path, epoch: int) -> Path:
    return output.with_name(f"{output.stem}_epoch{epoch:04d}{output.suffix}")


def rolling_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_resume{output.suffix}")


def parse_epoch_list(value: str) -> tuple[int, ...]:
    epochs = tuple(sorted(set(int(item) for item in re.split(r"[, ]+", value.strip()) if item)))
    if any(epoch < 1 for epoch in epochs):
        raise argparse.ArgumentTypeError("checkpoint epochs must be positive")
    return epochs


def get_rng_state(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["device"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "device" in state and device.type == "cuda":
        torch.cuda.set_rng_state(state["device"], device)
    elif "device" in state and device.type == "mps":
        torch.mps.set_rng_state(state["device"])


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


def make_schedule_state(
    config: Config,
    steps_per_epoch: int,
    next_step: int = 0,
) -> dict:
    """Build the serializable state for all step-based training schedules."""
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    total_steps = config.epochs * steps_per_epoch
    if not 0 <= next_step <= total_steps:
        raise ValueError("schedule next_step is outside the configured horizon")
    return {
        "schedule_version": SCHEDULE_VERSION,
        "target_epochs": config.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": min(config.warmup_epochs, config.epochs // 2) * steps_per_epoch,
        "teacher_warmup_steps": (
            min(config.teacher_warmup_epochs, config.epochs // 2) * steps_per_epoch
        ),
        "next_step": next_step,
    }


def schedule_values(config: Config, schedule: dict) -> dict[str, float]:
    """Return the values to apply at the schedule's next optimizer step."""
    step = schedule["next_step"]
    total_steps = schedule["total_steps"]
    return {
        "lr": scheduled_value(
            config.learning_rate,
            config.min_learning_rate,
            step,
            total_steps,
            schedule["warmup_steps"],
            warmup_start=config.min_learning_rate,
        ),
        "weight_decay": scheduled_value(
            config.weight_decay,
            config.final_weight_decay,
            step,
            total_steps,
        ),
        "momentum": scheduled_value(
            config.teacher_momentum,
            config.final_teacher_momentum,
            step,
            total_steps,
        ),
        "teacher_temperature": scheduled_value(
            config.teacher_temperature,
            config.teacher_temperature,
            step,
            total_steps,
            schedule["teacher_warmup_steps"],
            warmup_start=config.teacher_temperature_start,
        ),
    }


def recover_schedule_state(checkpoint: dict, config: Config, steps_per_epoch: int) -> dict:
    """Recover and validate the exact next schedule step from a checkpoint."""
    expected = make_schedule_state(config, steps_per_epoch, checkpoint["global_step"])
    saved = checkpoint.get("schedule")
    if saved is None:
        # Version-2 checkpoints predate explicit schedule state. They are safe
        # to infer only when resuming the same fixed target horizon.
        if checkpoint["config"].get("epochs") != config.epochs:
            raise ValueError("legacy checkpoint cannot change the schedule horizon")
        return expected

    required = set(expected)
    missing = sorted(required - set(saved))
    if missing:
        raise ValueError(f"checkpoint schedule is missing: {', '.join(missing)}")
    mismatches = sorted(
        key for key in required if key != "next_step" and saved[key] != expected[key]
    )
    if mismatches:
        raise ValueError(f"checkpoint schedule differs for: {', '.join(mismatches)}")
    if saved["next_step"] != checkpoint["global_step"]:
        raise ValueError("checkpoint schedule next_step differs from global_step")
    if saved["next_step"] != expected["next_step"]:
        raise ValueError("checkpoint schedule next_step is outside the current run")
    return {key: saved[key] for key in expected}


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
        photometric_augmentations=config.photometric_augmentations,
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


def checkpoint_payload(
    config: Config,
    model: StudentTeacher,
    optimizer: torch.optim.Optimizer,
    class_center: CenteredTeacher,
    patch_center: CenteredTeacher,
    history: list[dict],
    device: torch.device,
    schedule: dict,
) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": asdict(config),
        "history": history,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.student_parameters()),
        "completed_epoch": len(history),
        "global_step": schedule["next_step"],
        "schedule": dict(schedule),
        "teacher_backbone": model.teacher_backbone.state_dict(),
        "teacher_dino_head": model.teacher_dino_head.state_dict(),
        "teacher_ibot_head": model.teacher_ibot_head.state_dict(),
        "student_backbone": model.student_backbone.state_dict(),
        "student_dino_head": model.student_dino_head.state_dict(),
        "student_ibot_head": model.student_ibot_head.state_dict(),
        "class_center": class_center.state_dict(),
        "patch_center": patch_center.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": get_rng_state(device),
    }


def save_checkpoint(path: Path, payload: dict, write_metrics: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    if write_metrics:
        result = {
            key: payload[key]
            for key in (
                "config", "history", "device", "parameters", "completed_epoch",
                "global_step", "schedule",
            )
        }
        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"checkpoint={path} epoch={payload['completed_epoch']}", flush=True)


def resume_config_mismatches(
    saved_config: dict,
    current_config: dict,
    completed_epoch: int,
) -> list[str]:
    """Return incompatible configuration fields for a resumed run.

    Exact schedule recovery requires the target epoch horizon and every other
    architecture, data, optimizer, and seed setting to remain fixed.
    """
    # Checkpoints written before this field existed used the augmented pipeline.
    saved_config = {"photometric_augmentations": True, **saved_config}
    mismatches = [
        key
        for key, value in current_config.items()
        if saved_config.get(key) != value
    ]
    return mismatches


def restore_checkpoint(
    path: Path,
    config: Config,
    model: StudentTeacher,
    optimizer: torch.optim.Optimizer,
    class_center: CenteredTeacher,
    patch_center: CenteredTeacher,
    device: torch.device,
    steps_per_epoch: int,
) -> tuple[list[dict], int, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version", 0) < MIN_RESUMABLE_CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} predates resumable version-{MIN_RESUMABLE_CHECKPOINT_VERSION} checkpoints"
        )
    saved_config = checkpoint["config"]
    current_config = asdict(config)
    mismatches = resume_config_mismatches(
        saved_config, current_config, checkpoint["completed_epoch"]
    )
    if mismatches:
        raise ValueError(f"resume configuration differs for: {', '.join(mismatches)}")
    schedule = recover_schedule_state(checkpoint, config, steps_per_epoch)
    for name in (
        "teacher_backbone", "teacher_dino_head", "teacher_ibot_head",
        "student_backbone", "student_dino_head", "student_ibot_head",
    ):
        getattr(model, name).load_state_dict(checkpoint[name])
    class_center.load_state_dict(checkpoint["class_center"])
    patch_center.load_state_dict(checkpoint["patch_center"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    set_rng_state(checkpoint["rng_state"], device)
    start_epoch = checkpoint["completed_epoch"]
    print(
        f"resumed={path} completed_epoch={start_epoch} "
        f"next_schedule_step={schedule['next_step']}/{schedule['total_steps']}",
        flush=True,
    )
    return checkpoint["history"], start_epoch, schedule


def training_end_epoch(
    target_epochs: int, start_epoch: int, stop_after_epoch: int | None
) -> int:
    end_epoch = target_epochs if stop_after_epoch is None else stop_after_epoch
    if not 1 <= end_epoch <= target_epochs:
        raise ValueError(
            f"--stop-after-epoch must be between 1 and the target {target_epochs} epochs"
        )
    if end_epoch <= start_epoch:
        raise ValueError(
            f"--stop-after-epoch ({end_epoch}) must exceed the resumed epoch ({start_epoch})"
        )
    return end_epoch


def train(
    config: Config,
    output: Path,
    device: torch.device | None = None,
    checkpoint_epochs: tuple[int, ...] = (50, 75, 100),
    checkpoint_every: int = 50,
    resume: Path | None = None,
    stop_after_epoch: int | None = None,
) -> dict:
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
    schedule = make_schedule_state(config, len(loader))
    history: list[dict] = []
    start_epoch = 0
    if checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")
    if resume is not None:
        history, start_epoch, schedule = restore_checkpoint(
            resume, config, model, optimizer, class_center, patch_center, device, len(loader)
        )
    end_epoch = training_end_epoch(config.epochs, start_epoch, stop_after_epoch)
    print(
        f"device={device} samples={len(loader.dataset)} batches={len(loader)} "
        f"model=ViT-{config.depth}x{config.dim} patches={config.global_size // config.patch_size}x"
        f"{config.global_size // config.patch_size} preprocess={config.preprocess} "
        f"photometric_augmentations={config.photometric_augmentations}",
        flush=True,
    )

    model.train()
    for epoch in range(start_epoch, end_epoch):
        totals = {"loss": 0.0, "dino": 0.0, "ibot": 0.0, "koleo": 0.0}
        for views, _ in loader:
            global_crops = [crop.to(device) for crop in views["global"]]
            local_crops = [crop.to(device) for crop in views["local"]]
            batch_size = global_crops[0].size(0)
            masks = [
                make_masks(
                    batch_size,
                    config.global_size // config.patch_size,
                    torch.device("cpu"),
                    (config.mask_ratio_min, config.mask_ratio_max),
                    config.mask_probability,
                ).to(device)
                for _ in global_crops
            ]
            values = schedule_values(config, schedule)
            lr = values["lr"]
            wd = values["weight_decay"]
            momentum = values["momentum"]
            teacher_temp = values["teacher_temperature"]
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
                raise RuntimeError(
                    f"non-finite loss at step {schedule['next_step']}: {loss.item()}"
                )
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
            schedule["next_step"] += 1

        metrics = {key: value / len(loader) for key, value in totals.items()}
        metrics.update({"epoch": epoch + 1, "lr": lr, "weight_decay": wd, "momentum": momentum})
        history.append(metrics)
        print(
            f"epoch {epoch + 1:3d}/{config.epochs} loss={metrics['loss']:.4f} "
            f"dino={metrics['dino']:.4f} ibot={metrics['ibot']:.4f} "
            f"koleo={metrics['koleo']:.4f}",
            flush=True,
        )

        payload = checkpoint_payload(
            config, model, optimizer, class_center, patch_center,
            history, device, schedule,
        )
        completed_epoch = epoch + 1
        if completed_epoch in checkpoint_epochs:
            save_checkpoint(milestone_path(output, completed_epoch), payload)
        if checkpoint_every and completed_epoch % checkpoint_every == 0:
            save_checkpoint(rolling_path(output), payload, write_metrics=False)

    final_payload = checkpoint_payload(
        config, model, optimizer, class_center, patch_center,
        history, device, schedule,
    )
    temporary = rolling_path(output)
    if end_epoch < config.epochs:
        save_checkpoint(temporary, final_payload, write_metrics=False)
        print(
            f"paused_after_epoch={end_epoch} target_epochs={config.epochs} "
            f"resumable_checkpoint={temporary}",
            flush=True,
        )
        return {
            key: final_payload[key]
            for key in (
                "config", "history", "device", "parameters", "completed_epoch",
                "global_step", "schedule",
            )
        }

    save_checkpoint(output, final_payload)
    temporary.unlink(missing_ok=True)
    temporary.with_suffix(".json").unlink(missing_ok=True)
    print(f"completed={output} temporary_checkpoint_cleaned={temporary}", flush=True)
    result = {
        key: final_payload[key]
        for key in (
            "config", "history", "device", "parameters", "completed_epoch",
            "global_step", "schedule",
        )
    }
    return result


def parse_args() -> tuple:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset", type=int, default=0, help="0 uses all 60,000 samples")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
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
    parser.add_argument(
        "--photometric-augmentations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply brightness/contrast jitter, blur, and solarization after cropping "
            "(default: disabled; random resized crops remain enabled)."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--output", type=Path, default=MODELS_DIR / "dinov2_mnist_preproc.pt"
    )
    parser.add_argument(
        "--checkpoint-epochs", type=parse_epoch_list, default=(50, 75, 100),
        help="Comma-separated milestone epochs to preserve (default: 50,75,100).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Write a replaceable resumable checkpoint every N epochs; 0 disables it.",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Resume from a full training-state checkpoint.",
    )
    parser.add_argument(
        "--stop-after-epoch", type=int,
        help=(
            "Pause after this epoch while retaining the --epochs schedule horizon; "
            "the rolling checkpoint can be resumed with the same target."
        ),
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
        photometric_augmentations=args.photometric_augmentations,
    )
    return (
        config,
        args.output,
        torch.device(args.device) if args.device else None,
        args.checkpoint_epochs,
        args.checkpoint_every,
        args.resume,
        args.stop_after_epoch,
    )


if __name__ == "__main__":
    train(*parse_args())
