"""Calibrate the three nonlinear probes with temperatures and diagonal weights.

The calibrator has one positive temperature per model and one non-negative
model weight per output class.  It acts on per-sample centered logits so a
model's arbitrary common logit offset cannot become a class-dependent bias.
All calibration choices are made from MNIST training logits before test
prediction artifacts are loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from mnist_ssl.baselines.mae import make_transform
from mnist_ssl.dinov2.eval_frozen import backbone_fingerprint, seed_everything
from mnist_ssl.dinov2.nonlinear_probe import (
    SmallNonlinearProbe,
    batched_logits,
    file_sha256,
    load_feature_cache,
)
from mnist_ssl.dinov2.train import pick_device
from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.ijepa.nonlinear_probe import load_best_member
from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR


MODEL_NAMES = ("dino", "ijepa_300", "ijepa_500")
DEFAULT_DINO_BACKBONE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt"
)
DEFAULT_DINO_TRAIN_CACHE = (
    OUT_DIR / "dinov2_reranker_oof_10fold_ep50_75_seed0" / "train_features.pt"
)
DEFAULT_DINO_HEAD = (
    OUT_DIR / "dinov2_nonlinear_probe_50ep" / "nonlinear_probe.pt"
)
DEFAULT_IJEPA_300_BASE = (
    MODELS_DIR
    / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_"
    "base300ep_probe50ep.pt"
)
DEFAULT_IJEPA_300_HEAD = (
    OUT_DIR / "ijepa_nonlinear_probe_best300" / "nonlinear_probe_75ep.pt"
)
DEFAULT_IJEPA_500_BASE = (
    MODELS_DIR
    / "ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_"
    "base500ep_probe50ep.pt"
)
DEFAULT_IJEPA_500_HEAD = (
    OUT_DIR / "ijepa_nonlinear_probe_best500" / "nonlinear_probe_75ep.pt"
)
DEFAULT_DINO_TEST = (
    OUT_DIR / "dinov2_nonlinear_probe_50ep" / "predictions.pt"
)
DEFAULT_IJEPA_300_TEST = (
    OUT_DIR / "ijepa_nonlinear_probe_best300" / "predictions.pt"
)
DEFAULT_IJEPA_500_TEST = (
    OUT_DIR / "ijepa_nonlinear_probe_best500" / "predictions.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "nonlinear_temperature_diagonal_v1"


class TemperatureDiagonalEnsemble(nn.Module):
    """Centered-logit ensemble with a temperature and diagonal class weights."""

    def __init__(self, n_models: int = 3, n_classes: int = 10) -> None:
        super().__init__()
        self.log_temperatures = nn.Parameter(torch.zeros(n_models))
        self.weight_logits = nn.Parameter(torch.zeros(n_models, n_classes))

    @property
    def temperatures(self) -> torch.Tensor:
        return self.log_temperatures.exp()

    @property
    def class_weights(self) -> torch.Tensor:
        return self.weight_logits.softmax(dim=0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError("logits must have shape [examples, models, classes]")
        if logits.shape[1:] != self.weight_logits.shape:
            raise ValueError(
                "logit model/class dimensions disagree with the calibrator"
            )
        centered = logits - logits.mean(dim=-1, keepdim=True)
        scaled = centered / self.temperatures[None, :, None]
        return (self.class_weights[None, :, :] * scaled).sum(dim=1)


def parse_regularizations(value: str) -> tuple[float, ...]:
    try:
        result = tuple(
            sorted({float(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "regularizations must be comma-separated numbers"
        ) from exc
    if not result or result[0] < 0:
        raise argparse.ArgumentTypeError(
            "regularizations must contain non-negative values"
        )
    return result


def calibration_split(
    labels: torch.Tensor,
    *,
    selection_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic class-stratified fit and selection indices."""

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0 < selection_size < len(labels):
        raise ValueError("selection_size must be between zero and dataset size")

    counts = torch.bincount(labels)
    exact = counts.float() * (selection_size / len(labels))
    selected_per_class = exact.floor().long()
    remainder = selection_size - int(selected_per_class.sum().item())
    fractional_order = (exact - selected_per_class).argsort(descending=True)
    selected_per_class[fractional_order[:remainder]] += 1

    generator = torch.Generator().manual_seed(seed)
    selection_parts = []
    fit_parts = []
    for class_id, class_selection_size in enumerate(selected_per_class.tolist()):
        indices = labels.eq(class_id).nonzero(as_tuple=False).flatten()
        indices = indices[torch.randperm(len(indices), generator=generator)]
        selection_parts.append(indices[:class_selection_size])
        fit_parts.append(indices[class_selection_size:])
    selection = torch.cat(selection_parts).sort().values
    fit = torch.cat(fit_parts).sort().values
    return fit, selection


def _load_nonlinear_head(path: Path, device: torch.device) -> SmallNonlinearProbe:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("head_type") != "layernorm-mlp":
        raise ValueError(f"unsupported nonlinear head in {path}")
    head = SmallNonlinearProbe(
        checkpoint["in_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_classes=checkpoint.get("n_classes", 10),
        dropout=checkpoint["dropout"],
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head.eval()
    return head


@torch.no_grad()
def _extract_ijepa_training_logits(
    members: dict[str, tuple[nn.Module, SmallNonlinearProbe]],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    dataset = datasets.MNIST(
        str(DATASET_DIR),
        train=True,
        download=True,
        transform=make_transform(preproc=True),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    outputs = {
        name: torch.empty(len(dataset), 10, dtype=torch.float32)
        for name in members
    }
    labels = torch.empty(len(dataset), dtype=torch.long)
    offset = 0
    for images, targets in loader:
        images = images.to(device)
        end = offset + len(targets)
        labels[offset:end].copy_(targets)
        for name, (model, head) in members.items():
            features = model.encode(images, pool="flatten").float()
            outputs[name][offset:end].copy_(head(features).cpu())
        offset = end
    if offset != len(dataset):
        raise RuntimeError("failed to extract the complete MNIST training split")
    return outputs, labels


def _training_artifact_signature(args: argparse.Namespace) -> dict[str, str]:
    return {
        "dino_backbone_sha256": file_sha256(args.dino_backbone),
        "dino_head_sha256": file_sha256(args.dino_head),
        "ijepa_300_base_sha256": file_sha256(args.ijepa_300_base),
        "ijepa_300_head_sha256": file_sha256(args.ijepa_300_head),
        "ijepa_500_base_sha256": file_sha256(args.ijepa_500_base),
        "ijepa_500_head_sha256": file_sha256(args.ijepa_500_head),
    }


def _generate_training_artifact(
    args: argparse.Namespace,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    signature = _training_artifact_signature(args)
    dino_features, dino_labels, _ = load_feature_cache(
        args.dino_train_cache,
        checkpoint_sha256=signature["dino_backbone_sha256"],
        source_split="MNIST train (canonical order)",
        pool="cls",
    )
    dino_head = _load_nonlinear_head(args.dino_head, device)
    dino_logits = batched_logits(
        dino_head,
        dino_features,
        device,
        args.eval_batch_size,
    )
    del dino_features, dino_head

    ijepa_300, _, _ = load_best_member(
        args.ijepa_300_base,
        device,
        pretraining_epochs=300,
    )
    ijepa_500, _, _ = load_best_member(
        args.ijepa_500_base,
        device,
        pretraining_epochs=500,
    )
    ijepa_300_head = _load_nonlinear_head(args.ijepa_300_head, device)
    ijepa_500_head = _load_nonlinear_head(args.ijepa_500_head, device)
    fingerprints_before = {
        "ijepa_300": backbone_fingerprint(ijepa_300),
        "ijepa_500": backbone_fingerprint(ijepa_500),
    }
    ijepa_logits, ijepa_labels = _extract_ijepa_training_logits(
        {
            "ijepa_300": (ijepa_300, ijepa_300_head),
            "ijepa_500": (ijepa_500, ijepa_500_head),
        },
        device=device,
        batch_size=args.feature_batch_size,
        workers=args.workers,
    )
    fingerprints_after = {
        "ijepa_300": backbone_fingerprint(ijepa_300),
        "ijepa_500": backbone_fingerprint(ijepa_500),
    }
    if fingerprints_before != fingerprints_after:
        raise RuntimeError("an I-JEPA backbone changed during logit extraction")
    if not torch.equal(dino_labels, ijepa_labels):
        raise ValueError("DINO and I-JEPA training labels are not aligned")

    artifact = {
        "signature": signature,
        "model_order": MODEL_NAMES,
        "labels": dino_labels,
        "logits": torch.stack(
            [
                dino_logits,
                ijepa_logits["ijepa_300"],
                ijepa_logits["ijepa_500"],
            ],
            dim=1,
        ),
        "backbone_fingerprints_before": fingerprints_before,
        "backbone_fingerprints_after": fingerprints_after,
    }
    torch.save(artifact, path)
    return artifact


def _load_or_generate_training_artifact(
    args: argparse.Namespace,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    expected = _training_artifact_signature(args)
    if path.exists():
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("signature") != expected:
            raise ValueError("training-logit artifact signature mismatch")
        print(f"reused_training_logits={path}", flush=True)
    else:
        print("generating_training_logits=true", flush=True)
        artifact = _generate_training_artifact(args, path, device)
        print(f"training_logits={path}", flush=True)
    if tuple(artifact.get("model_order", ())) != MODEL_NAMES:
        raise ValueError("training-logit model order mismatch")
    if artifact["logits"].shape != (60_000, 3, 10):
        raise ValueError("training logits must have shape [60000,3,10]")
    if artifact["labels"].shape != (60_000,):
        raise ValueError("training labels must have shape [60000]")
    return artifact


def _regularized_loss(
    model: TemperatureDiagonalEnsemble,
    logits: torch.Tensor,
    labels: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    loss = F.cross_entropy(model(logits), labels)
    if regularization:
        uniform = torch.full_like(model.class_weights, 1.0 / logits.shape[1])
        penalty = model.log_temperatures.square().mean()
        penalty = penalty + (model.class_weights - uniform).square().mean()
        loss = loss + regularization * penalty
    return loss


@torch.no_grad()
def _metrics(
    model: TemperatureDiagonalEnsemble,
    logits: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    if include_mask is None:
        include_mask = torch.ones(len(labels), dtype=torch.bool)
    scores = model(logits)
    errors = int(scores[include_mask].argmax(dim=1).ne(labels[include_mask]).sum())
    count = int(include_mask.sum())
    nll = float(F.cross_entropy(scores[include_mask], labels[include_mask]))
    return {
        "scored_examples": count,
        "errors": errors,
        "accuracy_percent": 100.0 * (1.0 - errors / count),
        "nll": nll,
    }


def _fit_candidate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: torch.Tensor,
    selection_indices: torch.Tensor,
    *,
    regularization: float,
    steps: int,
    eval_every: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seed_everything(seed)
    model = TemperatureDiagonalEnsemble()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    best_nll = float("inf")
    best_errors = len(selection_indices)

    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            selection = _metrics(
                model,
                logits[selection_indices],
                labels[selection_indices],
            )
            selection_nll = float(selection["nll"])
            selection_errors = int(selection["errors"])
            if (selection_nll, selection_errors) < (best_nll, best_errors):
                best_nll = selection_nll
                best_errors = selection_errors
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
        if step == steps:
            break
        loss = _regularized_loss(
            model,
            logits[fit_indices],
            labels[fit_indices],
            regularization,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.load_state_dict(best_state)
    return best_state, {
        "regularization": regularization,
        "selected_step": best_step,
        "fit": _metrics(model, logits[fit_indices], labels[fit_indices]),
        "selection": _metrics(
            model,
            logits[selection_indices],
            labels[selection_indices],
        ),
    }


def _fit_from_training(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[TemperatureDiagonalEnsemble, dict[str, Any]]:
    fit_indices, selection_indices = calibration_split(
        labels,
        selection_size=args.selection_size,
        seed=args.split_seed,
    )
    candidates = []
    states = []
    for regularization in args.regularizations:
        state, result = _fit_candidate(
            logits,
            labels,
            fit_indices,
            selection_indices,
            regularization=regularization,
            steps=args.steps,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        states.append(state)
        candidates.append(result)
        print(
            f"regularization={regularization:g} "
            f"step={result['selected_step']} "
            f"selection_errors={result['selection']['errors']} "
            f"selection_nll={result['selection']['nll']:.8f}",
            flush=True,
        )
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["selection"]["nll"],
            candidates[index]["selection"]["errors"],
            candidates[index]["regularization"],
        ),
    )
    model = TemperatureDiagonalEnsemble()
    model.load_state_dict(states[selected_index])
    return model, {
        "fit_examples": len(fit_indices),
        "selection_examples": len(selection_indices),
        "split_seed": args.split_seed,
        "selection_rule": "minimum training-selection NLL; errors and regularization break ties",
        "candidates": candidates,
        "selected_candidate_index": selected_index,
        "selected": candidates[selected_index],
    }


def _load_test_artifacts(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str]]:
    paths = {
        "dino": args.dino_test,
        "ijepa_300": args.ijepa_300_test,
        "ijepa_500": args.ijepa_500_test,
    }
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    reference = payloads["dino"]
    for name, payload in payloads.items():
        for field in (
            "canonical_labels",
            "reviewed_labels",
            "reviewed_include_mask",
        ):
            if not torch.equal(reference[field], payload[field]):
                raise ValueError(f"{name} disagrees on {field}")
    logits = torch.stack(
        [
            payloads["dino"]["nonlinear_logits"].float(),
            payloads["ijepa_300"]["nonlinear_logits_by_epoch"][75].float(),
            payloads["ijepa_500"]["nonlinear_logits_by_epoch"][75].float(),
        ],
        dim=1,
    )
    if logits.shape != (10_000, 3, 10):
        raise ValueError("test logits must have shape [10000,3,10]")
    return (
        logits,
        reference["canonical_labels"].long(),
        reference["reviewed_labels"].long(),
        reference["reviewed_include_mask"].bool(),
        {name: file_sha256(path) for name, path in paths.items()},
    )


def _individual_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, dict[str, float | int]]:
    result = {}
    for index, name in enumerate(MODEL_NAMES):
        fixed = TemperatureDiagonalEnsemble()
        with torch.no_grad():
            fixed.weight_logits.fill_(-30.0)
            fixed.weight_logits[index].fill_(30.0)
        result[name] = _metrics(fixed, logits, labels, include_mask)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("steps and eval_every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "training_logits.pt"
    checkpoint_path = args.output_dir / "calibrator.pt"
    summary_path = args.output_dir / "summary.json"
    for output in (checkpoint_path, summary_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    device = torch.device(args.device) if args.device else pick_device()
    training = _load_or_generate_training_artifact(
        args,
        train_path,
        device,
    )
    train_logits = training["logits"].float()
    train_labels = training["labels"].long()
    calibrator, fit_summary = _fit_from_training(
        train_logits,
        train_labels,
        args,
    )
    train_metrics = _metrics(calibrator, train_logits, train_labels)

    # Deliberately load the test artifacts only after calibration is frozen.
    (
        test_logits,
        canonical_labels,
        reviewed_labels,
        reviewed_mask,
        test_hashes,
    ) = _load_test_artifacts(args)
    current_review = apply_mnist_test_label_policy(canonical_labels)
    if not torch.equal(current_review.labels, reviewed_labels):
        raise ValueError("saved reviewed labels differ from the current policy")
    if not torch.equal(current_review.include_mask, reviewed_mask):
        raise ValueError("saved reviewed mask differs from the current policy")

    canonical = _metrics(calibrator, test_logits, canonical_labels)
    reviewed = _metrics(
        calibrator,
        test_logits,
        reviewed_labels,
        reviewed_mask,
    )
    parameters = {
        "temperatures": {
            name: float(value)
            for name, value in zip(
                MODEL_NAMES,
                calibrator.temperatures.detach(),
            )
        },
        "class_weights": {
            name: [float(value) for value in row]
            for name, row in zip(
                MODEL_NAMES,
                calibrator.class_weights.detach(),
            )
        },
    }
    result = {
        "protocol": {
            "model_order": MODEL_NAMES,
            "members": {
                "dino": "epoch-75 backbone, 50-epoch nonlinear probe",
                "ijepa_300": "epoch-300 backbone, 75-epoch nonlinear probe",
                "ijepa_500": "epoch-500 backbone, 75-epoch nonlinear probe",
            },
            "test_loaded_after_calibration_selection": True,
            "test_labels_used_for_fitting": False,
            "logit_centering": "subtract each model's per-example class mean",
            "temperature_parameterization": "positive exponential",
            "class_weight_parameterization": "softmax across models for each class",
            "loss": "cross_entropy plus selected shrinkage toward T=1 and equal weights",
            "training_logits_sha256": file_sha256(train_path),
            "test_prediction_sha256": test_hashes,
            "seed": args.seed,
        },
        "fit": fit_summary,
        "parameters": parameters,
        "training": train_metrics,
        "canonical_test": {
            "calibrated": canonical,
            "individuals": _individual_metrics(test_logits, canonical_labels),
        },
        "reviewed_test": {
            "calibrated": reviewed,
            "individuals": _individual_metrics(
                test_logits,
                reviewed_labels,
                reviewed_mask,
            ),
            "policy": current_review.metadata,
        },
    }
    torch.save(
        {
            "state_dict": calibrator.state_dict(),
            "model_order": MODEL_NAMES,
            "parameters": parameters,
            "fit": fit_summary,
            "protocol": result["protocol"],
        },
        checkpoint_path,
    )
    result["protocol"]["calibrator_sha256"] = file_sha256(checkpoint_path)
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"canonical_errors={canonical['errors']} "
        f"canonical_accuracy={canonical['accuracy_percent']:.5f}% "
        f"reviewed_errors={reviewed['errors']} "
        f"reviewed_accuracy={reviewed['accuracy_percent']:.5f}%",
        flush=True,
    )
    print(f"summary={summary_path}", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-backbone", type=Path, default=DEFAULT_DINO_BACKBONE)
    parser.add_argument(
        "--dino-train-cache",
        type=Path,
        default=DEFAULT_DINO_TRAIN_CACHE,
    )
    parser.add_argument("--dino-head", type=Path, default=DEFAULT_DINO_HEAD)
    parser.add_argument(
        "--ijepa-300-base",
        type=Path,
        default=DEFAULT_IJEPA_300_BASE,
    )
    parser.add_argument(
        "--ijepa-300-head",
        type=Path,
        default=DEFAULT_IJEPA_300_HEAD,
    )
    parser.add_argument(
        "--ijepa-500-base",
        type=Path,
        default=DEFAULT_IJEPA_500_BASE,
    )
    parser.add_argument(
        "--ijepa-500-head",
        type=Path,
        default=DEFAULT_IJEPA_500_HEAD,
    )
    parser.add_argument("--dino-test", type=Path, default=DEFAULT_DINO_TEST)
    parser.add_argument(
        "--ijepa-300-test",
        type=Path,
        default=DEFAULT_IJEPA_300_TEST,
    )
    parser.add_argument(
        "--ijepa-500-test",
        type=Path,
        default=DEFAULT_IJEPA_500_TEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--regularizations",
        type=parse_regularizations,
        default=(0.0, 0.0001, 0.001, 0.01, 0.1),
    )
    parser.add_argument("--selection-size", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args(argv)
