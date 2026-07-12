"""Train XGBoost on frozen features from the best custom I-JEPA backbone.

The default uses the 300-epoch, 48-target / 16-context custom I-JEPA checkpoint
and mean-pools its 64 target-encoder tokens into 128 features. A deterministic,
class-balanced validation set is carved from MNIST's training split for early
stopping; the held-out test split is scored exactly once after fitting.

    uv run python -m ijepa_trials.xgboost_probe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ijepa_trials._ckpt import MODELS_DIR, set_seed
from ijepa_trials.mlp_probe import (
    DEFAULT_BACKBONE_EPOCHS,
    DEFAULT_N_TARGETS,
    load_best_backbone,
)
from trials.eval_classifier import extract_features
from trials.mae import pick_device


def stratified_validation_indices(
    labels: torch.Tensor, per_class: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic train/validation indices with equal validation classes."""
    generator = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0]
        if len(indices) <= per_class:
            raise ValueError(
                f"class {label.item()} has {len(indices)} samples; need more than {per_class}"
            )
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        validation_parts.append(shuffled[:per_class])
        train_parts.append(shuffled[per_class:])
    train_indices = torch.cat(train_parts)
    validation_indices = torch.cat(validation_parts)
    train_indices = train_indices[torch.randperm(len(train_indices), generator=generator)]
    validation_indices = validation_indices[
        torch.randperm(len(validation_indices), generator=generator)
    ]
    return train_indices, validation_indices


def classifier_params(args: argparse.Namespace) -> dict:
    """Regularized, CPU-friendly first-pass boosted-tree configuration."""
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_bin": args.max_bin,
        "early_stopping_rounds": args.early_stopping_rounds,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cpu") if args.feature_device == "cpu" else pick_device()
    model, backbone_path, backbone_config = load_best_backbone(
        device, args.backbone_epochs, args.n_targets
    )
    print(f"Seed: {args.seed}  feature device: {device}", flush=True)
    print(f"Frozen backbone: {backbone_path.name}", flush=True)
    print(f"Extracting {args.pool}-pooled embeddings...", flush=True)
    train_features, train_labels = extract_features(
        model, True, device, pool=args.pool, preproc=True
    )
    test_features, test_labels = extract_features(
        model, False, device, pool=args.pool, preproc=True
    )
    fit_indices, validation_indices = stratified_validation_indices(
        train_labels, args.validation_per_class, args.seed
    )
    X_fit = train_features[fit_indices]
    y_fit = train_labels[fit_indices]
    X_validation = train_features[validation_indices]
    y_validation = train_labels[validation_indices]
    print(
        f"Features: fit={tuple(X_fit.shape)} validation={tuple(X_validation.shape)} "
        f"test={tuple(test_features.shape)}",
        flush=True,
    )

    # PyTorch and XGBoost load incompatible native parallel runtimes in this
    # macOS environment. Persist the embeddings and fit XGBoost in a clean child
    # process that never imports torch.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{args.pool}_rows{round(args.subsample * 100)}_"
        f"cols{round(args.colsample_bytree * 100)}_depth{args.max_depth}_seed{args.seed}"
    )
    features_path = args.output_dir / f"ijepa_xgboost_{tag}_features.npz"
    job_path = args.output_dir / f"ijepa_xgboost_{tag}_job.json"
    model_path = args.output_dir / f"ijepa_xgboost_{tag}.json"
    results_path = args.output_dir / f"ijepa_xgboost_{tag}_results.json"
    np.savez(
        features_path,
        X_fit=X_fit.numpy(),
        y_fit=y_fit.numpy(),
        X_validation=X_validation.numpy(),
        y_validation=y_validation.numpy(),
        X_test=test_features.numpy(),
        y_test=test_labels.numpy(),
    )
    job = {
        "seed": args.seed,
        "backbone_checkpoint": backbone_path.name,
        "backbone_config": backbone_config,
        "backbone_frozen": True,
        "pool": args.pool,
        "feature_dim": int(train_features.shape[1]),
        "validation_per_class": args.validation_per_class,
        "fit_samples": len(fit_indices),
        "validation_samples": len(validation_indices),
        "xgboost_params": classifier_params(args),
        "verbose_every": args.verbose_every,
        "model_path": str(model_path),
        "results_path": str(results_path),
    }
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    print(f"Cached features -> {features_path}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_xgboost_fit.py")),
            "--features",
            str(features_path),
            "--job",
            str(job_path),
        ],
        check=True,
    )
    result = json.loads(results_path.read_text())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-epochs", type=int, default=DEFAULT_BACKBONE_EPOCHS)
    parser.add_argument("--n-targets", type=int, default=DEFAULT_N_TARGETS)
    parser.add_argument("--pool", choices=("mean", "flatten"), default="mean")
    parser.add_argument(
        "--feature-device",
        choices=("cpu", "auto"),
        default="auto",
        help="Device for frozen feature extraction; XGBoost always runs separately on CPU.",
    )
    parser.add_argument("--validation-per-class", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-child-weight", type=float, default=2.0)
    parser.add_argument("--subsample", type=float, default=0.5)
    parser.add_argument("--colsample-bytree", type=float, default=0.5)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--verbose-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
