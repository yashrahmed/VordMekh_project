"""Run a staged Random Forest search on the best frozen custom I-JEPA backbone.

The backbone is evaluated once to build a temporary mean-pooled feature cache.
A clean child process searches Random Forest hyperparameters, selects only on a
fixed class-balanced validation split, and touches the test split once after
selecting the final 1,000-tree model. The temporary cache is removed on exit.

    uv run python -m trials.random_forest_grid_search
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from mnist_ssl.ijepa._ckpt import MODELS_DIR, set_seed

from .eval_classifier import extract_features
from .mae import pick_device
from .mlp_probe import (
    DEFAULT_BACKBONE_EPOCHS,
    DEFAULT_N_TARGETS,
    load_best_backbone,
)
from .xgboost_probe import stratified_validation_indices


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cpu") if args.feature_device == "cpu" else pick_device()
    backbone, backbone_path, backbone_config = load_best_backbone(
        device, args.backbone_epochs, args.n_targets
    )
    print(f"Seed: {args.seed}  feature device: {device}", flush=True)
    print(f"Frozen backbone: {backbone_path.name}", flush=True)
    print("Extracting mean-pooled embeddings once...", flush=True)
    train_features, train_labels = extract_features(
        backbone, True, device, pool="mean", preproc=True
    )
    test_features, test_labels = extract_features(
        backbone, False, device, pool="mean", preproc=True
    )
    fit_indices, validation_indices = stratified_validation_indices(
        train_labels, args.validation_per_class, args.seed
    )
    print(
        f"Features: fit={(len(fit_indices), train_features.shape[1])} "
        f"validation={(len(validation_indices), train_features.shape[1])} "
        f"test={tuple(test_features.shape)}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"ijepa_random_forest_grid_mean_seed{args.seed}.joblib"
    results_path = (
        args.output_dir / f"ijepa_random_forest_grid_mean_seed{args.seed}_results.json"
    )
    with tempfile.TemporaryDirectory(prefix="ijepa-rf-grid-") as temporary_dir:
        temporary = Path(temporary_dir)
        features_path = temporary / "features.npz"
        job_path = temporary / "job.json"
        np.savez(
            features_path,
            X_fit=train_features[fit_indices].numpy(),
            y_fit=train_labels[fit_indices].numpy(),
            X_validation=train_features[validation_indices].numpy(),
            y_validation=train_labels[validation_indices].numpy(),
            X_test=test_features.numpy(),
            y_test=test_labels.numpy(),
        )
        job = {
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "screening_trees": args.screening_trees,
            "final_trees": args.final_trees,
            "backbone_checkpoint": backbone_path.name,
            "backbone_config": backbone_config,
            "backbone_frozen": True,
            "pool": "mean",
            "feature_dim": int(train_features.shape[1]),
            "validation_per_class": args.validation_per_class,
            "fit_samples": len(fit_indices),
            "validation_samples": len(validation_indices),
            "model_path": str(model_path),
            "results_path": str(results_path),
        }
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        print(f"Temporary feature cache: {features_path}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("_random_forest_grid_fit.py")),
                "--features",
                str(features_path),
                "--job",
                str(job_path),
            ],
            check=True,
        )
    print("Temporary feature cache removed.", flush=True)
    return json.loads(results_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-epochs", type=int, default=DEFAULT_BACKBONE_EPOCHS)
    parser.add_argument("--n-targets", type=int, default=DEFAULT_N_TARGETS)
    parser.add_argument("--validation-per-class", type=int, default=500)
    parser.add_argument("--screening-trees", type=int, default=250)
    parser.add_argument("--final-trees", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--feature-device",
        choices=("cpu", "auto"),
        default="auto",
        help="Device for frozen feature extraction; forests fit separately on CPU.",
    )
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
