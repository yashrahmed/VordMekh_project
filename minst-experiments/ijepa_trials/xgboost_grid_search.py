"""Test-tuned staged grid search for the frozen I-JEPA XGBoost probe.

This runner intentionally imports neither torch nor project model code. It uses
the cached feature split produced by :mod:`ijepa_trials.xgboost_probe`. Per the
experiment's explicit protocol, candidates are ranked directly by MNIST test
accuracy; validation metrics remain diagnostics rather than the selection rule.

    uv run python -m ijepa_trials.xgboost_grid_search
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from trials.mae import MODELS_DIR

DEFAULT_FEATURES = MODELS_DIR / "ijepa_xgboost_mean_rows50_cols50_depth8_seed0_features.npz"
DEFAULT_RESULTS = MODELS_DIR / "ijepa_xgboost_grid_mean_seed0_results.json"
DEFAULT_MODEL = MODELS_DIR / "ijepa_xgboost_grid_mean_seed0_best.json"


def base_params(seed: int, n_jobs: int) -> dict[str, Any]:
    return {
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 2.0,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 5.0,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_bin": 256,
        "early_stopping_rounds": 50,
        "random_state": seed,
        "n_jobs": n_jobs,
    }


def structural_grid(seed: int, n_jobs: int) -> list[dict[str, Any]]:
    """Coarse 3x3x3 grid over capacity and row/feature stochasticity."""
    common = base_params(seed, n_jobs)
    candidates = []
    for depth, rows, columns in itertools.product(
        (3, 5, 8), (0.5, 0.75, 1.0), (0.5, 0.75, 1.0)
    ):
        candidates.append(
            common | {"max_depth": depth, "subsample": rows, "colsample_bytree": columns}
        )
    return candidates


def regularization_grid(winner: dict[str, Any]) -> list[dict[str, Any]]:
    """Refine leaf support and L2 around the best structural configuration."""
    return [
        winner | {"min_child_weight": child_weight, "reg_lambda": reg_lambda}
        for child_weight, reg_lambda in itertools.product((1.0, 2.0, 5.0), (1.0, 5.0, 15.0))
    ]


def learning_rate_grid(winner: dict[str, Any]) -> list[dict[str, Any]]:
    """Check slower and faster shrinkage after structure/regularization selection."""
    return [winner | {"learning_rate": rate} for rate in (0.025, 0.05, 0.1)]


def signature(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def selection_key(trial: dict[str, Any]) -> tuple[float, float, float]:
    """Maximize test accuracy, then validation accuracy and validation log-loss."""
    return (
        trial["test_accuracy"],
        trial["validation_accuracy"],
        -trial["validation_logloss"],
    )


def best_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("cannot select from an empty trial list")
    return max(trials, key=selection_key)


def validation_accuracy(
    classifier: XGBClassifier, features: np.ndarray, labels: np.ndarray
) -> float:
    return float(np.mean(classifier.predict(features) == labels))


def fit_candidate(
    params: dict[str, Any],
    stage: str,
    trial_number: int,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], XGBClassifier]:
    classifier = XGBClassifier(**params)
    classifier.fit(
        X_fit,
        y_fit,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    trial = {
        "trial": trial_number,
        "stage": stage,
        "params": params,
        "best_iteration": int(classifier.best_iteration),
        "validation_logloss": float(classifier.best_score),
        "validation_accuracy": validation_accuracy(classifier, X_validation, y_validation),
        "test_accuracy": validation_accuracy(classifier, X_test, y_test),
    }
    return trial, classifier


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.features.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {args.features}. Run "
            "`python -m ijepa_trials.xgboost_probe` first."
        )
    arrays = np.load(args.features)
    X_fit, y_fit = arrays["X_fit"], arrays["y_fit"]
    X_validation, y_validation = arrays["X_validation"], arrays["y_validation"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]
    print(
        f"Grid data: fit={X_fit.shape} validation={X_validation.shape} test={X_test.shape}; "
        "selection is intentionally test-tuned",
        flush=True,
    )

    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    args.model.parent.mkdir(parents=True, exist_ok=True)

    def run_stage(stage: str, candidates: list[dict[str, Any]]) -> None:
        for params in candidates:
            key = signature(params)
            if key in seen:
                continue
            seen.add(key)
            trial, classifier = fit_candidate(
                params,
                stage,
                len(trials) + 1,
                X_fit,
                y_fit,
                X_validation,
                y_validation,
                X_test,
                y_test,
            )
            trials.append(trial)
            if best_trial(trials) is trial:
                classifier.save_model(args.model)
            print(
                f"[{trial['trial']:02d}] {stage:14s} "
                f"depth={params['max_depth']} rows={params['subsample']:.2f} "
                f"cols={params['colsample_bytree']:.2f} "
                f"child={params['min_child_weight']:.1f} l2={params['reg_lambda']:.1f} "
                f"lr={params['learning_rate']:.3f} iter={trial['best_iteration']} "
                f"val={trial['validation_accuracy']:.2%} test={trial['test_accuracy']:.2%} "
                f"logloss={trial['validation_logloss']:.5f}",
                flush=True,
            )
            args.results.parent.mkdir(parents=True, exist_ok=True)
            args.results.write_text(
                json.dumps({"complete": False, "trials": trials}, indent=2) + "\n"
            )

    run_stage("structure", structural_grid(args.seed, args.n_jobs))
    structure_winner = best_trial(trials)
    run_stage("regularization", regularization_grid(structure_winner["params"]))
    regularized_winner = best_trial(trials)
    run_stage("learning-rate", learning_rate_grid(regularized_winner["params"]))

    winner = best_trial(trials)
    classifier = XGBClassifier()
    classifier.load_model(args.model)
    fit_accuracy = float(np.mean(classifier.predict(X_fit) == y_fit))

    result = {
        "complete": True,
        "seed": args.seed,
        "features": str(args.features),
        "selection": (
            "maximize test accuracy; validation accuracy and lower validation logloss break ties"
        ),
        "test_used_during_search": True,
        "generalization_warning": (
            "The winner is test-tuned; its test accuracy is optimistic and is not an unbiased "
            "generalization estimate."
        ),
        "trial_count": len(trials),
        "winner": winner,
        "fit_accuracy": fit_accuracy,
        "test_accuracy": winner["test_accuracy"],
        "trials": trials,
    }
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"WINNER trial={winner['trial']} validation={winner['validation_accuracy']:.2%} "
        f"fit={fit_accuracy:.2%} test={winner['test_accuracy']:.2%}",
        flush=True,
    )
    print(f"Model -> {args.model}\nResults -> {args.results}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
