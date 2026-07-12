"""Native fitting subprocess for :mod:`trials.random_forest_grid_search`.

This file intentionally imports neither torch nor project model modules.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def signature(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def selection_key(trial: dict[str, Any]) -> tuple[float, float]:
    oob = trial["oob_accuracy"]
    return trial["validation_accuracy"], -1.0 if oob is None else oob


def ranked(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(trials, key=selection_key, reverse=True)


def stage_one_grid(job: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "n_estimators": job["screening_trees"],
        "criterion": "gini",
        "bootstrap": True,
        "max_samples": 0.8,
        "oob_score": True,
        "random_state": job["seed"],
        "n_jobs": job["n_jobs"],
    }
    return [
        common
        | {
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": features,
        }
        for depth, leaf, features in itertools.product(
            (None, 24), (1, 2, 4), ("sqrt", 0.25, 0.5)
        )
    ]


def stage_two_grid(
    leaders: list[dict[str, Any]], job: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = []
    sampling = ((True, 0.5), (True, 0.8), (True, 1.0), (False, None))
    for leader in leaders:
        geometry = leader["params"]
        for criterion, (bootstrap, max_samples) in itertools.product(
            ("gini", "entropy"), sampling
        ):
            candidates.append(
                geometry
                | {
                    "n_estimators": job["screening_trees"],
                    "criterion": criterion,
                    "bootstrap": bootstrap,
                    "max_samples": max_samples,
                    "oob_score": bootstrap,
                }
            )
    return candidates


def fit_candidate(
    params: dict[str, Any],
    stage: str,
    trial_number: int,
    arrays: Any,
) -> tuple[dict[str, Any], RandomForestClassifier]:
    classifier = RandomForestClassifier(**params)
    started = time.perf_counter()
    classifier.fit(arrays["X_fit"], arrays["y_fit"])
    fit_seconds = time.perf_counter() - started
    trial = {
        "trial": trial_number,
        "stage": stage,
        "params": params,
        "validation_accuracy": float(
            classifier.score(arrays["X_validation"], arrays["y_validation"])
        ),
        "oob_accuracy": float(classifier.oob_score_) if params["oob_score"] else None,
        "fit_seconds": fit_seconds,
    }
    return trial, classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()

    job = json.loads(args.job.read_text())
    arrays = np.load(args.features)
    print(
        f"Grid data: fit={arrays['X_fit'].shape} "
        f"validation={arrays['X_validation'].shape}; test held back",
        flush=True,
    )
    screening_trials: list[dict[str, Any]] = []
    final_trials: list[dict[str, Any]] = []
    seen: set[str] = set()

    def write_progress() -> None:
        Path(job["results_path"]).write_text(
            json.dumps(
                {
                    "complete": False,
                    "test_used_during_search": False,
                    "screening_trials": screening_trials,
                    "final_trials": final_trials,
                },
                indent=2,
            )
            + "\n"
        )

    def run_screening_stage(stage: str, candidates: list[dict[str, Any]]) -> None:
        for params in candidates:
            key = signature(params)
            if key in seen:
                continue
            seen.add(key)
            trial, _ = fit_candidate(params, stage, len(screening_trials) + 1, arrays)
            screening_trials.append(trial)
            print(
                f"[{trial['trial']:02d}] {stage:10s} trees={params['n_estimators']} "
                f"depth={params['max_depth']} leaf={params['min_samples_leaf']} "
                f"features={params['max_features']} criterion={params['criterion']} "
                f"bootstrap={params['bootstrap']} samples={params['max_samples']} "
                f"val={trial['validation_accuracy']:.2%} "
                f"oob={trial['oob_accuracy'] if trial['oob_accuracy'] is not None else 'n/a'} "
                f"time={trial['fit_seconds']:.1f}s",
                flush=True,
            )
            write_progress()

    run_screening_stage("structure", stage_one_grid(job))
    stage_one_leaders = ranked(screening_trials)[:2]
    run_screening_stage("sampling", stage_two_grid(stage_one_leaders, job))

    finalist_params = [
        trial["params"] | {"n_estimators": job["final_trees"]}
        for trial in ranked(screening_trials)[:3]
    ]
    best_model = None
    for params in finalist_params:
        trial, classifier = fit_candidate(params, "final", len(final_trials) + 1, arrays)
        final_trials.append(trial)
        if ranked(final_trials)[0] is trial:
            best_model = classifier
            joblib.dump(best_model, job["model_path"], compress=3)
        print(
            f"[F{trial['trial']}] trees={params['n_estimators']} "
            f"val={trial['validation_accuracy']:.2%} "
            f"oob={trial['oob_accuracy'] if trial['oob_accuracy'] is not None else 'n/a'} "
            f"time={trial['fit_seconds']:.1f}s",
            flush=True,
        )
        write_progress()

    winner = ranked(final_trials)[0]
    best_model = joblib.load(job["model_path"])
    fit_accuracy = float(best_model.score(arrays["X_fit"], arrays["y_fit"]))
    # First and only use of the test arrays, after validation-only selection.
    test_accuracy = float(best_model.score(arrays["X_test"], arrays["y_test"]))
    result = {
        key: value
        for key, value in job.items()
        if key not in {"model_path", "results_path"}
    }
    result.update(
        {
            "complete": True,
            "selection": "validation accuracy; OOB accuracy breaks ties",
            "test_used_during_search": False,
            "screening_trial_count": len(screening_trials),
            "final_trial_count": len(final_trials),
            "winner": winner,
            "fit_accuracy": fit_accuracy,
            "test_accuracy": test_accuracy,
            "model_bytes": Path(job["model_path"]).stat().st_size,
            "screening_trials": screening_trials,
            "final_trials": final_trials,
        }
    )
    Path(job["results_path"]).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"WINNER validation={winner['validation_accuracy']:.2%} "
        f"fit={fit_accuracy:.2%} test={test_accuracy:.2%}",
        flush=True,
    )
    print(f"Model -> {job['model_path']}\nResults -> {job['results_path']}", flush=True)


if __name__ == "__main__":
    main()
