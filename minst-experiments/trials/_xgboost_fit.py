"""Native XGBoost fitting subprocess for :mod:`xgboost_probe`.

This module deliberately imports neither torch nor any project module. Keeping
PyTorch and XGBoost in separate processes avoids a macOS native-runtime crash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier


def accuracy(model: XGBClassifier, features: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(model.predict(features) == labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()

    job = json.loads(args.job.read_text())
    arrays = np.load(args.features)
    classifier = XGBClassifier(**job["xgboost_params"])
    classifier.fit(
        arrays["X_fit"],
        arrays["y_fit"],
        eval_set=[(arrays["X_validation"], arrays["y_validation"])],
        verbose=job["verbose_every"],
    )

    result = {
        key: value
        for key, value in job.items()
        if key not in {"verbose_every", "model_path", "results_path"}
    }
    result.update(
        {
            "best_iteration": int(classifier.best_iteration),
            "fit_accuracy": accuracy(classifier, arrays["X_fit"], arrays["y_fit"]),
            "validation_accuracy": accuracy(
                classifier, arrays["X_validation"], arrays["y_validation"]
            ),
            "test_accuracy": accuracy(classifier, arrays["X_test"], arrays["y_test"]),
        }
    )
    model_path = Path(job["model_path"])
    results_path = Path(job["results_path"])
    classifier.save_model(model_path)
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"best_iteration={result['best_iteration']} fit={result['fit_accuracy']:.2%} "
        f"validation={result['validation_accuracy']:.2%} "
        f"test={result['test_accuracy']:.2%}",
        flush=True,
    )
    print(f"Model -> {model_path}\nResults -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
