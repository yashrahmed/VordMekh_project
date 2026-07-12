import pytest

from ijepa_trials.xgboost_grid_search import (
    base_params,
    best_trial,
    learning_rate_grid,
    regularization_grid,
    structural_grid,
)


def test_structural_grid_covers_depth_and_sampling_axes():
    grid = structural_grid(seed=0, n_jobs=8)

    assert len(grid) == 27
    assert {item["max_depth"] for item in grid} == {3, 5, 8}
    assert {item["subsample"] for item in grid} == {0.5, 0.75, 1.0}
    assert {item["colsample_bytree"] for item in grid} == {0.5, 0.75, 1.0}


def test_refinement_grids_hold_prior_winner_axes_fixed():
    winner = base_params(seed=0, n_jobs=8) | {
        "max_depth": 8,
        "subsample": 0.5,
        "colsample_bytree": 1.0,
    }

    regularization = regularization_grid(winner)
    learning_rates = learning_rate_grid(winner)

    assert len(regularization) == 9
    assert {item["min_child_weight"] for item in regularization} == {1.0, 2.0, 5.0}
    assert {item["reg_lambda"] for item in regularization} == {1.0, 5.0, 15.0}
    assert all(item["max_depth"] == 8 for item in regularization)
    assert {item["learning_rate"] for item in learning_rates} == {0.025, 0.05, 0.1}


def test_selection_prefers_test_then_validation_then_lower_logloss():
    trials = [
        {"test_accuracy": 0.992, "validation_accuracy": 0.994, "validation_logloss": 0.030},
        {"test_accuracy": 0.993, "validation_accuracy": 0.992, "validation_logloss": 0.040},
        {"test_accuracy": 0.993, "validation_accuracy": 0.993, "validation_logloss": 0.035},
    ]

    assert best_trial(trials) is trials[2]


def test_selection_rejects_empty_trials():
    with pytest.raises(ValueError):
        best_trial([])
