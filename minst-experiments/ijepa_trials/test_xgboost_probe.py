import argparse

import pytest
import torch

from ijepa_trials.xgboost_probe import classifier_params, stratified_validation_indices


def test_stratified_validation_indices_are_balanced_disjoint_and_reproducible():
    labels = torch.arange(3).repeat_interleave(10)

    train_a, validation_a = stratified_validation_indices(labels, per_class=2, seed=0)
    train_b, validation_b = stratified_validation_indices(labels, per_class=2, seed=0)

    assert torch.equal(train_a, train_b)
    assert torch.equal(validation_a, validation_b)
    assert set(train_a.tolist()).isdisjoint(validation_a.tolist())
    assert torch.bincount(labels[validation_a]).tolist() == [2, 2, 2]
    assert len(train_a) + len(validation_a) == len(labels)


def test_stratified_validation_rejects_oversized_request():
    with pytest.raises(ValueError):
        stratified_validation_indices(torch.tensor([0, 0, 1, 1]), per_class=2, seed=0)


def test_classifier_defaults_are_regularized_and_early_stopped():
    args = argparse.Namespace(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2.0,
        subsample=0.5,
        colsample_bytree=0.5,
        reg_alpha=0.05,
        reg_lambda=5.0,
        max_bin=256,
        early_stopping_rounds=50,
        seed=0,
        n_jobs=8,
    )

    params = classifier_params(args)

    assert params["tree_method"] == "hist"
    assert params["max_depth"] == 8
    assert params["subsample"] == 0.5
    assert params["colsample_bytree"] == 0.5
    assert params["reg_lambda"] > 1
    assert params["early_stopping_rounds"] == 50
    assert params["n_jobs"] == 8
