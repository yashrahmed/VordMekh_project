"""Unit tests for the frozen DINO nonlinear-probe experiment."""

from __future__ import annotations

import pytest
import torch

from mnist_ssl.dinov2.nonlinear_probe import (
    SmallNonlinearProbe,
    classification_metrics,
    compare_predictions,
    load_feature_cache,
    save_feature_cache,
)


def test_small_probe_shape_and_parameter_count() -> None:
    probe = SmallNonlinearProbe(128, hidden_dim=64, dropout=0.1)
    assert probe(torch.randn(7, 128)).shape == (7, 10)
    assert sum(parameter.numel() for parameter in probe.parameters()) == 9_162


def test_metrics_measure_top2_recovery_and_prediction_tradeoff() -> None:
    labels = torch.tensor([0, 1, 2, 1])
    baseline = torch.tensor(
        [
            [4.0, 2.0, 0.0],
            [3.0, 2.0, 1.0],
            [0.0, 3.0, 2.0],
            [3.0, 2.0, 1.0],
        ]
    )
    candidate = torch.tensor(
        [
            [4.0, 2.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 3.0, 2.0],
            [4.0, 2.0, 1.0],
        ]
    )
    metrics = classification_metrics(baseline, labels)
    comparison = compare_predictions(baseline, candidate, labels)
    assert metrics["errors"] == 3
    assert metrics["top2_recoverable_errors"] == 3
    assert metrics["top2_oracle_accuracy"] == 1.0
    assert comparison == {
        "changed_predictions": 1,
        "fixed_errors": 1,
        "new_errors": 0,
        "both_wrong": 2,
        "net_error_reduction": 1,
    }


def test_feature_cache_is_pinned_to_checkpoint_split_and_pool(tmp_path) -> None:
    path = tmp_path / "features.pt"
    features = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    save_feature_cache(
        path,
        features,
        labels,
        checkpoint_sha256="abc",
        source_split="train",
        pool="cls",
        backbone={"frozen": True},
    )
    loaded_features, loaded_labels, metadata = load_feature_cache(
        path,
        checkpoint_sha256="abc",
        source_split="train",
        pool="cls",
    )
    assert torch.equal(loaded_features, features)
    assert torch.equal(loaded_labels, labels)
    assert metadata == {"frozen": True}
    with pytest.raises(ValueError, match="signature mismatch"):
        load_feature_cache(
            path,
            checkpoint_sha256="different",
            source_split="train",
            pool="cls",
        )
