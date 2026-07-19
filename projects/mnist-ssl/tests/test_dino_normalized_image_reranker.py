"""Tests for the independent normalized-image pairwise reranker."""

from __future__ import annotations

import torch

from mnist_ssl.dinov2.normalized_image_reranker import (
    IndependentNormalizedReranker,
    derive_pair_metadata,
    gated_predictions,
    normalized_top2_margin,
    pairwise_ranking_loss,
)


def test_model_uses_only_normalized_28_by_28_images() -> None:
    model = IndependentNormalizedReranker()
    scores = model(torch.randn(4, 1, 28, 28))
    assert scores.shape == (4, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == 24_058


def test_pair_metadata_uses_true_class_and_hardest_wrong_class() -> None:
    logits = torch.tensor(
        [
            [4.0, 3.0, 1.0],
            [2.0, 4.0, 3.0],
            [3.0, 2.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 2, 1])
    metadata = derive_pair_metadata(logits, labels, gate_threshold=1.0)
    assert metadata["linear_top1"].tolist() == [0, 1, 2]
    assert metadata["linear_top2"].tolist() == [1, 2, 0]
    assert metadata["hardest_wrong"].tolist() == [1, 1, 2]
    assert torch.equal(
        metadata["normalized_margin"],
        normalized_top2_margin(logits),
    )


def test_pairwise_loss_prefers_true_class_over_hard_negative() -> None:
    labels = torch.tensor([0, 1])
    negatives = torch.tensor([1, 2])
    good = torch.tensor([[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]])
    bad = torch.tensor([[1.0, 4.0, 0.0], [0.0, 1.0, 4.0]])
    assert pairwise_ranking_loss(good, labels, negatives) < pairwise_ranking_loss(
        bad,
        labels,
        negatives,
    )


def test_gate_prohibits_changes_above_threshold() -> None:
    base_top1 = torch.tensor([0, 1, 2])
    base_top2 = torch.tensor([1, 2, 0])
    margin = torch.tensor([0.01, 0.20, 0.03])
    scores = torch.tensor(
        [
            [1.0, 4.0, 0.0],
            [0.0, 1.0, 4.0],
            [4.0, 0.0, 1.0],
        ]
    )
    predictions = gated_predictions(
        base_top1,
        base_top2,
        margin,
        scores,
        gate_threshold=0.0367,
    )
    assert predictions.tolist() == [1, 1, 0]
