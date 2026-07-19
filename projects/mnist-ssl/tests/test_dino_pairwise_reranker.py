"""Tests for candidate-conditioned DINO top-two reranking."""

from __future__ import annotations

import torch

from mnist_ssl.dinov2.pairwise_reranker import (
    hardest_wrong_classes,
    pairwise_ranking_loss,
    select_blend_alpha,
    top_two_predictions,
)


def test_hard_negative_never_uses_the_true_class() -> None:
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0],
            [4.0, 7.0, 6.0],
            [8.0, 2.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 1, 2])
    assert hardest_wrong_classes(logits, labels).tolist() == [1, 2, 0]


def test_pairwise_loss_rewards_true_score_above_hard_negative() -> None:
    labels = torch.tensor([0, 1])
    negatives = torch.tensor([1, 2])
    good = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    bad = torch.tensor([[1.0, 4.0, 0.0], [0.0, 1.0, 3.0]])
    assert pairwise_ranking_loss(good, labels, negatives) < pairwise_ranking_loss(
        bad, labels, negatives
    )


def test_alpha_selection_picks_only_net_helpful_switches() -> None:
    base = torch.tensor(
        [
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
        ]
    )
    labels = torch.tensor([1, 1, 0, 0])
    scorer = torch.tensor(
        [
            [0.0, 4.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.1, 0.0],
        ]
    )
    selected = select_blend_alpha(base, scorer, labels)
    predictions = top_two_predictions(base, scorer, selected["alpha"])
    assert selected["base_errors"] == 2
    assert selected["errors"] == 0
    assert selected["changed_predictions"] == 2
    assert predictions.tolist() == [1, 1, 0, 0]


def test_zero_alpha_exactly_preserves_base_top1() -> None:
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(12, 10, generator=generator)
    scorer = torch.randn(12, 10, generator=generator)
    assert torch.equal(
        top_two_predictions(base, scorer, 0.0), base.argmax(dim=1)
    )
    assert torch.equal(
        top_two_predictions(base, scorer, 0.0, normalize_base=True),
        base.argmax(dim=1),
    )


def test_alpha_selection_handles_sparse_runner_up_preferences() -> None:
    base = torch.tensor(
        [
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
        ]
    )
    labels = torch.tensor([1, 0, 0])
    scorer = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    selected = select_blend_alpha(base, scorer, labels)
    assert selected["errors"] == 0
    assert selected["changed_predictions"] == 1


def test_normalized_blend_is_invariant_to_per_sample_logit_scale() -> None:
    base = torch.tensor(
        [
            [4.0, 3.0, 0.0],
            [40.0, 30.0, 0.0],
        ]
    )
    scorer = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    predictions = top_two_predictions(
        base, scorer, alpha=1.0, normalize_base=True
    )
    assert predictions.tolist() == [1, 1]
