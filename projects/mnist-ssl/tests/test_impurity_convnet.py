"""Focused checks for the single neural impurity splitter."""

from __future__ import annotations

import pytest
import torch

from mnist_ssl.baselines.impurity_convnet import (
    SmallConvSplitter,
    leaf_memberships,
    split_statistics,
    weighted_leaf_impurity,
)


def test_splitter_emits_one_binary_membership_per_image() -> None:
    model = SmallConvSplitter()
    logits = model(torch.randn(7, 1, 28, 28))
    memberships = leaf_memberships(logits)

    assert logits.shape == (7,)
    assert memberships.shape == (7, 2)
    assert torch.all(memberships >= 0)
    assert torch.allclose(memberships.sum(dim=1), torch.ones(7), atol=1e-6)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_separating_label_groups_reduces_impurity(criterion: str) -> None:
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    unsplit = torch.full((4, 2), 0.5)

    assert weighted_leaf_impurity(separated, labels, criterion) < 1e-6
    assert weighted_leaf_impurity(unsplit, labels, criterion) > 0
    statistics = split_statistics(separated, labels, criterion)
    assert statistics["hard"]["impurity_reduction"] > 0


def test_impurity_backpropagates_through_the_single_splitter() -> None:
    torch.manual_seed(0)
    model = SmallConvSplitter()
    images = torch.randn(20, 1, 28, 28)
    labels = torch.arange(20) % 10

    memberships = leaf_memberships(model(images))
    loss = weighted_leaf_impurity(memberships, labels, "gini")
    loss.backward()

    assert model.features[0].weight.grad is not None
    assert model.features[0].weight.grad.abs().sum() > 0
    assert model.split.weight.grad is not None
    assert model.split.weight.grad.abs().sum() > 0
