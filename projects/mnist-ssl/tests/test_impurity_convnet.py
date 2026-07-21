"""Focused checks for the differentiable neural impurity tree."""

from __future__ import annotations

import pytest
import torch

from mnist_ssl.baselines.impurity_convnet import (
    NeuralImpurityTree,
    leaf_impurity,
    path_probabilities,
)


def test_path_probabilities_form_distribution() -> None:
    logits = torch.randn(7, 15)
    leaves, balance = path_probabilities(logits, depth=4)

    assert leaves.shape == (7, 16)
    assert torch.all(leaves >= 0)
    assert torch.allclose(leaves.sum(dim=1), torch.ones(7), atol=1e-6)
    assert balance.ndim == 0
    assert balance >= 0


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_pure_leaves_have_less_impurity_than_mixed_leaves(criterion: str) -> None:
    labels = torch.tensor([0, 0, 1, 1])
    pure = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    mixed = torch.full((4, 2), 0.5)

    assert leaf_impurity(pure, labels, criterion, n_classes=2) < 1e-6
    assert leaf_impurity(mixed, labels, criterion, n_classes=2) > 0.49


def test_tree_backpropagates_impurity_to_convnet() -> None:
    torch.manual_seed(0)
    model = NeuralImpurityTree(depth=2)
    images = torch.randn(12, 1, 28, 28)
    labels = torch.arange(12) % 3

    leaves, balance = model(images)
    loss = leaf_impurity(leaves, labels, "gini", n_classes=10) + balance
    loss.backward()

    assert model.trunk.layers[0].weight.grad is not None
    assert model.trunk.layers[0].weight.grad.abs().sum() > 0
    assert model.router.weight.grad is not None
    assert model.router.weight.grad.abs().sum() > 0
