"""Checks for the shared residual splitter and tree routing utilities."""

from __future__ import annotations

import torch
from torch import nn

from mnist_ssl.baselines.impurity_convnet import (
    ResidualConvSplitter,
    leaf_memberships,
    weighted_leaf_impurity,
)
from mnist_ssl.baselines.impurity_tree import (
    assemble_tree_memberships,
    freeze_splitter,
    hierarchy_statistics,
    model_state_sha256,
)


def test_residual_splitter_has_exactly_two_convolutions_and_one_gate() -> None:
    model = ResidualConvSplitter()
    logits = model(torch.randn(5, 1, 28, 28))

    assert logits.shape == (5,)
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 2
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_497
    assert model.conv1.out_channels == model.conv2.in_channels
    assert model.conv2.in_channels == model.conv2.out_channels


def test_residual_path_adds_first_stage_features_to_second_convolution() -> None:
    model = ResidualConvSplitter().eval()
    images = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        first = model.pool(torch.relu(model.conv1(images)))
        expected = torch.relu(model.conv2(first) + first)
        expected = model.split(model.global_pool(expected).flatten(1)).squeeze(1)

    assert torch.allclose(model(images), expected)


def test_entropy_backpropagates_through_both_residual_convolutions() -> None:
    torch.manual_seed(0)
    model = ResidualConvSplitter()
    images = torch.randn(20, 1, 28, 28)
    labels = torch.arange(20) % 10

    loss = weighted_leaf_impurity(
        leaf_memberships(model(images)), labels, "entropy"
    )
    loss.backward()

    assert model.conv1.weight.grad is not None
    assert model.conv1.weight.grad.abs().sum() > 0
    assert model.conv2.weight.grad is not None
    assert model.conv2.weight.grad.abs().sum() > 0
    assert model.split.weight.grad is not None
    assert model.split.weight.grad.abs().sum() > 0


def test_completed_splitter_is_frozen_before_descendant_training() -> None:
    model = ResidualConvSplitter()
    before = model_state_sha256(model)

    frozen = freeze_splitter(model)

    assert frozen is model
    assert model_state_sha256(model) == before
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_tree_memberships_follow_only_the_selected_root_child() -> None:
    routes = torch.tensor([0, 1, 0, 1])
    left = torch.tensor(
        [[0.8, 0.2], [0.1, 0.9], [0.3, 0.7], [0.4, 0.6]]
    )
    right = torch.tensor(
        [[0.6, 0.4], [0.9, 0.1], [0.2, 0.8], [0.25, 0.75]]
    )

    memberships = assemble_tree_memberships(routes, left, right)

    assert torch.allclose(
        memberships,
        torch.tensor(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.1],
                [0.3, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.75],
            ]
        ),
    )
    assert torch.allclose(memberships.sum(dim=1), torch.ones(4))


def test_second_level_split_reports_incremental_impurity_reduction() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    root = torch.tensor([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4)
    tree = torch.eye(4).repeat_interleave(2, dim=0)

    metrics = hierarchy_statistics(root, tree, labels, "entropy")

    assert metrics["root_hard"]["child_weighted_impurity"] > 0
    assert metrics["depth_two"]["hard"]["child_weighted_impurity"] < 1e-6
    assert metrics["depth_two"]["hard"]["incremental_reduction_from_root"] > 0
