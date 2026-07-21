"""Checks for independently grown depth-two neural impurity trees."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mnist_ssl.baselines.impurity_convnet import OriginalConvSplitter
from mnist_ssl.baselines.impurity_tree import (
    assemble_tree_memberships,
    hierarchy_statistics,
    load_frozen_root,
    model_state_sha256,
)


def test_original_splitter_has_three_convolutions_and_one_gate() -> None:
    model = OriginalConvSplitter()
    logits = model(torch.randn(5, 1, 28, 28))

    assert logits.shape == (5,)
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 3
    assert sum(parameter.numel() for parameter in model.parameters()) == 23_361


def test_root_checkpoint_is_loaded_frozen(tmp_path: Path) -> None:
    source = OriginalConvSplitter()
    checkpoint = tmp_path / "gini.pt"
    torch.save(
        {
            "criterion": "gini",
            "model_kind": "small_conv_binary_splitter",
            "model_state_dict": source.state_dict(),
        },
        checkpoint,
    )

    root = load_frozen_root(checkpoint, "gini", torch.device("cpu"))

    assert model_state_sha256(root) == model_state_sha256(source)
    assert not root.training
    assert all(not parameter.requires_grad for parameter in root.parameters())


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
    root = torch.tensor(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4
    )
    tree = torch.eye(4).repeat_interleave(2, dim=0)

    metrics = hierarchy_statistics(root, tree, labels, "entropy")

    assert metrics["root_hard"]["child_weighted_impurity"] > 0
    assert metrics["depth_two"]["hard"]["child_weighted_impurity"] < 1e-6
    assert metrics["depth_two"]["hard"]["incremental_reduction_from_root"] > 0


def test_state_fingerprint_changes_only_when_model_changes() -> None:
    model = OriginalConvSplitter()
    before = model_state_sha256(model)
    assert model_state_sha256(model) == before

    with torch.no_grad():
        model.split.bias.add_(1.0)

    assert model_state_sha256(model) != before
