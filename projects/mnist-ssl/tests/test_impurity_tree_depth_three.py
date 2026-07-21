"""Checks for entropy-only growth beneath depth-two leaves 0, 1, and 2."""

from __future__ import annotations

from pathlib import Path

import torch

from mnist_ssl.baselines.impurity_convnet import OriginalConvSplitter
from mnist_ssl.baselines.impurity_tree import model_state_sha256
from mnist_ssl.baselines.impurity_tree_depth_three import (
    assemble_depth_three_memberships,
    depth_three_statistics,
    load_frozen_depth_two_child,
)


def test_depth_two_child_checkpoint_is_loaded_frozen(tmp_path: Path) -> None:
    source = OriginalConvSplitter()
    checkpoint = tmp_path / "entropy_root_left.pt"
    torch.save(
        {
            "criterion": "entropy",
            "parent_leaf": "left",
            "model_kind": "original_three_conv_child_splitter",
            "model_state_dict": source.state_dict(),
        },
        checkpoint,
    )

    model = load_frozen_depth_two_child(
        checkpoint, "left", torch.device("cpu")
    )

    assert model_state_sha256(model) == model_state_sha256(source)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_depth_three_memberships_expand_three_leaves_and_keep_terminal() -> None:
    routes = torch.tensor([0, 1, 2, 3])
    expanded = {
        0: torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7], [0.4, 0.6]]),
        1: torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5], [0.7, 0.3]]),
        2: torch.tensor([[0.9, 0.1], [0.3, 0.7], [0.25, 0.75], [0.6, 0.4]]),
    }

    memberships = assemble_depth_three_memberships(routes, expanded)

    assert torch.allclose(
        memberships,
        torch.tensor(
            [
                [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.25, 0.75, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert torch.allclose(memberships.sum(dim=1), torch.ones(4))


def test_depth_three_statistics_report_incremental_gain() -> None:
    labels = torch.arange(7).repeat_interleave(2)
    root_routes = torch.tensor([0] * 8 + [1] * 6)
    root = torch.nn.functional.one_hot(root_routes, num_classes=2).float()
    depth_two_routes = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    )
    depth_two = torch.nn.functional.one_hot(
        depth_two_routes, num_classes=4
    ).float()
    depth_three = torch.eye(7).repeat_interleave(2, dim=0)

    metrics = depth_three_statistics(root, depth_two, depth_three, labels)

    baseline = metrics["depth_two"]["hard"]["child_weighted_impurity"]
    final = metrics["depth_three"]["hard"]
    assert baseline > 0
    assert final["child_weighted_impurity"] < 1e-6
    assert final["incremental_reduction_from_depth_two"] > 0
