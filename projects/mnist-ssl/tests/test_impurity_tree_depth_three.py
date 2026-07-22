"""Checks for greedy residual entropy-tree growth to seven leaves."""

from __future__ import annotations

from pathlib import Path

import torch

from mnist_ssl.baselines.impurity_convnet import ExperimentConfig
from mnist_ssl.baselines import impurity_tree_depth_three as experiment
from mnist_ssl.baselines.impurity_tree_depth_three import (
    assemble_depth_three_memberships,
    choose_terminal_leaf,
    expanded_depth_two_leaves,
    final_leaf_names,
    full_tree_statistics,
)


def test_terminal_leaf_is_selected_from_training_impurity_only() -> None:
    hard_metrics = {
        "leaf_impurity": [0.62, 0.04, 0.57, 0.71],
        "leaf_mass_fraction": [0.3, 0.1, 0.25, 0.35],
    }

    terminal = choose_terminal_leaf(hard_metrics)

    assert terminal == 1
    assert expanded_depth_two_leaves(terminal) == (0, 2, 3)
    assert final_leaf_names(terminal) == (
        "leaf_0/grandchild_left",
        "leaf_0/grandchild_right",
        "leaf_1/terminal",
        "leaf_2/grandchild_left",
        "leaf_2/grandchild_right",
        "leaf_3/grandchild_left",
        "leaf_3/grandchild_right",
    )


def test_depth_three_memberships_expand_nonterminal_parents() -> None:
    routes = torch.tensor([0, 1, 2, 3])
    expanded = {
        0: torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7], [0.4, 0.6]]),
        2: torch.tensor([[0.9, 0.1], [0.3, 0.7], [0.25, 0.75], [0.6, 0.4]]),
        3: torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5], [0.7, 0.3]]),
    }

    memberships = assemble_depth_three_memberships(routes, expanded, terminal_leaf=1)

    assert torch.allclose(
        memberships,
        torch.tensor(
            [
                [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.25, 0.75, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3],
            ]
        ),
    )
    assert torch.allclose(memberships.sum(dim=1), torch.ones(4))


def test_full_tree_statistics_report_both_incremental_stages() -> None:
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

    metrics = full_tree_statistics(
        root,
        depth_two,
        depth_three,
        labels,
        terminal_leaf=3,
    )

    assert metrics["depth_two"]["hard"]["incremental_reduction_from_root"] > 0
    assert metrics["depth_three"]["hard"]["child_weighted_impurity"] < 1e-6
    assert (
        metrics["depth_three"]["hard"][
            "incremental_reduction_from_depth_two"
        ]
        > 0
    )


def test_test_split_loads_only_after_complete_tree_training(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events = []
    train_dataset = object()
    test_dataset = object()

    def fake_load_mnist(dataset_dir: Path, train: bool):
        events.append(f"load_{'train' if train else 'test'}")
        return train_dataset if train else test_dataset

    def fake_train(dataset, config, device):
        assert dataset is train_dataset
        assert "load_test" not in events
        events.append("train_all_six")
        return {"all_nodes_frozen": True}

    def fake_finalize(state, dataset, config, device, output_dir):
        assert state["all_nodes_frozen"]
        assert dataset is test_dataset
        events.append("finalize_test")
        return {"terminal_depth_two_leaf": 3}

    monkeypatch.setattr(experiment, "load_mnist", fake_load_mnist)
    monkeypatch.setattr(experiment, "_train_complete_tree", fake_train)
    monkeypatch.setattr(experiment, "_finalize", fake_finalize)

    experiment.run_experiment(
        ExperimentConfig(device="cpu"),
        tmp_path / "dataset",
        tmp_path / "output",
    )

    assert events == ["load_train", "train_all_six", "load_test", "finalize_test"]
