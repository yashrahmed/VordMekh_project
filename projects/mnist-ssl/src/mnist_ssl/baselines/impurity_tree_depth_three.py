"""Train a complete entropy tree from uniform two-convolution residual nodes.

The root, both depth-two children, and three depth-three grandchildren all use
the same residual binary splitter.  Nodes are trained greedily and locally:
once a node finishes, it is frozen before any descendant is optimized.  After
the four depth-two leaves are known, the lowest-entropy training leaf remains
terminal and the other three receive one new splitter each.

The canonical test split is not constructed until all six splitters have
finished and the topology is fixed.  There is no end-to-end optimization and
no ten-class classification head.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from mnist_ssl.baselines.impurity_convnet import (
    ExperimentConfig,
    ResidualConvSplitter,
    pick_device,
    set_seed,
    split_statistics,
    train_splitter,
)
from mnist_ssl.baselines.impurity_tree import (
    DEPTH_TWO_LEAVES,
    ROOT_LEAVES,
    assemble_tree_memberships,
    collect_dataset_memberships,
    file_sha256,
    freeze_splitter,
    hard_routes,
    hierarchy_statistics,
    load_mnist,
    make_loader,
    model_state_sha256,
    tree_statistics,
)
from mnist_ssl.paths import DATASET_DIR, OUT_DIR


CRITERION = "entropy"
MODEL_KIND = "two_conv_residual_binary_splitter"
N_DEPTH_TWO_LEAVES = 4
N_FINAL_LEAVES = 7


def residual_parameter_count() -> int:
    return sum(parameter.numel() for parameter in ResidualConvSplitter().parameters())


def final_leaf_names(terminal_leaf: int) -> tuple[str, ...]:
    """Return seven stable names for three expanded and one terminal parent."""

    if terminal_leaf not in range(N_DEPTH_TWO_LEAVES):
        raise ValueError("terminal_leaf must be one of the four depth-two leaves")
    names = []
    for parent in range(N_DEPTH_TWO_LEAVES):
        if parent == terminal_leaf:
            names.append(f"leaf_{parent}/terminal")
        else:
            names.extend(
                (
                    f"leaf_{parent}/grandchild_left",
                    f"leaf_{parent}/grandchild_right",
                )
            )
    return tuple(names)


def expanded_depth_two_leaves(terminal_leaf: int) -> tuple[int, ...]:
    return tuple(
        leaf for leaf in range(N_DEPTH_TWO_LEAVES) if leaf != terminal_leaf
    )


def choose_terminal_leaf(depth_two_hard: dict) -> int:
    """Select exactly one terminal using only hard-routed training impurity."""

    impurities = depth_two_hard["leaf_impurity"]
    masses = depth_two_hard["leaf_mass_fraction"]
    if len(impurities) != N_DEPTH_TWO_LEAVES or len(masses) != N_DEPTH_TWO_LEAVES:
        raise ValueError("depth-two statistics must describe exactly four leaves")
    populated = [index for index, mass in enumerate(masses) if mass > 0]
    if not populated:
        raise ValueError("at least one depth-two leaf must be populated")
    return min(populated, key=lambda index: (impurities[index], index))


def assemble_depth_three_memberships(
    depth_two_routes: torch.Tensor,
    expanded_memberships: dict[int, torch.Tensor],
    terminal_leaf: int,
) -> torch.Tensor:
    """Map four hard parent routes into an irregular seven-leaf tree."""

    if depth_two_routes.ndim != 1:
        raise ValueError("depth_two_routes must have one entry per example")
    if not torch.all(
        (depth_two_routes >= 0) & (depth_two_routes < N_DEPTH_TWO_LEAVES)
    ):
        raise ValueError("depth-two routes must contain only 0, 1, 2, or 3")
    expected_parents = set(expanded_depth_two_leaves(terminal_leaf))
    if set(expanded_memberships) != expected_parents:
        raise ValueError("memberships must exist for every non-terminal parent")

    n_examples = len(depth_two_routes)
    reference = expanded_memberships[next(iter(sorted(expected_parents)))]
    output = reference.new_zeros((n_examples, N_FINAL_LEAVES))
    column = 0
    for parent in range(N_DEPTH_TWO_LEAVES):
        selected = depth_two_routes == parent
        if parent == terminal_leaf:
            output[selected, column] = 1.0
            column += 1
            continue
        memberships = expanded_memberships[parent]
        if tuple(memberships.shape) != (n_examples, 2):
            raise ValueError(
                f"leaf {parent} memberships must have shape {(n_examples, 2)}"
            )
        output[selected, column : column + 2] = memberships[selected]
        column += 2
    if column != N_FINAL_LEAVES:
        raise RuntimeError("depth-three layout did not produce seven leaves")
    return output


def full_tree_statistics(
    root_memberships: torch.Tensor,
    depth_two_memberships: torch.Tensor,
    depth_three_memberships: torch.Tensor,
    labels: torch.Tensor,
    terminal_leaf: int,
) -> dict:
    """Measure root, four-leaf, and seven-leaf entropy reductions."""

    first_two_levels = hierarchy_statistics(
        root_memberships,
        depth_two_memberships,
        labels,
        CRITERION,
    )
    depth_three = tree_statistics(
        depth_three_memberships,
        labels,
        CRITERION,
        final_leaf_names(terminal_leaf),
    )
    for mode in ("soft", "hard"):
        baseline = first_two_levels["depth_two"][mode][
            "child_weighted_impurity"
        ]
        final = depth_three[mode]
        incremental = baseline - final["child_weighted_impurity"]
        final["incremental_reduction_from_depth_two"] = incremental
        final["relative_incremental_reduction_from_depth_two"] = (
            incremental / max(baseline, 1e-8)
        )
    return {
        "root_hard": first_two_levels["root_hard"],
        "depth_two": first_two_levels["depth_two"],
        "depth_three": depth_three,
    }


def _train_node(
    name: str,
    training_dataset: Dataset,
    evaluation_dataset: Dataset,
    expected_labels: torch.Tensor | None,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    set_seed(config.seed)
    model = ResidualConvSplitter().to(device)
    print(f"entropy {name}: training_examples={len(training_dataset)}", flush=True)
    history = train_splitter(
        model,
        make_loader(training_dataset, config, shuffle=True),
        CRITERION,
        config,
        device,
    )
    memberships, labels = collect_dataset_memberships(
        model,
        evaluation_dataset,
        config,
        device,
    )
    if expected_labels is not None and not torch.equal(labels, expected_labels):
        raise RuntimeError(f"label order changed while evaluating {name}")
    freeze_splitter(model)
    return model, memberships, labels, history


def _fingerprints(
    root: nn.Module,
    depth_two_children: dict[int, nn.Module],
    depth_three_children: dict[int, nn.Module] | None = None,
) -> dict[str, str]:
    values = {"root": model_state_sha256(root)}
    values.update(
        {
            f"depth_two_leaf_{leaf}": model_state_sha256(model)
            for leaf, model in sorted(depth_two_children.items())
        }
    )
    if depth_three_children is not None:
        values.update(
            {
                f"depth_three_leaf_{leaf}": model_state_sha256(model)
                for leaf, model in sorted(depth_three_children.items())
            }
        )
    return values


def _train_complete_tree(
    train_dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> dict:
    """Train all six nodes before a canonical test dataset exists."""

    started = time.monotonic()
    root, root_memberships, train_labels, root_history = _train_node(
        "root",
        train_dataset,
        train_dataset,
        None,
        config,
        device,
    )
    root_routes = hard_routes(root_memberships)
    root_state_after_training = model_state_sha256(root)

    depth_two_children = {}
    depth_two_memberships_by_parent = {}
    depth_two_records = []
    for parent_leaf, parent_name in enumerate(ROOT_LEAVES):
        selected = root_routes == parent_leaf
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        if len(indices) < 2:
            raise ValueError(f"root {parent_name} has too few training examples")
        model, memberships, _, history = _train_node(
            f"depth_two_{parent_name}",
            Subset(train_dataset, indices.tolist()),
            train_dataset,
            train_labels,
            config,
            device,
        )
        depth_two_children[parent_leaf] = model
        depth_two_memberships_by_parent[parent_leaf] = memberships
        depth_two_records.append(
            {
                "parent_leaf": parent_leaf,
                "parent_name": parent_name,
                "training_examples": len(indices),
                "history": history,
                "metrics": {
                    "train": split_statistics(
                        memberships[selected],
                        train_labels[selected],
                        CRITERION,
                    )
                },
            }
        )

    if model_state_sha256(root) != root_state_after_training:
        raise RuntimeError("frozen root changed during depth-two training")
    depth_two_memberships = assemble_tree_memberships(
        root_routes,
        depth_two_memberships_by_parent[0],
        depth_two_memberships_by_parent[1],
    )
    depth_two_routes = depth_two_memberships.argmax(dim=1)
    depth_two_metrics = hierarchy_statistics(
        root_memberships,
        depth_two_memberships,
        train_labels,
        CRITERION,
    )
    terminal_leaf = choose_terminal_leaf(depth_two_metrics["depth_two"]["hard"])
    expanded_leaves = expanded_depth_two_leaves(terminal_leaf)
    print(
        "training-only topology: "
        f"terminal_leaf={terminal_leaf} expanded_leaves={expanded_leaves}",
        flush=True,
    )

    ancestors_before_depth_three = _fingerprints(root, depth_two_children)
    depth_three_children = {}
    depth_three_memberships_by_parent = {}
    depth_three_records = []
    for parent_leaf in expanded_leaves:
        selected = depth_two_routes == parent_leaf
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        if len(indices) < 2:
            raise ValueError(f"depth-two leaf {parent_leaf} has too few examples")
        model, memberships, _, history = _train_node(
            f"depth_three_leaf_{parent_leaf}",
            Subset(train_dataset, indices.tolist()),
            train_dataset,
            train_labels,
            config,
            device,
        )
        depth_three_children[parent_leaf] = model
        depth_three_memberships_by_parent[parent_leaf] = memberships
        depth_three_records.append(
            {
                "parent_leaf": parent_leaf,
                "training_examples": len(indices),
                "history": history,
                "metrics": {
                    "train": split_statistics(
                        memberships[selected],
                        train_labels[selected],
                        CRITERION,
                    )
                },
            }
        )

    ancestors_after_depth_three = _fingerprints(root, depth_two_children)
    if ancestors_after_depth_three != ancestors_before_depth_three:
        raise RuntimeError("a frozen ancestor changed during depth-three training")
    depth_three_memberships = assemble_depth_three_memberships(
        depth_two_routes,
        depth_three_memberships_by_parent,
        terminal_leaf,
    )
    all_fingerprints_before_test = _fingerprints(
        root,
        depth_two_children,
        depth_three_children,
    )
    return {
        "root": root,
        "root_history": root_history,
        "root_metrics": {
            "train": split_statistics(root_memberships, train_labels, CRITERION)
        },
        "root_state_after_training": root_state_after_training,
        "depth_two_children": depth_two_children,
        "depth_two_records": depth_two_records,
        "depth_three_children": depth_three_children,
        "depth_three_records": depth_three_records,
        "terminal_leaf": terminal_leaf,
        "expanded_leaves": expanded_leaves,
        "ancestor_fingerprints_before_depth_three": ancestors_before_depth_three,
        "ancestor_fingerprints_after_depth_three": ancestors_after_depth_three,
        "all_fingerprints_before_test": all_fingerprints_before_test,
        "train_metrics": full_tree_statistics(
            root_memberships,
            depth_two_memberships,
            depth_three_memberships,
            train_labels,
            terminal_leaf,
        ),
        "elapsed_training_seconds": time.monotonic() - started,
    }


def _save_node(
    path: Path,
    model: nn.Module,
    level: str,
    parent_leaf: int | None,
    config: ExperimentConfig,
    history: list[dict[str, float]],
    metrics: dict,
    ancestor_state_sha256: dict[str, str],
) -> str:
    torch.save(
        {
            "criterion": CRITERION,
            "model_kind": MODEL_KIND,
            "tree_level": level,
            "parent_leaf": parent_leaf,
            "parameter_count": residual_parameter_count(),
            "config": asdict(config),
            "ancestor_state_sha256": ancestor_state_sha256,
            "model_state_dict": model.state_dict(),
            "history": history,
            "metrics": metrics,
        },
        path,
    )
    return file_sha256(path)


def _finalize(
    state: dict,
    test_dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Evaluate the fixed tree once, verify freezing, and save all six nodes."""

    root_memberships, test_labels = collect_dataset_memberships(
        state["root"], test_dataset, config, device
    )
    root_routes = hard_routes(root_memberships)
    state["root_metrics"]["canonical_test"] = split_statistics(
        root_memberships,
        test_labels,
        CRITERION,
    )

    depth_two_memberships_by_parent = {}
    for record in state["depth_two_records"]:
        parent_leaf = record["parent_leaf"]
        memberships, labels = collect_dataset_memberships(
            state["depth_two_children"][parent_leaf],
            test_dataset,
            config,
            device,
        )
        if not torch.equal(labels, test_labels):
            raise RuntimeError("test label order changed at depth two")
        selected = root_routes == parent_leaf
        record["test_examples"] = int(selected.sum().item())
        record["metrics"]["canonical_test"] = split_statistics(
            memberships[selected], test_labels[selected], CRITERION
        )
        depth_two_memberships_by_parent[parent_leaf] = memberships

    depth_two_memberships = assemble_tree_memberships(
        root_routes,
        depth_two_memberships_by_parent[0],
        depth_two_memberships_by_parent[1],
    )
    depth_two_routes = depth_two_memberships.argmax(dim=1)
    depth_three_memberships_by_parent = {}
    for record in state["depth_three_records"]:
        parent_leaf = record["parent_leaf"]
        memberships, labels = collect_dataset_memberships(
            state["depth_three_children"][parent_leaf],
            test_dataset,
            config,
            device,
        )
        if not torch.equal(labels, test_labels):
            raise RuntimeError("test label order changed at depth three")
        selected = depth_two_routes == parent_leaf
        record["test_examples"] = int(selected.sum().item())
        record["metrics"]["canonical_test"] = split_statistics(
            memberships[selected], test_labels[selected], CRITERION
        )
        depth_three_memberships_by_parent[parent_leaf] = memberships

    depth_three_memberships = assemble_depth_three_memberships(
        depth_two_routes,
        depth_three_memberships_by_parent,
        state["terminal_leaf"],
    )
    test_metrics = full_tree_statistics(
        root_memberships,
        depth_two_memberships,
        depth_three_memberships,
        test_labels,
        state["terminal_leaf"],
    )
    fingerprints_after_test = _fingerprints(
        state["root"],
        state["depth_two_children"],
        state["depth_three_children"],
    )
    if fingerprints_after_test != state["all_fingerprints_before_test"]:
        raise RuntimeError("a frozen splitter changed during canonical evaluation")

    root_path = output_dir / "entropy_root.pt"
    root_sha = _save_node(
        root_path,
        state["root"],
        "root",
        None,
        config,
        state["root_history"],
        state["root_metrics"],
        {},
    )
    checkpoint_sha256 = {"root": root_sha}
    root_ancestor = {"root": state["root_state_after_training"]}
    for record in state["depth_two_records"]:
        parent_leaf = record["parent_leaf"]
        path = output_dir / f"entropy_depth_two_leaf_{parent_leaf}.pt"
        sha = _save_node(
            path,
            state["depth_two_children"][parent_leaf],
            "depth_two",
            parent_leaf,
            config,
            record.pop("history"),
            record["metrics"],
            root_ancestor,
        )
        record["checkpoint"] = path.name
        record["checkpoint_sha256"] = sha
        checkpoint_sha256[f"depth_two_leaf_{parent_leaf}"] = sha

    depth_two_ancestors = state["ancestor_fingerprints_before_depth_three"]
    for record in state["depth_three_records"]:
        parent_leaf = record["parent_leaf"]
        path = output_dir / f"entropy_depth_three_leaf_{parent_leaf}.pt"
        sha = _save_node(
            path,
            state["depth_three_children"][parent_leaf],
            "depth_three",
            parent_leaf,
            config,
            record.pop("history"),
            record["metrics"],
            depth_two_ancestors,
        )
        record["checkpoint"] = path.name
        record["checkpoint_sha256"] = sha
        checkpoint_sha256[f"depth_three_leaf_{parent_leaf}"] = sha

    hard = test_metrics["depth_three"]["hard"]
    print(
        "residual entropy tree hard test reduction: "
        f"total={hard['relative_impurity_reduction']:.2%} "
        f"incremental={hard['relative_incremental_reduction_from_depth_two']:.2%}",
        flush=True,
    )
    return {
        "criterion": CRITERION,
        "model_kind": MODEL_KIND,
        "parameter_count_each": residual_parameter_count(),
        "splitter_count": 6,
        "total_trainable_parameters_across_independent_nodes": (
            6 * residual_parameter_count()
        ),
        "terminal_depth_two_leaf": state["terminal_leaf"],
        "expanded_depth_two_leaves": state["expanded_leaves"],
        "terminal_selection": (
            "lowest hard-routed normalized Shannon leaf impurity on MNIST train; "
            "ties resolved by lowest leaf index"
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "root_checkpoint": root_path.name,
        "root_checkpoint_sha256": root_sha,
        "root_metrics": state["root_metrics"],
        "depth_two_nodes": state["depth_two_records"],
        "depth_three_nodes": state["depth_three_records"],
        "ancestor_fingerprints_before_depth_three": state[
            "ancestor_fingerprints_before_depth_three"
        ],
        "ancestor_fingerprints_after_depth_three": state[
            "ancestor_fingerprints_after_depth_three"
        ],
        "all_fingerprints_before_test": state["all_fingerprints_before_test"],
        "all_fingerprints_after_test": fingerprints_after_test,
        "all_frozen_states_unchanged": True,
        "elapsed_training_seconds": state["elapsed_training_seconds"],
        "metrics": {
            "train": state["train_metrics"],
            "canonical_test": test_metrics,
        },
    }


def run_experiment(
    config: ExperimentConfig,
    dataset_dir: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(config.device)
    print(f"device={device} output={output_dir}", flush=True)
    train_dataset = load_mnist(dataset_dir, train=True)
    trained = _train_complete_tree(train_dataset, config, device)
    print(
        "all six splitters trained and frozen; loading canonical test split",
        flush=True,
    )
    test_dataset = load_mnist(dataset_dir, train=False)
    result = _finalize(trained, test_dataset, config, device, output_dir)
    summary = {
        "experiment": "mnist_two_conv_residual_entropy_impurity_tree",
        "architecture": (
            "six independent binary splitters; each has exactly two 3x3 "
            "convolutions with 16 channels, one identity residual addition "
            "around the second convolution, adaptive average pooling, and one "
            "routing logit"
        ),
        "objective": (
            "sample-weighted normalized Shannon entropy of two differentiable "
            "leaves plus a label-free balance penalty; no classifier loss"
        ),
        "training_policy": (
            "greedy local training from scratch; freeze each node before "
            "training descendants; no end-to-end gradients"
        ),
        "topology_policy": (
            "after depth two, keep the lowest-entropy training leaf terminal "
            "and expand the other three"
        ),
        "test_policy": (
            "train and freeze all six splitters and fix topology before loading "
            "canonical MNIST test; no test selection"
        ),
        "final_leaf_names": final_leaf_names(result["terminal_depth_two_leaf"]),
        "config": asdict(config),
        "device": str(device),
        "dataset_dir": str(dataset_dir),
        "result": result,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary -> {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR / "neural_impurity_tree_residual_entropy",
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        balance_weight=args.balance_weight,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    run_experiment(config, args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
