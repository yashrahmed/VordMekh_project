"""Grow entropy leaves 0, 1, and 2 without updating the existing tree.

The frozen entropy root and its two frozen depth-two children define four hard
routes.  The near-pure digit-1 route (leaf 3) remains terminal.  Three fresh
three-convolution binary splitters are independently trained on the MNIST-train
subsets routed to leaves 0, 1, and 2.  Their only label-dependent objective is
normalized Shannon leaf entropy; there is no classification head and no
end-to-end optimization.

All three new splitters finish training before the canonical test split is
loaded.  Test follows the frozen root, the appropriate frozen child, and then
the new splitter where one exists.  No test result selects a model or setting.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from mnist_ssl.baselines.impurity_convnet import (
    ExperimentConfig,
    OriginalConvSplitter,
    pick_device,
    set_seed,
    split_statistics,
    train_splitter,
)
from mnist_ssl.baselines.impurity_tree import (
    FINAL_LEAVES,
    ROOT_LEAVES,
    assemble_tree_memberships,
    collect_dataset_memberships,
    file_sha256,
    hard_routes,
    load_frozen_root,
    load_mnist,
    make_loader,
    model_state_sha256,
    tree_statistics,
)
from mnist_ssl.paths import DATASET_DIR, OUT_DIR


CRITERION = "entropy"
EXPANDED_LEAVES = (0, 1, 2)
DEPTH_THREE_LEAVES = (
    "leaf_0/grandchild_left",
    "leaf_0/grandchild_right",
    "leaf_1/grandchild_left",
    "leaf_1/grandchild_right",
    "leaf_2/grandchild_left",
    "leaf_2/grandchild_right",
    "leaf_3/terminal",
)


def load_frozen_depth_two_child(
    path: Path,
    parent_leaf: str,
    device: torch.device,
) -> nn.Module:
    """Load and freeze one child from the completed depth-two experiment."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("criterion") != CRITERION:
        raise ValueError("depth-two child must use entropy impurity")
    if payload.get("parent_leaf") != parent_leaf:
        raise ValueError(
            f"parent leaf mismatch: expected {parent_leaf}, "
            f"found {payload.get('parent_leaf')}"
        )
    if payload.get("model_kind") != "original_three_conv_child_splitter":
        raise ValueError("unexpected depth-two child architecture")
    model = OriginalConvSplitter()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def assemble_depth_three_memberships(
    depth_two_routes: torch.Tensor,
    expanded_memberships: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Map three expanded parents and one terminal parent to seven leaves."""

    if depth_two_routes.ndim != 1:
        raise ValueError("depth_two_routes must have one entry per example")
    if not torch.all((depth_two_routes >= 0) & (depth_two_routes <= 3)):
        raise ValueError("depth-two routes must contain only 0, 1, 2, or 3")
    if set(expanded_memberships) != set(EXPANDED_LEAVES):
        raise ValueError("expanded memberships must be provided for leaves 0, 1, 2")

    n_examples = len(depth_two_routes)
    reference = expanded_memberships[0]
    for parent in EXPANDED_LEAVES:
        if tuple(expanded_memberships[parent].shape) != (n_examples, 2):
            raise ValueError(
                f"leaf {parent} memberships must have shape {(n_examples, 2)}"
            )
    output = reference.new_zeros((n_examples, 7))
    for parent in EXPANDED_LEAVES:
        selected = depth_two_routes == parent
        start = 2 * parent
        output[selected, start : start + 2] = expanded_memberships[parent][selected]
    output[depth_two_routes == 3, 6] = 1.0
    return output


@torch.no_grad()
def collect_depth_two_routes(
    root: nn.Module,
    depth_two_children: dict[str, nn.Module],
    dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return root memberships, four-leaf memberships, hard routes, and labels."""

    root_memberships, labels = collect_dataset_memberships(
        root, dataset, config, device
    )
    root_routes = hard_routes(root_memberships)
    child_memberships = {}
    for parent_name in ROOT_LEAVES:
        memberships, child_labels = collect_dataset_memberships(
            depth_two_children[parent_name], dataset, config, device
        )
        if not torch.equal(child_labels, labels):
            raise RuntimeError("label order changed during depth-two routing")
        child_memberships[parent_name] = memberships
    depth_two_memberships = assemble_tree_memberships(
        root_routes,
        child_memberships["left"],
        child_memberships["right"],
    )
    return (
        root_memberships,
        depth_two_memberships,
        depth_two_memberships.argmax(dim=1),
        labels,
    )


def depth_three_statistics(
    root_memberships: torch.Tensor,
    depth_two_memberships: torch.Tensor,
    depth_three_memberships: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Compare frozen root, frozen depth two, and the seven-leaf result."""

    root_hard = F.one_hot(hard_routes(root_memberships), num_classes=2).to(
        depth_three_memberships.dtype
    )
    root = split_statistics(root_hard, labels, CRITERION)["hard"]
    depth_two = tree_statistics(
        depth_two_memberships,
        labels,
        CRITERION,
        FINAL_LEAVES,
    )
    depth_three = tree_statistics(
        depth_three_memberships,
        labels,
        CRITERION,
        DEPTH_THREE_LEAVES,
    )
    baseline = depth_two["hard"]["child_weighted_impurity"]
    for mode in ("soft", "hard"):
        final = depth_three[mode]
        incremental = baseline - final["child_weighted_impurity"]
        final["incremental_reduction_from_depth_two"] = incremental
        final["relative_incremental_reduction_from_depth_two"] = (
            incremental / max(baseline, 1e-8)
        )
    return {
        "root_hard": root,
        "depth_two": depth_two,
        "depth_three": depth_three,
    }


def _fingerprints(
    root: nn.Module,
    depth_two_children: dict[str, nn.Module],
) -> dict[str, str]:
    return {
        "root": model_state_sha256(root),
        **{
            f"depth_two_{name}": model_state_sha256(model)
            for name, model in depth_two_children.items()
        },
    }


def _train_expanded_leaves(
    root_path: Path,
    depth_two_checkpoint_dir: Path,
    train_dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> dict:
    """Train the three new splitters without constructing or reading test."""

    root = load_frozen_root(root_path, CRITERION, device)
    depth_two_paths = {
        name: depth_two_checkpoint_dir / f"entropy_root_{name}.pt"
        for name in ROOT_LEAVES
    }
    depth_two_children = {
        name: load_frozen_depth_two_child(path, name, device)
        for name, path in depth_two_paths.items()
    }
    fingerprints_before = _fingerprints(root, depth_two_children)
    (
        root_train_memberships,
        depth_two_train_memberships,
        depth_two_train_routes,
        train_labels,
    ) = collect_depth_two_routes(
        root, depth_two_children, train_dataset, config, device
    )

    expanded_models = {}
    expanded_train_memberships = {}
    records = []
    started = time.monotonic()
    for parent_leaf in EXPANDED_LEAVES:
        selected = depth_two_train_routes == parent_leaf
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        if len(indices) < 2:
            raise ValueError(f"leaf {parent_leaf} has too few training examples")
        subset = Subset(train_dataset, indices.tolist())
        set_seed(config.seed)
        model = OriginalConvSplitter().to(device)
        print(
            f"entropy leaf_{parent_leaf}: training_examples={len(subset)}",
            flush=True,
        )
        history = train_splitter(
            model,
            make_loader(subset, config, shuffle=True),
            CRITERION,
            config,
            device,
        )
        memberships, labels = collect_dataset_memberships(
            model, train_dataset, config, device
        )
        if not torch.equal(labels, train_labels):
            raise RuntimeError("train label order changed during expanded evaluation")
        expanded_models[parent_leaf] = model
        expanded_train_memberships[parent_leaf] = memberships
        records.append(
            {
                "parent_leaf": parent_leaf,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "training_examples": len(indices),
                "history": history,
                "metrics": {
                    "train": split_statistics(
                        memberships[selected], train_labels[selected], CRITERION
                    )
                },
            }
        )

    fingerprints_after = _fingerprints(root, depth_two_children)
    if fingerprints_after != fingerprints_before:
        raise RuntimeError("a frozen ancestor changed during new leaf training")
    depth_three_train_memberships = assemble_depth_three_memberships(
        depth_two_train_routes,
        expanded_train_memberships,
    )
    return {
        "root_path": root_path,
        "depth_two_paths": depth_two_paths,
        "ancestor_file_sha256": {
            "root": file_sha256(root_path),
            **{
                f"depth_two_{name}": file_sha256(path)
                for name, path in depth_two_paths.items()
            },
        },
        "fingerprints_before": fingerprints_before,
        "root": root,
        "depth_two_children": depth_two_children,
        "expanded_models": expanded_models,
        "records": records,
        "train_metrics": depth_three_statistics(
            root_train_memberships,
            depth_two_train_memberships,
            depth_three_train_memberships,
            train_labels,
        ),
        "elapsed_training_seconds": time.monotonic() - started,
    }


def _finalize(
    state: dict,
    test_dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> dict:
    (
        root_test_memberships,
        depth_two_test_memberships,
        depth_two_test_routes,
        test_labels,
    ) = collect_depth_two_routes(
        state["root"],
        state["depth_two_children"],
        test_dataset,
        config,
        device,
    )
    expanded_test_memberships = {}
    for record in state["records"]:
        parent_leaf = record["parent_leaf"]
        model = state["expanded_models"][parent_leaf]
        memberships, labels = collect_dataset_memberships(
            model, test_dataset, config, device
        )
        if not torch.equal(labels, test_labels):
            raise RuntimeError("test label order changed during expanded evaluation")
        selected = depth_two_test_routes == parent_leaf
        record["test_examples"] = int(selected.sum().item())
        record["metrics"]["canonical_test"] = split_statistics(
            memberships[selected], test_labels[selected], CRITERION
        )
        expanded_test_memberships[parent_leaf] = memberships

        checkpoint_path = output_dir / f"entropy_leaf_{parent_leaf}.pt"
        torch.save(
            {
                "criterion": CRITERION,
                "parent_leaf": parent_leaf,
                "model_kind": "original_three_conv_depth_three_splitter",
                "ancestor_file_sha256": state["ancestor_file_sha256"],
                "ancestor_state_sha256": state["fingerprints_before"],
                "config": asdict(config),
                "model_state_dict": model.state_dict(),
                "history": record.pop("history"),
                "metrics": record["metrics"],
            },
            checkpoint_path,
        )
        record["checkpoint"] = checkpoint_path.name
        record["checkpoint_sha256"] = file_sha256(checkpoint_path)

    depth_three_test_memberships = assemble_depth_three_memberships(
        depth_two_test_routes,
        expanded_test_memberships,
    )
    test_metrics = depth_three_statistics(
        root_test_memberships,
        depth_two_test_memberships,
        depth_three_test_memberships,
        test_labels,
    )
    fingerprints_final = _fingerprints(
        state["root"], state["depth_two_children"]
    )
    if fingerprints_final != state["fingerprints_before"]:
        raise RuntimeError("a frozen ancestor changed during final evaluation")

    hard = test_metrics["depth_three"]["hard"]
    print(
        "entropy depth-three hard test reduction: "
        f"total={hard['relative_impurity_reduction']:.2%} "
        f"incremental={hard['relative_incremental_reduction_from_depth_two']:.2%}",
        flush=True,
    )
    return {
        "criterion": CRITERION,
        "expanded_depth_two_leaves": EXPANDED_LEAVES,
        "terminal_depth_two_leaf": 3,
        "ancestor_checkpoint_sha256": state["ancestor_file_sha256"],
        "ancestor_state_sha256_before": state["fingerprints_before"],
        "ancestor_state_sha256_after": fingerprints_final,
        "ancestors_unchanged": True,
        "elapsed_training_seconds": state["elapsed_training_seconds"],
        "new_splitters": state["records"],
        "metrics": {
            "train": state["train_metrics"],
            "canonical_test": test_metrics,
        },
    }


def run_experiment(
    config: ExperimentConfig,
    root_checkpoint: Path,
    depth_two_checkpoint_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> dict:
    required = [
        root_checkpoint,
        *(depth_two_checkpoint_dir / f"entropy_root_{name}.pt" for name in ROOT_LEAVES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing ancestor checkpoints: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(config.device)
    print(f"device={device} output={output_dir}", flush=True)
    train_dataset = load_mnist(dataset_dir, train=True)
    trained = _train_expanded_leaves(
        root_checkpoint,
        depth_two_checkpoint_dir,
        train_dataset,
        config,
        device,
    )
    print("all new splitters trained; loading canonical test split", flush=True)
    test_dataset = load_mnist(dataset_dir, train=False)
    result = _finalize(trained, test_dataset, config, device, output_dir)
    summary = {
        "experiment": "mnist_frozen_entropy_impurity_tree_depth_three",
        "architecture": (
            "frozen root and frozen depth-two splitters; independent original "
            "three-convolution splitters added only below leaves 0, 1, and 2; "
            "depth-two leaf 3 remains terminal; seven final leaves"
        ),
        "objective": (
            "sample-weighted normalized Shannon entropy of two differentiable "
            "leaves plus the same label-free balance penalty; no classifier loss"
        ),
        "training_policy": (
            "hard-route MNIST train through all frozen ancestors; train each new "
            "splitter only on its leaf subset; no end-to-end gradients"
        ),
        "test_policy": (
            "train all three new splitters before loading canonical MNIST test; "
            "no threshold, epoch, architecture, or parameter selected on test"
        ),
        "final_leaf_names": DEPTH_THREE_LEAVES,
        "config": asdict(config),
        "device": str(device),
        "root_checkpoint": str(root_checkpoint),
        "depth_two_checkpoint_dir": str(depth_two_checkpoint_dir),
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
    parser.add_argument(
        "--root-checkpoint",
        type=Path,
        default=OUT_DIR / "neural_impurity_stump_2026-07-21" / "entropy.pt",
    )
    parser.add_argument(
        "--depth-two-checkpoint-dir",
        type=Path,
        default=OUT_DIR / "neural_impurity_tree_depth_two_2026-07-21",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR / "neural_impurity_tree_depth_three_entropy",
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
    run_experiment(
        config,
        args.root_checkpoint,
        args.depth_two_checkpoint_dir,
        args.dataset_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
