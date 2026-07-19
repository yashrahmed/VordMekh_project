"""Train a top-two DINO reranker from unique out-of-fold hard pairs.

This is not a KEEP/SWITCH classifier.  Every training example contributes one
pair: its true label and the highest-scoring wrong label from a cross-fitted
linear probe.  A small class scorer learns to rank the true member above that
hard negative.  At inference, its scores are blended with the frozen linear
probe and only the probe's top two candidates are considered.

Protocol:

* folds 0-7: train the development scorer;
* fold 8: select scorer epoch and the single global blend weight;
* fold 9: untouched internal evaluation;
* folds 0-8: refit the selected scorer;
* fold 9: calibrate the final blend weight;
* MNIST test: evaluate once with the final frozen scorer and blend weight.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import MODELS_DIR, OUT_DIR

from .nonlinear_probe import (
    SmallNonlinearProbe,
    batched_logits,
    file_sha256,
    load_feature_cache,
    load_linear_probe,
)


DEFAULT_OOF_RECORDS = (
    OUT_DIR
    / "dinov2_reranker_oof_10fold_ep50_75_seed0"
    / "dinov2_reranker_oof.pt"
)
DEFAULT_TRAIN_FEATURES = (
    OUT_DIR
    / "dinov2_reranker_oof_10fold_ep50_75_seed0"
    / "train_features.pt"
)
DEFAULT_TEST_FEATURES = (
    OUT_DIR / "dinov2_nonlinear_probe_50ep" / "test_features.pt"
)
DEFAULT_BACKBONE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt"
)
DEFAULT_LINEAR_PROBE = (
    MODELS_DIR
    / "dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "dinov2_pairwise_reranker_v1"
DEFAULT_MILESTONES = (10, 20, 30, 40, 50)


def parse_milestones(value: str) -> tuple[int, ...]:
    try:
        milestones = tuple(
            sorted({int(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "milestones must be comma-separated integers"
        ) from exc
    if not milestones or milestones[0] < 1:
        raise argparse.ArgumentTypeError("milestones must contain positive epochs")
    return milestones


def hardest_wrong_classes(
    base_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return the highest-logit class other than the ground-truth class."""

    if base_logits.ndim != 2 or labels.shape != (len(base_logits),):
        raise ValueError("logits and labels have incompatible shapes")
    masked = base_logits.clone()
    masked[torch.arange(len(labels)), labels] = -torch.inf
    negatives = masked.argmax(dim=1)
    if negatives.eq(labels).any():
        raise RuntimeError("hard-negative selection returned a true label")
    return negatives


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    negatives: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy over true-class versus hard-negative scores."""

    rows = torch.arange(len(labels), device=scores.device)
    pair_logits = torch.stack(
        (scores[rows, labels], scores[rows, negatives]), dim=1
    )
    targets = torch.zeros(len(labels), dtype=torch.long, device=scores.device)
    return nn.functional.cross_entropy(pair_logits, targets)


def top_two_predictions(
    base_logits: torch.Tensor,
    candidate_scores: torch.Tensor,
    alpha: float,
    normalize_base: bool = False,
) -> torch.Tensor:
    """Blend scores and choose strictly between the frozen probe's top two."""

    if base_logits.shape != candidate_scores.shape:
        raise ValueError("base logits and candidate scores must have equal shapes")
    decision_logits = base_logits
    if normalize_base:
        scale = base_logits.std(dim=1, keepdim=True).clamp_min(1e-6)
        decision_logits = base_logits / scale
    top_values, top_classes = decision_logits.topk(2, dim=1)
    rows = torch.arange(len(base_logits))
    top_candidate_scores = candidate_scores[rows[:, None], top_classes]
    blended = top_values + float(alpha) * top_candidate_scores
    choices = blended.argmax(dim=1)
    return top_classes[rows, choices]


def prediction_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if include_mask is None:
        include_mask = torch.ones(len(labels), dtype=torch.bool)
    errors = include_mask & predictions.ne(labels)
    scored = int(include_mask.sum().item())
    error_count = int(errors.sum().item())
    return {
        "scored_examples": scored,
        "errors": error_count,
        "accuracy": 1.0 - error_count / scored,
    }


def top_two_oracle_metrics(
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if include_mask is None:
        include_mask = torch.ones(len(labels), dtype=torch.bool)
    top2 = base_logits.topk(2, dim=1).indices
    top1_wrong = top2[:, 0].ne(labels)
    recoverable = include_mask & top1_wrong & top2[:, 1].eq(labels)
    irrecoverable = include_mask & ~top2.eq(labels[:, None]).any(dim=1)
    scored = int(include_mask.sum().item())
    return {
        "top1_errors": int((include_mask & top1_wrong).sum().item()),
        "top2_recoverable_errors": int(recoverable.sum().item()),
        "top2_irrecoverable_errors": int(irrecoverable.sum().item()),
        "top2_oracle_accuracy": 1.0 - int(irrecoverable.sum().item()) / scored,
    }


def compare_predictions(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    labels: torch.Tensor,
    include_mask: torch.Tensor | None = None,
) -> dict[str, int]:
    if include_mask is None:
        include_mask = torch.ones(len(labels), dtype=torch.bool)
    baseline_correct = baseline.eq(labels)
    candidate_correct = candidate.eq(labels)
    fixed = include_mask & ~baseline_correct & candidate_correct
    broken = include_mask & baseline_correct & ~candidate_correct
    neutral = include_mask & ~baseline_correct & ~candidate_correct
    changed = include_mask & baseline.ne(candidate)
    return {
        "changed_predictions": int(changed.sum().item()),
        "fixed_errors": int(fixed.sum().item()),
        "new_errors": int(broken.sum().item()),
        "both_wrong": int(neutral.sum().item()),
        "net_error_reduction": int(fixed.sum().item() - broken.sum().item()),
    }


def select_blend_alpha(
    base_logits: torch.Tensor,
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
    normalize_base: bool = False,
) -> dict[str, Any]:
    """Select the least-changing alpha with minimum top-two error.

    As alpha grows from zero, an example changes from top1 to top2 only if the
    candidate scorer prefers top2.  Sorting those exact transition points makes
    selection deterministic and avoids an arbitrary alpha grid.
    """

    decision_logits = base_logits
    if normalize_base:
        scale = base_logits.std(dim=1, keepdim=True).clamp_min(1e-6)
        decision_logits = base_logits / scale
    top_values, top_classes = decision_logits.topk(2, dim=1)
    rows = torch.arange(len(base_logits))
    scorer_pair = candidate_scores[rows[:, None], top_classes]
    margin = top_values[:, 0] - top_values[:, 1]
    scorer_advantage = scorer_pair[:, 1] - scorer_pair[:, 0]
    eligible = scorer_advantage > 0

    baseline = top_classes[:, 0]
    runner_up = top_classes[:, 1]
    base_errors = int(baseline.ne(labels).sum().item())
    best_errors = base_errors
    best_changes = 0
    best_alpha = 0.0
    if not eligible.any():
        return {
            "alpha": best_alpha,
            "errors": best_errors,
            "changed_predictions": best_changes,
            "base_errors": base_errors,
        }

    thresholds = margin[eligible] / scorer_advantage[eligible]
    eligible_labels = labels[eligible]
    delta = (
        runner_up[eligible].ne(eligible_labels).to(torch.int64)
        - baseline[eligible].ne(eligible_labels).to(torch.int64)
    )
    order = thresholds.argsort(stable=True)
    thresholds = thresholds[order]
    delta = delta[order]
    unique_thresholds, counts = torch.unique_consecutive(
        thresholds, return_counts=True
    )

    cumulative_delta = 0
    changed = 0
    offset = 0
    for group_index, count_tensor in enumerate(counts):
        count = int(count_tensor.item())
        cumulative_delta += int(delta[offset : offset + count].sum().item())
        changed += count
        offset += count
        errors = base_errors + cumulative_delta
        if errors < best_errors or (
            errors == best_errors and changed < best_changes
        ):
            threshold = float(unique_thresholds[group_index].item())
            if group_index + 1 < len(unique_thresholds):
                next_threshold = float(
                    unique_thresholds[group_index + 1].item()
                )
                alpha = (threshold + next_threshold) / 2.0
            else:
                alpha = threshold + max(1e-6, abs(threshold) * 1e-6)
            best_errors = errors
            best_changes = changed
            best_alpha = alpha

    selected_predictions = top_two_predictions(
        base_logits,
        candidate_scores,
        best_alpha,
        normalize_base=normalize_base,
    )
    selected_errors = int(selected_predictions.ne(labels).sum().item())
    selected_changes = int(selected_predictions.ne(baseline).sum().item())
    if selected_errors != best_errors or selected_changes != best_changes:
        raise RuntimeError("blend-alpha sweep and prediction rule disagree")
    return {
        "alpha": best_alpha,
        "errors": best_errors,
        "changed_predictions": best_changes,
        "base_errors": base_errors,
    }


@torch.no_grad()
def score_features(
    scorer: nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    return batched_logits(scorer, features, device, batch_size)


def train_development_scorer(
    features: torch.Tensor,
    labels: torch.Tensor,
    negatives: torch.Tensor,
    base_logits: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    *,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    milestones: tuple[int, ...],
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    normalize_base: bool = False,
) -> tuple[nn.Module, int, float, list[dict[str, Any]]]:
    """Train on folds 0-7 and select epoch/alpha only on fold 8."""

    torch.manual_seed(seed)
    scorer = SmallNonlinearProbe(
        features.shape[1], hidden_dim=hidden_dim, dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            features[train_indices],
            labels[train_indices],
            negatives[train_indices],
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_alpha = 0.0
    best_errors = len(validation_indices) + 1
    best_changes = len(validation_indices) + 1

    for epoch in range(1, max(milestones) + 1):
        scorer.train()
        loss_sum = 0.0
        seen = 0
        for batch_features, batch_labels, batch_negatives in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_negatives = batch_negatives.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = scorer(batch_features)
            loss = pairwise_ranking_loss(
                scores, batch_labels, batch_negatives
            )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_labels)
            seen += len(batch_labels)

        record: dict[str, Any] = {"epoch": epoch, "loss": loss_sum / seen}
        if epoch in milestones:
            validation_scores = score_features(
                scorer, features[validation_indices], device
            )
            selection = select_blend_alpha(
                base_logits[validation_indices],
                validation_scores,
                labels[validation_indices],
                normalize_base=normalize_base,
            )
            record["validation"] = selection
            print(
                f"pairwise_epoch={epoch}/{max(milestones)} "
                f"loss={record['loss']:.6f} "
                f"validation_errors={selection['errors']} "
                f"alpha={selection['alpha']:.6g} "
                f"changes={selection['changed_predictions']}",
                flush=True,
            )
            selection_key = (
                selection["errors"],
                selection["changed_predictions"],
                epoch,
            )
            best_key = (best_errors, best_changes, best_epoch)
            if best_state is None or selection_key < best_key:
                best_state = copy.deepcopy(scorer.state_dict())
                best_epoch = epoch
                best_alpha = float(selection["alpha"])
                best_errors = int(selection["errors"])
                best_changes = int(selection["changed_predictions"])
        elif epoch % 10 == 0:
            print(
                f"pairwise_epoch={epoch}/{max(milestones)} "
                f"loss={record['loss']:.6f}",
                flush=True,
            )
        history.append(record)

    if best_state is None:
        raise RuntimeError("no development milestone was evaluated")
    scorer.load_state_dict(best_state)
    return scorer, best_epoch, best_alpha, history


def train_fixed_epoch_scorer(
    features: torch.Tensor,
    labels: torch.Tensor,
    negatives: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Refit the chosen architecture on folds 0-8 for final calibration."""

    torch.manual_seed(seed)
    scorer = SmallNonlinearProbe(
        features.shape[1], hidden_dim=hidden_dim, dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            features[train_indices],
            labels[train_indices],
            negatives[train_indices],
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history = []
    for epoch in range(1, epochs + 1):
        scorer.train()
        loss_sum = 0.0
        seen = 0
        for batch_features, batch_labels, batch_negatives in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_negatives = batch_negatives.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = pairwise_ranking_loss(
                scorer(batch_features), batch_labels, batch_negatives
            )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_labels)
            seen += len(batch_labels)
        history.append({"epoch": epoch, "loss": loss_sum / seen})
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"final_pairwise_epoch={epoch}/{epochs} "
                f"loss={history[-1]['loss']:.6f}",
                flush=True,
            )
    return scorer, history


def evaluate_split(
    base_logits: torch.Tensor,
    candidate_scores: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    include_mask: torch.Tensor | None = None,
    normalize_base: bool = False,
) -> dict[str, Any]:
    baseline = base_logits.argmax(dim=1)
    reranked = top_two_predictions(
        base_logits,
        candidate_scores,
        alpha,
        normalize_base=normalize_base,
    )
    return {
        "alpha": alpha,
        "baseline": prediction_metrics(baseline, labels, include_mask),
        "reranked": prediction_metrics(reranked, labels, include_mask),
        "comparison": compare_predictions(
            baseline, reranked, labels, include_mask
        ),
        "candidate_ceiling": top_two_oracle_metrics(
            base_logits, labels, include_mask
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    development_path = args.output_dir / "development_scorer.pt"
    final_path = args.output_dir / "final_scorer.pt"
    predictions_path = args.output_dir / "test_predictions.pt"
    outputs = (summary_path, development_path, final_path, predictions_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing experiment artifacts: {existing}"
        )

    backbone_sha = file_sha256(args.backbone)
    records = torch.load(
        args.oof_records, map_location="cpu", weights_only=False
    )
    metadata = records["metadata"]
    if metadata["backbone_checkpoint_sha256"] != backbone_sha:
        raise ValueError("OOF records belong to a different DINO backbone")
    if args.probe_epoch not in records["milestones"]:
        raise ValueError(
            f"OOF records do not contain probe epoch {args.probe_epoch}"
        )
    base_oof_logits = records["milestones"][args.probe_epoch]["logits"].float()
    labels = records["target"].long()
    fold_ids = records["fold_id"].long()

    train_features, cache_labels, cache_backbone = load_feature_cache(
        args.train_features,
        checkpoint_sha256=backbone_sha,
        source_split="MNIST train (canonical order)",
        pool="cls",
    )
    if not torch.equal(labels, cache_labels):
        raise ValueError("OOF labels and feature-cache labels differ")
    negatives = hardest_wrong_classes(base_oof_logits, labels)
    development_train = torch.where(fold_ids <= 7)[0]
    development_validation = torch.where(fold_ids == 8)[0]
    internal_evaluation = torch.where(fold_ids == 9)[0]
    if not all(
        len(indices) == expected
        for indices, expected in (
            (development_train, 48_000),
            (development_validation, 6_000),
            (internal_evaluation, 6_000),
        )
    ):
        raise ValueError("expected ten equal 6,000-example folds")

    print(
        f"device={device} unique_pairwise_train={len(development_train)} "
        f"validation={len(development_validation)} "
        f"internal_evaluation={len(internal_evaluation)}",
        flush=True,
    )
    development_scorer, selected_epoch, selected_alpha, development_history = (
        train_development_scorer(
            train_features,
            labels,
            negatives,
            base_oof_logits,
            development_train,
            development_validation,
            device=device,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            milestones=args.milestones,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
    )
    fold8_scores = score_features(
        development_scorer, train_features[development_validation], device
    )
    fold9_scores = score_features(
        development_scorer, train_features[internal_evaluation], device
    )
    development_validation_result = evaluate_split(
        base_oof_logits[development_validation],
        fold8_scores,
        labels[development_validation],
        selected_alpha,
    )
    untouched_internal_result = evaluate_split(
        base_oof_logits[internal_evaluation],
        fold9_scores,
        labels[internal_evaluation],
        selected_alpha,
    )
    print(
        "untouched_fold9 "
        f"base_errors={untouched_internal_result['baseline']['errors']} "
        f"reranked_errors={untouched_internal_result['reranked']['errors']} "
        f"net={untouched_internal_result['comparison']['net_error_reduction']}",
        flush=True,
    )
    torch.save(
        {
            "head_state_dict": development_scorer.state_dict(),
            "selected_epoch": selected_epoch,
            "selected_alpha": selected_alpha,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "in_dim": train_features.shape[1],
            "n_classes": 10,
        },
        development_path,
    )

    final_train = torch.where(fold_ids <= 8)[0]
    final_scorer, final_history = train_fixed_epoch_scorer(
        train_features,
        labels,
        negatives,
        final_train,
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=selected_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    final_calibration_scores = score_features(
        final_scorer, train_features[internal_evaluation], device
    )
    final_calibration = select_blend_alpha(
        base_oof_logits[internal_evaluation],
        final_calibration_scores,
        labels[internal_evaluation],
    )
    final_alpha = float(final_calibration["alpha"])
    final_calibration_result = evaluate_split(
        base_oof_logits[internal_evaluation],
        final_calibration_scores,
        labels[internal_evaluation],
        final_alpha,
    )

    test_features, test_labels, _ = load_feature_cache(
        args.test_features,
        checkpoint_sha256=backbone_sha,
        source_split="MNIST test (canonical order)",
        pool="cls",
    )
    linear_probe = load_linear_probe(args.linear_probe, device)
    test_base_logits = batched_logits(linear_probe, test_features, device)
    test_candidate_scores = score_features(final_scorer, test_features, device)
    canonical_test = evaluate_split(
        test_base_logits, test_candidate_scores, test_labels, final_alpha
    )
    reviewed = apply_mnist_test_label_policy(test_labels)
    reviewed_test = evaluate_split(
        test_base_logits,
        test_candidate_scores,
        reviewed.labels,
        final_alpha,
        reviewed.include_mask,
    )
    print(
        "reviewed_test "
        f"base_errors={reviewed_test['baseline']['errors']} "
        f"reranked_errors={reviewed_test['reranked']['errors']} "
        f"net={reviewed_test['comparison']['net_error_reduction']}",
        flush=True,
    )

    result = {
        "protocol": {
            "method": "candidate-conditioned pairwise hard-negative reranking",
            "keep_switch_labels_used": False,
            "oversampling_used": False,
            "unique_development_examples": len(development_train),
            "development_train_folds": list(range(8)),
            "model_and_alpha_selection_fold": 8,
            "untouched_internal_evaluation_fold": 9,
            "final_train_folds": list(range(9)),
            "final_alpha_calibration_fold": 9,
            "probe_epoch": args.probe_epoch,
            "backbone_frozen": True,
            "backbone_checkpoint": str(args.backbone),
            "backbone_checkpoint_sha256": backbone_sha,
            "backbone_fingerprint": cache_backbone.get("backbone_sha256"),
            "linear_probe": str(args.linear_probe),
            "linear_probe_sha256": file_sha256(args.linear_probe),
            "oof_records": str(args.oof_records),
            "seed": args.seed,
        },
        "architecture": {
            "type": "class-score layernorm-mlp",
            "in_dim": train_features.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "n_classes": 10,
            "parameters": sum(
                parameter.numel() for parameter in final_scorer.parameters()
            ),
            "inference": "argmax of blended scores restricted to base top two",
        },
        "optimization": {
            "loss": "pairwise cross-entropy: true class vs hardest OOF wrong class",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "milestones": list(args.milestones),
        },
        "development": {
            "selected_epoch": selected_epoch,
            "selected_alpha": selected_alpha,
            "history": development_history,
            "fold8_selection_result": development_validation_result,
            "fold9_untouched_result": untouched_internal_result,
        },
        "final_refit": {
            "epochs": selected_epoch,
            "history": final_history,
            "fold9_alpha_selection": final_calibration,
            "fold9_calibrated_result": final_calibration_result,
            "alpha": final_alpha,
        },
        "canonical_test": canonical_test,
        "reviewed_test": {
            "policy": reviewed.metadata,
            **reviewed_test,
        },
    }
    torch.save(
        {
            "head_state_dict": final_scorer.state_dict(),
            "epoch": selected_epoch,
            "alpha": final_alpha,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "in_dim": train_features.shape[1],
            "n_classes": 10,
            "protocol": result["protocol"],
        },
        final_path,
    )
    torch.save(
        {
            "canonical_labels": test_labels,
            "reviewed_labels": reviewed.labels,
            "reviewed_include_mask": reviewed.include_mask,
            "base_logits": test_base_logits,
            "candidate_scores": test_candidate_scores,
            "alpha": final_alpha,
            "baseline_predictions": test_base_logits.argmax(dim=1),
            "reranked_predictions": top_two_predictions(
                test_base_logits, test_candidate_scores, final_alpha
            ),
        },
        predictions_path,
    )
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"summary={summary_path}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-records", type=Path, default=DEFAULT_OOF_RECORDS)
    parser.add_argument(
        "--train-features", type=Path, default=DEFAULT_TRAIN_FEATURES
    )
    parser.add_argument(
        "--test-features", type=Path, default=DEFAULT_TEST_FEATURES
    )
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument(
        "--linear-probe", type=Path, default=DEFAULT_LINEAR_PROBE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-epoch", type=int, default=50)
    parser.add_argument(
        "--milestones", type=parse_milestones, default=DEFAULT_MILESTONES
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
