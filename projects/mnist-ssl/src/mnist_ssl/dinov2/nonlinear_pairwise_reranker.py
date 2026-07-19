"""Rerank the best nonlinear DINO probe's top two candidates.

The base nonlinear probe is cross-fitted with strict outer folds before the
pairwise scorer is trained.  Fold-probe margins are normalized per sample so
the final blend rule transfers to the full-data nonlinear probe without
depending on its absolute logit scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mnist_ssl.evaluation_labels import apply_mnist_test_label_policy
from mnist_ssl.paths import MODELS_DIR, OUT_DIR

from .eval_frozen import seed_everything
from .nonlinear_probe import (
    SmallNonlinearProbe,
    batched_logits,
    file_sha256,
    load_feature_cache,
    train_probe,
)
from .pairwise_reranker import (
    DEFAULT_OOF_RECORDS,
    DEFAULT_TRAIN_FEATURES,
    evaluate_split,
    hardest_wrong_classes,
    parse_milestones,
    score_features,
    select_blend_alpha,
    top_two_predictions,
    train_development_scorer,
    train_fixed_epoch_scorer,
)


DEFAULT_BACKBONE = (
    MODELS_DIR / "dinov2_mnist_augmented_cls_150ep_epoch0075.pt"
)
DEFAULT_FULL_PROBE = (
    OUT_DIR / "dinov2_nonlinear_probe_50ep" / "nonlinear_probe.pt"
)
DEFAULT_TEST_FEATURES = (
    OUT_DIR / "dinov2_nonlinear_probe_50ep" / "test_features.pt"
)
DEFAULT_OUTPUT_DIR = OUT_DIR / "dinov2_nonlinear_pairwise_reranker_v1"
DEFAULT_RERANKER_MILESTONES = (10, 20, 30, 40, 50)


def _index_signature(indices: torch.Tensor) -> dict[str, int]:
    return {
        "count": len(indices),
        "sum": int(indices.sum().item()),
        "squared_sum": int(indices.to(torch.int64).square().sum().item()),
    }


def train_or_load_base_fold(
    output_path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    held_out_indices: torch.Tensor,
    *,
    device: torch.device,
    fold_name: str,
    backbone_sha256: str,
    epochs: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> torch.Tensor:
    signature = {
        "fold_name": fold_name,
        "backbone_sha256": backbone_sha256,
        "train_indices": _index_signature(train_indices),
        "held_out_indices": _index_signature(held_out_indices),
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    if output_path.exists():
        saved = torch.load(output_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise ValueError(f"{output_path} belongs to a different OOF run")
        print(f"base_oof={fold_name} reused={output_path.name}", flush=True)
        return saved["logits"].float()

    print(
        f"base_oof={fold_name} train={len(train_indices)} "
        f"held_out={len(held_out_indices)}",
        flush=True,
    )
    seed_everything(seed)
    head, history = train_probe(
        features[train_indices],
        labels[train_indices],
        device=device,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    logits = batched_logits(head, features[held_out_indices], device)
    torch.save(
        {
            "signature": signature,
            "held_out_indices": held_out_indices,
            "logits": logits,
            "history": history,
        },
        output_path,
    )
    return logits


def generate_nested_oof_logits(
    output_dir: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: torch.Tensor,
    *,
    device: torch.device,
    backbone_sha256: str,
    epochs: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return development and final-refit OOF logits.

    Development records for folds 0-7 are cross-fitted strictly within those
    folds.  Fold 8 is predicted from 0-7, and fold 9 from 0-8.  Final-refit
    records for folds 0-8 are cross-fitted within 0-8.
    """

    aggregate_path = output_dir / "nonlinear_nested_oof.pt"
    aggregate_signature = {
        "backbone_sha256": backbone_sha256,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "fold_counts": torch.bincount(fold_ids, minlength=10).tolist(),
        "protocol": "dev crossfit 0-7; val 8; eval 9; final crossfit 0-8",
    }
    if aggregate_path.exists():
        saved = torch.load(aggregate_path, map_location="cpu", weights_only=False)
        if saved.get("signature") != aggregate_signature:
            raise ValueError(f"{aggregate_path} belongs to a different run")
        print(f"reused_nested_oof={aggregate_path}", flush=True)
        return saved["development_logits"].float(), saved["final_logits"].float()

    output_dir.mkdir(parents=True, exist_ok=True)
    development_logits = torch.full((len(labels), 10), torch.nan)
    final_logits = torch.full((len(labels), 10), torch.nan)

    for fold in range(8):
        held_out = torch.where(fold_ids == fold)[0]
        train_indices = torch.where((fold_ids <= 7) & (fold_ids != fold))[0]
        logits = train_or_load_base_fold(
            output_dir / f"base_development_fold_{fold:02d}.pt",
            features,
            labels,
            train_indices,
            held_out,
            device=device,
            fold_name=f"development-{fold}",
            backbone_sha256=backbone_sha256,
            epochs=epochs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )
        development_logits[held_out] = logits

    fold8 = torch.where(fold_ids == 8)[0]
    train0to7 = torch.where(fold_ids <= 7)[0]
    fold8_logits = train_or_load_base_fold(
        output_dir / "base_development_fold_08.pt",
        features,
        labels,
        train0to7,
        fold8,
        device=device,
        fold_name="development-8",
        backbone_sha256=backbone_sha256,
        epochs=epochs,
        hidden_dim=hidden_dim,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    development_logits[fold8] = fold8_logits

    fold9 = torch.where(fold_ids == 9)[0]
    train0to8 = torch.where(fold_ids <= 8)[0]
    fold9_logits = train_or_load_base_fold(
        output_dir / "base_development_fold_09.pt",
        features,
        labels,
        train0to8,
        fold9,
        device=device,
        fold_name="development-9",
        backbone_sha256=backbone_sha256,
        epochs=epochs,
        hidden_dim=hidden_dim,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    development_logits[fold9] = fold9_logits

    for fold in range(8):
        held_out = torch.where(fold_ids == fold)[0]
        train_indices = torch.where((fold_ids <= 8) & (fold_ids != fold))[0]
        logits = train_or_load_base_fold(
            output_dir / f"base_final_fold_{fold:02d}.pt",
            features,
            labels,
            train_indices,
            held_out,
            device=device,
            fold_name=f"final-{fold}",
            backbone_sha256=backbone_sha256,
            epochs=epochs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )
        final_logits[held_out] = logits
    final_logits[fold8] = fold8_logits
    final_logits[fold9] = fold9_logits

    if torch.isnan(development_logits).any() or torch.isnan(final_logits).any():
        raise RuntimeError("nested nonlinear OOF logits are incomplete")
    torch.save(
        {
            "signature": aggregate_signature,
            "development_logits": development_logits,
            "final_logits": final_logits,
            "labels": labels,
            "fold_ids": fold_ids,
        },
        aggregate_path,
    )
    return development_logits, final_logits


def load_full_nonlinear_probe(
    path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("head_type") != "layernorm-mlp":
        raise ValueError("full DINO probe is not the expected nonlinear head")
    head = SmallNonlinearProbe(
        checkpoint["in_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head.eval()
    return head, checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    development_path = args.output_dir / "development_pairwise_scorer.pt"
    final_path = args.output_dir / "final_pairwise_scorer.pt"
    predictions_path = args.output_dir / "test_predictions.pt"
    finished_outputs = (summary_path, development_path, final_path, predictions_path)
    existing = [path for path in finished_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing experiment artifacts: {existing}"
        )

    backbone_sha = file_sha256(args.backbone)
    fold_records = torch.load(
        args.fold_records, map_location="cpu", weights_only=False
    )
    labels = fold_records["target"].long()
    fold_ids = fold_records["fold_id"].long()
    train_features, cache_labels, cache_backbone = load_feature_cache(
        args.train_features,
        checkpoint_sha256=backbone_sha,
        source_split="MNIST train (canonical order)",
        pool="cls",
    )
    if not torch.equal(labels, cache_labels):
        raise ValueError("fold records and feature cache labels differ")

    development_base_logits, final_base_logits = generate_nested_oof_logits(
        args.output_dir,
        train_features,
        labels,
        fold_ids,
        device=device,
        backbone_sha256=backbone_sha,
        epochs=args.base_epochs,
        hidden_dim=args.base_hidden_dim,
        dropout=args.base_dropout,
        batch_size=args.base_batch_size,
        learning_rate=args.base_learning_rate,
        weight_decay=args.base_weight_decay,
        seed=args.seed,
    )

    development_train = torch.where(fold_ids <= 7)[0]
    fold8 = torch.where(fold_ids == 8)[0]
    fold9 = torch.where(fold_ids == 9)[0]
    development_negatives = hardest_wrong_classes(
        development_base_logits, labels
    )
    print(
        f"pairwise_development unique_train={len(development_train)} "
        f"validation={len(fold8)} untouched={len(fold9)} "
        f"normalized_margin=true",
        flush=True,
    )
    development_scorer, selected_epoch, selected_alpha, development_history = (
        train_development_scorer(
            train_features,
            labels,
            development_negatives,
            development_base_logits,
            development_train,
            fold8,
            device=device,
            hidden_dim=args.reranker_hidden_dim,
            dropout=args.reranker_dropout,
            milestones=args.reranker_milestones,
            batch_size=args.reranker_batch_size,
            learning_rate=args.reranker_learning_rate,
            weight_decay=args.reranker_weight_decay,
            seed=args.seed,
            normalize_base=True,
        )
    )
    fold8_scores = score_features(
        development_scorer, train_features[fold8], device
    )
    fold9_scores = score_features(
        development_scorer, train_features[fold9], device
    )
    fold8_result = evaluate_split(
        development_base_logits[fold8],
        fold8_scores,
        labels[fold8],
        selected_alpha,
        normalize_base=True,
    )
    fold9_untouched_result = evaluate_split(
        development_base_logits[fold9],
        fold9_scores,
        labels[fold9],
        selected_alpha,
        normalize_base=True,
    )
    print(
        "nonlinear_untouched_fold9 "
        f"base_errors={fold9_untouched_result['baseline']['errors']} "
        f"reranked_errors={fold9_untouched_result['reranked']['errors']} "
        f"net={fold9_untouched_result['comparison']['net_error_reduction']}",
        flush=True,
    )
    torch.save(
        {
            "head_state_dict": development_scorer.state_dict(),
            "selected_epoch": selected_epoch,
            "selected_alpha": selected_alpha,
            "normalized_margin": True,
            "in_dim": train_features.shape[1],
            "hidden_dim": args.reranker_hidden_dim,
            "dropout": args.reranker_dropout,
        },
        development_path,
    )

    final_train = torch.where(fold_ids <= 8)[0]
    final_negatives = hardest_wrong_classes(final_base_logits, labels)
    final_scorer, final_history = train_fixed_epoch_scorer(
        train_features,
        labels,
        final_negatives,
        final_train,
        device=device,
        hidden_dim=args.reranker_hidden_dim,
        dropout=args.reranker_dropout,
        epochs=selected_epoch,
        batch_size=args.reranker_batch_size,
        learning_rate=args.reranker_learning_rate,
        weight_decay=args.reranker_weight_decay,
        seed=args.seed,
    )
    calibration_scores = score_features(
        final_scorer, train_features[fold9], device
    )
    final_alpha_selection = select_blend_alpha(
        final_base_logits[fold9],
        calibration_scores,
        labels[fold9],
        normalize_base=True,
    )
    final_alpha = float(final_alpha_selection["alpha"])
    fold9_calibrated_result = evaluate_split(
        final_base_logits[fold9],
        calibration_scores,
        labels[fold9],
        final_alpha,
        normalize_base=True,
    )

    test_features, test_labels, _ = load_feature_cache(
        args.test_features,
        checkpoint_sha256=backbone_sha,
        source_split="MNIST test (canonical order)",
        pool="cls",
    )
    full_probe, full_probe_checkpoint = load_full_nonlinear_probe(
        args.full_probe, device
    )
    if full_probe_checkpoint["in_dim"] != train_features.shape[1]:
        raise ValueError("full nonlinear probe feature dimension differs")
    test_base_logits = batched_logits(full_probe, test_features, device)
    test_candidate_scores = score_features(final_scorer, test_features, device)
    canonical_test = evaluate_split(
        test_base_logits,
        test_candidate_scores,
        test_labels,
        final_alpha,
        normalize_base=True,
    )
    reviewed = apply_mnist_test_label_policy(test_labels)
    reviewed_test = evaluate_split(
        test_base_logits,
        test_candidate_scores,
        reviewed.labels,
        final_alpha,
        reviewed.include_mask,
        normalize_base=True,
    )
    print(
        "nonlinear_reviewed_test "
        f"base_errors={reviewed_test['baseline']['errors']} "
        f"reranked_errors={reviewed_test['reranked']['errors']} "
        f"net={reviewed_test['comparison']['net_error_reduction']}",
        flush=True,
    )

    result = {
        "protocol": {
            "method": "pairwise reranking of nonlinear DINO top two",
            "base_probe": "50-epoch layernorm-64-gelu nonlinear probe",
            "base_nested_oof": True,
            "keep_switch_labels_used": False,
            "oversampling_used": False,
            "base_margin_normalization": "top-two margin / per-sample ten-logit std",
            "development_train_folds": list(range(8)),
            "model_and_alpha_selection_fold": 8,
            "untouched_internal_evaluation_fold": 9,
            "final_train_folds": list(range(9)),
            "final_alpha_calibration_fold": 9,
            "backbone_frozen": True,
            "backbone_checkpoint": str(args.backbone),
            "backbone_checkpoint_sha256": backbone_sha,
            "backbone_fingerprint": cache_backbone.get("backbone_sha256"),
            "full_nonlinear_probe": str(args.full_probe),
            "full_nonlinear_probe_sha256": file_sha256(args.full_probe),
            "seed": args.seed,
        },
        "base_probe_training": {
            "epochs": args.base_epochs,
            "hidden_dim": args.base_hidden_dim,
            "dropout": args.base_dropout,
            "learning_rate": args.base_learning_rate,
            "weight_decay": args.base_weight_decay,
            "batch_size": args.base_batch_size,
        },
        "reranker": {
            "in_dim": train_features.shape[1],
            "hidden_dim": args.reranker_hidden_dim,
            "dropout": args.reranker_dropout,
            "parameters": sum(
                parameter.numel() for parameter in final_scorer.parameters()
            ),
            "loss": "true class versus hardest nonlinear-OOF wrong class",
            "milestones": list(args.reranker_milestones),
        },
        "development": {
            "selected_epoch": selected_epoch,
            "selected_alpha": selected_alpha,
            "history": development_history,
            "fold8_selection_result": fold8_result,
            "fold9_untouched_result": fold9_untouched_result,
        },
        "final_refit": {
            "epochs": selected_epoch,
            "history": final_history,
            "fold9_alpha_selection": final_alpha_selection,
            "fold9_calibrated_result": fold9_calibrated_result,
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
            "normalized_margin": True,
            "in_dim": train_features.shape[1],
            "hidden_dim": args.reranker_hidden_dim,
            "dropout": args.reranker_dropout,
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
                test_base_logits,
                test_candidate_scores,
                final_alpha,
                normalize_base=True,
            ),
        },
        predictions_path,
    )
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"summary={summary_path}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-records", type=Path, default=DEFAULT_OOF_RECORDS)
    parser.add_argument(
        "--train-features", type=Path, default=DEFAULT_TRAIN_FEATURES
    )
    parser.add_argument(
        "--test-features", type=Path, default=DEFAULT_TEST_FEATURES
    )
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--full-probe", type=Path, default=DEFAULT_FULL_PROBE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-epochs", type=int, default=50)
    parser.add_argument("--base-hidden-dim", type=int, default=64)
    parser.add_argument("--base-dropout", type=float, default=0.1)
    parser.add_argument("--base-batch-size", type=int, default=256)
    parser.add_argument("--base-learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--reranker-milestones",
        type=parse_milestones,
        default=DEFAULT_RERANKER_MILESTONES,
    )
    parser.add_argument("--reranker-hidden-dim", type=int, default=64)
    parser.add_argument("--reranker-dropout", type=float, default=0.1)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--reranker-learning-rate", type=float, default=1e-3)
    parser.add_argument("--reranker-weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
