"""Tests for fixed-gate normalized-image reranker evaluation."""

from __future__ import annotations

import torch

from mnist_ssl.dinov2.normalized_image_reranker_eval import policy_metrics


def test_policy_metrics_count_fixes_breaks_and_exclusions() -> None:
    labels = torch.tensor([1, 1, 2, 0, 2])
    top1 = torch.tensor([0, 1, 2, 1, 1])
    top2 = torch.tensor([1, 2, 0, 2, 2])
    predictions = torch.tensor([1, 2, 2, 2, 2])
    margin = torch.tensor([0.01, 0.02, 0.20, 0.01, 0.01])
    include = torch.tensor([True, True, True, True, False])
    metrics = policy_metrics(
        predictions,
        labels=labels,
        include_mask=include,
        base_top1=top1,
        base_top2=top2,
        normalized_margin=margin,
        gate_threshold=0.0367,
    )
    assert metrics["scored_examples"] == 4
    assert metrics["gate_eligible"] == 3
    assert metrics["fixed_errors"] == 1
    assert metrics["new_errors"] == 1
    assert metrics["wrong_to_wrong"] == 1
    assert metrics["net_error_reduction"] == 0
