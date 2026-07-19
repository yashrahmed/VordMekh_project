import torch

from mnist_ssl.dinov2.normalized_image_reranker_sweep import (
    exact_best_threshold,
    metrics_at_threshold,
)


def test_exact_best_threshold_balances_fixes_and_breaks() -> None:
    labels = torch.tensor([1, 1, 1, 0])
    top1 = torch.tensor([0, 0, 0, 0])
    top2 = torch.tensor([1, 1, 1, 1])
    margin = torch.tensor([0.1, 0.2, 0.3, 0.4])
    scores = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    threshold = exact_best_threshold(
        labels=labels,
        base_top1=top1,
        base_top2=top2,
        normalized_margin=margin,
        reranker_scores=scores,
    )
    metrics = metrics_at_threshold(
        threshold,
        labels=labels,
        base_top1=top1,
        base_top2=top2,
        normalized_margin=margin,
        reranker_scores=scores,
    )

    assert threshold == torch.tensor(0.3).item()
    assert metrics["fixed_errors"] == 3
    assert metrics["new_errors"] == 0
    assert metrics["net_error_reduction"] == 3
