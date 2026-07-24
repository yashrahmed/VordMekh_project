import torch

from mnist_ssl.ensembles.lora_logit_pair import (
    pair_weights,
    refinement_weights,
    score_grid,
    select_representative,
)


def test_pair_weights_cover_the_complete_simplex() -> None:
    assert pair_weights(4) == [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]


def test_refinement_weights_are_unique_clipped_and_sorted() -> None:
    assert refinement_weights(
        [0.0, 0.5, 1.0],
        denominator=10,
        radius=0.1,
    ) == [(0, 10), (1, 9), (4, 6), (5, 5), (6, 4), (9, 1), (10, 0)]


def test_grid_scores_raw_logits_and_prefers_equal_weight_on_error_ties() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0]],
            [[0.0, 4.0], [0.0, 4.0]],
        ]
    )
    labels = torch.tensor([0, 1])
    rows = score_grid(
        logits,
        labels,
        pair_weights(4),
        denominator=4,
        phase="test",
    )

    selected, exact_best = select_representative(rows)

    assert selected["dino_weight"] == 0.5
    assert selected["ijepa_500_weight"] == 0.5
    assert selected["train_errors"] == 0
    assert len(exact_best) == 3
