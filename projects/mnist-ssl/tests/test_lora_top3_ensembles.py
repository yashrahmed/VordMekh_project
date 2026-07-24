import torch

from mnist_ssl.ensembles.lora_top3_ensembles import (
    WEIGHT_FIELDS,
    _select_pair,
    _select_triplet,
    pair_refinement_weights,
    pair_weights,
    score_pair_grid,
    score_triplet_grid,
)
from mnist_ssl.ensembles.nonlinear_probe_triplet import simplex_weights


def test_pair_refinement_is_unique_clipped_and_sorted() -> None:
    assert pair_refinement_weights(
        [0.0, 0.5, 1.0],
        denominator=10,
        radius=0.1,
    ) == [(0, 10), (1, 9), (4, 6), (5, 5), (6, 4), (9, 1), (10, 0)]


def test_pair_grid_prefers_equal_weight_on_error_ties() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]],
            [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
        ]
    )
    labels = torch.tensor([0, 1])
    rows = score_pair_grid(
        logits,
        labels,
        pair_weights(4),
        denominator=4,
        phase="test",
        pair_name="first_second",
        indices=(0, 1),
    )

    selected, exact_best = _select_pair(rows, (0, 1))

    assert selected[WEIGHT_FIELDS[0]] == 0.5
    assert selected[WEIGHT_FIELDS[1]] == 0.5
    assert selected["train_errors"] == 0
    assert len(exact_best) == 3


def test_triplet_grid_prefers_the_closest_available_equal_mixture() -> None:
    logits = torch.tensor(
        [
            [[6.0, 0.0], [0.0, 6.0], [0.0, 6.0]],
            [[0.0, 6.0], [0.0, 6.0], [0.0, 6.0]],
        ]
    )
    labels = torch.tensor([0, 1])
    rows = score_triplet_grid(
        logits,
        labels,
        simplex_weights(2),
        denominator=2,
        phase="test",
    )

    selected, exact_best = _select_triplet(rows)

    assert selected[WEIGHT_FIELDS[0]] == 0.5
    assert selected[WEIGHT_FIELDS[1]] == 0.5
    assert selected[WEIGHT_FIELDS[2]] == 0.0
    assert selected["train_errors"] == 0
    assert len(exact_best) == 3
