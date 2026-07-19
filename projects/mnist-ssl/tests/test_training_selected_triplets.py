import torch

from mnist_ssl.ensembles.training_selected_triplets import (
    _score_grid,
    _select_representative,
)


def test_reduced_disagreement_scoring_matches_full_scoring() -> None:
    scores = torch.tensor(
        [
            [[3.0, 1.0], [4.0, 0.0], [2.0, 1.0]],
            [[3.0, 1.0], [0.0, 4.0], [2.0, 1.0]],
            [[1.0, 3.0], [0.0, 4.0], [1.0, 2.0]],
        ]
    )
    labels = torch.tensor([0, 1, 0])
    weights = [(2, 1, 1)]

    row = _score_grid(
        scores,
        labels,
        weights,
        denominator=4,
        group="linear",
        method="logit",
        phase="coarse",
    )[0]
    full_predictions = (
        2 * scores[:, 0] + scores[:, 1] + scores[:, 2]
    ).argmax(dim=1)

    assert row["train_errors"] == int(full_predictions.ne(labels).sum())


def test_representative_prefers_closest_to_equal_best_weight() -> None:
    rows = [
        {
            "train_errors": 1,
            "dino_weight": 0.8,
            "ijepa_300_weight": 0.1,
            "ijepa_500_weight": 0.1,
        },
        {
            "train_errors": 1,
            "dino_weight": 0.34,
            "ijepa_300_weight": 0.33,
            "ijepa_500_weight": 0.33,
        },
        {
            "train_errors": 2,
            "dino_weight": 1 / 3,
            "ijepa_300_weight": 1 / 3,
            "ijepa_500_weight": 1 / 3,
        },
    ]

    selected, exact_best = _select_representative(rows)

    assert len(exact_best) == 2
    assert selected["dino_weight"] == 0.34
