import torch

from mnist_ssl.ijepa.ensemble_triplets import (
    _score_grid,
    _select_representative,
)


def test_reduced_disagreement_scoring_matches_full_scoring() -> None:
    logits = torch.tensor(
        [
            [[3.0, 1.0], [4.0, 0.0], [2.0, 1.0]],
            [[3.0, 1.0], [0.0, 4.0], [2.0, 1.0]],
            [[1.0, 3.0], [0.0, 4.0], [1.0, 2.0]],
        ]
    )
    labels = torch.tensor([0, 1, 0])
    weights = [(2, 1, 1)]

    row = _score_grid(
        logits,
        labels,
        weights,
        denominator=4,
        phase="coarse",
    )[0]
    full_predictions = (
        2 * logits[:, 0] + logits[:, 1] + logits[:, 2]
    ).argmax(dim=1)

    assert row["train_errors"] == int(full_predictions.ne(labels).sum())


def test_representative_prefers_closest_to_equal_best_weight() -> None:
    rows = [
        {
            "train_errors": 1,
            "old28_500_weight": 0.8,
            "new56_300_weight": 0.1,
            "new56_500_weight": 0.1,
        },
        {
            "train_errors": 1,
            "old28_500_weight": 0.34,
            "new56_300_weight": 0.33,
            "new56_500_weight": 0.33,
        },
        {
            "train_errors": 2,
            "old28_500_weight": 1 / 3,
            "new56_300_weight": 1 / 3,
            "new56_500_weight": 1 / 3,
        },
    ]

    selected, exact_best = _select_representative(rows)

    assert len(exact_best) == 2
    assert selected["old28_500_weight"] == 0.34
