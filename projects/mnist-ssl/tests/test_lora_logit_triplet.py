import torch

from mnist_ssl.ensembles.lora_logit_triplet import (
    score_grid,
    select_representative,
)
from mnist_ssl.ensembles.nonlinear_probe_triplet import simplex_weights


def test_triplet_grid_uses_raw_logits() -> None:
    logits = torch.tensor(
        [
            [[6.0, 0.0], [0.0, 6.0], [0.0, 6.0]],
            [[0.0, 6.0], [0.0, 6.0], [0.0, 6.0]],
        ]
    )
    labels = torch.tensor([0, 1])

    rows = score_grid(
        logits,
        labels,
        simplex_weights(2),
        denominator=2,
        phase="test",
    )
    selected, exact_best = select_representative(rows)

    assert selected["train_errors"] == 0
    assert selected["dino_weight"] == 0.5
    assert selected["ijepa_300_weight"] == 0.5
    assert selected["ijepa_500_weight"] == 0.0
    assert len(exact_best) == 3
