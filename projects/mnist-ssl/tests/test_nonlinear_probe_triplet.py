from mnist_ssl.ensembles.nonlinear_probe_triplet import (
    refinement_weights,
    simplex_weights,
)


def test_simplex_weights_cover_complete_grid() -> None:
    weights = simplex_weights(100)

    assert len(weights) == 5_151
    assert len(set(weights)) == len(weights)
    assert all(sum(row) == 100 for row in weights)
    assert (100, 0, 0) in weights
    assert (0, 100, 0) in weights
    assert (0, 0, 100) in weights


def test_refinement_weights_stay_on_simplex_and_near_center() -> None:
    weights = refinement_weights(
        [(0.8, 0.1, 0.1)],
        denominator=1_000,
        radius=0.002,
    )

    assert weights
    assert all(sum(row) == 1_000 for row in weights)
    assert all(abs(row[0] - 800) <= 2 for row in weights)
    assert all(abs(row[1] - 100) <= 2 for row in weights)
    assert all(abs(row[2] - 100) <= 2 for row in weights)
