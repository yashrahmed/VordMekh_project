import torch

from mnist_ssl.ensembles.temperature_diagonal import (
    TemperatureDiagonalEnsemble,
    calibration_split,
)


def test_centering_removes_per_model_logit_offsets() -> None:
    torch.manual_seed(0)
    model = TemperatureDiagonalEnsemble()
    logits = torch.randn(7, 3, 10)
    offsets = torch.randn(7, 3, 1) * 100

    assert torch.allclose(model(logits), model(logits + offsets), atol=1e-5)


def test_diagonal_weights_are_positive_and_normalized_per_class() -> None:
    model = TemperatureDiagonalEnsemble()
    with torch.no_grad():
        model.weight_logits.normal_()

    weights = model.class_weights

    assert bool(weights.gt(0).all())
    assert torch.allclose(weights.sum(dim=0), torch.ones(10))


def test_calibration_split_is_reproducible_stratified_and_disjoint() -> None:
    labels = torch.arange(10).repeat_interleave(20)

    fit, selection = calibration_split(labels, selection_size=50, seed=7)
    fit_again, selection_again = calibration_split(
        labels,
        selection_size=50,
        seed=7,
    )

    assert torch.equal(fit, fit_again)
    assert torch.equal(selection, selection_again)
    assert len(fit) == 150
    assert len(selection) == 50
    assert not set(fit.tolist()) & set(selection.tolist())
    assert torch.equal(torch.bincount(labels[selection]), torch.full((10,), 5))
