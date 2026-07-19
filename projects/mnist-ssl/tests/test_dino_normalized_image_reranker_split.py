import torch

from mnist_ssl.dinov2.normalized_image_reranker_split import (
    stratified_correction_split,
)


def test_stratified_correction_split_is_balanced_disjoint_and_deterministic() -> None:
    labels = torch.arange(10).repeat_interleave(20)

    train_a, validation_a = stratified_correction_split(
        labels,
        validation_per_class=5,
        seed=7,
    )
    train_b, validation_b = stratified_correction_split(
        labels,
        validation_per_class=5,
        seed=7,
    )

    assert torch.equal(train_a, train_b)
    assert torch.equal(validation_a, validation_b)
    assert len(train_a) == 150
    assert len(validation_a) == 50
    assert not torch.isin(train_a, validation_a).any()
    assert torch.equal(
        torch.bincount(labels[validation_a], minlength=10),
        torch.full((10,), 5),
    )
    assert torch.equal(
        torch.cat((train_a, validation_a)).sort().values,
        torch.arange(len(labels)),
    )
