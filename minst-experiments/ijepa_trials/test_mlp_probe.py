import argparse

import pytest
import torch
import torch.nn as nn

from ijepa_trials.mlp_probe import parse_milestones
from ijepa_trials.train_probe import TwoLayerMLP


def test_two_layer_mlp_has_exactly_two_affine_layers():
    head = TwoLayerMLP(32, 16, n_classes=10, dropout=0.1)

    assert sum(isinstance(layer, nn.Linear) for layer in head) == 2
    assert head(torch.randn(4, 32)).shape == (4, 10)


def test_parse_milestones_sorts_and_deduplicates():
    assert parse_milestones("100,50,75,50") == (50, 75, 100)


@pytest.mark.parametrize("value", ["", "0,50", "nope"])
def test_parse_milestones_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_milestones(value)
