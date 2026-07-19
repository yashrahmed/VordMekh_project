"""Unit tests for the matched I-JEPA nonlinear-probe experiment."""

from __future__ import annotations

import argparse

import pytest
import torch

from mnist_ssl.dinov2.nonlinear_probe import SmallNonlinearProbe
from mnist_ssl.ijepa.nonlinear_probe import parse_milestones


def test_matched_ijepa_probe_parameter_count() -> None:
    probe = SmallNonlinearProbe(8192, hidden_dim=64, dropout=0.1)
    assert probe(torch.randn(2, 8192)).shape == (2, 10)
    assert sum(parameter.numel() for parameter in probe.parameters()) == 541_386


def test_milestones_are_positive_unique_and_sorted() -> None:
    assert parse_milestones("100,50,75,50") == (50, 75, 100)
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        parse_milestones("0,50")
