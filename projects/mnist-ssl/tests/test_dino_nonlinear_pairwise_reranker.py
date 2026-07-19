"""Tests for nested OOF generation used by nonlinear DINO reranking."""

from __future__ import annotations

import torch

from mnist_ssl.dinov2.nonlinear_pairwise_reranker import _index_signature


def test_index_signature_distinguishes_fold_membership() -> None:
    first = torch.tensor([0, 1, 4, 8])
    second = torch.tensor([0, 2, 3, 8])
    assert _index_signature(first) != _index_signature(second)
    assert _index_signature(first) == {
        "count": 4,
        "sum": 13,
        "squared_sum": 81,
    }
