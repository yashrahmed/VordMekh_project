"""Tests for canonical configs and deterministic ensemble bookkeeping."""

from __future__ import annotations

import hashlib

import torch

from mnist_ssl.ensembles.dino_ijepa import evaluate_logits as evaluate_pair_logits
from mnist_ssl.provenance import sha256_file


def test_pair_evaluator_reports_member_and_oracle_metrics() -> None:
    labels = torch.tensor([0, 1])
    dino_logits = torch.tensor([[8.0, 0.0], [0.0, 8.0]])
    ijepa_logits = torch.tensor([[0.0, 8.0], [0.0, 8.0]])
    summary, rows = evaluate_pair_logits(dino_logits, ijepa_logits, labels, step=50)
    assert summary["dino"] == {"test_accuracy": 100.0, "errors": 0}
    assert summary["ijepa"] == {"test_accuracy": 50.0, "errors": 1}
    assert summary["error_complementarity"]["shared_errors"] == 0
    assert summary["error_complementarity"]["oracle_accuracy"] == 100.0
    assert len(rows) == 6


def test_sha256_file(tmp_path) -> None:
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"manifest-pinned checkpoint")
    assert sha256_file(artifact) == hashlib.sha256(artifact.read_bytes()).hexdigest()
