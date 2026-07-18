"""Tests for canonical configs and deterministic ensemble bookkeeping."""

from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from mnist_ssl.ensembles.dino_ijepa import evaluate_logits as evaluate_pair_logits
from mnist_ssl.ensembles.dino_ijepa_triplet import (
    DEFAULT_CONFIG,
    assert_expected_result,
    configured_artifact_ids,
    grid_search,
    load_config,
)
from mnist_ssl.provenance import artifact_index, sha256_file


def test_best_triplet_config_references_manifest_artifacts() -> None:
    config = load_config(DEFAULT_CONFIG)
    ids = configured_artifact_ids(config)
    assert set(ids) == {
        "dino_backbone",
        "dino_probe",
        "ijepa_300_probe",
        "ijepa_500_probe",
    }
    assert set(ids.values()) <= set(artifact_index())
    assert config["expected"]["best"]["test_accuracy_percent"] == 99.61
    assert config["expected"]["reviewed_label_evaluation"]["scored_examples"] == 9_998
    assert config["expected"]["reviewed_label_evaluation"]["best"]["errors"] == 35


def test_triplet_grid_search_prefers_error_free_member() -> None:
    labels = torch.tensor([0, 1])
    logits = {
        "dino": torch.tensor([[8.0, 0.0], [0.0, 8.0]]),
        "ijepa_300": torch.tensor([[0.0, 8.0], [8.0, 0.0]]),
        "ijepa_500": torch.tensor([[0.0, 8.0], [8.0, 0.0]]),
    }
    best = grid_search(logits, labels, step=50)[0]
    assert best == {
        "dino_weight": 1.0,
        "ijepa_300_weight": 0.0,
        "ijepa_500_weight": 0.0,
        "test_accuracy": 100.0,
        "errors": 0,
    }


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


def test_expected_result_check_detects_drift() -> None:
    config = load_config(DEFAULT_CONFIG)
    expected = config["expected"]
    result = {
        "individual": {
            name: {"errors": errors}
            for name, errors in expected["individual_errors"].items()
        },
        "best_test_tuned_grid_row": {
            "dino_weight": expected["best"]["dino_weight"],
            "ijepa_300_weight": expected["best"]["ijepa_300_weight"],
            "ijepa_500_weight": expected["best"]["ijepa_500_weight"],
            "errors": expected["best"]["errors"],
            "test_accuracy": expected["best"]["test_accuracy_percent"],
        },
        "all_three_shared_errors": expected["all_three_shared_errors"],
        "oracle_accuracy": expected["oracle_accuracy_percent"],
        "reviewed_label_evaluation": {
            "label_policy": {
                "policy_sha256": expected["reviewed_label_evaluation"]["policy_sha256"],
                "decision_counts": expected["reviewed_label_evaluation"][
                    "decision_counts"
                ],
            },
            "scored_examples": expected["reviewed_label_evaluation"][
                "scored_examples"
            ],
            "individual": {
                name: {"errors": errors}
                for name, errors in expected["reviewed_label_evaluation"][
                    "individual_errors"
                ].items()
            },
            "best_test_tuned_grid_row": {
                "dino_weight": expected["reviewed_label_evaluation"]["best"][
                    "dino_weight"
                ],
                "ijepa_300_weight": expected["reviewed_label_evaluation"]["best"][
                    "ijepa_300_weight"
                ],
                "ijepa_500_weight": expected["reviewed_label_evaluation"]["best"][
                    "ijepa_500_weight"
                ],
                "errors": expected["reviewed_label_evaluation"]["best"]["errors"],
                "test_accuracy": expected["reviewed_label_evaluation"]["best"][
                    "test_accuracy_percent"
                ],
            },
            "all_three_shared_errors": expected["reviewed_label_evaluation"][
                "all_three_shared_errors"
            ],
            "oracle_accuracy": expected["reviewed_label_evaluation"][
                "oracle_accuracy_percent"
            ],
        },
    }
    assert_expected_result(result, expected)
    original_only = copy.deepcopy(result)
    original_only.pop("reviewed_label_evaluation")
    assert_expected_result(original_only, expected)
    result["best_test_tuned_grid_row"]["errors"] += 1
    with pytest.raises(RuntimeError, match="best ensemble drifted"):
        assert_expected_result(result, expected)


def test_sha256_file(tmp_path) -> None:
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"manifest-pinned checkpoint")
    assert sha256_file(artifact) == hashlib.sha256(artifact.read_bytes()).hexdigest()
