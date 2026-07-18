"""Tests for applying manually reviewed MNIST labels during evaluation."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from mnist_ssl.evaluation_labels import (
    DEFAULT_LABEL_POLICY,
    apply_mnist_test_label_policy,
    load_label_policy,
)
from mnist_ssl.label_review import DEFAULT_CANDIDATES, load_review_config


def canonical_labels() -> torch.Tensor:
    labels = torch.zeros(10_000, dtype=torch.long)
    for candidate in load_review_config(DEFAULT_CANDIDATES)["candidates"]:
        labels[candidate["index"]] = candidate["original_label"]
    return labels


def test_policy_applies_eight_relabels_and_two_exclusions() -> None:
    applied = apply_mnist_test_label_policy(canonical_labels())
    assert applied.metadata["decision_counts"] == {
        "exclude": 2,
        "keep": 5,
        "relabel": 8,
    }
    assert applied.metadata["scored_test_examples"] == 9_998
    assert applied.labels[947].item() == 9
    assert applied.labels[1621].item() == 0
    assert applied.include_mask[2462].item() is False
    assert applied.include_mask[3520].item() is False


def test_policy_is_pinned_to_reviewed_candidate_identity() -> None:
    policy = load_label_policy(DEFAULT_LABEL_POLICY)
    assert policy["source_review"]["candidate_set_sha256"] == (
        "7c056036a2bf0508657ba63b776827ac8875d261bc9984177553724bd60c40dd"
    )
    assert [decision["index"] for decision in policy["decisions"]] == [
        candidate["index"]
        for candidate in load_review_config(DEFAULT_CANDIDATES)["candidates"]
    ]


def test_policy_rejects_incomplete_decisions(tmp_path) -> None:
    policy = copy.deepcopy(load_label_policy(DEFAULT_LABEL_POLICY))
    policy["decisions"].pop()
    policy["source_review"]["reviewed_count"] -= 1
    path = tmp_path / "incomplete-policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="must decide every reviewed candidate"):
        load_label_policy(path)


def test_policy_rejects_noncanonical_dataset_order() -> None:
    labels = canonical_labels()
    labels[947] = 8 + 1
    with pytest.raises(ValueError, match="index 947 expected original label 8"):
        apply_mnist_test_label_policy(labels)
