"""Tests for the manual MNIST label-review artifact."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from mnist_ssl.label_review import (
    DEFAULT_CANDIDATES,
    build_review_payload,
    candidate_set_sha256,
    load_review_config,
    render_review_html,
)


def fake_mnist(config: dict) -> tuple[list[np.ndarray], list[int]]:
    image = np.arange(28 * 28, dtype=np.uint16).reshape(28, 28) % 256
    images = [image] * config["dataset"]["size"]
    labels = [0] * config["dataset"]["size"]
    for candidate in config["candidates"]:
        labels[candidate["index"]] = candidate["original_label"]
    return images, labels


def test_review_config_contains_only_paper_validated_issues() -> None:
    config = load_review_config(DEFAULT_CANDIDATES)
    assert [candidate["index"] for candidate in config["candidates"]] == [
        947,
        1621,
        1901,
        2130,
        2462,
        2597,
        2654,
        3520,
        3558,
        5937,
        6651,
        6783,
        8527,
        9679,
        9729,
    ]
    reason_counts = {
        candidate_set["id"]: candidate_set["count"]
        for candidate_set in config["candidate_sets"]
    }
    assert reason_counts == {"northcutt-validated-label-issues": 15}
    assert all(
        candidate["reasons"] == ["northcutt-validated-label-issues"]
        for candidate in config["candidates"]
    )
    assert all("member_predictions" not in candidate for candidate in config["candidates"])

    audit_categories = [
        candidate["published_audit"]["category"]
        for candidate in config["candidates"]
    ]
    assert audit_categories.count("correctable") == 10
    assert audit_categories.count("neither") == 3
    assert audit_categories.count("non_agreement") == 2


def test_review_payload_verifies_labels_and_embeds_only_candidate_images() -> None:
    config = load_review_config(DEFAULT_CANDIDATES)
    images, labels = fake_mnist(config)
    payload = build_review_payload(config, images, labels)
    assert payload["candidate_set_sha256"] == candidate_set_sha256(config)
    assert len(payload["candidates"]) == 15
    assert len(payload["candidates"][0]["pixels"]) == 28 * 28
    assert payload["candidates"][0]["pixels"][:4] == [0, 1, 2, 3]


def test_review_payload_rejects_noncanonical_dataset_order() -> None:
    config = load_review_config(DEFAULT_CANDIDATES)
    images, labels = fake_mnist(config)
    labels[config["candidates"][0]["index"]] = 9
    with pytest.raises(ValueError, match="dataset order does not match"):
        build_review_payload(config, images, labels)


def test_review_config_rejects_inconsistent_candidate_set_count(tmp_path) -> None:
    config = copy.deepcopy(load_review_config(DEFAULT_CANDIDATES))
    config["candidate_sets"][0]["count"] = 14
    path = tmp_path / "bad-candidates.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="declares 14 items but contains 15"):
        load_review_config(path)


def test_rendered_reviewer_is_standalone_and_exports_pinned_decisions() -> None:
    config = load_review_config(DEFAULT_CANDIDATES)
    images, labels = fake_mnist(config)
    payload = build_review_payload(config, images, labels)
    rendered = render_review_html(payload)
    assert rendered.startswith("<!doctype html>")
    assert "__PAYLOAD__" not in rendered
    assert "__STORAGE_KEY__" not in rendered
    assert payload["candidate_set_sha256"] in rendered
    assert "mnist-label-review-decisions.json" in rendered
    assert "localStorage.setItem" in rendered
    assert "fetch(" not in rendered
