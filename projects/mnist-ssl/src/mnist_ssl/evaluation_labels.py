"""Apply the manually reviewed MNIST test-label policy to evaluation tensors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mnist_ssl.label_review import (
    DEFAULT_CANDIDATES,
    candidate_set_sha256,
    load_review_config,
)
from mnist_ssl.paths import PROJECT_ROOT


DEFAULT_LABEL_POLICY = (
    PROJECT_ROOT / "configs" / "evaluation" / "mnist_label_corrections.json"
)
VALID_ACTIONS = {"keep", "relabel", "exclude"}


@dataclass(frozen=True)
class AppliedLabelPolicy:
    """Corrected labels, inclusion mask, and reproducibility metadata."""

    labels: torch.Tensor
    include_mask: torch.Tensor
    metadata: dict[str, Any]


def _policy_sha256(policy: dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_label_policy(
    path: Path = DEFAULT_LABEL_POLICY,
    candidates_path: Path = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Load a complete decision set and pin it to the exact reviewed candidates."""

    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise ValueError(f"unsupported label-policy schema in {path}")
    if policy.get("dataset") != {
        "name": "MNIST",
        "split": "test",
        "size": 10_000,
        "indexing": "zero-based canonical torchvision order",
    }:
        raise ValueError(f"{path} must describe the canonical MNIST test split")

    candidates = load_review_config(candidates_path)
    source = policy.get("source_review")
    if not isinstance(source, dict):
        raise ValueError(f"missing source_review in {path}")
    expected_candidate_sha = candidate_set_sha256(candidates)
    if source.get("candidate_set_name") != candidates["name"]:
        raise ValueError(f"{path} belongs to a different review candidate set")
    if source.get("candidate_set_sha256") != expected_candidate_sha:
        raise ValueError(f"{path} candidate-set identity does not match the review config")

    decisions = policy.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError(f"missing decisions in {path}")
    expected = {
        candidate["index"]: candidate["original_label"]
        for candidate in candidates["candidates"]
    }
    indexes = [decision.get("index") for decision in decisions]
    if indexes != sorted(expected):
        raise ValueError(f"{path} must decide every reviewed candidate in index order")
    if source.get("reviewed_count") != len(decisions):
        raise ValueError(f"{path} source review count does not match its decisions")

    for decision in decisions:
        index = decision["index"]
        if decision.get("original_label") != expected[index]:
            raise ValueError(f"{path} has an original-label mismatch at index {index}")
        action = decision.get("action")
        corrected = decision.get("corrected_label")
        if action not in VALID_ACTIONS:
            raise ValueError(f"{path} has an invalid action at index {index}")
        if action == "relabel":
            if (
                isinstance(corrected, bool)
                or not isinstance(corrected, int)
                or not 0 <= corrected <= 9
                or corrected == expected[index]
            ):
                raise ValueError(f"{path} has an invalid corrected label at index {index}")
        elif corrected is not None:
            raise ValueError(
                f"{path} must use a null corrected label for action {action!r}"
            )
    return policy


def apply_mnist_test_label_policy(
    labels: torch.Tensor,
    path: Path = DEFAULT_LABEL_POLICY,
    candidates_path: Path = DEFAULT_CANDIDATES,
) -> AppliedLabelPolicy:
    """Return reviewed labels and a mask that removes manually excluded examples."""

    policy = load_label_policy(path, candidates_path)
    expected_size = policy["dataset"]["size"]
    if labels.ndim != 1 or len(labels) != expected_size:
        raise ValueError(
            f"label policy expects {expected_size} one-dimensional test labels, "
            f"got shape {tuple(labels.shape)}"
        )

    corrected_labels = labels.clone()
    include_mask = torch.ones(len(labels), dtype=torch.bool, device=labels.device)
    counts = {action: 0 for action in sorted(VALID_ACTIONS)}
    for decision in policy["decisions"]:
        index = decision["index"]
        actual = int(labels[index].item())
        if actual != decision["original_label"]:
            raise ValueError(
                f"MNIST test index {index} expected original label "
                f"{decision['original_label']}, found {actual}"
            )
        action = decision["action"]
        counts[action] += 1
        if action == "relabel":
            corrected_labels[index] = decision["corrected_label"]
        elif action == "exclude":
            include_mask[index] = False

    metadata = {
        "name": policy["name"],
        "path": str(path),
        "policy_sha256": _policy_sha256(policy),
        **policy["source_review"],
        "decision_counts": counts,
        "original_test_examples": len(labels),
        "scored_test_examples": int(include_mask.sum().item()),
    }
    return AppliedLabelPolicy(corrected_labels, include_mask, metadata)
