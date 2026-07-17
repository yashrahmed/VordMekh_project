"""Checkpoint-manifest loading and artifact integrity verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .paths import PROJECT_ROOT


DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "checkpoint-manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    artifacts = manifest.get("artifacts", [])
    ids = [artifact["id"] for artifact in artifacts]
    if not artifacts or len(ids) != len(set(ids)):
        raise ValueError(f"invalid or duplicate artifact entries in {path}")
    return manifest


def artifact_index(path: Path = DEFAULT_MANIFEST) -> dict[str, dict]:
    manifest = load_manifest(path)
    return {artifact["id"]: artifact for artifact in manifest["artifacts"]}


def artifact_paths(
    artifact_ids: Iterable[str], path: Path = DEFAULT_MANIFEST
) -> dict[str, Path]:
    index = artifact_index(path)
    resolved = {}
    for artifact_id in artifact_ids:
        if artifact_id not in index:
            raise KeyError(f"artifact {artifact_id!r} is not present in {path}")
        artifact_path = Path(index[artifact_id]["path"])
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ValueError(f"unsafe artifact path: {artifact_path}")
        resolved[artifact_id] = PROJECT_ROOT / artifact_path
    return resolved


def verify_artifacts(
    artifact_ids: Iterable[str], path: Path = DEFAULT_MANIFEST
) -> dict[str, Path]:
    ids = tuple(artifact_ids)
    index = artifact_index(path)
    paths = artifact_paths(ids, path)
    for artifact_id, artifact_path in paths.items():
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        expected = index[artifact_id]["file_sha256"]
        actual = sha256_file(artifact_path)
        if actual != expected:
            raise RuntimeError(
                f"artifact hash mismatch for {artifact_id}: expected {expected}, got {actual}"
            )
    return paths
