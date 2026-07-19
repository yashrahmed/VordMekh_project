"""Repository-layout and provenance checks for the MNIST SSL project."""

from __future__ import annotations

import json
import importlib
import pkgutil
from pathlib import Path

import mnist_ssl
from mnist_ssl.paths import DATASET_DIR, MODELS_DIR, OUT_DIR, PROJECT_ROOT


CANONICAL_SCRIPTS = (
    "scripts/train/dinov2.py",
    "scripts/train/ijepa.py",
    "scripts/train/ijepa_probe.py",
    "scripts/train/mae.py",
    "scripts/evaluate/dinov2_frozen.py",
    "scripts/evaluate/dinov2_knn.py",
    "scripts/evaluate/ijepa_probe.py",
    "scripts/evaluate/knn.py",
    "scripts/reproduce/verify_artifacts.py",
    "scripts/reproduce/ijepa_9950.py",
    "scripts/reproduce/ijepa_members.py",
)


def test_project_paths_are_centralized() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert DATASET_DIR == PROJECT_ROOT / "dataset"
    assert MODELS_DIR == PROJECT_ROOT / "models"
    assert OUT_DIR == PROJECT_ROOT / "out"


def test_canonical_scripts_are_discoverable() -> None:
    missing = [path for path in CANONICAL_SCRIPTS if not (PROJECT_ROOT / path).is_file()]
    assert not missing


def test_checkpoint_manifest_uses_safe_relative_paths() -> None:
    manifest_path = PROJECT_ROOT / "results" / "checkpoint-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest["artifacts"]
    assert artifacts
    assert len({artifact["id"] for artifact in artifacts}) == len(artifacts)
    for artifact in artifacts:
        path = Path(artifact["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert path.parts[0] == manifest["artifact_root"]
        assert len(artifact["file_sha256"]) == 64


def test_all_package_modules_import() -> None:
    modules = pkgutil.walk_packages(
        mnist_ssl.__path__, prefix=f"{mnist_ssl.__name__}."
    )
    for module in modules:
        importlib.import_module(module.name)
