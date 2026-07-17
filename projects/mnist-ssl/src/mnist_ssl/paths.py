"""Canonical filesystem locations for the MNIST SSL project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "out"
IMAGES_DIR = PROJECT_ROOT / "images"
