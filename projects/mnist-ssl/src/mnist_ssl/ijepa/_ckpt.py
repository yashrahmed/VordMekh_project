"""Resumable-checkpoint helpers shared by the ``mnist_ssl.ijepa`` modules.

Generic over a filename ``stem`` (e.g. ``"ijepa_mnist_scatter"`` for pretraining
or ``"ijepa_clf_probe_flatten"`` for the probe): a run of ``total`` epochs writes
a resumable partial ``<stem>_e<epoch>of<total>.partial.pt`` every
:data:`CKPT_INTERVAL` epochs, resumes from the most-trained partial, and prunes
the partials once the final ``<stem>_<total>ep.pt`` is written.

The ``ijepa_*`` stems are disjoint from ``trials``' ``mae_mnist_`` / ``clf_mnist_``
names, so both packages can share ``models/`` without ever clobbering each other.
Mirrors ``mnist_ssl.baselines.mae`` partial machinery; kept separate
because that one hard-codes the ``mae_mnist_`` prefix.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch

from mnist_ssl.paths import MODELS_DIR

CKPT_INTERVAL = 50  # save a resumable partial every N epochs

__all__ = [
    "CKPT_INTERVAL",
    "MODELS_DIR",
    "set_seed",
    "final_path",
    "partial_path",
    "find_latest_partial",
    "clear_partials",
]


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/Torch so a run is reproducible.

    Covers encoder/predictor weight init, the block-position sampling and the
    training-loader shuffle (all draw from torch's global RNG).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def final_path(stem: str, epochs: int) -> Path:
    """Final checkpoint path for a ``stem`` trained ``epochs`` epochs.

    The epoch count is in the name so runs of different length don't clobber each
    other (``..._50ep.pt`` vs ``..._300ep.pt``).
    """
    return MODELS_DIR / f"{stem}_{epochs}ep.pt"


def partial_path(stem: str, epoch: int, total: int) -> Path:
    """Path for a resumable mid-run checkpoint at ``epoch`` of a ``total``-epoch run.

    Ends in ``.partial.pt`` (never matched by the ``*ep.pt`` final-checkpoint
    glob) and carries ``of<total>`` so a partial stays bound to its exact run
    length, since the LR schedule depends on the total epoch count.
    """
    return MODELS_DIR / f"{stem}_e{epoch}of{total}.partial.pt"


def _partial_glob(stem: str, total: int) -> tuple[str, re.Pattern[str]]:
    glob = f"{stem}_e*of{total}.partial.pt"
    pat = re.compile(rf"^{re.escape(stem)}_e(\d+)of{total}\.partial\.pt$")
    return glob, pat


def find_latest_partial(stem: str, total: int) -> tuple[Path, int] | None:
    """Most-trained partial for this exact ``(stem, total)`` run, or ``None``."""
    glob, pat = _partial_glob(stem, total)
    found = []
    for p in MODELS_DIR.glob(glob):
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        return None
    epoch, path = max(found)
    return path, epoch


def clear_partials(stem: str, total: int, keep: int | None = None) -> None:
    """Delete this run's partial checkpoints, optionally sparing epoch ``keep``."""
    glob, pat = _partial_glob(stem, total)
    for p in MODELS_DIR.glob(glob):
        m = pat.match(p.name)
        if m and (keep is None or int(m.group(1)) != keep):
            p.unlink()
