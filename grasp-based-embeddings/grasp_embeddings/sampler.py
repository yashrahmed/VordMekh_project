"""Generate labeled grasp samples (p, h, l1..l5) over a shape."""

from __future__ import annotations

import numpy as np

from .hand import Hand
from .shapes import Shape


def generate_dataset(
    shape: Shape,
    hand: Hand,
    n: int,
    rng: np.random.Generator | None = None,
    margin: float = 1.0,
    bounds=None,
):
    """Sample ``n`` random poses and record the grasp label at each.

    Positions are sampled uniformly in the shape's bounding box expanded by
    ``margin`` (or in an explicit ``bounds`` = (min_x, min_y, max_x, max_y)).
    Headings are uniform in [0, 2*pi).

    Returns a dict of arrays:
        position : (n, 2)
        heading  : (n,)
        lengths  : (n, n_fingers)
    """
    rng = rng or np.random.default_rng()
    if bounds is None:
        min_x, min_y, max_x, max_y = shape.bounds()
        min_x -= margin; min_y -= margin
        max_x += margin; max_y += margin
    else:
        min_x, min_y, max_x, max_y = bounds

    positions = np.column_stack([
        rng.uniform(min_x, max_x, n),
        rng.uniform(min_y, max_y, n),
    ])
    headings = rng.uniform(0.0, 2 * np.pi, n)

    lengths = np.empty((n, len(hand.fingers)))
    for i in range(n):
        lengths[i] = hand.at(positions[i], headings[i]).sense(shape)

    return {"position": positions, "heading": headings, "lengths": lengths}


def save_dataset(path: str, data: dict) -> None:
    np.savez_compressed(path, **data)


def load_dataset(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
