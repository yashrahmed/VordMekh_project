"""Primitive 2D geometry: ray casting against polygon edges.

Everything here works on plain numpy arrays so it stays cheap to vectorize
later. A "ray" is an origin point plus a unit direction; a "segment" is a pair
of endpoints (a, b).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def _cross(v: np.ndarray, w: np.ndarray) -> float:
    """2D scalar cross product v.x*w.y - v.y*w.x."""
    return float(v[0] * w[1] - v[1] * w[0])


def ray_segment_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float | None:
    """Distance along the ray to where it first hits segment ``a-b``.

    Solves ``origin + t*direction = a + u*(b-a)`` for t >= 0 and u in [0, 1].
    ``direction`` is assumed to be (approximately) unit length, so the returned
    ``t`` is a Euclidean distance. Returns ``None`` if there is no hit.
    """
    edge = b - a
    denom = _cross(direction, edge)
    if abs(denom) < EPS:
        return None  # ray parallel to the edge
    diff = a - origin
    t = _cross(diff, edge) / denom
    u = _cross(diff, direction) / denom
    if t >= 0.0 and -EPS <= u <= 1.0 + EPS:
        return t
    return None


def point_in_polygon(point: np.ndarray, rings: list[np.ndarray]) -> bool:
    """Even-odd (ray crossing) test over a set of rings.

    ``rings`` is a list of (N, 2) vertex arrays: ring 0 is the outer boundary,
    any further rings are holes. The even-odd rule across all rings handles
    holes automatically (a point inside an odd number of rings is "inside").
    """
    x, y = float(point[0]), float(point[1])
    inside = False
    for ring in rings:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[j]
            # Does the horizontal ray at height y cross edge (j -> i)?
            if (yi > y) != (yj > y):
                x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
                if x < x_cross:
                    inside = not inside
            j = i
    return inside
