"""2D shapes as polygons with optional holes, plus a few example shapes.

A ``Shape`` is a list of rings. Ring 0 is the outer boundary; any subsequent
rings are holes (e.g. the triangular counter inside the letter 'A'). Winding
order does not matter because both the inside test and the ray test use the
even-odd rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import point_in_polygon, ray_segment_intersection


@dataclass
class Shape:
    rings: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rings = [np.asarray(r, dtype=float) for r in self.rings]

    def contains(self, point) -> bool:
        return point_in_polygon(np.asarray(point, dtype=float), self.rings)

    def edges(self):
        """Yield (a, b) endpoint pairs for every edge of every ring."""
        for ring in self.rings:
            n = len(ring)
            for i in range(n):
                yield ring[i], ring[(i + 1) % n]

    def ray_distance(self, origin: np.ndarray, direction: np.ndarray,
                     max_length: float) -> float:
        """First contact distance of a ray, clamped to ``max_length``.

        If the origin is inside the shape the distance is 0 (the finger starts
        embedded). Otherwise it is the nearest edge hit within ``max_length``,
        or ``max_length`` if nothing is hit (finger fully extended into space).
        """
        if self.contains(origin):
            return 0.0
        best = max_length
        for a, b in self.edges():
            t = ray_segment_intersection(origin, direction, a, b)
            if t is not None and t < best:
                best = t
        return best

    def bounds(self):
        """(min_x, min_y, max_x, max_y) over all rings."""
        pts = np.vstack(self.rings)
        return (pts[:, 0].min(), pts[:, 1].min(),
                pts[:, 0].max(), pts[:, 1].max())


# --------------------------------------------------------------------------- #
# Example shapes
# --------------------------------------------------------------------------- #

def square(side: float = 2.0, center=(0.0, 0.0)) -> Shape:
    cx, cy = center
    h = side / 2.0
    ring = [(cx - h, cy - h), (cx + h, cy - h),
            (cx + h, cy + h), (cx - h, cy + h)]
    return Shape([ring])


def circle(radius: float = 1.0, center=(0.0, 0.0), n: int = 64) -> Shape:
    cx, cy = center
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([cx + radius * np.cos(ang),
                            cy + radius * np.sin(ang)])
    return Shape([ring])


def letter_a(height: float = 4.0, width: float = 3.0,
             stroke: float = 0.55, center=(0.0, 0.0)) -> Shape:
    """A blocky capital 'A' (outer outline + triangular counter hole).

    Roughly matches the hand-drawn 'A' in the project notes. Built as a wide
    triangular outer wedge with a triangular hole and a crossbar carved by the
    even-odd rule via the outline path.
    """
    cx, cy = center
    top = cy + height / 2.0
    bot = cy - height / 2.0
    half = width / 2.0

    # Outer outline of an 'A': go up the left leg, across the apex, down the
    # right leg, then back along the bottom. We trace it as a single ring.
    apex = np.array([cx, top])
    # leg outer/inner x at the bottom
    lo = cx - half
    ro = cx + half
    li = cx - half + stroke
    ri = cx + half - stroke

    outer = [
        (lo, bot),                 # bottom-left outer
        (apex[0] - stroke, top),   # apex left
        (apex[0] + stroke, top),   # apex right
        (ro, bot),                 # bottom-right outer
        (ri, bot),                 # bottom-right inner
        # up the inner right edge to the crossbar
        (cx + 0.18 * width, cy - 0.1 * height),
        (cx - 0.18 * width, cy - 0.1 * height),
        # down the inner left edge
        (li, bot),
    ]

    # Triangular counter (the enclosed hole above the crossbar).
    hole = [
        (cx - 0.16 * width, cy + 0.05 * height),
        (cx + 0.16 * width, cy + 0.05 * height),
        (cx, top - 1.4 * stroke),
    ]
    return Shape([outer, hole])
