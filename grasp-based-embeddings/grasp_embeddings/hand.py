"""The 'feeling hand': a posed set of fingers that probe a shape by ray casting.

A finger is defined in the hand's local frame by an origin offset and an angle
offset. The hand has a world position and heading; placing the hand maps every
finger's origin and direction into world space, and ``sense`` extends each
finger until contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import Shape


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass
class Finger:
    """A single finger in the hand's local frame.

    ``origin`` is its base offset from the hand position; ``angle`` is its
    pointing direction relative to the hand heading (radians). Default: based at
    the hand origin, pointing along the heading (+x local).
    """
    origin: np.ndarray = None
    angle: float = 0.0

    def __post_init__(self) -> None:
        if self.origin is None:
            self.origin = np.zeros(2)
        self.origin = np.asarray(self.origin, dtype=float)


@dataclass
class Hand:
    fingers: list[Finger]
    max_length: float = 3.0
    position: np.ndarray = None
    heading: float = 0.0

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = np.zeros(2)
        self.position = np.asarray(self.position, dtype=float)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def fan(cls, n_fingers: int = 5, spread_deg: float = 80.0,
            max_length: float = 3.0, base_offset: float = 0.0) -> "Hand":
        """A fan of ``n_fingers`` rays emanating from (near) the hand origin.

        ``spread_deg`` is the total angular span; fingers are spaced evenly
        across it and centered on the heading. ``base_offset`` optionally
        spreads the finger *bases* sideways (a palm width) to add parallax.
        """
        spread = np.deg2rad(spread_deg)
        if n_fingers == 1:
            angles = np.array([0.0])
            offs = np.array([0.0])
        else:
            angles = np.linspace(-spread / 2, spread / 2, n_fingers)
            offs = np.linspace(-base_offset / 2, base_offset / 2, n_fingers)
        fingers = [Finger(origin=(0.0, o), angle=a) for a, o in zip(angles, offs)]
        return cls(fingers=fingers, max_length=max_length)

    # -- posing ------------------------------------------------------------- #
    def at(self, position, heading: float) -> "Hand":
        """Return a copy of this hand placed at a world pose (non-mutating)."""
        return Hand(fingers=self.fingers, max_length=self.max_length,
                    position=np.asarray(position, dtype=float), heading=heading)

    def world_fingers(self):
        """Yield (origin_world, direction_world) for each finger."""
        R = _rot(self.heading)
        for f in self.fingers:
            o = self.position + R @ f.origin
            d = _rot(self.heading + f.angle) @ np.array([1.0, 0.0])
            yield o, d

    # -- sensing ------------------------------------------------------------ #
    def sense(self, shape: Shape) -> np.ndarray:
        """Extend every finger and return the contact distances l1..ln."""
        return np.array([
            shape.ray_distance(o, d, self.max_length)
            for o, d in self.world_fingers()
        ])

    def contact_points(self, shape: Shape):
        """Return (origin, contact_point, length) per finger (for plotting)."""
        out = []
        for o, d in self.world_fingers():
            length = shape.ray_distance(o, d, self.max_length)
            out.append((o, o + length * d, length))
        return out
