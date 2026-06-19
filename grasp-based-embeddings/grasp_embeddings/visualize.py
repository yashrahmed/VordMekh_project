"""Matplotlib visualization of shapes, probes, and datasets."""

from __future__ import annotations

import numpy as np

from .hand import Hand
from .shapes import Shape


def _draw_shape(ax, shape: Shape):
    from matplotlib.patches import Polygon as MplPolygon
    # Outer ring filled, holes punched as white.
    for i, ring in enumerate(shape.rings):
        color = "#e74c3c" if i == 0 else "white"
        ax.add_patch(MplPolygon(ring, closed=True, facecolor=color,
                                edgecolor="#922" if i == 0 else "#922",
                                lw=1.5, zorder=1))


def draw_probe(shape: Shape, hand: Hand, ax=None, show_misses=True):
    """Draw a shape and one posed hand's fingers + contact points."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    _draw_shape(ax, shape)

    for o, contact, length in hand.contact_points(shape):
        hit = length < hand.max_length - 1e-9
        if length == 0.0:
            # finger started inside the shape
            ax.plot(*o, "o", color="purple", zorder=4, ms=8)
            continue
        if not hit and not show_misses:
            continue
        ax.plot([o[0], contact[0]], [o[1], contact[1]],
                color="#2255cc" if hit else "#aacc", lw=2, zorder=3)
        ax.plot(*o, "o", color="black", ms=4, zorder=4)
        if hit:
            ax.plot(*contact, "o", color="#2255cc", ms=6, zorder=5)

    ax.plot(*hand.position, "*", color="green", ms=14, zorder=6)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.15)
    return ax


def draw_dataset(shape: Shape, data: dict, ax=None, finger: int = 0):
    """Scatter sampled positions colored by one finger's contact distance."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    _draw_shape(ax, shape)
    pos = data["position"]
    vals = data["lengths"][:, finger]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=vals, cmap="viridis",
                    s=8, zorder=3, alpha=0.8)
    plt.colorbar(sc, ax=ax, label=f"finger {finger} contact distance")
    ax.set_aspect("equal")
    ax.margins(0.05)
    return ax
