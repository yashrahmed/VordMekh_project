"""Sanity checks for the geometry and sensing primitives.

Run with:  uv run pytest    (or: uv run python tests/test_geometry.py)
"""

import numpy as np

from grasp_embeddings.geometry import ray_segment_intersection, point_in_polygon
from grasp_embeddings.hand import Hand
from grasp_embeddings.shapes import square, circle


def test_ray_hits_segment():
    o = np.array([0.0, 0.0])
    d = np.array([1.0, 0.0])
    t = ray_segment_intersection(o, d, np.array([2.0, -1.0]), np.array([2.0, 1.0]))
    assert t is not None and abs(t - 2.0) < 1e-6


def test_ray_misses_segment():
    o = np.array([0.0, 0.0])
    d = np.array([1.0, 0.0])
    # segment is behind the ray
    t = ray_segment_intersection(o, d, np.array([-2.0, -1.0]), np.array([-2.0, 1.0]))
    assert t is None


def test_point_in_polygon_square():
    sq = square(side=2.0)
    assert point_in_polygon(np.array([0.0, 0.0]), sq.rings)
    assert not point_in_polygon(np.array([5.0, 5.0]), sq.rings)


def test_finger_inside_shape_is_zero():
    sq = square(side=2.0)
    hand = Hand.fan(n_fingers=5, spread_deg=80, max_length=10.0)
    lengths = hand.at((0.0, 0.0), 0.0).sense(sq)  # center -> all inside
    assert np.allclose(lengths, 0.0)


def test_finger_from_outside_hits_square():
    sq = square(side=2.0)
    hand = Hand.fan(n_fingers=1, spread_deg=0, max_length=10.0)
    # placed left of the square, pointing +x (heading 0): center finger -> wall at x=-1
    lengths = hand.at((-5.0, 0.0), 0.0).sense(sq)
    assert abs(lengths[0] - 4.0) < 1e-6


def test_miss_clamps_to_max_length():
    sq = square(side=2.0)
    hand = Hand.fan(n_fingers=1, spread_deg=0, max_length=3.0)
    # pointing away from the square -> no contact -> clamps to max_length
    lengths = hand.at((-5.0, 0.0), np.pi).sense(sq)
    assert abs(lengths[0] - 3.0) < 1e-6


def test_circle_radial_distance():
    c = circle(radius=1.0, n=256)
    hand = Hand.fan(n_fingers=1, spread_deg=0, max_length=10.0)
    # from (-3,0) pointing +x, contact at x=-1 -> distance 2
    lengths = hand.at((-3.0, 0.0), 0.0).sense(c)
    assert abs(lengths[0] - 2.0) < 1e-2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
