"""grasp-based-embeddings: local 2D shape descriptors from a 'feeling hand'."""

from .geometry import ray_segment_intersection, point_in_polygon
from .shapes import Shape, square, circle, letter_a
from .hand import Hand, Finger
from .sampler import generate_dataset, save_dataset, load_dataset

__all__ = [
    "ray_segment_intersection",
    "point_in_polygon",
    "Shape",
    "square",
    "circle",
    "letter_a",
    "Hand",
    "Finger",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
]
