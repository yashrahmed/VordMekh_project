"""Shift an image in 4 directions, subtract each from the original, average.

Usage:
    uv run python shift_subtract.py <filename>      # e.g. bulb.png

The file is read from the images/ folder next to this script. We shift it by
3 pixels left, right, up, and down; subtract each shift from the original; and
average the four magnitudes. This is an isotropic edge response (edges in any
orientation light up), unlike a single-direction shift.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SHIFT = 3
IMAGES_DIR = Path(__file__).parent / "images"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python shift_subtract.py <filename>  (from images/ folder)")

    path = IMAGES_DIR / sys.argv[1]
    img = mpimg.imread(path).astype(np.float64)  # HxW or HxWxC, range [0,1] or [0,255]

    # Shift by SHIFT pixels in each of the 4 directions. The vacated edge
    # (where there is no data to pull from) is filled with zeros.
    def shift(a, dy, dx):
        out = np.zeros_like(a)
        ys_src = slice(max(0, dy), a.shape[0] + min(0, dy))
        ys_dst = slice(max(0, -dy), a.shape[0] + min(0, -dy))
        xs_src = slice(max(0, dx), a.shape[1] + min(0, dx))
        xs_dst = slice(max(0, -dx), a.shape[1] + min(0, -dx))
        out[ys_dst, xs_dst] = a[ys_src, xs_src]
        return out

    directions = [(0, SHIFT), (0, -SHIFT), (SHIFT, 0), (-SHIFT, 0)]  # L, R, U, D
    diffs = [np.abs(img - shift(img, dy, dx)) for dy, dx in directions]
    result = np.mean(diffs, axis=0)
    if result.ndim == 3:  # collapse colour channels into one edge map
        result = result.sum(axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    cmap = None if img.ndim == 3 else "gray"
    axes[0].imshow(img, cmap=cmap)
    axes[0].set_title("original")
    axes[1].imshow(result, cmap="gray")
    axes[1].set_title(f"avg |original - shift| (4 dirs, {SHIFT}px)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
