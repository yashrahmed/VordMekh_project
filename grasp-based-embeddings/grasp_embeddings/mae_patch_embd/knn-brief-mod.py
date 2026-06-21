"""k-NN classification on MNIST with *structured* BRIEF descriptors.

A variant of ``knn-brief.py`` that replaces the random comparison pairs with a
regular, hand-designed lattice. The image is divided into a ``grid x grid`` of
cells; at each cell a centre box is compared against its **four neighbours**
(up / down / left / right), giving up to **4 bits per location**. Border
locations whose outward neighbour box would leave the frame drop that feature,
so the total is ``4 * grid * (grid - 1)`` bits. This is the census-/LBP-flavoured
counterpart to the random sampling in ``knn-brief.py``: every bit is anchored to
a fixed spatial location and measures a *local* intensity gradient (centre vs
neighbour) instead of a random, possibly long-range, comparison. Both share
:mod:`brief`.

The probes **tile** the frame. Coordinates live in a normalised space of side
``grid`` (one unit per cell), so a location sits at its cell centre
``(i+0.5, j+0.5)`` and each neighbour box is offset by half a cell. With a box
side of half a cell, neighbour boxes butt up against the centre box edge-to-edge
-- the plus-shaped probe in the diagram -- and the plus of one cell meets the
plus of the next, covering the digit (only the cell corners are left bare, as a
plus shape must). On a 28px image each box is ``14 / grid`` px on a side.

    python knn-brief-mod.py                    # describe MNIST + run k-NN
    python knn-brief-mod.py --grid 8 --k 5
    python knn-brief-mod.py --viz-only --grid 4
    python knn-brief-mod.py --viz-only --grid 4 --save brief-mod.png
"""

from __future__ import annotations

import argparse

from grasp_embeddings.mae_patch_embd import brief


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=int, default=4,
        help="GxG lattice of locations; up to 4 bits each -> 4*grid*(grid-1) bits.",
    )
    parser.add_argument("--k", type=int, default=5, help="Neighbours for k-NN.")
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Only show/save the comparison pattern; skip the MNIST k-NN eval.",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save the figure (PNG)."
    )
    args = parser.parse_args()

    features = brief.generate_structured_features(args.grid)
    print(f"Generated structured BRIEF: {args.grid}x{args.grid} grid, "
          f"{len(features)} bits (<=4 per location; edge-clipped to stay in frame).")

    if args.viz_only:
        brief.visualize_structured(features, args.grid, args.save)
        return

    extent = args.grid  # coordinate space side == grid (one unit per cell)
    print("Describing MNIST train split with structured BRIEF...")
    train_desc, train_labels = brief.describe_split(features, extent, train=True)
    print("Describing MNIST test split with structured BRIEF...")
    test_desc, test_labels = brief.describe_split(features, extent, train=False)
    print(f"  train: {train_desc.shape}   test: {test_desc.shape}")

    pred = brief.knn_predict(test_desc, train_desc, train_labels, args.k)
    acc = (pred == test_labels).mean()

    print(f"\n--- {args.k}-NN on {len(features)}-bit structured BRIEF (Hamming) ---")
    print(f"Test accuracy: {acc:.2%}  (error {1 - acc:.2%})")


if __name__ == "__main__":
    main()
