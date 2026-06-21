"""k-NN classification on MNIST with handcrafted (random) BRIEF descriptors.

BRIEF (Calonder et al. 2010) summarises a patch as a string of binary intensity
comparisons: each feature is an ordered pair of locations ``(start, end)`` and
its bit is ``1`` iff ``mean(start) < mean(end)``, where each location's intensity
is the mean over a small box, computed with an integral image. The comparison
pattern is sampled *once* (fixed seed) and reused for every image.

Each MNIST image becomes an ``n``-bit descriptor; test digits are classified by
a majority vote over their nearest training descriptors in **Hamming** distance.
This is the handcrafted, zero-learning counterpart to the learned encoders in
``knn.py`` -- a "do something but don't learn" baseline. See ``knn-brief-mod.py``
for the structured (lattice) variant; both share :mod:`brief`.

    python knn-brief.py                       # describe MNIST + run k-NN
    python knn-brief.py --n 64 --k 5 --seed 0
    python knn-brief.py --viz-only            # just show the comparison pattern
    python knn-brief.py --viz-only --save brief.png
"""

from __future__ import annotations

import argparse

from grasp_embeddings.mae_patch_embd import brief


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", type=int, default=4, help="Frame side length.")
    parser.add_argument("--n", type=int, default=64, help="Number of features.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5, help="Neighbours for k-NN.")
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Only show/save the feature placements; skip the MNIST k-NN eval.",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save the figure (PNG)."
    )
    args = parser.parse_args()

    features = brief.generate_random_features(args.patch, args.n, args.seed)
    print(f"Generated {len(features)} BRIEF features over a "
          f"{args.patch}x{args.patch} frame (seed {args.seed}).")

    if args.viz_only:
        brief.visualize_random(features, args.patch, args.save)
        return

    print("Describing MNIST train split with BRIEF...")
    train_desc, train_labels = brief.describe_split(features, args.patch, train=True)
    print("Describing MNIST test split with BRIEF...")
    test_desc, test_labels = brief.describe_split(features, args.patch, train=False)
    print(f"  train: {train_desc.shape}   test: {test_desc.shape}")

    pred = brief.knn_predict(test_desc, train_desc, train_labels, args.k)
    acc = (pred == test_labels).mean()

    print(f"\n--- {args.k}-NN on {args.n}-bit BRIEF descriptors (Hamming) ---")
    print(f"Test accuracy: {acc:.2%}  (error {1 - acc:.2%})")


if __name__ == "__main__":
    main()
