"""k-NN classification on MNIST with handcrafted BRIEF descriptors.

BRIEF (Calonder et al. 2010) summarises a patch as a string of binary
intensity comparisons: each feature is an ordered pair of locations
``(start, end)`` and its bit is ``1`` iff ``mean(start) < mean(end)``, where
each location's intensity is the mean over a small box (the drawn square),
computed with an integral image. The comparison pattern is sampled *once*
(fixed seed) and reused for every image.

Each MNIST image becomes an ``n``-bit descriptor; test digits are classified by
a majority vote over their nearest training descriptors in **Hamming** distance.
This is the handcrafted, zero-learning counterpart to the learned encoders in
``knn.py`` -- a "do something but don't learn" baseline.

    python knn-brief.py                       # describe MNIST + run k-NN
    python knn-brief.py --n 64 --k 5 --seed 0
    python knn-brief.py --viz-only            # just show the comparison pattern
    python knn-brief.py --viz-only --save brief.png
"""

from __future__ import annotations

import argparse

import numpy as np

# Side length (in patch units) of the start/end marker square. Sampling is inset
# by half of this so every square lies completely within the image.
SQUARE = 0.5


def generate_features(
    patch_size: int = 4,
    n_features: int = 64,
    seed: int | None = 0,
) -> np.ndarray:
    """Sample ``n_features`` BRIEF comparison pairs over a ``patch_size`` patch.

    Each feature is an ordered ``(start, end)`` pair of points drawn from
    *continuous* random positions inside the patch (not snapped to grid cells).
    Centres are inset by ``SQUARE / 2`` so the drawn marker square always lies
    completely within the image, and the two points of a pair are kept at least
    ``SQUARE`` apart (Chebyshev) so their squares never overlap. Returns a float
    array of shape ``(n_features, 2, 2)`` where the last axis is ``(row, col)``:
    ``features[i, 0]`` is the start point and ``features[i, 1]`` is the end point.
    """
    rng = np.random.default_rng(seed)
    half = SQUARE / 2
    lo, hi = half, patch_size - half
    # (n_features, 2 points, 2 coords) uniform over the inset region.
    feats = rng.uniform(lo, hi, size=(n_features, 2, 2))
    # Resample any pair whose two squares overlap (Chebyshev gap < SQUARE).
    while True:
        gap = np.abs(feats[:, 0] - feats[:, 1]).max(axis=1)  # (n_features,)
        overlap = gap < SQUARE
        if not overlap.any():
            break
        feats[overlap] = rng.uniform(lo, hi, size=(int(overlap.sum()), 2, 2))
    return feats


def integral_image(img: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero top/left border.

    ``ii[i, j]`` is the sum of ``img[:i, :j]``, so the sum over any axis-aligned
    box ``img[r0:r1, c0:c1]`` is
    ``ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]`` in O(1).
    """
    img = np.asarray(img, dtype=np.float64)
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    return ii


def _box_means(ii, rows, cols, half_r, half_c):
    """Mean intensity of the box around each ``(row, col)`` centre, via ``ii``.

    ``rows``/``cols`` and the half-extents are in *pixel* units. Boxes are
    clamped to the image and always span at least one pixel.
    """
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    r0 = np.clip(np.rint(rows - half_r).astype(int), 0, H - 1)
    c0 = np.clip(np.rint(cols - half_c).astype(int), 0, W - 1)
    r1 = np.clip(np.rint(rows + half_r).astype(int), r0 + 1, H)
    c1 = np.clip(np.rint(cols + half_c).astype(int), c0 + 1, W)
    total = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
    return total / ((r1 - r0) * (c1 - c0))


def evaluate(img: np.ndarray, features: np.ndarray, patch_size: int) -> np.ndarray:
    """Evaluate the BRIEF descriptor of ``img`` for a comparison pattern.

    ``features`` are in patch-unit coordinates (see :func:`generate_features`)
    and are scaled to ``img``'s pixel grid. Each point's intensity is the mean
    over its ``SQUARE`` box -- the same square drawn in the visualization --
    computed with an integral image so every box sum is O(1) regardless of box
    size. The descriptor bit is ``1`` iff the start box is darker than the end
    box (``mean(start) < mean(end)``).

    Returns a ``uint8`` array of shape ``(len(features),)``.
    """
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    ii = integral_image(img)
    scale_r, scale_c = H / patch_size, W / patch_size
    half_r, half_c = (SQUARE / 2) * scale_r, (SQUARE / 2) * scale_c

    start, end = features[:, 0], features[:, 1]  # each (n_features, 2) = (row, col)
    m_start = _box_means(
        ii, start[:, 0] * scale_r, start[:, 1] * scale_c, half_r, half_c
    )
    m_end = _box_means(ii, end[:, 0] * scale_r, end[:, 1] * scale_c, half_r, half_c)
    return (m_start < m_end).astype(np.uint8)


def evaluate_batch(imgs: np.ndarray, features: np.ndarray, patch_size: int) -> np.ndarray:
    """Vectorised :func:`evaluate` over a batch of images ``(B, H, W)``.

    The box corners depend only on the (fixed) comparison pattern, not on pixel
    values, so they are computed once and reused across the whole batch via a
    single batched integral image. Returns a ``uint8`` array ``(B, n_features)``.
    """
    imgs = np.asarray(imgs, dtype=np.float64)
    B, H, W = imgs.shape
    ii = np.zeros((B, H + 1, W + 1), dtype=np.float64)
    ii[:, 1:, 1:] = imgs.cumsum(axis=1).cumsum(axis=2)

    scale_r, scale_c = H / patch_size, W / patch_size
    half_r, half_c = (SQUARE / 2) * scale_r, (SQUARE / 2) * scale_c

    pts = features.reshape(-1, 2)  # (2*n, 2): [f0_start, f0_end, f1_start, ...]
    rows, cols = pts[:, 0] * scale_r, pts[:, 1] * scale_c
    r0 = np.clip(np.rint(rows - half_r).astype(int), 0, H - 1)
    c0 = np.clip(np.rint(cols - half_c).astype(int), 0, W - 1)
    r1 = np.clip(np.rint(rows + half_r).astype(int), r0 + 1, H)
    c1 = np.clip(np.rint(cols + half_c).astype(int), c0 + 1, W)

    # ii[:, r1, c1] etc. broadcast over the batch -> (B, 2*n) box sums.
    total = ii[:, r1, c1] - ii[:, r0, c1] - ii[:, r1, c0] + ii[:, r0, c0]
    means = (total / ((r1 - r0) * (c1 - c0))).reshape(B, -1, 2)  # (B, n, start/end)
    return (means[:, :, 0] < means[:, :, 1]).astype(np.uint8)


def _draw(ax, feature, patch_size, marker=SQUARE):
    """Render one BRIEF pair: red start square, blue end square, pink arrow."""
    from matplotlib.patches import Rectangle

    (sr, sc), (er, ec) = feature

    # Light grid for spatial reference (positions are continuous, not on it).
    for i in range(patch_size + 1):
        ax.axhline(i, color="0.85", lw=0.8, zorder=0)
        ax.axvline(i, color="0.85", lw=0.8, zorder=0)

    # Small squares centred on the sampled points.
    half = marker / 2
    ax.add_patch(Rectangle((sc - half, sr - half), marker, marker,
                           color="red", alpha=0.9, zorder=1))
    ax.add_patch(Rectangle((ec - half, er - half), marker, marker,
                           color="blue", alpha=0.9, zorder=1))

    # Pink arrow from the start point to the end point.
    ax.annotate(
        "",
        xy=(ec, er),
        xytext=(sc, sr),
        arrowprops=dict(arrowstyle="-|>", color="deeppink", lw=2, shrinkA=0, shrinkB=0),
        zorder=2,
    )

    ax.set_xlim(0, patch_size)
    ax.set_ylim(0, patch_size)
    ax.invert_yaxis()  # row 0 at the top, like an image
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_features(features: np.ndarray, patch_size: int, save: str | None = None):
    """Show each feature's placement in its own cell of a subplot grid."""
    import matplotlib.pyplot as plt

    n = len(features)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, feature in enumerate(features):
        _draw(axes[i], feature, patch_size)
        axes[i].set_title(f"#{i}", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"BRIEF comparison pattern ({patch_size}x{patch_size}, {n} features)\n"
        "red = start, blue = end",
        fontsize=10,
    )
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"Saved figure -> {save}")
    else:
        plt.show()


N_CLASSES = 10


def describe_split(features, patch_size, train, chunk=10000):
    """BRIEF-describe a full MNIST split. Returns ``(N, n_features)`` bits, labels.

    Reads the raw ``uint8`` images directly (no transform) and scales them to
    ``[0, 1]`` before the integral-image evaluation.
    """
    from torchvision import datasets

    from grasp_embeddings.mae_patch_embd.mae import DATASET_DIR

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(root=str(DATASET_DIR), train=train, download=True)
    imgs = ds.data.numpy().astype(np.float64) / 255.0  # (N, 28, 28)
    labels = ds.targets.numpy()

    descs = [
        evaluate_batch(imgs[i : i + chunk], features, patch_size)
        for i in range(0, len(imgs), chunk)
    ]
    return np.concatenate(descs), labels


def knn_predict(test_desc, train_desc, train_labels, k, chunk=512):
    """Majority-vote k-NN over BRIEF descriptors using Hamming distance.

    For 0/1 bit vectors the Hamming distance is
    ``a . (1 - b) + (1 - a) . b``, so a chunk of distances is two matmuls.
    """
    A = test_desc.astype(np.float32)
    B = train_desc.astype(np.float32)
    Bc = 1.0 - B
    preds = []
    for i in range(0, len(A), chunk):
        a = A[i : i + chunk]  # (c, n_features)
        ham = a @ Bc.T + (1.0 - a) @ B.T  # (c, N_train) Hamming distances
        idx = np.argpartition(ham, k, axis=1)[:, :k]  # k smallest (unordered)
        votes = train_labels[idx]  # (c, k)
        preds.append(
            np.array([np.bincount(v, minlength=N_CLASSES).argmax() for v in votes])
        )
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", type=int, default=4, help="Patch side length.")
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

    features = generate_features(args.patch, args.n, args.seed)
    print(f"Generated {len(features)} BRIEF features over a "
          f"{args.patch}x{args.patch} patch (seed {args.seed}).")

    if args.viz_only:
        visualize_features(features, args.patch, args.save)
        return

    print("Describing MNIST train split with BRIEF...")
    train_desc, train_labels = describe_split(features, args.patch, train=True)
    print("Describing MNIST test split with BRIEF...")
    test_desc, test_labels = describe_split(features, args.patch, train=False)
    print(f"  train: {train_desc.shape}   test: {test_desc.shape}")

    pred = knn_predict(test_desc, train_desc, train_labels, args.k)
    acc = (pred == test_labels).mean()

    print(f"\n--- {args.k}-NN on {args.n}-bit BRIEF descriptors (Hamming) ---")
    print(f"Test accuracy: {acc:.2%}  (error {1 - acc:.2%})")


if __name__ == "__main__":
    main()
