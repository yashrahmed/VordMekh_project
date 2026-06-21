"""Handcrafted BRIEF descriptors -- shared core for the BRIEF experiments.

BRIEF (Calonder et al. 2010) summarises an image as a string of binary intensity
comparisons: each feature is an ordered pair of locations and its bit is ``1``
iff ``mean(first) < mean(second)``, where each location's intensity is the mean
over a small box, computed with an integral image (O(1) per box). A comparison
pattern is built *once* and reused for every image.

This module is the common home for two sampling strategies and the descriptor /
k-NN machinery they share:

* :func:`generate_random_features` -- the original BRIEF: ``n`` ordered
  ``(start, end)`` pairs sampled uniformly at random over the frame (used by
  ``knn-brief.py``).
* :func:`generate_structured_features` -- a regular ``grid x grid`` lattice where
  each location compares a centre box against its 4 neighbours (up/down/left/
  right), the census-/LBP-flavoured variant (used by ``knn-brief-mod.py``).

Both produce a ``(n_features, 2, 2)`` float array in a normalised coordinate
space of side ``extent`` (last axis ``(row, col)``; ``[i, 0]`` first point,
``[i, 1]`` second point), consumed by the same :func:`evaluate_batch`,
:func:`describe_split` and :func:`knn_predict`. The ``retrieve``/``classify``
scripts use :func:`make_features` + :func:`evaluate_batch` to turn images into
descriptors without a learned encoder.
"""

from __future__ import annotations

import numpy as np

# Side length (in coordinate-space units) of every marker box. For the random
# pattern the frame is ``patch`` units wide and sampling is inset by half of
# this; for the structured pattern the frame is ``grid`` units wide (one unit per
# cell) and this is half a cell, so a neighbour box touches the centre box
# edge-to-edge. On a 28px image a box is ``SQUARE / extent * 28`` px on a side.
SQUARE = 0.5

N_CLASSES = 10


# --------------------------------------------------------------------------- #
# Comparison-pattern generators
# --------------------------------------------------------------------------- #
def generate_random_features(
    patch_size: int = 4,
    n_features: int = 64,
    seed: int | None = 0,
) -> np.ndarray:
    """Sample ``n_features`` random BRIEF comparison pairs over a ``patch_size`` frame.

    Each feature is an ordered ``(start, end)`` pair of points drawn from
    *continuous* random positions inside the frame (not snapped to grid cells).
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


def _in_bounds(point: np.ndarray, extent: int, half: float = SQUARE / 2) -> bool:
    """True iff the ``SQUARE`` box centred at ``point`` lies fully in ``[0, extent]``."""
    return bool((point - half >= 0).all() and (point + half <= extent).all())


def generate_structured_features(grid: int = 4) -> np.ndarray:
    """Build the structured BRIEF comparison pattern for a ``grid x grid`` lattice.

    Coordinates live in a normalised space of side ``grid`` (one unit per cell).
    A location sits at the centre of its cell ``(i+0.5, j+0.5)``; its four
    neighbour boxes are offset by half a cell, so each neighbour box butts up
    against the centre box edge-to-edge (the plus-shaped probe).

    A feature is **dropped if either of its boxes would leave the frame**: the
    border locations' outward-pointing neighbours (the top row's up, the bottom
    row's down, etc.) spill past ``[0, grid]`` and are not placed. Centre boxes
    always fit, so every location keeps at least two features (corners 2, edges
    3, interior 4). The kept count is ``4*grid*(grid-1)``.

    Returns a float array of shape ``(n_features, 2, 2)`` where the last axis is
    ``(row, col)``: ``features[i, 0]`` is the centre point and ``features[i, 1]``
    the neighbour point. Features are emitted in row-major location order, each
    location's surviving neighbours ordered up, down, left, right.
    """
    off = SQUARE  # half a cell -> neighbour boxes touch the centre box
    dirs = np.array([(-off, 0.0), (off, 0.0), (0.0, -off), (0.0, off)])  # U D L R
    feats = []
    for i in range(grid):
        for j in range(grid):
            centre = np.array([i + 0.5, j + 0.5])
            for d in dirs:
                nbr = centre + d
                if _in_bounds(centre, grid) and _in_bounds(nbr, grid):
                    feats.append([centre, nbr])
    return np.asarray(feats, dtype=np.float64)


def make_features(
    kind: str,
    *,
    patch: int = 4,
    n: int = 64,
    seed: int | None = 0,
    grid: int = 8,
) -> tuple[np.ndarray, int, str]:
    """Build a comparison pattern by ``kind``; return ``(features, extent, label)``.

    ``kind`` is ``"brief"`` (random) or ``"brief-mod"`` (structured). ``extent``
    is the coordinate-space side length to pass to :func:`evaluate_batch` /
    :func:`describe_split`; ``label`` is a short human description for logging.
    """
    if kind == "brief":
        feats = generate_random_features(patch, n, seed)
        return feats, patch, (
            f"random BRIEF ({len(feats)} bits, {patch}x{patch} frame, seed {seed})"
        )
    if kind == "brief-mod":
        feats = generate_structured_features(grid)
        return feats, grid, (
            f"structured BRIEF ({len(feats)} bits, {grid}x{grid} grid)"
        )
    raise ValueError(f"unknown BRIEF kind: {kind!r} (expected 'brief' or 'brief-mod')")


# --------------------------------------------------------------------------- #
# Descriptor evaluation
# --------------------------------------------------------------------------- #
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


def evaluate(img: np.ndarray, features: np.ndarray, extent: int) -> np.ndarray:
    """Evaluate the BRIEF descriptor of one image for a comparison pattern.

    ``features`` are in the normalised coordinate space of side ``extent`` and
    are scaled to ``img``'s pixel grid. Each point's intensity is the mean over
    its :data:`SQUARE` box, computed with an integral image. The descriptor bit
    is ``1`` iff the first box is darker than the second (``mean(a) < mean(b)``).

    Returns a ``uint8`` array of shape ``(len(features),)``.
    """
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    ii = integral_image(img)
    scale_r, scale_c = H / extent, W / extent
    half_r, half_c = (SQUARE / 2) * scale_r, (SQUARE / 2) * scale_c

    a, b = features[:, 0], features[:, 1]  # each (n_features, 2) = (row, col)
    m_a = _box_means(ii, a[:, 0] * scale_r, a[:, 1] * scale_c, half_r, half_c)
    m_b = _box_means(ii, b[:, 0] * scale_r, b[:, 1] * scale_c, half_r, half_c)
    return (m_a < m_b).astype(np.uint8)


def evaluate_batch(imgs: np.ndarray, features: np.ndarray, extent: int) -> np.ndarray:
    """Vectorised :func:`evaluate` over a batch of images ``(B, H, W)``.

    The box corners depend only on the (fixed) comparison pattern, not on pixel
    values, so they are computed once and reused across the whole batch via a
    single batched integral image. Returns a ``uint8`` array ``(B, n_features)``.
    """
    imgs = np.asarray(imgs, dtype=np.float64)
    B, H, W = imgs.shape
    ii = np.zeros((B, H + 1, W + 1), dtype=np.float64)
    ii[:, 1:, 1:] = imgs.cumsum(axis=1).cumsum(axis=2)

    scale_r, scale_c = H / extent, W / extent
    half_r, half_c = (SQUARE / 2) * scale_r, (SQUARE / 2) * scale_c

    pts = features.reshape(-1, 2)  # (2*n, 2): [f0_a, f0_b, f1_a, f1_b, ...]
    rows, cols = pts[:, 0] * scale_r, pts[:, 1] * scale_c
    r0 = np.clip(np.rint(rows - half_r).astype(int), 0, H - 1)
    c0 = np.clip(np.rint(cols - half_c).astype(int), 0, W - 1)
    r1 = np.clip(np.rint(rows + half_r).astype(int), r0 + 1, H)
    c1 = np.clip(np.rint(cols + half_c).astype(int), c0 + 1, W)

    # ii[:, r1, c1] etc. broadcast over the batch -> (B, 2*n) box sums.
    total = ii[:, r1, c1] - ii[:, r0, c1] - ii[:, r1, c0] + ii[:, r0, c0]
    means = (total / ((r1 - r0) * (c1 - c0))).reshape(B, -1, 2)  # (B, n, a/b)
    return (means[:, :, 0] < means[:, :, 1]).astype(np.uint8)


# --------------------------------------------------------------------------- #
# MNIST description + k-NN
# --------------------------------------------------------------------------- #
def describe_split(features, extent, train, chunk=10000):
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
        evaluate_batch(imgs[i : i + chunk], features, extent)
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


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def _draw_pair(ax, feature, extent, marker=SQUARE):
    """Render one random BRIEF pair: red start square, blue end square, pink arrow."""
    from matplotlib.patches import Rectangle

    (sr, sc), (er, ec) = feature

    # Light grid for spatial reference (positions are continuous, not on it).
    for i in range(extent + 1):
        ax.axhline(i, color="0.85", lw=0.8, zorder=0)
        ax.axvline(i, color="0.85", lw=0.8, zorder=0)

    half = marker / 2
    ax.add_patch(Rectangle((sc - half, sr - half), marker, marker,
                           color="red", alpha=0.9, zorder=1))
    ax.add_patch(Rectangle((ec - half, er - half), marker, marker,
                           color="blue", alpha=0.9, zorder=1))
    ax.annotate(
        "",
        xy=(ec, er),
        xytext=(sc, sr),
        arrowprops=dict(arrowstyle="-|>", color="deeppink", lw=2, shrinkA=0, shrinkB=0),
        zorder=2,
    )

    ax.set_xlim(0, extent)
    ax.set_ylim(0, extent)
    ax.invert_yaxis()  # row 0 at the top, like an image
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_random(features: np.ndarray, extent: int, save: str | None = None):
    """Show each random BRIEF pair in its own cell of a subplot grid."""
    import matplotlib.pyplot as plt

    n = len(features)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, feature in enumerate(features):
        _draw_pair(axes[i], feature, extent)
        axes[i].set_title(f"#{i}", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"BRIEF comparison pattern ({extent}x{extent}, {n} features)\n"
        "red = start, blue = end",
        fontsize=10,
    )
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"Saved figure -> {save}")
    else:
        plt.show()


def _group_by_location(features: np.ndarray) -> list[np.ndarray]:
    """Split the feature list into per-location blocks of features sharing a centre.

    Features are emitted contiguously per location (see
    :func:`generate_structured_features`), so a single pass that breaks whenever
    the centre changes recovers the groups -- which may hold 2-4 features each
    after edge clipping.
    """
    groups, cur, cur_centre = [], [], None
    for f in features:
        c = tuple(f[0])
        if c != cur_centre:
            if cur:
                groups.append(np.asarray(cur))
            cur, cur_centre = [f], c
        else:
            cur.append(f)
    if cur:
        groups.append(np.asarray(cur))
    return groups


def _draw_location(ax, group, extent, marker=SQUARE):
    """Render one location's plus probe: blue centre, red neighbours, pink arrows."""
    from matplotlib.patches import Rectangle

    centre = group[0, 0]  # all features in a group share the centre

    for i in range(extent + 1):
        ax.axhline(i, color="0.85", lw=0.8, zorder=0)
        ax.axvline(i, color="0.85", lw=0.8, zorder=0)

    half = marker / 2
    cr, cc = centre
    ax.add_patch(Rectangle((cc - half, cr - half), marker, marker,
                           color="blue", alpha=0.9, zorder=1))
    for _, end in group:  # (centre, neighbour) pairs
        er, ec = end
        ax.add_patch(Rectangle((ec - half, er - half), marker, marker,
                               color="red", alpha=0.6, zorder=1))
        ax.annotate(
            "",
            xy=(ec, er),
            xytext=(cc, cr),
            arrowprops=dict(arrowstyle="-|>", color="deeppink", lw=1.5,
                            shrinkA=0, shrinkB=0),
            zorder=2,
        )

    ax.set_xlim(-marker, extent + marker)
    ax.set_ylim(-marker, extent + marker)
    ax.invert_yaxis()  # row 0 at the top, like an image
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_structured(features: np.ndarray, grid: int, save: str | None = None):
    """Show each location's plus probe in its own cell of a subplot grid."""
    import matplotlib.pyplot as plt

    groups = _group_by_location(features)
    n = len(groups)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, group in enumerate(groups):
        _draw_location(axes[idx], group, grid)
        i = int(round(group[0, 0, 0] - 0.5))
        j = int(round(group[0, 0, 1] - 0.5))
        axes[idx].set_title(f"({i},{j}) {len(group)}b", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Structured BRIEF ({grid}x{grid} grid, edge-clipped, "
        f"{len(features)} bits)\nblue = centre, red = neighbour",
        fontsize=10,
    )
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"Saved figure -> {save}")
    else:
        plt.show()
