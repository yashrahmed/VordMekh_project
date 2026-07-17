"""Intensity-geodesic distance matrix over a 2x-downsampled image.

A scratch/experiment script for the shape-descriptor track (todo item 3). Given
an image it:

1. takes an image (an MNIST sample by ``--index``, or any file via ``--image``);
2. downsamples it by 2 with **average** pooling (a 28x28 MNIST digit -> 14x14);
3. builds a grid graph over the downsampled pixels -- each pixel is a node, with
   edges to its 8- (or ``--connectivity 4``) neighbours weighted by
   ``|dI| + ALPHA`` (the absolute **intensity difference** across the edge plus a
   fixed per-step cost) -- and computes the all-pairs **geodesic** (shortest-path)
   distance matrix with Dijkstra.

The geodesic between two pixels is then the cheapest path between them under that
edge cost: travelling within a flat region is near-free (only the ``ALPHA`` step
cost), while crossing the digit's stroke edges is expensive, so the matrix mixes
the image's intensity terrain with its layout. ``ALPHA`` is locked to
:data:`ALPHA` (0.1) -- not a flag -- so every consumer builds the same geodesic.

    python -m trials.geodesic --index 0
    python -m trials.geodesic --image path/to/img.png
    python -m trials.geodesic --index 7 --connectivity 4

With ``--save`` (default ``geodesic.png`` under the project ``images/`` dir, which
is gitignored) it writes a figure: the downsampled image, the geodesic distance
field from a chosen source pixel, and the full NxN distance matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from torchvision import datasets, transforms

from .mae import DATASET_DIR, PROJECT_ROOT

# ``--arch`` value exposed by retrieve.py / knn.py for the geodesic signature.
GEODESIC_ARCH = "geodesic"

# Graph settings, locked so every consumer (this script, retrieve, knn) builds the
# same geodesic. ALPHA is the fixed per-step spatial cost added to each edge
# (weight = |dI| + ALPHA) so distance also grows with path length, not just
# intensity climb; 8-connectivity gives near-isotropic geodesics on the grid.
ALPHA = 0.1
DEFAULT_CONNECTIVITY = 8


def load_mnist_image(index: int, train: bool) -> np.ndarray:
    """Return one MNIST image as a float32 (28, 28) array in [0, 1]."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(
        root=str(DATASET_DIR), train=train, download=True, transform=transforms.ToTensor()
    )
    if not 0 <= index < len(ds):
        raise IndexError(f"--index {index} out of range (0..{len(ds) - 1})")
    img, label = ds[index]
    print(f"MNIST {'train' if train else 'test'}[{index}]: label {label}")
    return img.squeeze(0).numpy().astype(np.float32)


def load_file_image(path: Path) -> np.ndarray:
    """Load an arbitrary image file as a float32 grayscale array in [0, 1]."""
    import matplotlib.image as mpimg

    arr = mpimg.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:  # RGB(A) -> luminance, drop any alpha channel
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if arr.max() > 1.0:  # 8-bit images come in as 0..255
        arr /= 255.0
    print(f"Loaded {path.name}: {arr.shape}")
    return arr


def downsample_avg(img: np.ndarray, factor: int = 2) -> np.ndarray:
    """Average-pool ``img`` by ``factor`` (28x28 -> 14x14 at factor=2).

    Uses ``avg_pool2d``; the side length must be divisible by ``factor`` (true
    for MNIST's 28). The kernel averages each non-overlapping factor x factor
    block, so it is a true mean downsample, not a strided subsample.
    """
    h, w = img.shape
    if h % factor or w % factor:
        raise ValueError(f"image {h}x{w} not divisible by factor {factor}")
    t = torch.from_numpy(img)[None, None]  # (1, 1, H, W)
    pooled = F.avg_pool2d(t, kernel_size=factor, stride=factor)
    return pooled[0, 0].numpy()


def _neighbour_offsets(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    raise ValueError("--connectivity must be 4 or 8")


def geodesic_matrix(
    grid: np.ndarray, connectivity: int = DEFAULT_CONNECTIVITY, alpha: float = ALPHA
) -> np.ndarray:
    """All-pairs geodesic distances over the pixel grid.

    Each of the ``H*W`` pixels is a node (row-major id ``r*W + c``); neighbouring
    pixels are joined by an edge of weight ``|I_a - I_b| + alpha``. Returns the
    dense ``(H*W, H*W)`` shortest-path (Dijkstra) distance matrix. ``alpha``
    defaults to the locked :data:`ALPHA`.
    """
    h, w = grid.shape
    flat = grid.reshape(-1)
    rows, cols, data = [], [], []
    for dr, dc in _neighbour_offsets(connectivity):
        # Shift the grid by (dr, dc); valid cells get an edge to their neighbour.
        rr = np.arange(h)[:, None] + dr
        cc = np.arange(w)[None, :] + dc
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        src = (np.arange(h)[:, None] * w + np.arange(w)[None, :])[valid]
        dst = (rr * w + cc)[valid]
        weight = np.abs(flat[src] - flat[dst]) + alpha
        rows.append(src)
        cols.append(dst)
        data.append(weight)
    n = h * w
    graph = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    # Undirected: symmetric weights, both (dr,dc) and (-dr,-dc) are emitted above.
    return shortest_path(graph, method="D", directed=False)


# --------------------------------------------------------------------------- #
# D2 shape-distribution descriptor (shared by knn / eval_classifier)
# --------------------------------------------------------------------------- #
GEODESIC_BINS = 64


def geodesic_histogram(grid: np.ndarray, bins: int = GEODESIC_BINS) -> np.ndarray:
    """One image's geodesic D2 shape distribution.

    Builds the all-pairs geodesic matrix for ``grid``, scales the upper-triangle
    distances by their own max (so the descriptor is scale-invariant), and
    histograms them into ``bins`` density-normalised bins over [0, 1] -- an
    Osada-style D2 descriptor. Permutation-invariant: it keeps the distribution
    of geodesic gaps, not where they sit.
    """
    dist = geodesic_matrix(grid)
    vals = dist[np.triu_indices(dist.shape[0], k=1)]
    vmax = vals.max()
    if vmax > 0:
        vals = vals / vmax
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0), density=True)
    return hist.astype(np.float32)


def geodesic_features(loader, bins: int = GEODESIC_BINS):
    """Describe every image from ``loader`` as a geodesic D2 histogram.

    ``loader`` yields ``(imgs, labels)`` batches of MNIST tensors (B, 1, 28, 28).
    Each image is 2x average-downsampled, then turned into a
    :func:`geodesic_histogram`. Returns ``(X, y)`` with ``X`` the L2-normalised
    histograms (so cosine k-NN / nearest-centroid compare them) -- a fixed,
    zero-learning descriptor, no training required.

    retrieve.py flattens the whole distance matrix instead, but that is ~38k-dim
    per image; over tens of thousands of images this compact distribution stands
    in for it.
    """
    feats, labels = [], []
    for imgs, y in loader:
        arr = imgs.squeeze(1).numpy().astype(np.float32)  # (B, 28, 28)
        for img in arr:
            feats.append(torch.from_numpy(geodesic_histogram(downsample_avg(img), bins)))
        labels.append(y)
    return F.normalize(torch.stack(feats), dim=-1), torch.cat(labels)


def visualize(grid: np.ndarray, dist: np.ndarray, source: int, out: Path) -> None:
    import matplotlib.pyplot as plt

    h, w = grid.shape
    sr, sc = divmod(source, w)
    field = dist[source].reshape(h, w)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(grid, cmap="gray")
    axes[0].set_title(f"downsampled {h}x{w}")
    axes[0].scatter([sc], [sr], c="red", s=40, marker="x")

    im1 = axes[1].imshow(field, cmap="viridis")
    axes[1].set_title(f"geodesic from pixel {source} (r{sr},c{sc})")
    axes[1].scatter([sc], [sr], c="red", s=40, marker="x")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(dist, cmap="magma")
    axes[2].set_title(f"distance matrix {dist.shape[0]}x{dist.shape[1]}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved figure -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--index", type=int, default=0, help="MNIST sample index.")
    src.add_argument("--image", type=str, default=None, help="Path to an image file.")
    parser.add_argument(
        "--train", action="store_true", help="Use the MNIST train split (default: test)."
    )
    parser.add_argument("--factor", type=int, default=2, help="Downsample factor.")
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=DEFAULT_CONNECTIVITY,
        help="Grid neighbourhood.",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Source pixel id for the distance-field plot (default: grid centre).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Figure path (default: <project>/images/geodesic.png).",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip the figure.")
    args = parser.parse_args()

    if args.image is not None:
        img = load_file_image(Path(args.image))
    else:
        img = load_mnist_image(args.index, train=args.train)

    grid = downsample_avg(img, args.factor)
    print(f"Downsampled to {grid.shape}  (range {grid.min():.3f}..{grid.max():.3f})")

    dist = geodesic_matrix(grid, connectivity=args.connectivity)
    finite = dist[np.isfinite(dist)]
    print(
        f"Geodesic matrix: {dist.shape}  "
        f"max {finite.max():.3f}  mean {finite.mean():.3f}"
        + ("" if np.isfinite(dist).all() else "  (graph not fully connected: some inf)")
    )

    if not args.no_viz:
        h, w = grid.shape
        source = args.source if args.source is not None else (h // 2) * w + (w // 2)
        out = Path(args.save) if args.save else PROJECT_ROOT / "images" / "geodesic.png"
        visualize(grid, dist, source, out)


if __name__ == "__main__":
    main()
