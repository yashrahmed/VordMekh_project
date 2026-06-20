# grasp-based-embeddings

Generate labeled data describing the **local shape** of 2D objects as sensed by a
"feeling hand" — five fingers that extend in straight lines from a posed hand
until they touch the shape. A sample is `(p, h, l1..l5)`: the hand pose plus the
five contact distances.

See [`todo.md`](todo.md) for the full idea, design notes, and roadmap.

## Install

This is a [uv](https://docs.astral.sh/uv/) project. `uv sync` creates the
virtualenv, resolves dependencies from `uv.lock`, and installs the package
(plus dev tools like pytest):

```bash
uv sync
```

## Quick start

```bash
# Render a single hand probing the "A" shape -> out/demo.png
uv run python -m grasp_embeddings.demo

# Generate a labeled dataset and a scatter visualization
uv run python -m grasp_embeddings.demo --dataset 2000

# Run the tests
uv run pytest
```

Or from Python:

```python
import numpy as np
from grasp_embeddings.shapes import letter_a
from grasp_embeddings.hand import Hand
from grasp_embeddings.sampler import generate_dataset

shape = letter_a()
hand = Hand.fan(n_fingers=5, spread_deg=80, max_length=3.0)

# One reading at a chosen pose:
lengths = hand.at(position=(0.0, 0.0), heading=0.0).sense(shape)

# A whole dataset of (p, h, l1..l5):
data = generate_dataset(shape, hand, n=2000, rng=np.random.default_rng(0))
```

## Layout

- `geometry.py` — ray–segment intersection, point-in-polygon (even-odd, holes).
- `shapes.py` — `Shape` (rings = outer + holes) and example shapes.
- `hand.py` — `Hand` / `Finger`; `sense()` casts the five rays.
- `sampler.py` — `generate_dataset()` and save/load helpers.
- `visualize.py` — matplotlib rendering of a probe and of a dataset.
- `demo.py` — runnable entry point.
- `mae_patch_embd/` — a small ViT-style **Masked Autoencoder** (He et al. 2021)
  trained on MNIST, with a nearest-neighbor retrieval demo over its encoder
  embeddings. See below.

## MAE patch embeddings (`mae_patch_embd`)

A self-contained subpackage: patchify an image, drop ~75% of the patches,
encode only the visible ones with a Transformer, then reconstruct the missing
pixels (MSE on masked patches only).

```bash
# Train (downloads MNIST to dataset/, writes weights to models/mae_mnist.pt)
uv run python -m grasp_embeddings.mae_patch_embd.mae --epochs 50

# Nearest-neighbor retrieval with the trained encoder
uv run python -m grasp_embeddings.mae_patch_embd.retrieve --seed 0 --save out.png
```

`dataset/`, `models/`, and `*.png` are gitignored.
