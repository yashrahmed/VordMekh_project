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
- `mae_patch_embd/` — small **self-supervised encoders** trained on MNIST: a
  Masked Autoencoder (He et al. 2021) in ViT and conv-net flavors, plus an
  I-JEPA (Assran et al. 2023), with nearest-neighbor retrieval, k-NN, and
  classification evals over their encoder embeddings. See below.

## MAE patch embeddings (`mae_patch_embd`)

A self-contained subpackage with three self-supervised architectures, selected
via `--arch {vit,cnn,jepa}`:

- **`vit`** — patchify, *drop* ~75% of the patches, encode the visible ones with
  a Transformer, reconstruct the missing pixels (MSE on masked patches).
- **`cnn`** — a masked conv autoencoder (Context-Encoder style): masked patches
  are *zeroed* in the input, a conv encoder/decoder reconstructs the image.
- **`jepa`** — an I-JEPA (Assran et al. 2023): a context encoder sees the visible
  patches, an EMA target encoder sees the full image, and a predictor predicts
  the target's *latent* representations at the masked positions (no pixel
  decoder — the loss is in representation space).

```bash
# Train (downloads MNIST to dataset/, writes models/mae_mnist_<arch>.pt)
uv run python -m grasp_embeddings.mae_patch_embd.mae --arch vit --epochs 50
uv run python -m grasp_embeddings.mae_patch_embd.mae --arch cnn --epochs 50
uv run python -m grasp_embeddings.mae_patch_embd.mae --arch jepa --epochs 50

# Nearest-neighbor retrieval with the trained encoder
uv run python -m grasp_embeddings.mae_patch_embd.retrieve --arch vit --seed 0 --save out.png

# Classification: linear probe (frozen) or end-to-end fine-tune (--unfreeze)
uv run python -m grasp_embeddings.mae_patch_embd.classify --arch cnn --unfreeze --epochs 50

# k-NN over the frozen embeddings (full test set); --arch no-enc = raw-pixel floor
uv run python -m grasp_embeddings.mae_patch_embd.knn --arch vit --k 5
uv run python -m grasp_embeddings.mae_patch_embd.knn --arch no-enc
```

`dataset/`, `models/`, and `*.png` are gitignored.
