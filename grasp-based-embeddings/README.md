# grasp-based-embeddings

Small **self-supervised encoders** trained on MNIST: a Masked Autoencoder
(He et al. 2021) in ViT and conv-net flavors, plus an I-JEPA (Assran et al.
2023), with nearest-neighbor retrieval, k-NN, and classification evals over
their encoder embeddings. The driving question — can we beat the MNIST
benchmark with representations learned without label supervision?

See [`todo.md`](todo.md) for the full idea, design notes, findings, and roadmap.

## Install

This is a [uv](https://docs.astral.sh/uv/) project. `uv sync` creates the
virtualenv, resolves dependencies from `uv.lock`, and installs the package
(plus dev tools like pytest):

```bash
uv sync
```

## MAE patch embeddings (`trials`)

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
uv run python -m trials.mae --arch vit --epochs 50
uv run python -m trials.mae --arch cnn --epochs 50
uv run python -m trials.mae --arch jepa --epochs 50

# Nearest-neighbor retrieval with the trained encoder
uv run python -m trials.retrieve --arch vit --seed 0 --save out.png

# Classification: linear probe (frozen) or end-to-end fine-tune (--unfreeze)
uv run python -m trials.classify --arch cnn --unfreeze --epochs 50

# k-NN over the frozen embeddings (full test set); --arch no-enc = raw-pixel floor
uv run python -m trials.knn --arch vit --k 5
uv run python -m trials.knn --arch no-enc
```

`dataset/`, `models/`, and `*.png` are gitignored.
