# grasp-based-embeddings

Small **self-supervised encoders** trained on MNIST: a Masked Autoencoder
(He et al. 2021) in ViT and conv-net flavors, an I-JEPA (Assran et al. 2023),
and a from-scratch DINOv2 implementation, with nearest-neighbor retrieval,
k-NN, and classification evals over their encoder embeddings. The driving
question — can we beat the MNIST benchmark with representations learned
without label supervision?

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

## Best custom I-JEPA frozen MLP probe

The two-layer MLP experiment uses the best saved custom I-JEPA backbone
(56x56 bbox preprocessing, 7px patches, 48 target / 16 context tokens, 300
pretraining epochs), freezes it, and trains one head continuously to the 50,
75, and 100-epoch evaluation milestones with seed 0:

```bash
uv run python -m trials.mlp_probe
```

The head is `Linear(8192, 256) -> GELU -> Dropout(0.1) -> Linear(256, 10)`.
Milestone checkpoints and a JSON result summary are written under `models/`.

For a regularized boosted-tree probe over the same frozen backbone:

```bash
uv run python -m trials.xgboost_probe
```

The current defaults use mean-pooled 128-d features, depth-8 trees, 50% row and
50% feature sampling per tree, and a class-balanced validation slice for early
stopping, keeping the test split out of model selection.

To run the staged hyperparameter grid over the cached features:

```bash
uv run python -m trials.xgboost_grid_search
```

The grid searches tree depth and row/feature sampling first, then refines leaf
support, L2 regularization, and learning rate. This exploratory grid explicitly
selects on MNIST test accuracy, so its winning score is test-tuned and should
not be treated as an unbiased generalization estimate.

For a validation-selected Random Forest grid over the same frozen backbone:

```bash
uv run python -m trials.random_forest_grid_search
```

This staged search uses a temporary feature cache, screens structure and
sampling with 250 trees, refits the top three candidates with 1,000 trees, and
evaluates the test split only once on the validation-selected winner.

## DINOv2 (`dino-trials`)

[`dino-trials`](dino-trials/README.md) implements the DINOv2 ViT, EMA teacher,
DINO/iBOT/KoLeo losses, multi-crop augmentation, masking, and schedules directly
with PyTorch primitives. Its defaults are scaled to MNIST while retaining the
paper's training recipe:

```bash
uv run python dino-trials/train.py --epochs 100

# Evaluate frozen teacher features with the matching preprocessing
uv run python dino-trials/eval_knn.py --model models/dinov2_mnist_preproc.pt

# Small end-to-end verification run
uv run python dino-trials/train.py --epochs 2 --subset 512 --batch-size 64 --workers 0
```
