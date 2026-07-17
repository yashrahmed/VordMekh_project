# MNIST self-supervised learning

From-scratch self-supervised vision experiments on MNIST, including DINOv2,
I-JEPA, ViT/conv masked autoencoders, frozen linear probes, nearest-neighbor
evaluation, and classifier ensembles.

The project asks how far label-free representation learning can push MNIST
classification once the learned backbone is frozen. The current best individual
model is a DINOv2 frozen CLS probe at **99.42%**. A test-tuned DINO/I-JEPA
triplet reaches **99.61%**; see the selection caveat in the results document.

## Start here

- [Curated results](docs/results.md)
- [Reproduce the best current ensemble](docs/reproduce-best.md)
- [Active roadmap](ROADMAP.md)
- [Full experiment log](docs/experiment-log.md)
- [DINOv2 implementation notes](dino-trials/README.md)
- [Checkpoint manifest](results/checkpoint-manifest.json)

## Install

This directory is an independent [uv](https://docs.astral.sh/uv/) project:

```bash
cd projects/mnist-ssl
uv sync
```

Datasets, checkpoints, plots, and run logs are stored under `dataset/`,
`models/`, `images/`, and `out/`. They are intentionally ignored by Git.
Tracked result summaries and hashes live under `results/`.

## Canonical commands

Train the custom 56x56 I-JEPA backbone and its frozen flattened probe:

```bash
caffeinate -i uv run python -m ijepa_trials.custom_ijepa \
  --epochs 500 --n-targets 48 --save-epoch 300 --seed 0
caffeinate -i uv run python -m ijepa_trials.run_best_t48_500
```

Train the MNIST-scale DINOv2 implementation with a fixed 150-epoch schedule:

```bash
caffeinate -i uv run python dino-trials/train.py \
  --epochs 150 \
  --checkpoint-epochs 50,75,100,125,150 \
  --checkpoint-every 10 \
  --photometric-augmentations \
  --output models/dinov2_mnist_augmented_cls_150ep.pt
```

Evaluate a frozen DINOv2 teacher backbone:

```bash
uv run python dino-trials/eval_frozen.py \
  --model models/dinov2_mnist_augmented_cls_150ep_epoch0075.pt \
  --pool cls \
  --output models/dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt
```

Re-run the current best three-model grid:

```bash
uv run python dino-trials/ensemble_ijepa_triplet.py --workers 0
```

These paths remain compatible during the repository cleanup. A later package
refactor will replace the hyphenated `dino-trials` directory and consolidate
the experiment entry points without changing the model implementations.

## Current code map

| Path | Responsibility |
|---|---|
| `dino-trials/` | DINOv2 model, losses, training, frozen evaluation, and DINO/I-JEPA ensembles |
| `ijepa_trials/` | Custom and CNN-stem I-JEPA training, frozen probes, sweeps, and I-JEPA ensembles |
| `trials/` | MAE baselines, handcrafted descriptors, retrieval, k-NN, and alternative probe heads |
| `docs/` | Results, reproduction instructions, and the historical lab notebook |
| `results/` | Tracked metrics and checkpoint provenance |

The current organization is an intentionally compatible intermediate state.
The target is an importable `src/mnist_ssl/` package with thin scripts grouped
by training, evaluation, reproduction, and analysis.
