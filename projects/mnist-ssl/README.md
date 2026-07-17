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
- [DINOv2 implementation notes](docs/dinov2.md)
- [Best-run configurations](configs/best/)
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
caffeinate -i uv run python scripts/train/ijepa.py \
  --epochs 500 --n-targets 48 --save-epoch 300 --seed 0
caffeinate -i uv run python scripts/reproduce/ijepa_members.py
```

Train the MNIST-scale DINOv2 implementation with a fixed 150-epoch schedule:

```bash
caffeinate -i uv run python scripts/train/dinov2.py \
  --epochs 150 \
  --checkpoint-epochs 50,75,100,125,150 \
  --checkpoint-every 10 \
  --photometric-augmentations \
  --output models/dinov2_mnist_augmented_cls_150ep.pt
```

Evaluate a frozen DINOv2 teacher backbone:

```bash
uv run python scripts/evaluate/dinov2_frozen.py \
  --model models/dinov2_mnist_augmented_cls_150ep_epoch0075.pt \
  --pool cls \
  --output models/dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt
```

Re-run the current best three-model grid:

```bash
uv run python scripts/reproduce/verify_artifacts.py
uv run python scripts/reproduce/best_ensemble.py --workers 0
```

## Code map

| Path | Responsibility |
|---|---|
| `src/mnist_ssl/dinov2/` | DINOv2 model, losses, training, and frozen evaluation |
| `src/mnist_ssl/ijepa/` | Custom and CNN-stem I-JEPA training, frozen probes, sweeps, and I-JEPA ensembles |
| `src/mnist_ssl/baselines/` | MAE baselines, handcrafted descriptors, retrieval, k-NN, and alternative probe heads |
| `src/mnist_ssl/ensembles/` | Cross-family ensemble evaluation and grid searches |
| `scripts/` | Thin, discoverable entry points grouped by train, evaluate, reproduce, sweeps, and analysis |
| `configs/best/` | Machine-readable settings and expected metrics for canonical results |
| `docs/` | Results, reproduction instructions, and the historical lab notebook |
| `results/` | Tracked metrics and checkpoint provenance |

Implementations live in the importable `mnist_ssl` package. Scripts contain no
model logic; they only expose stable commands for common workflows.
