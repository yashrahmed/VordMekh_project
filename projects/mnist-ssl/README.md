# MNIST self-supervised learning

From-scratch self-supervised vision experiments on MNIST, including DINOv2,
I-JEPA, ViT/conv masked autoencoders, frozen linear probes, nearest-neighbor
evaluation, and classifier ensembles.

The project asks how far label-free representation learning can push MNIST
classification once the learned backbone is frozen. The current best individual
model is a small nonlinear probe on frozen DINOv2 CLS features at **99.52%**.
A test-tuned probability mixture of the DINOv2, I-JEPA-300, and I-JEPA-500
nonlinear probes reaches **99.64%** on the original labels and **99.67%** under
the manually reviewed label policy at the same weights. Separately tuning to
the reviewed labels reaches **99.68%**; see the selection caveat in the results
document.

## Best reviewed-label result

The best reviewed-label triplet averages nonlinear-probe softmax probabilities
with weights `0.530 * DINOv2-75 + 0.253 * I-JEPA-300 + 0.217 *
I-JEPA-500`. Under the completed manual-review policy, which relabels eight
examples and excludes two ambiguous examples, it makes **32 errors among 9,998
scored examples: 99.67994% accuracy**. The individual nonlinear probes make
45, 59, and 54 errors respectively. Fourteen errors are shared by all three,
placing the label-oracle ceiling at **99.85997%**.

These weights were selected on the reviewed test labels. The
canonical-label-selected weights are `0.547/0.270/0.183`; they make 36
canonical errors and 33 reviewed errors. The result demonstrates complementary
errors but is a test-tuned diagnostic, not a validation-clean estimate. Exact
metrics and hashes are preserved in the
[nonlinear-ensemble reproduction record](results/reproductions/2026-07-18-nonlinear-ensembles.json).

## Start here

- [Curated results](docs/results.md)
- [Reproduce the best current ensemble](docs/reproduce-best.md)
- [Active roadmap](ROADMAP.md)
- [Full experiment log](docs/experiment-log.md)
- [DINOv2 implementation notes](docs/dinov2.md)
- [Best-run configurations](configs/best/)
- [Checkpoint manifest](results/checkpoint-manifest.json)
- [Artifact and checkpoint policy](docs/artifact-policy.md)

## Install

This directory is an independent [uv](https://docs.astral.sh/uv/) project:

```bash
cd projects/mnist-ssl
uv sync
```

Datasets, checkpoints, plots, and run logs are stored under `dataset/`,
`models/`, `images/`, and `out/`. They are intentionally ignored by Git.
Tracked result summaries and hashes live under `results/`; the
[artifact policy](docs/artifact-policy.md) explains promotion and cleanup.

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

Re-run the preserved linear triplet:

```bash
uv run python scripts/reproduce/verify_artifacts.py
uv run python scripts/reproduce/best_ensemble.py --workers 0
uv run python scripts/reproduce/best_ensemble.py \
  --workers 0 \
  --apply-known-corrections
```

With the ignored nonlinear prediction artifacts present, re-run the latest
diagnostic grids:

```bash
uv run python scripts/analysis/grid_dino_ijepa500_nonlinear_ensemble.py
uv run python scripts/analysis/grid_dino_ijepa_nonlinear_triplet.py
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

To inspect the 15 paper-validated MNIST label issues relevant to the current
label-aware upper-bound work, build the
[manual MNIST label reviewer](docs/review-mnist-labels.md).
