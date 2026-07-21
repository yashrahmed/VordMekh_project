# MNIST self-supervised learning

From-scratch self-supervised vision experiments on MNIST, including DINOv2,
I-JEPA, ViT/conv masked autoencoders, frozen linear probes, nearest-neighbor
evaluation, and classifier ensembles.

The project asks how far label-free representation learning can push MNIST
classification with frozen or parameter-efficiently adapted backbones. The
current best frozen individual is a nonlinear probe on DINOv2 CLS features at
**99.52%**. Rank-8 LoRA adaptation of the I-JEPA-500 target tower with a
nonlinear head reaches an observed **99.58%** without changing any pretrained
backbone tensor.
The current best ensemble selected without test labels averages the DINOv2,
I-JEPA-300, and I-JEPA-500 nonlinear probabilities. It reaches **99.63%** on
the original labels and **99.66%** under the manually reviewed label policy.

## Best train-selected ensemble

The reported triplet averages nonlinear-probe softmax probabilities with
weights `0.556 * DINOv2-75 + 0.222 * I-JEPA-300 + 0.222 *
I-JEPA-500`. The weights and probability score space were selected using
MNIST train; all choices were frozen before test prediction artifacts were
loaded. It makes **37 errors on the 10,000 canonical test examples: 99.63%**.

Under the completed manual-review policy, which relabels eight examples and
excludes two ambiguous examples, the same frozen rule makes **34 errors among
9,998 scored examples: 99.65993%**. Exact metrics and hashes are preserved in the
[train-selected reproduction record](results/reproductions/2026-07-18-training-selected-triplets.json).

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

Run the complete LoRA backbone/probe matrix. The command follows fixed
150-epoch trajectories and reports the prespecified 50/75/100/150 milestones:

```bash
caffeinate -i uv run python scripts/analysis/train_lora_backbone_probes.py
```

Run the convolutional neural decision-stump comparison. Each model emits one
binary split and is trained only to reduce the label impurity of its two leaves;
there is no digit-classification head:

```bash
caffeinate -i uv run python scripts/analysis/train_impurity_convnet.py \
  --criteria gini,entropy --epochs 20 --batch-size 1024 --seed 0
```

Re-run the current train-selected comparison:

```bash
uv run python scripts/analysis/grid_train_selected_probe_triplets.py
```

Verify every manifest-pinned artifact:

```bash
uv run python scripts/reproduce/verify_artifacts.py
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
