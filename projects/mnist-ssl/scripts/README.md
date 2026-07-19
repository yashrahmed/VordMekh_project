# Experiment commands

Scripts are intentionally thin entry points. All reusable implementation code
lives under `src/mnist_ssl`; this directory answers “what should I run?” without
mixing orchestration into model modules.

## Train

| Script | Purpose |
|---|---|
| `train/dinov2.py` | Train or resume the MNIST-scale DINOv2 backbone |
| `train/ijepa.py` | Train the custom I-JEPA target/context model |
| `train/ijepa_probe.py` | Train a frozen or unfrozen I-JEPA classifier |
| `train/mae.py` | Train ViT, convolutional, or JEPA-style MAE baselines |

## Evaluate

| Script | Purpose |
|---|---|
| `evaluate/dinov2_frozen.py` | Frozen weighted k-NN plus linear probe |
| `evaluate/dinov2_knn.py` | Frozen weighted k-NN only |
| `evaluate/ijepa_probe.py` | Evaluate a saved I-JEPA classifier |
| `evaluate/knn.py` | k-NN for MAE and handcrafted baselines |

## Reproduce

| Script | Purpose |
|---|---|
| `reproduce/verify_artifacts.py` | Verify manifest-pinned checkpoint file hashes |
| `reproduce/best_ensemble.py` | Reproduce the DINO/I-JEPA triplet; add `--apply-known-corrections` for reviewed labels |
| `reproduce/ijepa_members.py` | Rebuild the 300/500-epoch 56x56 I-JEPA members |
| `reproduce/ijepa_9950.py` | Verify the historical 99.50% I-JEPA-only triplet |

The best-ensemble command reads `configs/best/dino_ijepa_triplet.json`, resolves
its checkpoint IDs through `results/checkpoint-manifest.json`, verifies the
files, and fails if the recorded metrics drift.

## Sweeps and analysis

`sweeps/` contains deliberate multi-run searches. `analysis/` contains
post-training inspection and visualization tools. New one-off commands should
go into one of these categories rather than `out/`, which is reserved for
generated artifacts.

| Script | Purpose |
|---|---|
| `analysis/review_mnist_labels.py` | Build the standalone manual label reviewer |
| `analysis/ijepa_errors.py` | Render errors for a saved I-JEPA probe |
| `analysis/shift_subtract.py` | Inspect the shift/subtract image transform |
| `analysis/train_dino_nonlinear_probe.py` | Compare a small nonlinear head with the frozen DINO linear probe |
| `analysis/train_ijepa_nonlinear_probe.py` | Compare the matched small nonlinear head on the best frozen I-JEPA member |

Do not add a second implementation to `scripts/`: put reusable code in
`src/mnist_ssl/` and expose it through a thin entry point here. Checkpoint and
output retention is defined in [`docs/artifact-policy.md`](../docs/artifact-policy.md).
