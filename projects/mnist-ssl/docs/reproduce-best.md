# Reproduce the best DINOv2 + I-JEPA ensemble

The reported best ensemble combines the frozen nonlinear probes from DINOv2,
I-JEPA-300, and I-JEPA-500. Its probability score space and weights were
selected on MNIST train, then frozen before test prediction artifacts were
loaded:

| Member | Probability weight | Individual test accuracy |
|---|---:|---:|
| Augmented DINOv2 epoch-75 CLS nonlinear-50 | 0.556 | 99.52% |
| 56x56 I-JEPA-300 flattened nonlinear-75 | 0.222 | 99.35% |
| 56x56 I-JEPA-500 flattened nonlinear-75 | 0.222 | 99.42% |

The frozen mixture makes 37 errors on canonical test labels: **99.63%**. Under
the manual-review policy it makes 34 errors over 9,998 examples:
**99.65993%**.

## Reproduce from existing checkpoints

From the repository root:

```bash
cd projects/mnist-ssl
uv sync
```

The run needs the three preserved nonlinear-probe prediction artifacts and
heads, plus their frozen backbone/linear-probe checkpoints. Exact file hashes
are recorded in
[`2026-07-18-training-selected-triplets.json`](../results/reproductions/2026-07-18-training-selected-triplets.json).
Run the train-selected grids into a fresh output directory:

```bash
uv run python scripts/analysis/grid_train_selected_probe_triplets.py \
  --output-dir out/training_selected_probe_triplets_reproduction
```

Expected summary:

```text
group=linear method=logit weights=0.248/0.541/0.211 train_errors=22
group=nonlinear method=probability weights=0.556/0.222/0.222 train_errors=3
group=linear selected_method=logit canonical_errors=55 reviewed_errors=51
group=nonlinear selected_method=probability canonical_errors=37 reviewed_errors=34
```

The script searches both probe groups using only training logits. It writes the
full grid under `out/`; compare the emitted training-logit, grid, and input
hashes with the tracked reproduction record.

## Rebuild the I-JEPA members

This single seed-0 trajectory saves both the 300- and 500-epoch target encoder
milestones and then trains the corresponding frozen probes:

```bash
caffeinate -i uv run python scripts/reproduce/ijepa_members.py
```

The runner uses 56x56 bbox-normalized MNIST, 7x7 patches, 48 target tokens, 16
context tokens, and 50-epoch flattened linear probes. Rebuilt checkpoints must
be evaluated and fingerprinted before the manifest is updated.

Train the matched nonlinear heads without modifying the frozen encoders:

```bash
uv run python scripts/analysis/train_ijepa_nonlinear_probe.py
uv run python scripts/analysis/train_ijepa_nonlinear_probe.py \
  --linear-probe models/ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base500ep_probe50ep.pt \
  --pretraining-epochs 500 \
  --output-dir out/ijepa_nonlinear_probe_best500
```

## Rebuild the DINOv2 member

Launch the final horizon as 150 epochs from the start; the cosine schedules are
defined by this horizon and cannot be reproduced by extending a shorter run:

```bash
caffeinate -i uv run python scripts/train/dinov2.py \
  --epochs 150 \
  --seed 0 \
  --checkpoint-epochs 50,75,100,125,150 \
  --checkpoint-every 10 \
  --photometric-augmentations \
  --output models/dinov2_mnist_augmented_cls_150ep.pt
```

Train the frozen epoch-75 CLS probe:

```bash
caffeinate -i uv run python scripts/evaluate/dinov2_frozen.py \
  --model models/dinov2_mnist_augmented_cls_150ep_epoch0075.pt \
  --pool cls \
  --linear-epochs 50 \
  --output models/dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt
```

Train the matched nonlinear head:

```bash
uv run python scripts/analysis/train_dino_nonlinear_probe.py
```

Finally, rerun the train-selected comparison. Stochastic retraining may not
reproduce the exact same checkpoint bytes across PyTorch or hardware versions;
the required sanity check is the individual metrics, frozen-backbone
invariants, selected training weights, and final ensemble score, with
environment differences recorded alongside the result.

The older I-JEPA-only triplet requires a historical 28x28 member. Its
train-selected weights reach 99.42%; see its
[reproduction guide](reproduce-ijepa-9950.md).
