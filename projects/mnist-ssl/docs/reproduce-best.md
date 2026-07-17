# Reproduce the best DINOv2 + I-JEPA ensemble

The current exploratory best combines the frozen logits of one DINOv2 probe
and two I-JEPA probes:

| Member | Weight | Individual accuracy |
|---|---:|---:|
| Augmented DINOv2, fixed-150 epoch-75 EMA-teacher CLS probe | 0.84 | 99.42% |
| 56x56 I-JEPA, 300-epoch target encoder, flattened probe | 0.06 | 99.36% |
| 56x56 I-JEPA, 500-epoch target encoder, flattened probe | 0.10 | 99.34% |

The weighted-logit triplet makes 39 errors on the 10,000-example MNIST test
set: **99.61%**.

The weights were selected by a one-percent grid evaluated directly on test
labels. This is an exploratory measurement of complementary signal, not a
validation-clean model-selection result.

## Fast verification from existing checkpoints

From the repository root:

```bash
cd projects/mnist-ssl
uv sync
```

Verify that the five required local artifacts match
[`results/checkpoint-manifest.json`](../results/checkpoint-manifest.json):

```bash
shasum -a 256 \
  models/dinov2_mnist_augmented_cls_150ep_epoch0075.pt \
  models/dinov2_mnist_augmented_cls_150ep_epoch0075_cls_linear50ep.pt \
  models/ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base300ep_probe50ep.pt \
  models/ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base500ep_probe50ep.pt
```

Run the frozen evaluation and weight grid:

```bash
uv run python scripts/reproduce/best_ensemble.py --workers 0
```

Expected summary:

```text
dino: 99.42% (58 errors)
ijepa_300: 99.36% (64 errors)
ijepa_500: 99.34% (66 errors)
best: 99.61% (39 errors), DINO=0.84, I-JEPA-300=0.06, I-JEPA-500=0.10
all_three_shared_errors=21 oracle=99.79%
```

The evaluator fingerprints every frozen backbone before and after inference and
fails if any fingerprint changes. The full weight grid is written under
`out/`; the compact audited reproduction is tracked under
[`results/reproductions/`](../results/reproductions/).

## Rebuild the I-JEPA members

This single seed-0 trajectory saves both the 300- and 500-epoch target encoder
milestones and then trains the corresponding frozen probes:

```bash
caffeinate -i uv run python scripts/reproduce/ijepa_members.py
```

The runner uses 56x56 bbox-normalized MNIST, 7x7 patches, 48 target tokens, 16
context tokens, and 50-epoch flattened linear probes. Rebuilt checkpoints must
be evaluated and fingerprinted before the manifest is updated.

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

Finally, rerun the triplet command. Stochastic retraining may not reproduce the
exact same checkpoint bytes across PyTorch or hardware versions; the required
sanity check is the individual metrics, frozen-backbone invariants, and final
ensemble score, with environment differences recorded alongside the result.

The older I-JEPA-only 99.50% triplet requires a historical 28x28 member and has
its own [reproduction guide](reproduce-ijepa-9950.md).
