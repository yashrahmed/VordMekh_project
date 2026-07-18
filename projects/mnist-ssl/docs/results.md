# Curated MNIST results

All linear-probe and k-NN results use frozen self-supervised backbones. The
tables below separate individual models and prespecified combinations from
weights selected directly on the MNIST test labels.

## Leaderboard

| Method | Selection | Test accuracy | Errors |
|---|---|---:|---:|
| DINOv2 + I-JEPA-300, equal logits | Prespecified weights | 99.47% | 53 |
| DINOv2 CLS, epoch 75 of fixed-150 schedule | Best individual observed | 99.42% | 58 |
| I-JEPA 56x56 flatten, epoch 300 | Best individual I-JEPA observed | 99.36% | 64 |
| I-JEPA 56x56 flatten, epoch 500 | Individual comparison | 99.34% | 66 |
| I-JEPA-only triplet | Test-tuned weights | 99.50% | 50 |
| DINOv2 + I-JEPA-300 | Test-tuned weights | 99.57% | 43 |
| **DINOv2 + I-JEPA-300 + I-JEPA-500** | **Test-tuned weights** | **99.61%** | **39** |

The 99.50%, 99.57%, and 99.61% rows are diagnostics demonstrating
complementary errors. Their weights were selected on the MNIST test labels, so
they are not unbiased held-out estimates. The next rigorous experiment must
choose weights on a validation split and evaluate the test set once.

## Manually reviewed label view

The original benchmark scores above remain unchanged for comparability. A
manual review of the 15 MNIST issues validated by Northcutt et al. adopted
eight relabels, excluded two ambiguous examples, and retained five original
labels. Applying that policy leaves 9,998 scored examples.
Pass `--apply-known-corrections` to the ensemble evaluator to produce this
additional view; without the flag, evaluation uses only the original labels.

| Method | Reviewed-label accuracy | Errors |
|---|---:|---:|
| DINOv2 CLS | 99.43% | 57 |
| I-JEPA 56x56 flatten, epoch 300 | 99.40% | 60 |
| I-JEPA 56x56 flatten, epoch 500 | 99.38% | 62 |
| **DINOv2 + I-JEPA-300 + I-JEPA-500** | **99.65%** | **35** |

The same 0.84/0.06/0.10 weights win the reviewed-label grid. Nineteen reviewed
errors are shared by all three members, so the label-oracle ceiling is
**99.81%**. This is four fewer ensemble errors and two fewer shared errors than
the original-label view; it does not turn the test-tuned weights into a
validation-clean result.

## Best individual DINOv2

- Augmented DINOv2, fixed 150-epoch cosine schedule.
- Milestone: epoch 75.
- Frozen EMA teacher, CLS readout, 50-epoch linear probe.
- Test accuracy: **99.42%** (58 errors).
- Frozen-backbone SHA-256:
  `518aca8f613af6c3a4e255b9b9dc80f6a9d15120b6a50658d6d374debd3495e7`.

| Backbone epoch | 5-NN | Linear train | Linear test |
|---:|---:|---:|---:|
| 50 | 99.24% | 99.57% | 99.34% |
| **75** | 99.08% | 99.64% | **99.42%** |
| 100 | 99.19% | 99.67% | 99.40% |
| 125 | 99.16% | 99.67% | 99.31% |
| 150 | 99.16% | 99.66% | 99.34% |

## Best individual I-JEPA

- Custom I-JEPA with 56x56 bbox-normalized inputs and 7x7 patches.
- 48 target and 16 context tokens.
- 300 backbone epochs; frozen flattened 50-epoch linear probe.
- Test accuracy: **99.36%** (64 errors).
- Frozen-backbone SHA-256:
  `4f944247839d3853b0ccc35f8410bc377d5fb2d9ae4693ee204560a15f7f0da0`.

## Best current ensemble

The exploratory winning logit rule is:

```text
0.84 * DINOv2-75-CLS logits
+ 0.06 * I-JEPA-300-flatten logits
+ 0.10 * I-JEPA-500-flatten logits
```

It makes 39 errors for **99.61%** accuracy. Only 21 errors are shared by all
three members, giving a label-oracle ceiling of 99.79% for this triplet.

Under the manually reviewed policy, the same weights make 35 errors among
9,998 included examples for **99.65%** accuracy. Nineteen errors remain shared,
giving a reviewed-label oracle ceiling of **99.81%**.

The result was reproduced from the preserved checkpoints on 2026-07-16 at
commit `6d79240c375285203c0892b6378d28f9b5c504cd`. Frozen-backbone fingerprints
matched before and after evaluation. See the [reproduction record](../results/reproductions/2026-07-16-best-triplet.json)
and [checkpoint manifest](../results/checkpoint-manifest.json). The reviewed
view has its own
[2026-07-18 reproduction record](../results/reproductions/2026-07-18-reviewed-label-triplet.json).

Long-horizon DINOv2 tables, MAE baselines, probe alternatives, and negative
results remain available in the [experiment log](experiment-log.md).
