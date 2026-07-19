# Curated MNIST results

All linear-probe and k-NN results use frozen self-supervised backbones. Ensemble
rows in the reported leaderboard use rules fixed without test labels;
historically observed individual milestones remain labeled as such. Mixtures
selected directly on test labels are separated into a diagnostic-ceiling
table.

## Reported leaderboard

| Method | Selection | Test accuracy | Errors |
|---|---|---:|---:|
| DINOv2 + I-JEPA-300, equal logits | Prespecified weights | 99.47% | 53 |
| DINOv2 CLS, epoch 75 of fixed-150 schedule | Best individual observed | 99.42% | 58 |
| DINOv2 CLS nonlinear-64 probe | Best nonlinear individual observed | 99.52% | 48 |
| I-JEPA 56x56 flatten, epoch 300 | Best individual I-JEPA observed | 99.36% | 64 |
| I-JEPA 56x56 flatten, epoch 500 | Individual comparison | 99.34% | 66 |
| I-JEPA-500 flatten nonlinear-64 probe | Best nonlinear milestone observed | 99.42% | 58 |
| DINOv2 + I-JEPA-300 + I-JEPA-500 linear probes | Train-selected logits | 99.45% | 55 |
| **DINOv2 + I-JEPA-300 + I-JEPA-500 nonlinear probes** | **Train-selected probabilities** | **99.63%** | **37** |

The train-selected nonlinear triplet is the reported best ensemble. Its
probability score space and `0.556/0.222/0.222` weights were frozen before test
prediction artifacts were loaded.

## Test-selected diagnostic ceilings

| Method | Test-selected accuracy | Errors |
|---|---:|---:|
| I-JEPA-only triplet | 99.50% | 50 |
| DINOv2 + I-JEPA-300 | 99.57% | 43 |
| DINOv2 + I-JEPA-300 + I-JEPA-500 linear probes | 99.61% | 39 |
| DINOv2 + I-JEPA-500 nonlinear probabilities | 99.61% | 39 |
| DINOv2 + I-JEPA-300 + I-JEPA-500 nonlinear probabilities | 99.64% | 36 |

These rows demonstrate complementary errors and upper-bound headroom. Because
their weights were selected on MNIST test labels, they are not reported model
performance.

## Training-selected ensemble weights

The scalar-mixture search was repeated using only the 60,000 MNIST training
labels. Both raw logits and softmax probabilities used a full 1% simplex grid
and a local 0.1% refinement. All method and weight choices were frozen before
test prediction artifacts were loaded.

| Probe group | Train-selected score/weights (DINO/300/500) | Train errors | Canonical test | Reviewed test |
|---|---|---:|---:|---:|
| Linear | logits, 0.248/0.541/0.211 | 22 | 99.45% (55) | 99.49% (51) |
| **Nonlinear** | **probabilities, 0.556/0.222/0.222** | **3** | **99.63% (37)** | **99.66% (34)** |

The nonlinear result is only one canonical and one reviewed error behind the
canonical test-tuned mixture. The linear search transfers poorly: its member
training errors are 219/34/75, so it places 54.1% of the weight on I-JEPA-300,
whereas DINO is the stronger test model. Exact plateau sizes and hashes are in
the [training-selected reproduction record](../results/reproductions/2026-07-18-training-selected-triplets.json).

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
| DINOv2 CLS nonlinear-64 probe | 99.55% | 45 |
| I-JEPA-300 flatten nonlinear-64 probe | 99.41% | 59 |
| I-JEPA-500 flatten nonlinear-64 probe | 99.46% | 54 |
| Linear triplet, train-selected weights | 99.49% | 51 |
| **Nonlinear probability triplet, train-selected weights** | **99.66%** | **34** |

For diagnostic context, the canonical-test-selected nonlinear weights make 33
reviewed errors, and weights selected directly on the reviewed labels make 32.
Those are upper-bound measurements rather than reported ensemble results.
Fourteen reviewed errors are shared by all three nonlinear members, giving a
label-oracle ceiling of **99.86%**.

## Top-two reranking: negative result

A compact normalized-image ConvNet was trained to choose between the frozen
DINOv2 linear probe's top two classes. The correction data used a deterministic
class-balanced 50,000/10,000 training/validation split. Both the reranker epoch
and the normalized logit-margin gate were selected only on the 10,000-example
correction validation set; test was evaluated once after freezing the choice.

Validation selected epoch 40 and threshold `0.121541038`. It reduced validation
errors from 42 to 36 with nine fixes and three regressions. On canonical test
labels it made ten fixes but introduced seven regressions, reducing errors only
from 58 to 55 (**99.45%**). The reviewed-label view similarly moved from 57 to
54 errors (**99.46%**). Even a perfect decision rule inside the selected test
gate could reach only 99.61%, because the gate exposed 19 of the 58 canonical
errors.

The reranker therefore neither approaches the 99.7% target nor provides the
desired no-regression behavior. Top-two reranking is retained as a reproducible
negative result and is not a current research direction. See the
[machine-readable record](../results/reproductions/2026-07-18-top2-reranking.json).

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

The reported ensemble averages nonlinear-probe probabilities:

```text
0.556 * softmax(DINOv2-75-CLS nonlinear logits)
+ 0.222 * softmax(I-JEPA-300-flatten nonlinear logits)
+ 0.222 * softmax(I-JEPA-500-flatten nonlinear logits)
```

The score space and weights were selected on MNIST train. They make 37
canonical errors for **99.63%** accuracy and 34 reviewed errors among 9,998
examples for **99.66%**. Only 18 canonical and 14 reviewed errors are shared by
all three members, giving oracle ceilings of **99.82%** and **99.86%**.

The nonlinear grid used the DINO 50-epoch nonlinear head and the I-JEPA-300
and I-JEPA-500 75-epoch nonlinear heads. Exact training selection, plateau
sizes, input hashes, and test metrics are in the
[train-selected reproduction record](../results/reproductions/2026-07-18-training-selected-triplets.json).
The test-selected nonlinear ceiling remains in a separate
[diagnostic record](../results/reproductions/2026-07-18-nonlinear-ensembles.json),
and the historical linear diagnostic remains in its
[reproduction record](../results/reproductions/2026-07-16-best-triplet.json)
and [checkpoint manifest](../results/checkpoint-manifest.json).

Long-horizon DINOv2 tables, MAE baselines, probe alternatives, and negative
results remain available in the [experiment log](experiment-log.md).
