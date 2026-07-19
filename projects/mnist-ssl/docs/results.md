# Curated MNIST results

All linear-probe and k-NN results use frozen self-supervised backbones. The
tables below separate individual models and prespecified combinations from
weights selected directly on the MNIST test labels.

## Leaderboard

| Method | Selection | Test accuracy | Errors |
|---|---|---:|---:|
| DINOv2 + I-JEPA-300, equal logits | Prespecified weights | 99.47% | 53 |
| DINOv2 CLS, epoch 75 of fixed-150 schedule | Best individual observed | 99.42% | 58 |
| DINOv2 CLS nonlinear-64 probe | Best nonlinear individual observed | 99.52% | 48 |
| I-JEPA 56x56 flatten, epoch 300 | Best individual I-JEPA observed | 99.36% | 64 |
| I-JEPA 56x56 flatten, epoch 500 | Individual comparison | 99.34% | 66 |
| I-JEPA-500 flatten nonlinear-64 probe | Best nonlinear milestone observed | 99.42% | 58 |
| I-JEPA-only triplet | Test-tuned weights | 99.50% | 50 |
| DINOv2 + I-JEPA-300 | Test-tuned weights | 99.57% | 43 |
| DINOv2 + I-JEPA-300 + I-JEPA-500 linear probes | Test-tuned weights | 99.61% | 39 |
| DINOv2 + I-JEPA-500 nonlinear probabilities | Test-tuned weights | 99.61% | 39 |
| **DINOv2 + I-JEPA-300 + I-JEPA-500 nonlinear probabilities** | **Test-tuned weights** | **99.64%** | **36** |

All rows marked test-tuned are diagnostics demonstrating complementary errors.
Their weights were selected on the MNIST test labels, so they are not unbiased
held-out estimates. The nonlinear triplet's canonical winner averages softmax
probabilities with DINO/I-JEPA-300/I-JEPA-500 weights
`0.547/0.270/0.183`. The next rigorous experiment must choose weights on a
validation split and evaluate the test set once.

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
| Nonlinear probability triplet, canonical-selected weights | 99.67% | 33 |
| **Nonlinear probability triplet, reviewed-selected weights** | **99.68%** | **32** |

The reviewed-selected probability weights are `0.530/0.253/0.217`; they make
37 errors on the canonical labels. The canonical-selected weights make 36
canonical and 33 reviewed errors. Fourteen reviewed errors are shared by all
three nonlinear members, so the reviewed-label oracle ceiling is **99.86%**.
Direct reviewed-label selection does not make the result validation-clean.

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

The exploratory canonical-label winner averages nonlinear-probe probabilities:

```text
0.547 * softmax(DINOv2-75-CLS nonlinear logits)
+ 0.270 * softmax(I-JEPA-300-flatten nonlinear logits)
+ 0.183 * softmax(I-JEPA-500-flatten nonlinear logits)
```

It makes 36 errors for **99.64%** accuracy. Only 18 canonical errors are shared
by all three members, giving a label-oracle ceiling of **99.82%**.

Under the manually reviewed policy, the same weights make 33 errors among
9,998 included examples for **99.67%** accuracy. Separately selecting on the
reviewed labels changes the weights to `0.530/0.253/0.217` and reduces the
reviewed count to 32, while increasing the canonical count to 37. Fourteen
reviewed errors remain shared, giving a reviewed-label oracle ceiling of
**99.86%**.

The nonlinear grid used the DINO 50-epoch nonlinear head and the 75-epoch
I-JEPA-300 and I-JEPA-500 nonlinear heads. Exact input hashes, checkpoint
hashes, cross-view scores, and grid hashes are in the
[nonlinear-ensemble reproduction record](../results/reproductions/2026-07-18-nonlinear-ensembles.json).
The previous linear triplet remains preserved in its
[reproduction record](../results/reproductions/2026-07-16-best-triplet.json)
and [checkpoint manifest](../results/checkpoint-manifest.json).

Long-horizon DINOv2 tables, MAE baselines, probe alternatives, and negative
results remain available in the [experiment log](experiment-log.md).
