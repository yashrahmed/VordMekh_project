# Review the paper-validated MNIST label issues

The review queue contains only the 15 MNIST test examples validated as label
issues in:

> Curtis G. Northcutt, Anish Athalye, and Jonas Mueller. “Pervasive Label Errors
> in Test Sets Destabilize Machine Learning Benchmarks.” NeurIPS 2021 Datasets
> and Benchmarks.

The canonical publication is on
[OpenReview](https://openreview.net/forum?id=XccDXrDNLek), with the
[paper PDF](https://openreview.net/pdf?id=XccDXrDNLek). The paper sent 100
algorithmically identified MNIST candidates to five MTurk reviewers and
validated 15 examples for which fewer than three reviewers accepted the
original label. The published breakdown is 10 correctable examples, two with
no reviewer agreement, and three for which the majority selected neither
proposed label.

No model-derived errors from this project are included. The visualizer hides
the published suggestion and votes behind a disclosure control so the raw
digit can be judged first.

## Build and open the reviewer

From `projects/mnist-ssl`:

```bash
uv run python scripts/analysis/review_mnist_labels.py --open
```

The command downloads MNIST if necessary, verifies every configured original
label against the canonical torchvision test order, and writes the standalone
reviewer to `out/mnist_label_review.html`. To use an existing dataset at a
different location:

```bash
uv run python scripts/analysis/review_mnist_labels.py \
  --dataset-root /path/to/dataset \
  --no-download \
  --open
```

For each example, choose one decision:

- **Keep original label** when the supplied MNIST label is acceptable.
- **Relabel** and choose a digit when another label is clearly correct.
- **Exclude as ambiguous** when no single digit label is defensible.

Notes are optional. Decisions are saved in browser-local storage as the review
progresses. **Download decisions** exports a portable
`mnist-label-review-decisions.json`; an incomplete export remains valid and
records which examples are still unreviewed. The same file can be imported to
resume on another browser.

The exported file pins the SHA-256 identity of the exact candidate set. A later
evaluation change should reject decisions whose candidate-set identity,
original labels, or MNIST ordering do not match.

## Candidate provenance

The tracked candidate definition is
[`configs/evaluation/mnist_label_review_candidates.json`](../configs/evaluation/mnist_label_review_candidates.json).
It pins the authors' external audit annotations to
[`cleanlab/label-errors@6d5d6b31a13216290afc40e5c6319399c4d15c06`](https://github.com/cleanlab/label-errors/blob/6d5d6b31a13216290afc40e5c6319399c4d15c06/mturk/mnist_mturk.json),
records their SHA-256 digest, and applies the paper's `mturk.given < 3`
selection rule.

The published suggestions and votes are context, not automatic corrections.
The completed manual review is tracked separately in
[`configs/evaluation/mnist_label_corrections.json`](../configs/evaluation/mnist_label_corrections.json)
and is applied by the ensemble evaluators when
`--apply-known-corrections` is passed.
