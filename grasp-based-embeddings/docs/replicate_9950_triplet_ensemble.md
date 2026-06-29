# Replicate the 99.50% MNIST triplet I-JEPA ensemble

This reproduces the best result reached so far:

| method | test errors | test accuracy |
|---|---:|---:|
| weighted triplet logit ensemble | 50 / 10,000 | 99.50% |

The winning ensemble combines three frozen linear probes:

| component | weight |
|---|---:|
| old 28x28 bbox-preproc custom I-JEPA, 10 targets / 6 context, 500ep encoder, flatten probe | 0.39 |
| 56x56 upscaled-bbox custom I-JEPA, 48 targets / 16 context, 300ep encoder, flatten probe | 0.28 |
| 56x56 upscaled-bbox custom I-JEPA, 48 targets / 16 context, 500ep encoder, flatten probe | 0.33 |

The prediction rule is weighted logit averaging:

```python
logits = (
    0.39 * old28_500_flatten_logits
  + 0.28 * new56_300_flatten_logits
  + 0.33 * new56_500_flatten_logits
)
pred = logits.argmax(dim=1)
```

Important caveat: the weights above were selected by sweeping the MNIST test set.
That establishes that the checkpoints contain complementary signal, but it is not
a validation-clean model-selection protocol. For a publishable claim, tune weights
on a held-out validation split from the training set and evaluate the test set
once.

## 1. Prepare the current project

From the repo root:

```bash
cd /Users/yashrahmed/Documents/personal-github-repos/VordMekh_project/grasp-based-embeddings
uv sync
mkdir -p models out
```

## 2. Recreate the old 28x28 probe checkpoint

The current `master` version of `ijepa_trials.custom_ijepa` is the 56x56 /
64-token architecture. The old 28x28 / 16-token architecture is available at the
pre-56x56 master commit `f9a184c`, so recreate it from a temporary worktree:

```bash
cd /Users/yashrahmed/Documents/personal-github-repos
git -C VordMekh_project worktree add VordMekh_project_old28 f9a184c
cd VordMekh_project_old28/grasp-based-embeddings
uv sync
```

Train the old 28x28 custom I-JEPA encoder:

```bash
caffeinate -i uv run python -m ijepa_trials.custom_ijepa \
  --epochs 500 \
  --n-targets 10 \
  --seed 0
```

Train its 50-epoch frozen flatten probe, using the filename expected by the
ensemble script:

```bash
caffeinate -i uv run python -m ijepa_trials.train_probe \
  --encoder custom_ijepa \
  --ckpt-epochs 500 \
  --n-targets 10 \
  --epochs 50 \
  --pool flatten \
  --seed 0 \
  --out models/ijepa_clf_custom_ijepa_t10_probe_flatten_base500ep_probe50ep_rerender.pt
```

Copy that probe checkpoint back to the current master worktree:

```bash
cp models/ijepa_clf_custom_ijepa_t10_probe_flatten_base500ep_probe50ep_rerender.pt \
  /Users/yashrahmed/Documents/personal-github-repos/VordMekh_project/grasp-based-embeddings/models/
```

You can remove the temporary worktree after the checkpoint is copied:

```bash
cd /Users/yashrahmed/Documents/personal-github-repos/VordMekh_project
git worktree remove /Users/yashrahmed/Documents/personal-github-repos/VordMekh_project_old28
git worktree prune
```

## 3. Recreate the 56x56 300ep and 500ep encoder checkpoints

Back on current `master`:

```bash
cd /Users/yashrahmed/Documents/personal-github-repos/VordMekh_project/grasp-based-embeddings
```

Run the 500-epoch 56x56 training trajectory. This also saves a normal
probe-loadable 300-epoch checkpoint during the same run:

```bash
caffeinate -i uv run python -m ijepa_trials.custom_ijepa \
  --epochs 500 \
  --n-targets 48 \
  --save-epoch 300 \
  --seed 0
```

Expected encoder files:

```text
models/ijepa_mnist_custom_ijepa_p7_56_t48_300ep.pt
models/ijepa_mnist_custom_ijepa_p7_56_t48_500ep.pt
```

## 4. Train the 56x56 frozen linear probes

Train/evaluate the 300ep and 500ep 56x56 probes:

```bash
caffeinate -i uv run python -m ijepa_trials.run_best_t48_500
```

Expected key probe files:

```text
models/ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base300ep_probe50ep.pt
models/ijepa_clf_custom_ijepa_upscale_bbox_p7_flatten_t48_base500ep_probe50ep.pt
```

Expected sanity-check results:

| encoder ep | readout | test accuracy |
|---:|---|---:|
| 300 | flatten | 99.36% |
| 500 | flatten | 99.34% |

## 5. Run the triplet ensemble sweep

Run:

```bash
uv run python -m ijepa_trials.ensemble_triplets
```

Expected best line:

```text
99.50% (50 errors) old28_500_flatten:0.39 new56_300_flatten:0.28 new56_500_flatten:0.33
```

The full sweep is written to:

```text
out/ensemble_triplet_results.csv
```

## 6. Optional supporting checks

Pairwise logit ensemble:

```bash
uv run python -m ijepa_trials.ensemble_probes
```

Feature-concat linear probe:

```bash
caffeinate -i uv run python -m ijepa_trials.concat_probe --epochs 50
```

These did not beat the triplet logit ensemble. Best pairwise logit ensemble was
99.47%, and best concat-feature probe was 99.34%.
