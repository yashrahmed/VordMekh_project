# Replicate the train-selected MNIST triplet I-JEPA ensemble

This reproduces the pre-DINOv2 three-member linear-probe ensemble using a
test-clean selection protocol:

| selection | train errors | canonical test | reviewed test |
|---|---:|---:|---:|
| train-selected raw-logit weights | 20 / 60,000 | 99.42% (58 / 10,000) | 99.48% (52 / 9,998) |

The ensemble combines three frozen linear probes:

| component | weight |
|---|---:|
| old 28x28 bbox-preproc custom I-JEPA, 10 targets / 6 context, 500ep encoder, flatten probe | 0.278 |
| 56x56 upscaled-bbox custom I-JEPA, 48 targets / 16 context, 300ep encoder, flatten probe | 0.472 |
| 56x56 upscaled-bbox custom I-JEPA, 48 targets / 16 context, 500ep encoder, flatten probe | 0.250 |

The prediction rule is weighted logit averaging:

```python
logits = (
    0.278 * old28_500_flatten_logits
  + 0.472 * new56_300_flatten_logits
  + 0.250 * new56_500_flatten_logits
)
pred = logits.argmax(dim=1)
```

The weights are selected on all 60,000 training examples with a complete 1%
simplex grid and a local 0.1% refinement. When multiple weights have the same
training error count, the representative closest to equal weights is selected.
The test split is loaded only after this choice is frozen. This uses the probe
training split rather than a separate validation split, but it does not use
test labels for model selection.

The historical test-selected weights `0.39/0.28/0.33` reached 99.50% on the
canonical test set but made 26 training errors, versus 20 for the clean
train-selected mix. They are retained only as evidence of test-selection
optimism and are not reported model performance.

## 1. Prepare the current project

From the repository root:

```bash
cd projects/mnist-ssl
uv sync
mkdir -p models out
```

## 2. Recreate the old 28x28 probe checkpoint

The current `master` implementation in `src/mnist_ssl/ijepa/custom_ijepa.py` is the 56x56 /
64-token architecture. The old 28x28 / 16-token architecture is available at the
pre-56x56 master commit `f9a184c`, so recreate it from a temporary worktree:

```bash
cd ..  # parent directory containing the current repository
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
  ../../VordMekh_project/projects/mnist-ssl/models/
```

You can remove the temporary worktree after the checkpoint is copied:

```bash
cd ../../VordMekh_project
git worktree remove ../VordMekh_project_old28
git worktree prune
```

## 3. Recreate the 56x56 300ep and 500ep encoder checkpoints

Back on current `master`:

```bash
cd projects/mnist-ssl
```

Run the 500-epoch 56x56 training trajectory. This also saves a normal
probe-loadable 300-epoch checkpoint during the same run:

```bash
caffeinate -i uv run python scripts/train/ijepa.py \
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
caffeinate -i uv run python scripts/reproduce/ijepa_members.py
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

## 5. Select the weights on train and evaluate test

Run:

```bash
uv run python scripts/reproduce/ijepa_train_selected_triplet.py
```

Expected key lines:

```text
selected_weights=0.278/0.472/0.250 train_errors=20
canonical_errors=58 reviewed_errors=52
```

The cached training logits, full grid, and raw summary are written to:

```text
out/ijepa_train_selected_triplet_v1/training_logits.pt
out/ijepa_train_selected_triplet_v1/grid.csv
out/ijepa_train_selected_triplet_v1/summary.json
```

## 6. Optional supporting checks

Pairwise logit ensemble:

```bash
uv run python -m mnist_ssl.ijepa.ensemble_probes
```

Feature-concat linear probe:

```bash
caffeinate -i uv run python -m mnist_ssl.ijepa.concat_probe --epochs 50
```

These did not beat the triplet logit ensemble. Best pairwise logit ensemble was
99.47%, and best concat-feature probe was 99.34%.
