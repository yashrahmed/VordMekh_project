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
| `reproduce/ijepa_members.py` | Rebuild the 300/500-epoch 56x56 I-JEPA members |
| `reproduce/ijepa_train_selected_triplet.py` | Select the three-I-JEPA linear-probe mixture on MNIST train, then evaluate test once |

The reported current ensemble is selected by
`analysis/grid_train_selected_probe_triplets.py`.

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
| `analysis/train_ijepa_nonlinear_probe.py` | Compare the matched small nonlinear head on a specified frozen 300- or 500-epoch I-JEPA member |
| `analysis/train_impurity_convnet.py` | Train one two-convolution binary splitter to reduce Gini or Shannon impurity across two leaves |
| `analysis/train_impurity_tree_depth_two.py` | Freeze each original impurity stump and independently train one child splitter on each routed leaf |
| `analysis/grid_dino_ijepa500_nonlinear_ensemble.py` | Grid raw-logit and probability mixtures of the best DINO and I-JEPA-500 nonlinear probes |
| `analysis/grid_dino_ijepa_nonlinear_triplet.py` | Coarse-to-fine raw-logit and probability grids over all three nonlinear probes |
| `analysis/calibrate_nonlinear_triplet.py` | Fit training-only temperatures and class-specific diagonal weights over the three nonlinear probes |
| `analysis/grid_train_selected_probe_triplets.py` | Select scalar linear- and nonlinear-probe triplet mixtures on train, then evaluate test once |
| `analysis/train_dino_pairwise_reranker.py` | Cross-fit a linear top-two scorer and select its blend without test tuning |
| `analysis/train_dino_nonlinear_pairwise_reranker.py` | Cross-fit the matched nonlinear top-two scorer |
| `analysis/train_dino_normalized_image_reranker.py` | Train an image-only pairwise reranker behind a fixed linear-probe margin gate |
| `analysis/eval_dino_normalized_image_reranker.py` | Evaluate the frozen normalized-image reranker without test-time tuning |
| `analysis/sweep_dino_normalized_image_reranker_threshold.py` | Select a margin gate by sweeping a frozen normalized-image reranker |
| `analysis/train_dino_normalized_image_reranker_split.py` | Train the reranker on 50k examples and select its epoch and gate on a held-out 10k |
| `analysis/eval_dino_normalized_image_reranker_split.py` | Evaluate the validation-selected split reranker once on the test set |

Do not add a second implementation to `scripts/`: put reusable code in
`src/mnist_ssl/` and expose it through a thin entry point here. Checkpoint and
output retention is defined in [`docs/artifact-policy.md`](../docs/artifact-policy.md).
