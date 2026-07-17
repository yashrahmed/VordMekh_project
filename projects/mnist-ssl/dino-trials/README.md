# DINOv2 on MNIST, from scratch

This experiment is a compact implementation of the training method in
**DINOv2: Learning Robust Visual Features without Supervision**. It uses only
PyTorch and Torchvision primitives; no DINO, DINOv2, iBOT, timm, or other
algorithm implementation is imported.

## What is retained from DINOv2

- ViT patch encoder with a class token, learned positions, pre-norm attention,
  SwiGLU feed-forward layers, LayerScale, and stochastic depth.
- Identically initialized student and stop-gradient EMA teacher.
- Two global views plus multiple local views.
- Cross-view DINO class-token loss with teacher centering and sharpening.
- Same-view iBOT loss on block-masked student patch tokens; the teacher sees the
  corresponding unmasked global image.
- Independent normalized prototype heads for the DINO class-token and iBOT
  patch-token objectives, matching the final paper recipe.
- KoLeo entropy regularization on each global view's student class tokens.
- AdamW, gradient clipping, last-layer freezing, and cosine learning-rate,
  weight-decay, and teacher-momentum schedules.

## MNIST configuration

The original recipe targets hundreds of millions of RGB images and very large
ViTs. For MNIST, global crops are 56x56 and local crops 28x28, with 7x7 patches
(8x8 and 4x4 token grids). The default encoder matches the custom I-JEPA scale:
dimension 128, four blocks, four heads, and 1,024 prototypes. Four local crops
replace the paper's eight to avoid excessive duplicate content and compute on
tiny digits. The crop-only ablation is now the default: random resized crops
remain enabled, while brightness/contrast jitter, Gaussian blur, and
solarization are disabled. Pass `--photometric-augmentations` to recover the
previous DINOv2-inspired photometric pipeline. Horizontal flips remain disabled
because they are not label-preserving for digits. The iBOT mask interval (10%-50%),
masking probability (50%), loss weights, temperatures, KoLeo weight, and the
0.994-to-1.0 teacher momentum schedule follow the paper/repository recipe.

The custom-I-JEPA preprocessing pipeline is enabled by default. Each raw MNIST
image is converted to a tensor, upscaled from 28x28 to 56x56, cropped to the
tight nonzero digit bounding box, and stretched back to 56x56. DINO global and
local views are sampled only after this common preprocessing step. Use
`--no-preprocess` for a raw-image ablation.

Registers are deliberately absent: they were introduced in the later *Vision
Transformers Need Registers* work and are not part of the original DINOv2
paper's training architecture.

## Run

From `projects/mnist-ssl`:

```bash
uv sync
uv run python dino-trials/train.py \
  --epochs 100 --checkpoint-epochs 50,75,100 --checkpoint-every 10

# Reproduce the earlier augmented baseline
uv run python dino-trials/train.py \
  --epochs 100 --checkpoint-epochs 50,75,100 --checkpoint-every 10 \
  --photometric-augmentations

# Frozen-teacher weighted k-NN evaluation; preprocessing is read from checkpoint
uv run python dino-trials/eval_knn.py \
  --model models/dinov2_mnist_preproc.pt --k 5
```

A quick end-to-end verification run is:

```bash
uv run python dino-trials/train.py \
  --epochs 2 --subset 512 --batch-size 64 --workers 0 \
  --output models/dinov2_mnist_smoke.pt
```

The 100-epoch command is one continuous run. It preserves
`dinov2_mnist_preproc_epoch0050.pt`, `..._epoch0075.pt`, and
`..._epoch0100.pt`, as well as the final `dinov2_mnist_preproc.pt`. A rolling
`..._resume.pt` is replaced every 10 epochs and removed only after successful
completion. If a run is interrupted after a rolling save, resume the same
schedule and output with:

```bash
uv run python dino-trials/train.py \
  --epochs 100 --checkpoint-epochs 50,75,100 --checkpoint-every 10 \
  --resume models/dinov2_mnist_preproc_resume.pt
```

The target epoch horizon is part of the saved schedule and must match on
resume. Extending a completed 100-epoch checkpoint by changing `--epochs` would
reinterpret its cosine progress and is therefore rejected. Start a fresh run
with the final horizon fixed at launch, or implement an explicit second-phase
schedule instead.

To make a planned evaluation decision partway through a fixed schedule, use
`--stop-after-epoch` without changing `--epochs`. For example, this pauses at
epoch 300 while retaining the exact 500-epoch cosine schedule:

```bash
uv run python dino-trials/train.py \
  --epochs 500 --stop-after-epoch 300 \
  --checkpoint-epochs 50,75,100,125,150,300,500 --checkpoint-every 10 \
  --output models/dinov2_mnist_fixed500.pt
```

After evaluating the epoch-300 milestone, continue with the same `--epochs 500`
configuration and `--resume models/dinov2_mnist_fixed500_epoch0300.pt`. A
planned pause writes the rolling checkpoint and does not create the final model
or remove resumable state.

Every `.pt` checkpoint contains both networks, separate DINO/iBOT head weights,
teacher centers, AdamW optimizer state, completed epoch/global step, RNG state,
the full configuration, explicit LR/weight-decay/teacher schedule state, and
epoch metrics. Resume validates the saved target, steps per epoch, warmups, and
next schedule step before continuing. Version-2 checkpoints are supported by
inferring this state only for the same target horizon. This makes both milestone
and rolling checkpoints resumable. The adjacent JSON files contain
configuration, schedule state, and metrics without tensors. For downstream
features, instantiate
`StudentTeacher` from the stored configuration, load `teacher_backbone`, and
use `encode`.

`eval_knn.py` applies the same deterministic preprocessing to both the MNIST
reference and test splits. It follows the checkpoint's saved `preprocess` value
by default and warns when `--preprocess` or `--no-preprocess` overrides it.

`eval_frozen.py` caches frozen EMA-teacher features, evaluates weighted 5-NN,
trains a resumable linear probe, and fingerprints the backbone before and after
probe training. It supports `--pool cls`, `--pool mean`, and `--pool concat`:

```bash
uv run python dino-trials/eval_frozen.py \
  --model models/dinov2_mnist_preproc_epoch0100.pt \
  --pool cls --output models/dinov2_cls_base100ep_linear50ep.pt
```

## Crop-only results

### 10-epoch pilot

A seed-0 MPS run trained on all 60,000 MNIST training images with two global
and four local random resized crops. Brightness/contrast jitter, Gaussian blur,
and solarization were disabled; bbox preprocessing, normalization, iBOT masks,
and all loss/schedule settings remained enabled. The run completed 4,680 steps,
all student/teacher checkpoint tensors were finite, and its final metrics were:

| epoch | total pretext loss | DINO | iBOT | KoLeo |
|---:|---:|---:|---:|---:|
| 10 | 6.8937 | 4.9252 | 1.9835 | -0.1499 |

The full-state artifacts are `models/dinov2_mnist_crop_only_10ep.pt` and
`models/dinov2_mnist_crop_only_10ep_epoch0010.pt`.

### 100-epoch run and frozen CLS evaluation

The matched seed-0 run completed all 46,800 steps and preserved full-state
checkpoints at epochs 50, 75, and 100. Frozen evaluation used the EMA teacher's
official-style CLS readout, weighted 5-NN, and a 50-epoch linear probe. Every
backbone SHA-256 fingerprint matched before and after probe training.

| pretrain epoch | pretext loss | weighted 5-NN | linear train | linear test |
|---:|---:|---:|---:|---:|
| 50 | 2.6651 | 98.83% | 99.44% | 99.19% |
| 75 | 2.4730 | 99.01% | 99.59% | 99.25% |
| 100 | 2.3854 | 99.09% | 99.60% | 99.32% |

All saved student and teacher backbone tensors were finite. The rolling
checkpoint was removed after successful completion. The controlled comparison
against the earlier augmented run is recorded below.

## Augmented 100-epoch MNIST result

One continuous seed-0 MPS run using the earlier photometric pipeline saved full
training-state checkpoints at epochs 50, 75, and 100. The frozen evaluation
below used mean-pooled EMA-teacher patch tokens and a 50-epoch linear head; the
backbone fingerprint was identical before and after every evaluation.

| pretrain epoch | pretext loss | weighted 5-NN | linear train | linear test |
|---:|---:|---:|---:|---:|
| 50 | 3.3547 | 97.31% | 97.93% | 97.81% |
| 75 | 2.9480 | 97.62% | 98.11% | 98.13% |
| 100 | 2.8391 | 97.74% | 98.09% | 98.16% |

The downstream gains flattened between epochs 75 and 100 even though the
self-supervised loss continued to fall. These are mean-patch results, not the
official DINO CLS readout; CLS-plus-mean evaluation remains a follow-up.

### Controlled augmentation and pooling comparison

Both seed-0 training configurations and all evaluation hyperparameters match;
the only training configuration difference is whether brightness/contrast
jitter, Gaussian blur, and solarization are enabled. Every one of the 12 frozen
evaluations preserved its backbone fingerprint.

| training | readout | epoch | weighted 5-NN | linear test |
|---|---|---:|---:|---:|
| augmented | mean patches | 50 | 97.31% | 97.81% |
| augmented | mean patches | 75 | 97.62% | 98.13% |
| augmented | mean patches | 100 | 97.74% | 98.16% |
| augmented | CLS | 50 | 98.98% | **99.32%** |
| augmented | CLS | 75 | 99.19% | 99.30% |
| augmented | CLS | 100 | **99.20%** | 99.28% |
| crop-only | mean patches | 50 | 95.88% | 97.08% |
| crop-only | mean patches | 75 | 96.43% | 97.20% |
| crop-only | mean patches | 100 | 96.52% | 97.17% |
| crop-only | CLS | 50 | 98.83% | 99.19% |
| crop-only | CLS | 75 | 99.01% | 99.25% |
| crop-only | CLS | 100 | 99.09% | **99.32%** |

Removing photometric augmentation has little effect on CLS quality: augmented
training leads by 0.11-0.18 percentage points in 5-NN, while linear-test scores
are within 0.13 points and crop-only is ahead by 0.04 points at epoch 100. The
patch-token average is much more augmentation-sensitive: augmented training
leads by 1.19-1.43 points in 5-NN and 0.73-0.99 points in linear-test accuracy.

### Verified smoke run

The two-epoch smoke command was run on a 512-image subset (8 batches per epoch)
using the default 3,254,080-parameter student. It completed two epochs with
finite losses
and finite student/teacher checkpoint weights. The iBOT term changed from
3.4010 to 3.3017 and KoLeo from 8.1622 to 4.2933. The DINO term changed from
6.2564 to 6.8687 while the teacher temperature warmed from 0.04 to 0.07; these
raw cross-entropy values are therefore not directly comparable as a convergence
curve. The final student and EMA-teacher class-token parameters were distinct
(mean absolute difference 0.00156), confirming that optimization and the EMA
path both executed rather than merely completing a forward pass.

## Sources studied

- Paper: <https://arxiv.org/abs/2304.07193>
- Official implementation: <https://github.com/facebookresearch/dinov2>

The implementation follows the method rather than copying the official
distributed/xFormers code, which is unnecessary for this single-device MNIST
experiment.
