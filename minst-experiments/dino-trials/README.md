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
tiny digits. Brightness and contrast replace RGB color jitter, and horizontal
flips are disabled because they are not label-preserving for digits. The iBOT
mask interval (10%-50%),
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

From `minst-experiments`:

```bash
uv sync
uv run python dino-trials/train.py \
  --epochs 100 --checkpoint-epochs 50,75,100 --checkpoint-every 50

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
`..._resume.pt` is replaced every 50 epochs and removed only after successful
completion. If a run is interrupted after a rolling save, resume the same
schedule and output with:

```bash
uv run python dino-trials/train.py \
  --epochs 100 --checkpoint-epochs 50,75,100 --checkpoint-every 50 \
  --resume models/dinov2_mnist_preproc_resume.pt
```

Every `.pt` checkpoint contains both networks, separate DINO/iBOT head weights,
teacher centers, AdamW optimizer state, completed epoch/global step, RNG state,
the full configuration, and epoch metrics. This makes both milestone and
rolling checkpoints resumable. The adjacent JSON files contain configuration
and metrics without tensors. For downstream features, instantiate
`StudentTeacher` from the stored configuration, load `teacher_backbone`, and
use `encode`.

`eval_knn.py` applies the same deterministic preprocessing to both the MNIST
reference and test splits. It follows the checkpoint's saved `preprocess` value
by default and warns when `--preprocess` or `--no-preprocess` overrides it.

### Verified smoke run

The command above was run on a 512-image subset (8 batches per epoch) using the
default 3,254,080-parameter student. It completed two epochs with finite losses
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
