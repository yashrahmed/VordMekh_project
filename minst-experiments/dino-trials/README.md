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
- A shared normalized prototype head for the class and patch objectives.
- KoLeo entropy regularization on each global view's student class tokens.
- AdamW, gradient clipping, last-layer freezing, and cosine learning-rate,
  weight-decay, and teacher-momentum schedules.

## MNIST configuration

The original recipe targets hundreds of millions of RGB images and very large
ViTs. For MNIST, global crops are 56x56 and local crops 28x28, with 7x7 patches
(8x8 and 4x4 token grids). The default encoder has dimension 192, six blocks,
six heads, and 1,024 prototypes. Four local crops replace the paper's eight to
avoid excessive duplicate content and compute on tiny digits. Brightness and
contrast replace RGB color jitter, and horizontal flips are disabled because
they are not label-preserving for digits. The iBOT mask interval (10%-50%),
masking probability (50%), loss weights, temperatures, KoLeo weight, and the
0.994-to-1.0 teacher momentum schedule follow the paper/repository recipe.

Registers are deliberately absent: they were introduced in the later *Vision
Transformers Need Registers* work and are not part of the original DINOv2
paper's training architecture.

## Run

From `minst-experiments`:

```bash
uv sync
uv run python dino-trials/train.py --epochs 100
```

A quick end-to-end verification run is:

```bash
uv run python dino-trials/train.py \
  --epochs 2 --subset 512 --batch-size 64 --workers 0 \
  --output models/dinov2_mnist_smoke.pt
```

The checkpoint contains both networks, the full configuration, and epoch
metrics; the adjacent JSON file contains the configuration and metrics without
model weights. For downstream features, instantiate `StudentTeacher` from the
stored configuration, load `teacher_backbone`, and use `encode`.

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
