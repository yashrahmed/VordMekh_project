# mae-patch-embd

A small, modern **Masked Autoencoder** (ViT-style, He et al. 2021) trained on
MNIST. Patchify the image, drop ~75% of the patches, encode only the visible
ones with a Transformer, then reconstruct the missing pixels from the visible
tokens plus learned mask tokens. The loss is the MSE on the masked patches only.

This is a [uv](https://docs.astral.sh/uv/) project.

```bash
uv sync                                            # create venv + install deps
uv run python -m mae_patch_embd.mae                # train, defaults (10 epochs)
uv run python -m mae_patch_embd.mae --epochs 50 --mask-ratio 0.85
```

MNIST is downloaded to `dataset/` and the trained weights are written to
`models/mae_mnist.pt` (both gitignored).

## Layout

- `mae_patch_embd/mae.py` — patchify/unpatchify, the MAE model, and the
  training loop. Auto-selects CUDA → MPS → CPU.
