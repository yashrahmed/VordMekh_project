# Grasp-based embeddings

## Goal - The only experimental project in the current track.

## The idea

Generate labeled data describing the **local shape** of 2D objects, where the
label is what a "feeling hand" would sense at a given pose.

### The feeling hand
- The hand has **5 fingers**. Each finger extends in a **straight line** up to a
  maximum length `L_max`.
- A finger's extension **stops on contact** with a shape edge; its measured
  value is the distance traveled until contact.
- The hand has a **position `p`** and a **heading `h`** (orientation). Fingers
  are arranged in the hand's local frame and rotate/translate with it.
- The hand can be placed in **any position and orientation**.
- Fingers start **fully retracted**.
- The hand may **collide** with / overlap the shape. If a finger's origin starts
  **inside** the shape, its extension length is `0`.
- Units are arbitrary — everything is relative.

## Relation to Signed Distance Fields

This is the **ray-based dual of a signed distance function (SDF)**. With `f(x)`
the SDF (`f = 0` on the boundary, `< 0` inside), a finger `(O, D)` returns the
first positive root of `f(O + tD) = 0` — i.e. each finger reading is a
**ray-march (sphere trace) of the shape's SDF along one beam**. So generating the
labels is exactly "ray-march the SDF along 5 rays," and a net that predicts
`(O, D) → ℓ` is learning a *directed distance field* (cf. PRIF, DeepSDF's
ray-based cousins). DeepSDF : SDF :: the planned net : a directed distance field.


## What I wish to test.
1. Learning Patch embeddings.
   1. [ ] Train an auto encoder with handcrafted BRIEF.
   2. [ ] Train a classifier with brief.
   3. [ ] Test shape descriptors on MNIST and measure similarity.
   4. [x] Train a VIT/MAE on MNIST and measure similarity. 
   5. [x] Train a VIT/MAE on MNIST and measure classification accuracy after finetuning.
   6. [ ] Train a Conv-net MAE on MNIST and measure similarity. 
   7. [ ] Train a Conv-net MAE on MNIST and measure classification accuracy after finetuning.
   8. [ ] Train an I-JEPA on MNIST and measure similarity. 
   9. [ ] Train an I-JEPA on MNIST and measure classification accuracy (maybe after finetuning?).
2. BRIEF like features.
   1. A brief feature that moves. Similar to a hand but with a single finger.
   2. Imagine that this moves from one point to another sampling values. And there are two of these. Then shape descriptors could be used to find correspondence and similarity perhaps.
   3. What is unclear is correspondence is by shape descriptor value but how do I account for the value of the feature "head" i.e. the sum of the area where the value is sampled?
3. Can shape descriptors meant for 3-d meshes be extended to 5d point where a point is (x, y, dr, dg, db)? Worth looking into. --- Use r,g,b as "locations" and connect adjacent points to create the mesh. The position dimensions are not necessary. r,g,b as locations may not work as it creates a risk of the mesh folding in weird ways as color does not have spatil separation gaurantees. But I think it is worth try with a single channel image.


## Work log
### 2026-06-20
- Built a ViT-style **Masked Autoencoder** (`mae_patch_embd/mae.py`): patchify,
  75% random masking, encode visible patches only, decode with mask tokens,
  MSE loss on masked patches. Trained on MNIST (50 epochs).
- Wrote a nearest-neighbor **retrieval** demo (`mae_patch_embd/retrieve.py`):
  samples 5 images per class (0-9), picks a random query, embeds with the
  (unmasked) encoder, and shows the query + top-3 cosine matches. Trained
  encoder retrieves the right class; a random encoder (`--no-model-init`)
  collapses to ~0.98 cosine for everything (item 1.4 — similarity).
- Wrote a **classification** eval (`mae_patch_embd/classify.py`): linear head on
  the encoder, frozen by default (linear probe) or fine-tuned with `--unfreeze`;
  reports train/test error. The encoder is the ViT pretrained by MAE; the head
  is the only thing that trains in probe mode (item 1.5 — classification).

  Results (MNIST, mean-pooled tokens -> linear head):
  | setup | encoder | train acc | test acc |
  |---|---|---|---|
  | linear probe (50 ep) | trained, frozen | 97.3% | 97.4% |
  | linear probe (50 ep) | random (`--no-model-init`) | 59.4% | 59.1% |
  | fine-tune (10 ep) | trained, unfrozen | 99.7% | 98.8% |
  | fine-tune (50 ep, wd=0) | trained, unfrozen | 99.8% | 98.7% |
  | fine-tune (50 ep, wd=0.05) | trained, unfrozen | 99.7% | 98.7% |

  Notes: weight decay (now default 0.05) only slightly narrows the train/test
  gap — the model isn't overfitting much. Test accuracy plateaus ~98.7-98.8%
  and stays below CNN-level MNIST SOTA (~99.3%+). The cap is architectural, not
  capacity: a plain ViT has no conv inductive bias (data-hungry on small
  images), 7x7 patches give only 16 coarse tokens, and mean-pooling is a lossy
  readout. Next levers: conv stem / smaller patches / a CLS token (motivates
  items 1.6-1.7, the conv-net MAE).

