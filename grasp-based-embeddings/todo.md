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
   6. [x] Train a Conv-net MAE on MNIST and measure similarity. 
   7. [x] Train a Conv-net MAE on MNIST and measure classification accuracy after finetuning.
   8. [x] Test KNN with ConvNet and VIT.
   9. [x] Train an I-JEPA on MNIST and measure similarity. 
   10. [x] Train an I-JEPA on MNIST and measure classification accuracy (maybe after finetuning?).
   11. [x] Test KNN with I-JEPA.
2. BRIEF like features.
   1. A brief feature that moves. Similar to a hand but with a single finger.
   2. Imagine that this moves from one point to another sampling values. And there are two of these. Then shape descriptors could be used to find correspondence and similarity perhaps.
   3. What is unclear is correspondence is by shape descriptor value but how do I account for the value of the feature "head" i.e. the sum of the area where the value is sampled?
3. Can shape descriptors meant for 3-d meshes be extended to 5d point where a point is (x, y, dr, dg, db)? Worth looking into. --- Use r,g,b as "locations" and connect adjacent points to create the mesh. The position dimensions are not necessary. r,g,b as locations may not work as it creates a risk of the mesh folding in weird ways as color does not have spatil separation gaurantees. But I think it is worth try with a single channel image.


## Work log
### 2026-06-20
Built three self-supervised encoders on MNIST in `mae_patch_embd/`, plus evals
over their frozen and fine-tuned embeddings (items 1.4-1.11):

- **`mae.py`** (`--arch {vit,cnn,jepa}`, checkpoints `models/mae_mnist_<arch>.pt`):
  - `vit` — ViT MAE (He 2021): drop 75% of patches, encode the visible ones,
    decode the missing pixels (MSE on masked patches).
  - `cnn` — conv MAE (Context-Encoder, Pathak 2016): masked patches *zeroed*,
    conv enc/dec reconstructs (MSE on masked pixels).
  - `jepa` — I-JEPA (Assran 2023): context encoder + EMA target encoder +
    predictor; predicts the target's *latent* representations at masked
    positions (MSE in representation space, no pixel decoder).
- **`retrieve.py`** — cosine nearest-neighbour retrieval demo (item 1.4).
- **`classify.py`** — linear probe (frozen) or `--unfreeze` end-to-end fine-tune.
- **`knn.py`** — 5-NN over the full 60k/10k splits; `--arch no-enc` = a raw-pixel
  Euclidean floor any learned embedding should beat.

**Headline results (MNIST test set).** ViT/CNN trained 50 epochs; I-JEPA swept
by pretraining length (see below).

| eval | ViT | CNN | I-JEPA | raw pixels |
|---|---|---|---|---|
| 5-NN, frozen embedding | 97.16% | 91.29% | **97.91%** (300 ep) | 96.88% |
| linear probe (frozen head) | 97.4% | -- | -- | -- |
| fine-tune (unfrozen, 50 ep) | 98.7% | **99.0%** | 98.69% | -- |

I-JEPA frozen-5-NN vs pretraining length: 50 ep **93.30%** → 200 ep **97.59%** →
300 ep **97.91%** (pretext MSE 0.140 → 0.047 → 0.039; diminishing returns).

**What we learned:**
1. **Reconstruction ≠ good embeddings.** The conv MAE's frozen 5-NN (91.29%) is
   *below* the raw-pixel floor (96.88%) — its inpainting features destroy
   similarity info bare pixels keep — yet it's the best *trainable* extractor
   (fine-tune 99.0%). Better trainable features, worse off-the-shelf embedding.
2. **Latent prediction (I-JEPA) gives the best frozen embedding — once trained
   long enough.** At 50 ep it was undertrained (93.30%, below the pixel floor);
   at 300 ep it's the best off-the-shelf embedding (97.91%), clearing the floor
   and beating the ViT MAE. Its EMA target evolves slowly, so it needs a longer
   schedule than the MAEs (OneCycle: more epochs = higher sustained LR, not a
   resumable add-on).
3. **Fine-tuning erases the differences** — all three land 98.7-99.0%. On a task
   this easy (pixel floor ~97%) the pretext barely matters once labels are added.

**Open caveat — not yet epoch-matched:** I-JEPA's frozen-embedding win is at
300 ep vs the ViT/CNN MAEs at 50 ep. Re-train the ViT MAE at 300 ep before
treating "JEPA > MAE for embeddings" as settled. To give any pretext real
headroom, move off MNIST (CIFAR-10) or into the scarce-label regime.

