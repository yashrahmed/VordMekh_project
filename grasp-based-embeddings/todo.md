# Grasp-based embeddings

## Goal - The only experimental project in the current track.
- Can I beat the MNIST benchmark using representations learned without label supervision (and if possible in a sample efficient way)?
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
   1. [x] Train a classifier with brief.
   2. [ ] Test shape descriptors on MNIST and measure similarity.
   3. [x] Train a VIT/MAE on MNIST and measure similarity. 
   4. [x] Train a VIT/MAE on MNIST and measure classification accuracy after finetuning.
   5. [x] Train a Conv-net MAE on MNIST and measure similarity. 
   6. [x] Train a Conv-net MAE on MNIST and measure classification accuracy after finetuning.
   7. [x] Test KNN with ConvNet and VIT.
   8. [x] Train an I-JEPA on MNIST and measure similarity. 
   9.  [x] Train an I-JEPA on MNIST and measure classification accuracy (maybe after finetuning?).
   10. [x] Test KNN with I-JEPA.
   11. [x] Test KNN with handcrafted BRIEF (random sampling).
   12. [x] Test KNN with *structured* (designed) BRIEF sampling.
2. BRIEF like features.
   1. A brief feature that moves. Similar to a hand but with a single finger.
   2. Imagine that this moves from one point to another sampling values. And there are two of these. Then shape descriptors could be used to find correspondence and similarity perhaps.
   3. What is unclear is correspondence is by shape descriptor value but how do I account for the value of the feature "head" i.e. the sum of the area where the value is sampled?
3. Can shape descriptors meant for 3-d meshes be extended to 5d point where a point is (x, y, dr, dg, db)? Worth looking into. --- Use r,g,b as "locations" and connect adjacent points to create the mesh. The position dimensions are not necessary. r,g,b as locations may not work as it creates a risk of the mesh folding in weird ways as color does not have spatil separation gaurantees. But I think it is worth try with a single channel image.


## Work log

All evals on the MNIST test set (10k), full 60k train split. Code lives in
`grasp_embeddings/mae_patch_embd/`:

- **`mae.py`** (`--arch {vit,cnn,jepa}`) — three self-supervised encoders:
  `vit` = ViT MAE (He 2021; drop 75% of patches, decode the missing pixels),
  `cnn` = conv MAE (Pathak 2016; zero the masked patches, conv reconstruct),
  `jepa` = I-JEPA (Assran 2023; EMA target encoder + predictor in latent space).
- **`brief.py`** + **`knn-brief.py`** / **`knn-brief-mod.py`** — handcrafted BRIEF
  (Calonder 2010), zero learning: random pairs vs a structured (census/LBP-style)
  lattice. Bit = `mean(box_a) < mean(box_b)`, box means via an integral image.
- **`knn.py`** — 5-NN over frozen embeddings (`--arch no-enc` = raw-pixel floor).
- **`classify.py`** — linear probe (frozen) or `--unfreeze` fine-tune; also
  `--brief` / `--brief-mod` (probe directly on the bit vector).
- **`retrieve.py`** — cosine-NN retrieval demo; also `--brief` / `--brief-mod`.

### Results — frozen embedding (5-NN, no labels reach the encoder)

| method | 5-NN acc | |
|---|---|---|
| I-JEPA (300 ep) | **98.18%** | tied best |
| ViT MAE (300 ep) | **98.17%** | ties I-JEPA once epoch-matched |
| ViT MAE (50 ep) | 97.16% | the earlier gap was epochs, not pretext |
| **raw pixels** | **96.88%** | floor — any learned embedding should beat this |
| BRIEF, random (512 bits) | 93.77% | best random budget |
| BRIEF, structured (224 bits) | 93.42% | |
| conv MAE (50 ep) | 91.29% | below the floor |

> **Note:** I-JEPA's 300-ep frozen **5-NN (98.18%) is basically the same as its
> frozen linear probe (98.40%)** — a non-parametric neighbour vote over the raw
> embeddings nearly matches a *trained* linear head, so the unsupervised
> representation is already as linearly class-separable as labels would make it.

### Results — with labels (linear probe = frozen encoder; fine-tune = unfrozen, 50 ep)

| method | linear probe | fine-tune |
|---|---|---|
| I-JEPA | **98.40%** | 98.69% |
| ViT MAE (300 ep) | 98.13% | 98.7% |
| ViT MAE (50 ep) | 97.4% | — |
| conv MAE | — | **99.0%** |
| BRIEF, structured (224 bits) | 88.65% | n/a — no parameters |
| BRIEF, random (64 bits) | 77.37% | n/a — no parameters |

### Key findings

1. **Reconstruction ≠ good embeddings.** The conv MAE has the *worst* frozen
   embedding (91.29%, below the pixel floor) yet the *best* fine-tuned result
   (99.0%) — its inpainting features destroy off-the-shelf similarity but make a
   great trainable starting point.

2. **The best frozen embeddings barely need the labels.** I-JEPA at 300 ep:
   frozen 5-NN 98.18% ≈ linear probe 98.40% ≈ fine-tune 98.69% (all within
   ~0.5 pts) — the unsupervised representation already does almost all the work
   (the 300-ep ViT MAE is the same story; see finding 6). I-JEPA's EMA target
   evolves slowly, so it needs a longer schedule than the MAEs to get there:

   | I-JEPA frozen 5-NN | 50 ep | 200 ep | 300 ep |
   |---|---|---|---|
   | acc | 94.12% | 97.89% | 98.18% |

3. **I-JEPA target-LayerNorm fix.** `forward()` was missing the LayerNorm on the
   target tokens that Meta applies before the MSE; adding it improved every
   downstream metric, most when undertrained (the EMA target is least stable
   early). Judge by downstream acc, not pretext MSE — LN rescales the target so
   the loss jumps (~0.04 → ~0.42) even as quality rises.

   | eval | no LN | +LN |
   |---|---|---|
   | 5-NN, 50 ep | 93.30% | 94.12% |
   | 5-NN, 300 ep | 97.91% | 98.18% |
   | probe, 300 ep | 98.24% | 98.40% |

4. **Handcrafted BRIEF is a zero-learning control, and behaves like one.**
   - Both variants sit ~3 pts *below* the raw-pixel floor — crude binary
     intensity comparisons discard similarity structure the bare pixels keep
     (same failure mode as the conv MAE). It clears only the conv MAE, nothing
     else.
   - **Structured > random at a smaller bit budget** — anchoring each bit to a
     fixed local gradient spends bits better than random, possibly long-range
     pairs:

     | sampling | bits | 5-NN |
     |---|---|---|
     | random | 64 | 79.42% |
     | structured | 48 | **84.21%** |
     | random | 256 | 92.57% |
     | structured | 224 | **93.42%** |

     Random saturates at ~94% by 512 bits (each doubling buys ~⅓ the last), so the
     ceiling is the representation, not the budget.
   - **The linear probe is *below* BRIEF's own 5-NN** (77.4% / 88.7% vs 79.4% /
     93.4%): a single hyperplane underfits the 0/1 bit space where Hamming k-NN
     can carve non-linear, per-digit neighbourhoods. Unlike the learned encoders
     (probe ≥ 5-NN), BRIEF is the one place non-parametric matching wins.
   - **No fine-tuning path** — the descriptor has no parameters and a
     non-differentiable threshold, so it structurally cannot join the 98.7–99.0%
     fine-tune cluster. Its value is as the floor that shows what learning buys.

5. **Fine-tuning erases the encoder differences** — all three learned encoders
   land 98.7–99.0%. On a task this easy (pixel floor ~97%) the pretext barely
   matters once labels are added.

6. **Epoch-matched, the ViT MAE catches I-JEPA — the "win" was the budget.**
   I-JEPA's apparent frozen-embedding lead was at 300 ep vs the MAEs at 50 ep.
   Re-training the ViT MAE at 300 ep closes the gap: frozen 5-NN 97.16% → 98.17%
   (≈ I-JEPA's 98.18%), probe 97.4% → 98.13%. So on MNIST latent prediction and
   pixel reconstruction are essentially equivalent for frozen embeddings — the
   differentiator was training length, not the pretext.

   | ViT MAE frozen | 50 ep | 300 ep |
   |---|---|---|
   | 5-NN | 97.16% | 98.17% |
   | probe | 97.4% | 98.13% |

**Caveat now flips to the task.** With the epoch confound removed, MNIST's ~97%
pixel floor is simply too high to separate these pretexts. For real headroom,
move to CIFAR-10 or a scarce-label regime where a better representation can
actually show.
