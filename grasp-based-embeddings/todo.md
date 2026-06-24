# Grasp-based embeddings

## Goal - The only experimental project in the current track.
- Can I beat the MNIST benchmark using representations learned without label supervision (and if possible in a sample efficient way)?

## What I wish to test.
1. Learning Patch embeddings.
   1. [x] Train a classifier with brief.
   2. [x] Test shape descriptors on MNIST and measure similarity / classification accuracy.
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
2. Looks like I am going to have to double down on IJEPA.
   1. [x] Run a test using cropped and scaled images — bbox-crop + stretch-to-frame is a large win at 50 ep; see finding 10.
   2. [x] Run a test using a different patching scheme (Closer to canonical IJEPA) — canonical block masking underperforms scatter masking at this resolution; see finding 9.
   3. [ ] Look into augmentation techniques.
   4. [ ] Maybe? try increased emebedding dims?
   5. [ ] Re-run the bbox-preproc scatter JEPA at **300 ep** — finding 10 is 50 ep only; does the preproc edge hold (or grow) once raw scatter has caught up at full budget?
   6. [ ] Account for known MNIST **label errors** when reading these results — the test set has ~15 human-validated mislabels (~0.15%), a soft ~99.8% ceiling that several frozen probes are now brushing against. See the corrected-test-set viewer / indices: [labelerrors.com](https://labelerrors.com) ([Northcutt et al., NeurIPS 2021](https://arxiv.org/pdf/2103.14749); [cleanlab/label-errors](https://github.com/cleanlab/label-errors)).
3. Additional material follow up -
   1. [ ] [Le-JEPA](https://arxiv.org/pdf/2511.08544)
   2. [ ] [V-JEPA](https://arxiv.org/pdf/2601.14354)
## Results — frozen embedding (5-NN, no labels reach the encoder)

| method | 5-NN acc | |
|---|---|---|
| I-JEPA (300 ep) | **98.18%** | tied best |
| ViT MAE (300 ep) | **98.17%** | ties I-JEPA once epoch-matched |
| ViT MAE (50 ep) | 97.16% | the earlier gap was epochs, not pretext |
| **raw pixels** | **96.88%** | floor — any learned embedding should beat this |
| BRIEF, random (512 bits) | 93.77% | best random budget |
| BRIEF, structured (224 bits) | 93.42% | |
| conv MAE (50 ep) | 91.29% | below the floor |
| geodesic D2 hist (64 bins) | 44.21% | layout-blind shape distribution |

## Results — with labels (linear probe = frozen encoder; fine-tune = unfrozen, 50 ep)

| method | linear probe (mean) | linear probe (flatten) | fine-tune |
|---|---|---|---|
| I-JEPA | 98.40% | **99.11%** | 98.69% |
| ViT MAE (300 ep) | 98.13% | 98.87% | 98.7% |
| ViT MAE (50 ep) | 97.4% | — | — |
| conv MAE | — | — | **99.0%** |
| BRIEF, structured (224 bits) | 88.65% | n/a | n/a — no parameters |
| BRIEF, random (64 bits) | 77.37% | n/a | n/a — no parameters |

Flatten-probe column from finding 7 (concatenated patch tokens). The I-JEPA
**99.11%** is the 300-ep head and is the best result here -- a frozen encoder
with a linear head edges past every fine-tuned model.

## Key findings

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

6. **Epoch-matched, the ViT MAE is almost as good as I-JEPA — the "win" was the
   budget.** I-JEPA's apparent frozen-embedding lead was at 300 ep vs the MAEs at
   50 ep. Re-training the ViT MAE at 300 ep all but closes it: frozen 5-NN
   97.16% → 98.17%, a **tie** with I-JEPA's 98.18%; on the linear probe
   97.4% → 98.13%, where I-JEPA keeps a marginal **~0.27 pt** edge (98.40%).
   So on MNIST latent prediction and pixel reconstruction are essentially
   equivalent — the differentiator was training length, not the pretext.

   | ViT MAE frozen | 50 ep | 300 ep |
   |---|---|---|
   | 5-NN | 97.16% | 98.17% |
   | probe | 97.4% | 98.13% |

7. **Flattening the patch grid helps the linear probe, not k-NN.** Default pooling
   mean-pools the patch tokens; concatenating them instead (`--flatten`,
   D = N·embed_dim = 2048) keeps *where* each feature sits. k-NN barely moves
   (cosine averages over patches either way), but the linear probe gains ~0.7 pt
   for both encoders — a hyperplane can exploit the spatial layout mean-pooling
   discards. Frozen I-JEPA + flattened probe (**99.05%**, **99.11%** with a
   300-ep head) edges past the best *fine-tuned* model (conv MAE 99.0%) with no
   encoder gradients at all.

   | 300-ep encoder | 5-NN mean → flatten | probe mean → flatten |
   |---|---|---|
   | ViT MAE | 98.17% → 98.10% | 98.13% → **98.87%** |
   | I-JEPA | 98.18% → 98.23% | 98.40% → **99.05%** |

   Flatten-probe numbers are seed-locked (`train_classifier --seed 0`, 50-ep
   head) and reproducible. Extending only the I-JEPA head to 300 ep lifts it to a
   stable **99.11%** (train err 0.05%; the loss plateaus at ~0.0057 and the run
   reproduces exactly — MPS kernel jitter that wobbles the 50-ep number by
   ~0.1 pt washes out once the head converges).

8. **A geodesic shape-distribution descriptor is far below the floor (44.21%).**
   Downsampling 2x and taking the all-pairs intensity-geodesic distance matrix
   (`|dI| + 0.1` edges, 8-connected), then summarizing it as a 64-bin histogram
   of pairwise distances (an Osada-style D2 descriptor), lands at 44.21% 5-NN --
   far under even the conv MAE. The histogram is **permutation-invariant**, so it
   throws away *where* structure sits and keeps only the distribution of geodesic
   gaps; on near-identical digit topologies that signal is too coarse to separate
   classes. (The full flattened matrix that `retrieve` uses keeps the layout but
   is ~38k-dim -- impractical to k-NN over 70k images, hence the histogram here.)
   A spatially-aware reduction (e.g. per-pixel mean geodesic distance) is the
   obvious next variant if this thread is worth pursuing.

   A second, weaker read confirms it: under a training-free **nearest
   class-centroid** classifier (`eval_classifier --arch geodesic`) the same
   descriptor scores only **30.87%** test -- below its own 44.21% 5-NN, as
   expected (a single linear prototype per class can't carve the multimodal
   neighbourhoods k-NN exploits). Both lenses agree the descriptor, not the
   classifier, is the bottleneck. A sparsity check explains why: on the 14x14
   graph ~74% of nodes are background and ~66% of 8-conn edges are ~flat (cost
   ~ALPHA), so most of the distance matrix encodes an empty grid identical across
   digits -- masking the background before building the graph is the first
   correction to try.

9. **Canonical block masking underperforms random scatter masking on MNIST —
   the image is too small for a contiguous block to carry information.** The
   `ijepa-canonical` arch implements Assran et al.'s block masking (a contiguous
   2×4/4×2 context band carved disjoint from 4 independently-sampled 2-patch
   "domino" targets, per-block latent prediction, K=1 batch-shared layout) on the
   same 4×4 patch grid. Epoch-matched against the existing random-scatter `jepa`,
   it loses on **every** frozen metric, at both 50 and 75 ep:

   | frozen eval | canon 50ep | canon 75ep | scatter 50ep | scatter 75ep |
   |---|---|---|---|---|
   | 5-NN (mean) | 88.01% | 87.28% | 93.44% | **95.24%** |
   | probe (mean) | 86.28% | 88.15% | 94.30% | **95.74%** |
   | probe (flatten) | 95.92% | 96.23% | 97.72% | **98.09%** |

   The gap is ~7–8 pts on the pooled views and ~1.9 pts on flatten, and it
   *widens* slightly from 50→75 ep (scatter keeps improving; canonical's 5-NN
   even ticks down 88.0→87.3, i.e. it has plateaued). **Why:** on a 28×28 digit
   the 4×4 grid makes a "block" trivially small — the context band is already
   half the image and a domino target is 2 patches, so a contiguous block carries
   almost no information that its neighbours don't, and predicting it is too easy
   to force good features. Random scatter masking poses a harder, spatially
   distributed prediction task that learns better representations at this
   resolution. Canonical block masking's intended advantage (predicting *semantic*
   regions) needs a finer grid (e.g. PATCH_SIZE=2 → 14×14, as in the original
   I-JEPA) to have room to express itself — the bottleneck here is image/grid
   resolution, not the objective. Canonical also leans hardest on the flatten
   probe (mean→flatten jumps ~8 pts vs scatter's ~2–3), consistent with block
   prediction spreading signal across patch positions that mean-pooling collapses.

10. **Bounding-box normalization is a large win — 50 ep of preproc ≈ 300 ep of
   raw.** Cropping each digit to its tight bounding box and stretching it to fill
   the full 28x28 frame (`--preproc`, aspect ratio *not* preserved) before
   patchifying removes scale/translation nuisance variation up front, so the
   encoder no longer spends capacity learning to be invariant to it. Trained on
   the **same** scatter I-JEPA for 50 ep, every frozen metric jumps:

   | frozen eval | scatter 50 ep (raw) | scatter 50 ep (preproc) | scatter 300 ep (raw) |
   |---|---|---|---|
   | 5-NN (mean) | 93.44% | **98.38%** | 98.18% |
   | probe (mean) | 94.30% | **98.61%** | 98.40% |
   | probe (flatten) | 97.72% | **98.93%** | 99.11% |

   The gain is +1.2–4.9 pts over raw 50 ep, and on 5-NN and the mean probe the
   preproc'd 50-ep model **matches or beats the raw 300-ep model** — normalizing
   geometry buys roughly what 6x the pretraining budget bought. It helps most
   where the metric is most geometry-sensitive: **5-NN +4.94** (aligned digits
   make cosine matching trivial) and **mean probe +4.31**, versus only **+1.21**
   on the flatten probe, which already encodes per-patch layout and sat near the
   300-ep ceiling. Open question (action item 2.5): re-run at 300 ep to see if
   the edge holds once raw scatter has its full budget, or whether preproc just
   *front-loads* a representation raw eventually reaches. Cheap, label-free, and
   stackable with everything else here — the most promising lever found so far.

**Caveat now flips to the task.** With the epoch confound removed, MNIST's ~97%
pixel floor is simply too high to separate these pretexts. For real headroom,
move to CIFAR-10 or a scarce-label regime where a better representation can
actually show.

## Original intent

This project started from a different idea than the embedding study above, kept
here for context. The plan was to learn shape from a **"feeling hand"**: 5
fingers extending in straight lines from a posed origin `(p, h)`, each stopping
on contact with a shape edge and returning the distance travelled (0 if it
starts inside the shape). A pose maps to 5 contact distances — a local shape
label.

This is the **ray-based dual of a signed distance function (SDF)**: with `f` the
SDF, a finger `(O, D)` returns the first positive root of `f(O + tD) = 0`, so
each reading is a ray-march of the SDF along one beam and a net predicting
`(O, D) → ℓ` learns a *directed distance field* (cf. PRIF; DeepSDF : SDF :: this
net : a directed distance field). The geodesic and BRIEF-like descriptor threads
are the surviving offshoots of this shape-descriptor line of thinking.
