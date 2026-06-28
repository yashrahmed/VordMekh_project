# Grasp-based embeddings

## Goal - The only experimental project in the current track.
- Beat **99.5% test accuracy on MNIST** using representations learned without label supervision (and if possible in a sample efficient way).

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
   4. [x] Tried increased embedding dims — didn't help.
   7. [x] Try a **Conv-net stem** (replace the linear patch embedding with a small conv front-end) — old stem `Conv3 s2 p1 -> Conv2 s2 p1` underperformed custom 10-6 I-JEPA; see finding 16.
   5. [x] Re-run the bbox-preproc scatter JEPA at **500 ep** (run past the planned 300; flatten only, per request) — see finding 11: the preproc edge **holds** (flatten probe 99.15%, flatten 5-NN 99.01% — new bests) but is mostly *front-loading*; raw scatter nearly matches the probe at full budget.
   6. [ ] Account for known MNIST **label errors** when reading these results — the test set has ~15 human-validated mislabels (~0.15%), a soft ~99.8% ceiling that several frozen probes are now brushing against. See the corrected-test-set viewer / indices: [labelerrors.com](https://labelerrors.com) ([Northcutt et al., NeurIPS 2021](https://arxiv.org/pdf/2103.14749); [cleanlab/label-errors](https://github.com/cleanlab/label-errors)).
3. Additional material follow up -
   1. [ ] [Le-JEPA](https://arxiv.org/pdf/2511.08544)
   2. [ ] [V-JEPA](https://arxiv.org/pdf/2601.14354)
## Results — frozen embedding (5-NN, no labels reach the encoder)

| method | 5-NN acc | |
|---|---|---|
| I-JEPA (preproc, 500 ep, flatten) | **99.01%** | new best — geometry-normalized (finding 11) |
| I-JEPA (300 ep) | **98.18%** | best raw (ties ViT MAE) |
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
| I-JEPA (preproc, 500 ep) | — | **99.15%** | — |
| CNN-stem I-JEPA (preproc, old stem, best) | — | **99.06%** | — |
| ViT MAE (300 ep) | 98.13% | 98.87% | 98.7% |
| ViT MAE (50 ep) | 97.4% | — | — |
| conv MAE | — | — | **99.0%** |
| BRIEF, structured (224 bits) | 88.65% | n/a | n/a — no parameters |
| BRIEF, random (64 bits) | 77.37% | n/a | n/a — no parameters |

Flatten-probe column from finding 7 (concatenated patch tokens). The best result
here is now I-JEPA **+ preproc at 500 ep, 99.15%** (finding 11); the raw 300-ep
head's **99.11%** is a hair behind. Either way a frozen encoder with a linear
head edges past every fine-tuned model.

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
   `custom_ijepa` arch (formerly `ijepa-canonical`) implements Assran et al.'s block masking (a contiguous
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
   300-ep ceiling. Resolved at 500 ep (finding 11):
   the edge **holds** (flatten probe 99.15%, flatten 5-NN 99.01% — new bests) but
   is mostly *front-loading* — raw scatter nearly catches the probe at full
   budget, though preproc keeps a real ~0.8-pt 5-NN lead. Cheap, label-free, and
   stackable with everything else here — the most promising lever found so far.

11. **Preproc holds at full budget — the edge is mostly front-loading, not a
   higher ceiling.** Re-running scatter I-JEPA + `--preproc` at **500 ep** (action
   item 2.5, past the planned 300; flatten-only this run, per request) gives a
   frozen **flatten probe 99.15%** (probe train err 0.00%) and **flatten 5-NN
   99.01%** — both the best numbers in the study. But against raw scatter at its
   own full budget the probe margin is thin: 99.15% vs raw 300-ep's 99.11%
   (~0.04 pt), and the real preproc story is still the *50-ep* result (98.93%
   flatten probe ≈ raw 300-ep) — preproc **front-loads** a representation raw
   eventually reaches rather than raising the asymptote. The exception is k-NN:
   flatten 5-NN 99.01% clears every prior frozen 5-NN (raw 300-ep 98.23% flatten
   / 98.18% mean) by ~0.8 pt — aligned, scale-normalized digits make cosine
   matching markedly easier, and that gain does *not* wash out with budget. Both
   frozen probes now sit ~0.7 pt below the soft ~99.8% MNIST label-error ceiling
   (action item 2.6). The pretext MSE rose then plateaued ~0.34 over the run
   (LayerNorm'd moving target — see finding 3; not a reconstruction loss).

   | frozen eval (flatten) | preproc 50 ep | preproc 500 ep | raw 300 ep |
   |---|---|---|---|
   | linear probe | 98.93% | **99.15%** | 99.11% |
   | 5-NN | — | **99.01%** | 98.23% |

12. **Independent per-patch target prediction did not help.** Predicting each masked patch in its own predictor pass (no cross-target attention) instead of all jointly was neutral-to-worse than the default scatter I-JEPA (50 ep, no preproc, flatten): probe 97.60% vs 97.72%, 5-NN 93.84% vs 94.70% — the joint prediction's cross-target coupling is mildly useful; not pursued.

13. **Overlapping 14x14 patches did not help.** Replacing the 7x7 non-overlapping grid (16 patches, 2048-d flatten) with 14x14 patches at stride 7 (3x3 = 9 overlapping patches, 1152-d flatten) on scatter I-JEPA looked promising *raw* — 50 ep flatten probe 97.86% vs 97.72% and 5-NN 96.79% vs 94.70%, holding at 100 ep (probe 98.59%, 5-NN 97.85%). But the gain was just the larger windows absorbing scale/translation nuisance — exactly what bbox preproc (finding 10) already does, more cheaply. With `--preproc` the advantage vanished: at 50 ep patch-14 gave probe 98.70% / 5-NN 97.51%, *below* patch-7 preproc's 98.93% flatten probe, and well below the study best (patch-7 preproc 500 ep, 99.15% probe / 99.01% 5-NN). Patch size and geometry normalization are substitutes, not complements; not pursued.

14. **Scattered single-patch targets — best 50-ep result so far (98.99%).** Converging the canonical block design toward scatter one step at a time (`ijepa_trials/custom_ijepa.py`): replacing the 4 contiguous target *blocks* with **8 single patches** picked at random as the targets (the other 8 patches the context), all predicted **jointly** in one predictor pass (one conceptual block, intra-block attention on). Same patch-7 / 4x4 grid, enc_dim 128, preproc, frozen flatten probe, seed 0, 50 ep. **Test 98.99% (train 99.89%)** — the top 50-ep flatten-probe number on record, edging the prior 50-ep best (scatter preproc 98.93%) and matching enc32 custom I-JEPA at 300 ep. The jump is all in the masking *shape*: the same setup with contiguous blocks scored only 98.60% (+0.39 pt from scattering the targets), confirming finding 9's claim that block targets are a worse pretext here — a block hides a whole local region with no visible interior anchors, while scattered targets each sit among visible neighbours. Latent_mse settles lower too (~0.244 vs ~0.30 for blocks). **Caveats:** the 0.06-pt margin over scatter is ~6 test images, single-seed — a statistical tie, not a decisive win; and it's still below the longer scatter runs (300 ep 99.11%, 500 ep 99.15%). Next convergence step: 12 targets / 4 context to fully match scatter's mask ratio (swept in finding 15).

15. **Target/context split sweep — the optimum is 10 targets / 6 context, not the 8-8 default.** Made the single-patch target count configurable (`custom_ijepa.py` `--n-targets`; split written as targets-context) and swept `n_targets ∈ {4,6,8,10,12}` at two pretraining budgets, frozen flatten probe, patch-7 / 4x4 grid, enc_dim 128, preproc, seed 0.

   | split (t-c) | n_targets | test @50 ep | test @75 ep |
   |---|---|---|---|
   | 4-12 | 4 | 98.60% | 98.32% |
   | 6-10 | 6 | 98.85% | 98.54% |
   | 8-8 *(default)* | 8 | 98.93% | 98.98% |
   | **10-6** | **10** | **98.96%** | **99.06%** |
   | 12-4 | 12 | 98.72% | 98.94% |

   Two clean effects. (1) **Inverted-U in the mask ratio, peaking at n_targets=10** at both budgets: too few targets starves the predictor of supervision (4-12), too few context patches starves the context encoder (12-4), and 10-6 is the sweet spot. The best result, **10-6 @ 75 ep = 99.06%**, is a new 75-ep best and clears the 8-8 default's 98.98% by +0.08 pt. (2) **More epochs split the field by mask ratio:** heavy masking *improves* 50→75 ep (8-8 +0.05, 10-6 +0.10, 12-4 +0.22 — hard pretext keeps paying off), while light masking *regresses* (4-12 −0.28, 6-10 −0.31 — an easy pretext overfits with more budget). Train accuracies were all 99.5–99.9% (test acc is the discriminating metric). **Caveats:** the 10-6 vs 8-8 margin is single-seed and ~8 test images — directionally consistent across both budgets but not decisive. Default kept at 8-8 per spec; 10-6 is the recommended operating point.

   **300-ep confirmation.** Ran 10-6 at the full 300-ep budget (same protocol): **flatten probe test 99.12%** (train 99.86%) — a new best 300-ep number, edging the prior scatter 300-ep (99.11%) and the 8-8 line, and confirming the 75-ep edge holds. But the gain over its own 75-ep result is only +0.06 pt (99.06 → 99.12) for 4x the compute — the same front-loading / diminishing-returns pattern as finding 11. Pretext MSE plateaued ~0.29.

   **500-ep — new study best (99.21%).** Ran 10-6 at 500 ep (same protocol): **flatten probe test 99.21%** (train 99.82%) — the best frozen-flatten-probe number on record, clearing the prior study best (scatter 500-ep preproc 99.15%, finding 11) by +0.06 pt. Unlike the 75→300 step (+0.06 pt for 4x compute, pure front-loading), the **300→500 step gained +0.09 pt** (99.12 → 99.21), so 10-6 is *not* plateaued at 300 — extra budget still buys real accuracy, nudging the apparent asymptote up from ~99.15% toward ~99.2%+. Net: the split optimum holds at every budget tested, and at 500 ep both *raises* the ceiling over scatter and reaches it. Progression: 99.06 (75) → 99.12 (300) → 99.21 (500). Still ~0.6 pt under the soft ~99.8% MNIST label-error floor.

16. **The old CNN stem is usable but not better than custom 10-6 I-JEPA.** Added
   `ijepa_trials/cnn_stem_ijepa.py` as a feature-space I-JEPA variant: bbox
   preproc stays on, a dense stem maps `1x28x28 -> Conv3 s2 p1 -> 32x14x14 ->
   GELU -> Conv2 s2 p1 -> 64x8x8 -> GELU`, then the `8x8x64` feature map is
   split into the same 16-token `4x4` grid of `2x2x64` feature patches. Default
   split is still **10 targets / 6 context**; masking happens after the dense
   stem, so this is a feature-space JEPA rather than strict raw-patch I-JEPA.

   | old CNN-stem run | encoder ep | probe ep | train acc | test acc |
   |---|---:|---:|---:|---:|
   | frozen flatten probe | 50 | 50 | 99.90% | 98.80% |
   | frozen flatten probe | 75 | 25 | 99.68% | 98.87% |
   | frozen flatten probe | 75 | 50 | 99.90% | 99.02% |
   | frozen flatten probe | 75 | 75 | 99.94% | 98.97% |
   | frozen flatten probe | 300 | 50 | 99.87% | 99.06% |

   Split sweep, same old CNN stem, frozen flatten probe, 50 probe epochs:

   | split (t-c) | n_targets | train @50 ep | test @50 ep |
   |---|---:|---:|---:|
   | 2-14 | 2 | 99.37% | 97.76% |
   | 4-12 | 4 | 99.57% | 98.15% |
   | 6-10 | 6 | 99.68% | 98.42% |
   | 8-8 | 8 | 99.81% | 98.62% |
   | **10-6** | **10** | **99.90%** | **98.80%** |
   | 12-4 | 12 | 99.77% | 98.53% |

   | split (t-c) | n_targets | train @75 ep | test @75 ep |
   |---|---:|---:|---:|
   | 2-14 | 2 | 99.11% | 97.70% |
   | 4-12 | 4 | 99.61% | 98.19% |
   | 6-10 | 6 | 99.53% | 98.38% |
   | 8-8 | 8 | 99.82% | 98.60% |
   | **10-6** | **10** | **99.90%** | **99.02%** |
   | 12-4 | 12 | 99.90% | 98.91% |

   Probe length does not explain the gap: on the same 75-ep encoder, 50 probe
   epochs were best (99.02%), while 75 probe epochs overfit/slipped slightly
   (98.97%) and 25 probe epochs undertrained (98.87%). More encoder pretraining
   helped only modestly: 50→75→300 encoder epochs gave 98.80→99.02→99.06. The
   split sweep confirms the same best operating point as custom I-JEPA — **10
   targets / 6 context** — with 12-4 as the closest runner-up at 75 ep. However,
   the CNN stem is still **not outperforming custom I-JEPA**: its best 75-ep
   result (99.02%) trails custom 10-6 at 75 ep (99.06%), and its 300-ep result
   (99.06%) trails custom 10-6 at 300 ep (99.12%) and 500 ep (99.21%). Verdict:
   keep the old CNN stem as a documented branch, but it is not the path to the
   99.5% goal unless the architecture/objective changes substantially.

17. **1000 ep does not beat the 500-ep custom 10-6 ceiling.** For completeness,
   ran the best custom I-JEPA setup from scratch for **1000 encoder epochs**
   (bbox preproc, 4x4 patch grid, **10 target / 6 context** scattered
   single-patch joint targets, enc_dim 128, seed 0), then trained the standard
   **50-ep frozen flatten probe**. Result: **99.20% test accuracy** (80 errors,
   train 99.77%), essentially tied with but slightly below the 500-ep best
   (**99.21%**, finding 15). A separate warm-start continuation from the 500-ep
   checkpoint also failed to help: 50/100/500-ep probes on that continued encoder
   scored 99.10% / 99.16% / 99.15%, respectively. Implication: the current
   custom 10-6 feature recipe is not compute-limited at 500 ep; longer encoder
   training is at best flat and can degrade the downstream linear geometry.
   Future attempts to reach 99.5% need a representation/objective change rather
   than simply more epochs on this exact setup.

18. **56x56 upscaled-bbox with a finer 8x8 patch grid sets a new best, but still
   has not reached 99.5%.** Changed the preprocessing to
   upscale MNIST to **56x56**, bbox-crop the upscaled digit, then stretch the crop
   to 56x56. Changed custom I-JEPA from 14px patches on the 56x56 image
   (4x4 = 16 tokens) to **7px patches** (8x8 = 64 tokens), and swept scaled
   target/context ratios with both mean and flattened frozen linear probes
   (50 probe epochs, seed 0).

   Completed 50-ep encoder results:

   | split (t-c) | n_targets | mean probe | flatten probe |
   |---|---:|---:|---:|
   | 8-56 | 8 | 96.25% | 98.13% |
   | 16-48 | 16 | 97.29% | 98.44% |
   | 24-40 | 24 | 97.66% | 98.52% |
   | 32-32 | 32 | 98.21% | 98.87% |
   | 36-28 | 36 | 98.43% | 98.93% |
   | 40-24 | 40 | 98.75% | 98.97% |
   | **44-20** | **44** | **98.95%** | **99.14%** |
   | 48-16 | 48 | 98.33% | 98.87% |

   This established a new best **50-ep** result: **44 targets / 20 context,
   flatten probe = 99.14%**, beating the prior 16-token 75-ep split-sweep best
   (10-6 @ 75 ep = 99.06%) and the 56x56/14px-patch branch (10-6 @ 75 ep =
   98.96%).

   Completed 75-ep encoder results:

   | split (t-c) | n_targets | mean probe | flatten probe |
   |---|---:|---:|---:|
   | 8-56 | 8 | 96.43% | 98.08% |
   | 16-48 | 16 | 97.11% | 98.27% |
   | 24-40 | 24 | 97.94% | 98.43% |
   | 32-32 | 32 | 98.39% | 98.81% |
   | 36-28 | 36 | 98.66% | 98.82% |
   | 40-24 | 40 | 98.82% | 98.92% |
   | 44-20 | 44 | 99.20% | 99.21% |
   | **48-16** | **48** | **99.28%** | **99.15%** |

   The completed sweep sets a new study best: **48 targets / 16 context at 75 ep,
   mean probe = 99.28%**. This clears the previous all-time custom I-JEPA result
   (**10-6 @ 500 ep, flatten = 99.21%**) by +0.07 pt, using much less encoder
   pretraining. It is still short of the 99.5% goal.

   Two readout/masking patterns matter. First, flattened readout remains strongest
   across most of the grid and peaks at **44-20 @ 75 ep = 99.21%**, but it drops
   at 48-16. Second, mean pooling becomes competitive only under very heavy
   masking: **48-16 @ 75 ep mean = 99.28%** is the only mean-probe winner. This
   suggests that the finer 8x8 grid plus heavy masking learns a globally useful
   per-token representation, while flatten readout is more sensitive to the
   context becoming too sparse. The 75-ep light/mid splits mostly underperformed
   their 50-ep versions, so the benefit of extra epochs appears concentrated in
   the heaviest mask-ratio regime.

   **High-mask completion sweep.** For completeness, also swept 50-14, 52-12,
   and 54-10 at both encoder budgets. These did not improve on 48-16 @ 75 ep.

   | split (t-c) | n_targets | mean @50 ep | flatten @50 ep | mean @75 ep | flatten @75 ep |
   |---|---:|---:|---:|---:|---:|
   | 50-14 | 50 | 98.48% | 98.82% | 99.22% | 99.23% |
   | 52-12 | 52 | 93.71% | 98.29% | 98.96% | 99.12% |
   | 54-10 | 54 | 95.67% | 98.67% | 98.70% | 98.99% |

   The high-mask tail confirms the peak is around **48/64 targets**, not beyond
   it. At 50 ep, performance collapses sharply once context falls below 16
   patches, especially for mean pooling. At 75 ep, 50-14 partly recovers
   (99.23% flatten), but 52-12 and 54-10 remain clearly below the 48-16 mean
   winner. Net: **48-16 @ 75 ep mean = 99.28%** remains the best operating point
   found in this sweep.

**Caveat now flips to the task.** With the epoch confound removed, MNIST's ~97%
pixel floor leaves little room to separate these pretexts, but the explicit goal
is still to push the unsupervised MNIST pipeline past **99.5%**. Future work
should focus on changes that plausibly close the remaining ~0.2 pt gap from the
current 99.28% best, while accounting for known MNIST label errors.

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
