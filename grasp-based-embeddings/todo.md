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

### Procedure to generate one labeled sample
1. Sample a position `p` and orientation `h`.
2. Place and orient the hand.
3. Extend all fingers until each makes contact (or hits `L_max`).
4. Measure each extension.

A sample is then: `(p, h, l1, l2, l3, l4, l5)`.

(The `(p, h)` is the *query pose*; the `l_i` vector is the *grasp label* /
local-shape descriptor at that pose.)

## Open questions / analysis to do
- Is there any point to this idea? (utility of the descriptor)
- Has something similar been tried? (shape context, ray/radial signatures,
  2D LiDAR scans, tactile/haptic shape recognition, DeepSDF-style probes)
- How to relate **multiple measurements** into a coherent **shape context**?

## Next steps (original notes, kept)
- Check similarities with shape context methods.
- Find a way to calculate hand grasps on high planar shapes.
- See the "Biology of LLM" paper by anthropic.
- TBD....
- Its a LOOONG shot..but if there exists a way to map any complicated embedding
  space into the space of grasp embeddings, then a lot of human cognitive feats
  can be replicated.

## Work log
### 2026-06-20
- Built a ViT-style **Masked Autoencoder** (`mae_patch_embd/mae.py`): patchify,
  75% random masking, encode visible patches only, decode with mask tokens,
  MSE loss on masked patches. Trained on MNIST (50 epochs).
- Wrote a nearest-neighbor **retrieval** demo (`mae_patch_embd/retrieve.py`):
  samples 5 images per class (0-9), picks a random query, embeds with the
  (unmasked) encoder, and shows the query + top-3 cosine matches.

