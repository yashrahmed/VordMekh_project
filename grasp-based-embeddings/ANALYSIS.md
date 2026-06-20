# Analysis

## The idea

A "feeling hand" with 5 fingers is placed at a pose `(p, h)` (position + heading).
Each finger extends in a straight line until it touches a shape edge, recording
the contact distance (0 if the finger's origin starts inside the shape, clamped
to `L_max` on a miss). One sample is `(p, h, l1..l5)` — the hand pose plus its
five contact distances. The goal is to learn these labels for image patches as a
local shape descriptor.

## Relation to Signed Distance Fields

This is the **ray-based dual of a signed distance function (SDF)**. With `f(x)`
the SDF (`f = 0` on the boundary, `< 0` inside), a finger `(O, D)` returns the
first positive root of `f(O + tD) = 0` — i.e. each finger reading is a
**ray-march (sphere trace) of the shape's SDF along one beam**. So generating the
labels is exactly "ray-march the SDF along 5 rays," and a net that predicts
`(O, D) → ℓ` is learning a *directed distance field* (cf. PRIF, DeepSDF's
ray-based cousins). DeepSDF : SDF :: the planned net : a directed distance field.

## Notes

1. Patch embeddings via auto encoders.
   1. Train an auto encoder with handcrafted BRIEF.
   2. Train a classifier with brief.
   3. Test shape descriptors on MNIST and measure similarity.
   4. Train a regular VAE on MNIST. 
2. BRIEF like features.
   1. A brief feature that moves. Similar to a hand but with a single finger.
   2. Imagine that this moves from one point to another sampling values. And there are two of these. Then shape descriptors could be used to find correspondence and similarity perhaps.
   3. What is unclear is correspondence is by shape descriptor value but how do I account for the value of the feature "head" i.e. the sum of the area where the value is sampled?
3. Can shape descriptors meant for 3-d meshes be extended to 5d point where a point is (x, y, dr, dg, db)? Worth looking into. --- Use r,g,b as "locations" and connect adjacent points to create the mesh. The position dimensions are not necessary. r,g,b as locations may not work as it creates a risk of the mesh folding in weird ways as color does not have spatil separation gaurantees. But I think it is worth try with a single channel image.
