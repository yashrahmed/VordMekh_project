# Analysis

Three questions from the project notes. TL;DR up front, detail below.

- **Is there a point?** Yes — it's a *sensor-grounded, pose-conditioned local
  shape descriptor*. It carries more than a plain SDF sample (direction +
  anisotropy + an inside/outside bit), but as stated (5 rays from one origin,
  finite reach) it is lossy and ambiguous. Easy upgrades sharpen it a lot.
- **Has it been tried?** The *parts* are well-trodden (radial/ray signatures,
  2D LiDAR scans, tactile/haptic probing, SDF ray-marching). The *framing* —
  arbitrary-pose few-finger contact readings used as a learned embedding/label
  space — is the underexplored bit.
- **Relating measurements into a shape context?** Express every reading in the
  *query's local frame*, then aggregate. Four routes below, from a cheap
  shape-context histogram to a DeepSDF-style latent code (recommended bridge to
  the neural-net goal).

---

## 1. Is there any point to this idea?

What one reading actually is: place a probe at pose `(p, h)`, shoot 5 rays at
fixed relative angles, record the first-contact distance per ray (0 if the
origin starts inside). That's a **directional depth probe**.

Why that's more than nothing:

- **More information than a single SDF sample.** A signed-distance value at a
  point is one scalar (distance to nearest boundary), rotation-invariant, and
  direction-free. Five rays at known relative angles capture *which way* the
  boundary is and *how it bends* — local orientation and a crude curvature — in
  one shot. The reading is pose-*equivariant*, not collapsed.
- **Free occupancy signal.** The "origin inside → 0" rule means each reading
  also reports inside/outside. So one label encodes occupancy *and* boundary
  geometry together.
- **Physically grounded.** It is exactly the measurement a sparse range sensor
  or a finger array would produce. A net trained to predict it is learning
  "what a sensor would feel here" — a useful intermediate representation for
  robotics and active perception, not just an abstract feature.

Where the point is weaker (and how to fix it):

- **Lossy / ambiguous.** With one fan origin all rays share a vertex, so you get
  a *star-shaped* sample of the boundary visible from a single point — many
  distinct local shapes give the same 5 numbers. Concave pockets, thin features
  past `L_max`, and occluded edges are invisible.
- **Fix 1 — distinct finger origins (palm width).** Spread the bases sideways
  (`Hand.fan(base_offset=...)`). Parallax lets you triangulate boundary
  *orientation and curvature*, not just distance; even 2–3 readings then pin a
  local arc.
- **Fix 2 — richer per-finger output.** Return the **surface normal at contact**
  and an explicit **hit/miss bit** alongside distance. Big information gain for
  little cost.

Verdict: worthwhile as a *sensor-grounded local shape code*. The speculative
bet in `todo.md` — that the *space of grasp readings* could be a useful
universal coordinate system — is unproven but in the same spirit as
"representations that predict sensorimotor outcomes," which is a respectable
research direction.

## 2. Has something similar been tried? (prior art)

The idea sits at the intersection of four established lines:

- **Shape context (Belongie & Malik, 2000).** The closest *conceptual* cousin:
  a local descriptor that summarizes shape *relative to a reference point* via a
  log-polar histogram of all other boundary points. Differences: it needs the
  full boundary point set, is passive, and is rotation-normalized; the grasp
  idea uses a *few physical rays* and is a *partial, movable* probe. Worth a
  direct comparison — and it directly inspires aggregation route (d) below.
  - https://en.wikipedia.org/wiki/Shape_context
  - https://papers.nips.cc/paper/1913-shape-context-a-new-descriptor-for-shape-matching-and-object-recognition

- **Radial / centroid-distance "ray" signatures + Fourier descriptors.** Classic
  2D shape signatures shoot rays from the centroid at regular angles and record
  boundary distance → a 1D function. The single-origin finger fan is literally a
  *local, un-anchored* version of this. The differentiator is arbitrary
  placement + learning, not the sensing.
  - https://www.researchgate.net/publication/253247572_Ray_casting_approach_for_boundary_extraction_and_Fourier_shape_descriptor_characterization

- **2D LiDAR / range scans + learned descriptors** (place recognition). A LiDAR
  scan is a dense version of this hand — hundreds of rays from one origin. Scan
  Context, OverlapTransformer, PointNetVLAD, OREOS, etc. learn descriptors from
  such scans. The grasp idea = a *sparse* scan from *arbitrary* poses, used as a
  *label* rather than for localization. Their rotation-invariant ring/sector
  encodings are directly reusable for question 3.
  - https://arxiv.org/pdf/2203.03397 (OverlapTransformer)
  - https://arxiv.org/pdf/2106.10458 (place-recognition deep-learning survey)

- **Tactile / haptic shape recognition & active touch.** The closest *embodied*
  analog: a hand/finger contacting an object to infer shape. Crucially, the
  **active perception** branch already studies *where to probe next*, which is
  exactly the multi-measurement question.
  - https://arxiv.org/abs/2107.09584 (Active 3D Shape Reconstruction from Vision and Touch)
  - https://arxiv.org/pdf/2203.09149 (Active Visuo-Haptic Object Shape Completion)
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC6651744/ (Active Haptic Perception review)

- **DeepSDF / implicit neural representations + ray probing.** A finger length is
  approximately the first zero-crossing of an SDF/occupancy field along the ray,
  i.e. *ray-marching an implicit field* (cf. differentiable sphere tracing).
  Predicting grasp labels from a patch ≈ learning a ray-traced view of an
  implicit field — connecting the idea to the most active branch of neural shape
  modeling, and motivating aggregation route (c).
  - https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf
  - https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_DIST_Rendering_Deep_Implicit_Signed_Distance_Function_With_Differentiable_Sphere_CVPR_2020_paper.pdf

Bottom line: the *components* are mature. What I did **not** find as a named
method: arbitrary-pose, few-finger contact readings used as a *learned embedding
/ label space* meant as a universal local-shape code. So the sensing isn't
novel; the framing has room.

## 3. Relating multiple measurements into a shape context

One reading is partial and pose-dependent. To build a "shape context," aggregate
many readings into a representation with the right invariances.

**Invariance recipe (applies to all routes):** express every measurement in the
*query pose's local frame* (subtract/rotate by the query pose) so the context is
invariant to absolute placement; keep finger *order* fixed; optionally normalize
lengths by `L_max` or median contact distance for scale invariance.

Four routes, increasing sophistication:

- **(d) Polar/ring binning — cheapest, interpretable baseline.** Anchor at the
  query point; bin each finger's *contact point* into log-polar (radius × angle)
  cells and accumulate (count, mean length, hit-rate). This is literally *shape
  context computed from probe contacts* instead of given boundary points. Great
  first sanity check that the readings carry shape info.

- **(a) Common-frame fusion → point cloud.** Each finger reading gives a world
  contact point (and a normal, if added). Pool contacts from many poses → a
  local boundary point cloud / occupancy estimate; length-0 readings constrain
  the interior. Then apply any standard descriptor (shape context, FPFH-like, or
  a small PointNet). "Turn probes into geometry, then describe the geometry."

- **(b) Pose-tagged set → permutation-invariant net.** Treat the readings around
  a region as a *set* of tuples `(Δp, Δh, l1..l5)` in the query frame and learn
  a Deep Sets / Set Transformer aggregator. Handles a variable number of
  measurements and yields a context vector directly; frame-relative inputs give
  translation/rotation invariance for free.

- **(c) Implicit field regression — recommended bridge to the NN goal.** Use the
  readings as supervision to fit a *local field* (occupancy or SDF) with a small
  MLP conditioned on a latent **context code** (DeepSDF-style auto-decoder). Each
  finger length is a ray-marching constraint (field empty up to `l_i`, boundary
  at `l_i`). The optimized context code *is* your shape-context embedding. This
  unifies measurements, is natively multi-measurement, gives a compact
  embedding, and matches the end goal: train an encoder `patch → context code`,
  decode to grasp readings.

**Suggested sequencing:** start with (d)/(a) to confirm readings are
informative, then move to (b)/(c) for the learned embedding and the patch→label
net.

**Active-perception bonus:** once readings are predictable, pick the *next* pose
that most reduces uncertainty (the active-touch loop) — turning "relate multiple
measurements" into "choose which measurement to take."
