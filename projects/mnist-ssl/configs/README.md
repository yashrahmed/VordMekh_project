# Canonical experiment configurations

`best/` records the exact recipes behind the curated leaderboard. These files
are small, reviewable provenance records; datasets and checkpoints remain under
ignored artifact directories.

- `dinov2_augmented_fixed150.json`: training recipe for the best individual
  DINOv2 trajectory.
- `ijepa_56_t48_fixed500.json`: training/probe recipe that produces the two
  56x56 I-JEPA ensemble members.

Changing a fixed training horizon creates a different schedule and therefore a
different experiment. Add a new config rather than editing a completed recipe.

`evaluation/` contains tracked, non-training inputs for reproducible evaluation
and audit workflows. `mnist_label_review_candidates.json` defines the exact
manual-review queue of 15 issues validated in the original NeurIPS paper. It
includes candidate provenance but no project-adopted label corrections.
`mnist_label_corrections.json` records the completed project review: eight
relabels, two exclusions, and five retained original labels.
