# Results and provenance

- `leaderboard.csv` is the compact, curated index of comparable measurements.
- `checkpoint-manifest.json` maps stable artifact IDs to ignored local files and
  their integrity hashes.
- `reproductions/` contains small audited records from completed sanity checks.
  The LoRA matrix record reports every prespecified test milestone and pins the
  exact source-checkpoint and unchanged-base fingerprints.

Full grids, logs, checkpoints, and plots belong in the ignored `out/`, `models/`,
and `images/` directories. See [the artifact policy](../docs/artifact-policy.md)
before promoting or deleting experiment artifacts.
