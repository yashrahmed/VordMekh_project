# Artifact and checkpoint policy

This repository tracks enough metadata to identify and reproduce important
results without committing datasets, multi-megabyte checkpoints, or generated
plots.

## What belongs where

| Path | Contents | Tracked by Git? |
|---|---|---:|
| `configs/best/` | Exact recipes and expected metrics for canonical runs | Yes |
| `results/leaderboard.csv` | Curated result index | Yes |
| `results/checkpoint-manifest.json` | Stable artifact IDs, paths, sizes, and SHA-256 hashes | Yes |
| `results/reproductions/` | Audited compact reproduction records | Yes |
| `models/` | Local backbone, optimizer-state, and probe checkpoint files | No |
| `dataset/` | Downloaded MNIST data | No |
| `out/` | Logs, full grids, temporary JSON/CSV output, and analyses | No |
| `images/` | Generated figures | No |

Python environments, bytecode, test caches, and package build metadata are
also ignored and may be deleted at any time.

## Promote a result

When an experiment becomes a result worth preserving:

1. Put its checkpoint files under `models/` with names that include the method,
   fixed training horizon, milestone epoch, pool/readout, and probe horizon.
2. Add each required file to `results/checkpoint-manifest.json`. Record its byte
   size, full-file SHA-256, role, and the results that require it. For a frozen
   backbone, also record the deterministic state-dict fingerprint emitted by
   the evaluator.
3. Add or update a machine-readable recipe under `configs/best/`. Refer to
   checkpoints by manifest artifact ID, not by duplicating paths in code.
4. Add the summary metric to `results/leaderboard.csv` and the relevant guide
   under `docs/`.
5. Re-run the frozen evaluation and store a compact audit record under
   `results/reproductions/`.

For the current best ensemble, the complete verification is:

```bash
uv run python scripts/reproduce/verify_artifacts.py
uv run python scripts/reproduce/best_ensemble.py --workers 0
```

The second command independently verifies the four required file hashes,
fingerprints all three frozen backbones before and after inference, and asserts
the recorded 99.61% result.

## Retention and cleanup

- Keep every local file referenced by `results/checkpoint-manifest.json` while
  the result remains on the curated leaderboard.
- Rolling resume checkpoints and non-winning milestones may be removed only
  after their experiment has finished and the retained milestones have been
  evaluated.
- Treat `out/` and `images/` as reproducible scratch space. Copy only concise
  metrics and provenance into tracked `results/`; do not commit raw logs.
- Never infer that an ignored checkpoint is disposable merely because Git does
  not show it. Compare against the manifest first.

The manifest identifies local files but is not binary storage. Backing up or
publishing the pinned checkpoint files requires a separate artifact store; if
one is added later, record immutable download locations alongside the hashes.
