# VordMekh Project

VordMekh is a research monorepo. Each project owns its dependencies, runnable
commands, and documentation under [`projects/`](projects/).

## Projects

| Project | Status | Purpose |
|---|---|---|
| [MNIST self-supervised learning](projects/mnist-ssl/) | Active | From-scratch DINOv2, I-JEPA, MAE, frozen probes, and ensembles. The current exploratory best is 99.61%. |
| [ChaiGPT](projects/chai-gpt/) | Historical experiments | Seven conversational-planning prototypes plus an equipment-search experiment. |
| [Local Secrets Store](projects/local-secrets-store/) | Local utility | PIN-locked, encrypted credentials vault with an HTML interface. |
| [Logic Programming](projects/logic-programming/) | Planned | Logic-programming experiments with Clingo. |
| [Classical Planning](projects/classical-planning/) | Planned | Classical-planning experiments with Unified Planning. |
| [3D Modeling](projects/3d-modeling/) | Placeholder | Future 3D-modeling work. |
| [Building Reasoning Models](projects/building-reasoning-models/) | Placeholder | Future reasoning-model work. |
| [Graphic Design](projects/graphic-design/) | Placeholder | Future graphic-design work. |
| [Claude Code Clone](projects/claude-code-clone/) | Reference | Notes and source material for a Claude Code clone. |

## Repository conventions

- Run commands from the project directory that documents them.
- Project dependencies do not belong at repository root.
- Generated data, model checkpoints, caches, and logs remain untracked.
- Small result summaries and checkpoint hashes are tracked so reported results
  can be audited without committing large binary artifacts.

The MNIST project has the most complete implementation and experiment record.
Start with its [README](projects/mnist-ssl/README.md), [results](projects/mnist-ssl/docs/results.md),
or [best-result reproduction guide](projects/mnist-ssl/docs/reproduce-best.md).
