# VordMekh Project

VordMekh is a research monorepo. Each project owns its dependencies, runnable
commands, and documentation under [`projects/`](projects/).

## Projects

| Project | Status | Purpose |
|---|---|---|
| [MNIST self-supervised learning](projects/mnist-ssl/) | Active | From-scratch DINOv2, I-JEPA, MAE, frozen probes, and ensembles. The current exploratory best is 99.61%. |
| [ChaiGPT](projects/chai-gpt/) | Historical experiments | Seven conversational-planning prototypes plus an equipment-search experiment. |
| [Local Secrets Store](projects/local-secrets-store/) | Local utility | PIN-locked, encrypted credentials vault with an HTML interface. |

## Repository conventions

- Run commands from the project directory that documents them.
- Project dependencies do not belong at repository root.
- Generated data, model checkpoints, caches, and logs remain untracked.
- Small result summaries and checkpoint hashes are tracked so reported results
  can be audited without committing large binary artifacts.

The MNIST project has the most complete implementation and experiment record.
Start with its [README](projects/mnist-ssl/README.md), [results](projects/mnist-ssl/docs/results.md),
or [best-result reproduction guide](projects/mnist-ssl/docs/reproduce-best.md).
