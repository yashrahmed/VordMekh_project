# ChaiGPT experiments

Seven iterations exploring conversational state collection, talking forms,
grounded planning, frame reasoning, and recipe-tool search.

Each version is preserved in its own package. Start with the version-specific
README files for [V1](chai_gpt/README.md), [V2](chai_gpt_v2/README.md), and
[V3](chai_gpt_v3/README.md). Later versions are historical prototypes whose
design conclusions are summarized in the repository root README.

Install the dependencies in this directory:

```bash
python3 -m pip install -r requirements.txt
```

Launch commands must be run from `projects/chai-gpt` because the historical
prototypes load `keys-config.yml` and other resources relative to that working
directory. The key file is ignored by Git.
