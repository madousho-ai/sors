# External dataset assets

Training, serving, dataset adapters and sampling live in this repository. Authored
datasets, validation schemas and authoring tools live in the separate
`decidophobia-dataset` repository. Its root directly contains
`synth-intents-v2.5/`, `synth-intents-v3/`, `synth-intents-v5.1/`,
`synth-simple-eval/`, `label-descriptions/` and `public-decisions/`.

## Select a checkout

From the training repository:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --datasets-dir ../decidophobia-dataset --dataset synth-v5.1
```

The order is an explicit `--datasets-dir` / API argument, then
`DECIDOPHOBIA_DATASETS_DIR`, then a sibling `decidophobia-dataset` checkout.
Training and invariance evaluation expose the flag; resume also accepts it.
Loaders accept `datasets_dir=` and retain their existing explicit asset-directory
or file arguments. An unavailable requested asset produces a configuration error.

The `data/` directory holds downloaded third-party inputs and caches.
`--data-dir` still selects the Banking77 cache. `--public-manifest` continues to
select a manifest with locally prepared official decision data. These locations
are independent of `--datasets-dir`; see the data repository's
`public-decisions/README.md` for preparation instructions.

The v3 and v5 loaders execute Python validators from the asset checkout. Use a
trusted checkout and pin its revision for reproducible runs.

## Reproduce and resume

Training records the resolved asset path, both repositories' revisions and their
dirty status in `sampling_provenance`. Complete training-state checkpoints also
store a content fingerprint covering sampling code, the resolver, resume checks,
the v5 schema and every v5 JSON file. Logical filenames keep this fingerprint
stable when the same files move between directories. File contents remain checked.

```bash
PYTHONPATH=src .venv/bin/python scripts/resume.py \
  --run runs/example --datasets-dir ../decidophobia-dataset
```

Resume chooses an explicit flag first, then the environment, then the saved
asset path, then the sibling checkout. A checkpoint whose sampling code or data
changed is refused. A narrowly pinned migration record in
`src/decidophobia/training/dataset_split_compat.json` allows pre-split fingerprints
that matched the source snapshot at migration time. This exception requires the
exact audited post-split fingerprint; subsequent code or data edits invalidate it.

For an old weights-only checkpoint, retain `--allow-optimizer-reset` and verify
both versions with `--expected-code-commit CODE_REV --expected-data-commit DATA_REV`.
The migration record also recognizes original full commit IDs (or unambiguous
prefixes of at least seven characters) with identical pre-split sampling inputs.
Older or independently modified sampling versions still require their matching
code and data. The migration performs CPU checkpoint/sampling validation; actual
GPU continuation belongs on the original training host with matching dependencies.

## Tests

Asset-backed integration tests use the same root setting:

```bash
DECIDOPHOBIA_DATASETS_DIR=../decidophobia-dataset \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_train_cli.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_dataset_paths.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_external_resume.py
```

Schema and authoring-tool tests are maintained in the dataset repository.
Loader and sampler tests remain here; their small temporary fixtures reuse the
schema from the selected asset checkout. Benchmark integration tests additionally
need the appropriate existing third-party caches.
