# External dataset assets

SORS training, serving, dataset adapters and sampling live in this repository. Authored
datasets, validation schemas and authoring tools live in the separate
`decidophobia-dataset` repository. Its root directly contains
`synth-intents-v2.5/`, `synth-intents-v3/`, `synth-intents-v5.3/`,
`synth-simple-eval/`, `label-descriptions/` and `public-decisions/`.

## Select a checkout

From the training repository:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --datasets-dir ../decidophobia-dataset --dataset synth-v5.3
```

`synth-intents-v5.1` became `synth-intents-v5.2` when targeted hard questions were
added to every domain, and `synth-intents-v5.2` became `synth-intents-v5.3` when the
`contrast` goal (confusable-intent menus and minimal-edit counterfactuals) was added.
`--dataset synth-v5.1` and `--dataset synth-v5.3` are rejected with a pointer to the
new name; `synth-v5` remains an alias of the current data. Complete states saved on
older data fail the resume fingerprint check against v5.3. A `--mix` written for v5.2
must add a `contrast=` share, since every goal in the data keeps a positive share.

The order is an explicit `--datasets-dir` / API argument, then
`SORS_DATASETS_DIR`, then the legacy `DECIDOPHOBIA_DATASETS_DIR`, then a sibling
`decidophobia-dataset` checkout. The separate data repository retains its current name.
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
The v5 loader uses this code checkout's JevBench cache for overlap checks, even
when either repository has a custom directory name. `JEVBENCH_DIR` explicitly
overrides that cache location.

## Public decision training

The public adapters also support `quality` (QuALITY v1.0.1 HTML-stripped),
`reclor`, and `logiqa2` (English MRC). Use their official train partitions and
preserve source licenses: QuALITY is CC BY 4.0, ReClor is restricted to
non-commercial research, and LogiQA 2.0 is CC BY-NC-SA 4.0. ReClor reads the
`answers` field with a zero-based label; QuALITY uses the validated one-based
`gold_label`. QuALITY retains both author sets for each article. LogiQA IDs
include the source row index because the official numeric IDs repeat.
The three adapters count and exclude complete questions with duplicate option
text; LogiQA additionally excludes every row in a conflicting-label group.

`--dataset-weights` sets relative question counts across training sources,
before consistency pairing. Give a positive weight for every selected dataset:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --datasets-dir ../decidophobia-dataset \
  --dataset synth-v5.3+sharc+toolace \
  --dataset-weights synth-v5.3=27,sharc=6,toolace=3 \
  --batch-size 36 --consistency 1
```

This draws 27 synth questions, six ShARC questions and three ToolACE questions,
then creates 72 prompts. Fractions use deterministic largest-remainder
allocation. Legacy `synth` shares its dataset quota between its two samplers.
Omitting the flag preserves the existing equal-sampler allocation and random
sequence. `--mix` and `--passes` continue to control only the synth goals.

A public manifest source may specify `include_ids` as a JSON file of adapter-local
string IDs, plus `include_ids_sha256`. This selects complete questions while
preserving every option and the original label; unknown or repeated IDs are
rejected. The loader records selection hashes and exclusion counts alongside
the raw source hashes. `max_menu_tokens` in `sors.data.public_decisions` computes
the maximum complete context-first prompt length over all menu permutations;
its text-length callback must use the run tokenizer with
`add_special_tokens=False, split_special_tokens=True`. Length screening can then
write an allowlist before training, preserving complete evidence within the
configured token budget.

## Reproduce and resume

Training records the resolved asset path, both repositories' revisions and their
dirty status in `sampling_provenance`. Complete training-state checkpoints also
record these versions. Exported checkouts and runtimes without Git record unknown
version fields; content fingerprint verification remains available. Complete states
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
`src/sors/training/dataset_split_compat.json` allows pre-split fingerprints
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
SORS_DATASETS_DIR=../decidophobia-dataset \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_train_cli.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_dataset_paths.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_external_resume.py
```

Schema and authoring-tool tests are maintained in the dataset repository.
Loader and sampler tests remain here; their small temporary fixtures reuse the
schema from the selected asset checkout. Benchmark integration tests additionally
need the appropriate existing third-party caches.
