# External dataset assets

SORS training, serving, dataset adapters and sampling live in this repository. Authored
datasets, validation schemas and authoring tools live in the separate
`decidophobia-dataset` repository. Its root directly contains
`synth-intents-v2.5/`, `synth-intents-v3/`, `synth-intents-v5.13/`,
`synth-simple-eval/`, `label-descriptions/` and `public-decisions/`.

## Select a checkout

From the training repository:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --datasets-dir ../decidophobia-dataset --dataset synth-v5.13
```

`synth-intents-v5.1` became `synth-intents-v5.2` when targeted hard questions were
added to every domain, `synth-intents-v5.2` became `synth-intents-v5.3` when the
`contrast` goal (confusable-intent menus and minimal-edit counterfactuals) was added, and
`synth-intents-v5.3` became `synth-intents-v5.4` when 1,299 JevBench-style hard items
(rule documents, state tracking, lead qualification and score items, 100 of them with
two-level soft answers) and five new domains were added, and `synth-intents-v5.4`
became `synth-intents-v5.5` when every counterfactual material recorded the material
it was edited from (`contrast_of`), so training puts the original and the edited item
into the same batch: drawing an edited item also draws the same question on its source
material, and the pair counts as two questions of the batch (`split.synth_v5_contrast_items`,
and `synth_v5_contrast_draws` per round with `--passes`), and `synth-intents-v5.5`
became `synth-intents-v5.6` when every question gained ability categories
(`categories/<domain>.json`: fact_lookup, fact_compose, negation_update, intent,
computation_easy, computation_hard, plus long_context for materials of 1,000+ chars) and
1,188 `factread` materials were added: 297 long materials, each in four versions (the
original, two edits of different text facts that change the answer, one edit that keeps
it), none of which touches a digit, with 144 held-out groups kept out of training.
`--exclude-categories computation_easy,computation_hard` drops the items that need
arithmetic or date reasoning (`split.synth_v5_excluded_categories`; run dir `-xc...`).
`synth-intents-v5.6` became `synth-intents-v5.7` when 1,152 `scoresem` materials were
added: 192 score questions whose levels are described in words only (no counting), each
with a hard pair crossing to the adjacent level by one text edit, a hard anchor and three
soft materials between two adjacent levels; 48 held-out questions are kept out of training.
`synth-intents-v5.7` became `synth-intents-v5.8` when 5,758 `curric` materials were
added: 2,879 graded computation pairs (date deadlines and amount thresholds in five levels,
from symbolic relations to cross-month dates and discount-plus-surcharge amounts), each an
item plus a one-number flip with the opposite answer.
`synth-intents-v5.8` became `synth-intents-v5.9` when every curric binding gained `cot`, the
program-written working for that item; it is a training-only language-model target.
`synth-intents-v5.9` became `synth-intents-v5.10` when the curric working was rewritten in the
solver's own voice: what the question asks, the rule and each number quoted from the material,
the arithmetic step by step, and the matching option ("So I choose yes").
`synth-intents-v5.10` became `synth-intents-v5.11` when the phrasing of every step of that
working started to vary per item, so no fixed sentence opens or closes all of them.
`synth-intents-v5.11` became `synth-intents-v5.12` when 24,060 judge items were added
(judge_math, judge_code, judge_text and ordinal; data README section 19): bindings 94,453 ->
118,513, score items 3,023 -> 12,643. Runs before 2026-10-21 (up to v75) used v5.11, data
repository at b06ff62.
`synth-intents-v5.12` became `synth-intents-v5.13` when sampling was grouped by source x
question type and the hard sources were raised to about 7,200 items each (curric choice
items, ordinal, and nine hand-written sources: data README section 20): bindings 118,513 ->
169,419, score items 12,643 -> 37,710. Runs before 2026-10-22 used v5.12 or older.
`--dataset synth-v5.1` through `synth-v5.12` are rejected with a pointer to the new name; `synth-v5`
remains an alias of the current data. Complete states saved on older data fail the
resume fingerprint check against v5.13. A `--mix` written for v5.2 must add a
`contrast=` share, since every goal in the data keeps a positive share. Sors-2B was
trained on v5.3; reproduce it with the data repository at f613f66 (before the hard items).
The v5.4 model of 2026-10-10 (2,600 steps) used the data repository at e01bf36; the v5.5
runs of 2026-10-11/12 (v55, v56c, v56s, v57) used eca83c1.

Two options act on these pairs. `--contrast-balance` makes drawing the original item bring its
edited item too, so both appear equally often (without it the original appears once more per
edit, and since their answers are opposite, cross-entropy leans toward the original's answer;
`split.synth_v5_contrast_balance` records it). `--contrast-weight w` adds a pairwise
log-odds term: with answers A (original) and B (edited), each row's d = log p(B) - log p(A)
is averaged over the rows that answer B and over those that answer A, and the loss is
w · softplus(m - (mean_B d - mean_A d)) with m = `--contrast-margin` (default 2). Features
the two materials share cancel in the difference, so the gradient comes from the edited
fact. Pairs with the same answer, soft labels or a missing option row are skipped. A batch
keeps each pair and its consistency copies adjacent and, with `--accumulate-gradients`,
in one `--micro-tokens` group (required). TensorBoard logs `train/contrast_margin`.
`--contrast-pairs n` (with `--passes`) builds every batch the same way: exactly n
original/edited pairs, and the remaining slots split among goals in proportion to their
single draws per round (largest remainder, at least one each; recorded as
`split.synth_v5_quota`). Each goal and the pairs keep their own shuffled queue, so the
pair term is present in every step at the same size instead of 0-9 pairs per batch.

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
  --dataset synth-v5.13+sharc+toolace \
  --dataset-weights synth-v5.13=27,sharc=6,toolace=3 \
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
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/test_train_cli.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/test_dataset_paths.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/test_external_resume.py
```

Schema and authoring-tool tests are maintained in the dataset repository.
Loader and sampler tests remain here; their small temporary fixtures reuse the
schema from the selected asset checkout. Benchmark integration tests additionally
need the appropriate existing third-party caches.
