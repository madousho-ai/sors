# SORS

**State-conditioned Option Ranking System**

A Jev-style decision model you can train on one laptop GPU.

[![License: MIT](https://img.shields.io/badge/license-MIT-0a0a0a.svg?style=for-the-badge&labelColor=000000)](LICENSE)
[![Base: Qwen3-0.6B-Base](https://img.shields.io/badge/BASE-Qwen3--0.6B--Base-0a0a0a.svg?style=for-the-badge&labelColor=000000)](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
[![Python 3.13](https://img.shields.io/badge/PYTHON-3.13-0a0a0a.svg?style=for-the-badge&labelColor=000000)](.python-version)
[![Trainable: 0.43%](https://img.shields.io/badge/TRAINABLE-0.43%25-0a0a0a.svg?style=for-the-badge&labelColor=000000)](#how-it-works)

**English** | [简体中文](README.zh-CN.md)

SORS is an open-source implementation of the [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) decision model, built on Qwen3 base models. You hand it a state and a set of typed questions; it answers each one with a probability distribution over the options you defined, and nothing else.

The premise is that the base model already knows the answers. What it lacks is a way to say so in a form software can consume — its answer is a string, and its probability mass is spread over 151936 vocabulary entries. **So the only thing trained here is the output format.** The knowledge stays in weights that never receive a gradient: 2.56 M parameters move, 0.43% of the model.

## Highlights

- 256 anonymous decision slots living in Qwen3's unused vocabulary rows. No embedding resize, no new head — the caller binds slots to option names per request.
- Only the output format is trained: 259 embedding rows plus a LoRA on attention. The 596 M-parameter backbone is frozen.
- 17 of Banking77's 77 intents are held out of training entirely. The model reads their names for the first time at evaluation and scores **0.921**, against 0.948 on the intents it trained on.
- One KV-cache prefix per state, one branch per question, so N questions about one state cost one pass over that state.
- 2000 steps in 14 minutes on an 8 GB laptop GPU, with a thermal gate for machines that cook.
- Zero-shot baseline scripts that dump hidden states and logits, so probes, temperature scaling and ablations fit offline.

## Quick Start

You'll need Python 3.13 and [uv](https://docs.astral.sh/uv/). A CUDA GPU with 6 GB is enough for the default recipe.

```bash
git clone <this repo> sors && cd sors
uv sync
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset banking77 --steps 2000
```

Banking77 downloads itself into `data/banking77` from a pinned upstream commit, checked by md5 and row count. You'll see a progress line every 20 steps and a full evaluation every 100:

```text
step  1980  loss 0.0304  845s  tctl 66°C
{"step": 2000, "train_loss": 0.0299, "t": 852.1, "tctl_c": 65.8, "thermal_waits": 10,
 "seen":   {"accuracy": 0.948, "nll": 0.176, "ece": 0.009, "n": 2400},
 "unseen": {"accuracy": 0.921, "nll": 0.296, "ece": 0.029, "n": 680}}
```

`seen` is held-out *messages* from the 60 intents used in training. `unseen` is the 17 intents that never appeared during training at all — neither their examples nor their names. The gap between those two columns is the whole point: 0.921 says the matching came from the base model, and training supplied the answer format.

Watch it live with `.venv/bin/tensorboard --logdir runs`.

```bash
# both datasets at once, one set of slots serving two question types
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset both --steps 2000

# readout only — backbone bit-for-bit unchanged, only the 259 embedding rows move
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset banking77 --trainable d-only --steps 2000

# evaluate a checkpoint on another task without training it
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset boolq --init runs/<run>/trained.pt --steps 0
```

## How It Works

A request becomes one prompt. The answer token sits immediately after `Answer:`; one forward pass, read the logits at that position, softmax over this question's slot ids:

```text
Passage: <state>

Question: <question>
Options:
<|D0|>. no
<|D1|>. yes

Answer:
```

**The 256 slots.** Qwen3's tokenizer occupies 151669 of the 151936 rows its embedding matrix declares, leaving 267 unused. 256 slot tokens `<|D0|>` … `<|D255|>` plus 3 type markers fit in that tail, so the matrix keeps its shape and no checkpoint has to be resized. The slot ↔ option binding is resampled for every training example, so an intent lands on a different slot each time it appears — what gets learned is "point at the matching entry among the ones I was given", which is what has to hold when the caller defines the options.

**One parameter, both ends.** `tie_word_embeddings` holds for Qwen3, so the vector read off a menu line and the vector scored at the answer position are the same row. Scoring slot *k* is `h · E[D_k]`, algebraically the *k*-th row of an `nn.Linear(hidden, 256)` head. `SlotEmbedding` does the frozen lookup and overwrites the slot positions with its own 259 × 1024 parameter; `SlotHead` computes the frozen `h @ Wᵀ` and replaces those 259 columns with `h @ rowsᵀ`, referencing the same tensor. AdamW therefore carries moments for 259 rows instead of 151936 — the earlier version marked the whole matrix trainable and zeroed the gradient with a hook, which trains identically and costs 1.2 GiB of optimizer state that is 99.8% zeros.

**The loss** is a softmax restricted to the k slots present in that example, cross-entropy to the correct one. The other ~151k vocabulary entries never enter the denominator: the training shape matches the inference shape, where the caller gives k options and the answer must be one of them.

**Question isolation and the state prefix.** `context-first` (default) puts the state ahead of the menu, so N questions about one state share a single KV-cache prefix — `cache.py` runs the state once and branches per question, each branch seeing the state and its own question. `menu-first` is the control group; it loses on all 9 evaluation points (0.888 vs 0.903 held-out accuracy, NLL 0.341 vs 0.310) and has no shareable prefix.

**Question types.** `<|choice|>` is k unordered options, `<|bool|>` is a choice at k=2, `<|score|>` is k ordered levels with the expectation as the score. Under `--type-marker` the question label is written `Question (<|bool|>):` and those rows train alongside the slots. `<|score|>` currently holds its id and nothing more; see [Limitations](#limitations).

## Results

The default recipe is `--trainable attn --lr-schedule cosine --layout context-first`, which is what the tables below use: Qwen3-0.6B-Base, 2000 steps, LoRA r=8 on attention, cosine LR with 100 warmup steps, 10-option evaluation menus.

**Banking77** — 77 intents, 17 held out of training entirely:

| | accuracy | NLL | Brier | ECE |
|---|---|---|---|---|
| seen intents (n=2400) | 0.948 | 0.176 | 0.081 | 0.009 |
| **held-out intents (n=680)** | **0.921** | 0.296 | 0.126 | 0.029 |
| held-out, before training | 0.206 | 2.800 | 0.942 | 0.241 |

Chance on a 10-option menu is 0.100.

**BoolQ** — yes/no as a 2-option menu, validation n=3270, majority class 0.622:

| | accuracy | AUROC | Brier | ECE |
|---|---|---|---|---|
| trained 1500 steps | 0.805 | 0.877 | 0.277 | 0.032 |
| `--dataset both`, 2000 steps | 0.786 | 0.857 | 0.298 | 0.036 |
| Banking77 checkpoint, no BoolQ training | 0.612 | 0.645 | 0.502 | 0.168 |

The last row is a checkpoint trained only on customer-service intents, pointed at reading comprehension with `--steps 0`. AUROC 0.645 says the slot mechanism crossed the task boundary; accuracy just under the 0.622 majority rate says the task itself still has to be trained.

**Trainable surface** — how much of the model needs to move:

| `--trainable` | what moves | seen | held-out | peak VRAM |
|---|---|---|---|---|
| `d-only` | 259 embedding rows; backbone frozen | 0.812 | 0.813 | 4.1 GiB |
| `attn` (default) | the above + LoRA on `q/k/v/o` | 0.934 | 0.871 | 5.0 GiB |
| `attn-mlp` | the above + LoRA on `gate/up/down` | 0.945 | 0.878 | 6.0 GiB |

`d-only` reaches 0.813 on intents it has never seen with the backbone bit-for-bit unchanged — that number is pure readout. Attention is where routing lives, which is why opening it is worth +0.058 and opening the MLPs on top of it is worth +0.007. These three ran before the layout flag existed, under `menu-first` with a constant LR, so compare them to each other rather than to the tables above.

**Wall clock**, one RTX 3070 Ti Laptop (8 GB), including thermal pauses: Banking77 2000 steps 14.2 min, BoolQ 1500 steps 16.5 min, `both` 2000 steps 23.0 min.

## Training

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --help
```

The knobs that matter: `--dataset banking77 | boolq | both`, `--trainable d-only | attn | attn-mlp`, `--layout context-first | menu-first`, `--type-marker`, `--held-out 17`, `--k-min 2 --k-max 10` (training menu length, resampled per example), `--k-eval 10`.

Defaults: LoRA r=8, alpha=16, dropout=0.05; `1e-4` for LoRA and `1e-3` for the embedding rows, which start from scratch and need a faster clock; cosine decay with 100 warmup steps. A constant LR left the held-out curve swinging ±3 points between evaluations, larger than the n=680 noise floor.

Each run writes `runs/<timestamp>-<dataset>-<trainable>-<schedule>-<layout>/`:

| file | contents |
|---|---|
| `log.jsonl` | one line per evaluation; step 0 is the pre-training or post-`--init` baseline |
| `result.json` | arguments, class split, full evaluation history, peak VRAM |
| `trained.pt` | LoRA weights plus the 259 embedding rows — the base model reloads from `--model` |
| `tb/` | TensorBoard: `train/loss`, `train/lr_*`, `eval/<set>/<metric>`, `sys/tctl_c` |

`--init runs/<run>/trained.pt` resumes from a checkpoint; combined with `--steps 0` it evaluates without training.

## Evaluation

Every evaluation reports accuracy, top-5, NLL, Brier, ECE and mean confidence over the slot distribution; binary sets add AUROC, predicted positive rate and binary Brier. `n_saturated` counts rows whose correct-class probability fell below the NLL clamp, which makes the distortion in that metric auditable.

`scripts/baseline-boolq.py` and `scripts/baseline-banking77.py` measure the untrained readout in the same deployment shape. They are standalone and import nothing from the package.

BoolQ validation, n=3270, majority class 0.622:

| model | accuracy | Brier | ECE | AUROC |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.650 | 0.213 | 0.058 | 0.709 |
| Qwen3-0.6B (instruct) | 0.657 | 0.255 | 0.206 | 0.721 |
| Qwen3-1.7B-Base | 0.785 | 0.149 | 0.020 | 0.858 |
| Qwen3-1.7B (instruct) | 0.754 | 0.231 | 0.225 | 0.845 |

Instruct tuning leaves discriminative power roughly where it was and wrecks calibration — ECE 0.206 against 0.058 at 0.6B. The base models are the starting point here for that reason.

Banking77 test, n=3080, 77-option menu, Qwen3-1.7B-Base: accuracy 0.260 / 0.269 across two permutation seeds, top-5 0.443 / 0.470, with 45% of the probability mass landing outside the 77 codes. Chance is 0.013. The knowledge is there; the readout is what's missing.

Each baseline writes a JSON of metrics plus an NPZ holding hidden states, slot logits, full-vocabulary logsumexp and top-k, so probes, temperature scaling and ablations fit offline without a second forward pass. The Banking77 script binds each intent to a two-letter code that tokenizes as a single token both with and without a leading space, and runs two permutation seeds to establish the preference-noise floor.

## Export and serve a complete model

Publish one complete model directory. Its `model.safetensors` contains trained
parameters, frozen embeddings and decision layers together; shared tensors are
stored once. `config.json` records the architecture and parameter precision, and
the tokenizer files preserve the trained token IDs. The directory is sufficient
for offline loading with the Sors runtime.

```bash
PYTHONPATH=src .venv/bin/python scripts/export-model.py \
  --init runs/<run>/trained.safetensors \
  --base-model /path/to/pinned-base \
  --out ../Sors-0.8B --device cpu --dtype bfloat16 --local-files-only

PYTHONPATH=src .venv/bin/python scripts/serve.py \
  --init ../Sors-0.8B --local-files-only --warmup
```

The exporter accepts a new output directory and preserves the loaded model.
The serving CLI also accepts complete Sors models hosted on Hugging Face:

```bash
PYTHONPATH=src .venv/bin/python scripts/serve.py \
  --init SakuraYuyuko/Sors-0.8B --demo --port 9999 --host 0.0.0.0
```

Hub references use the standard Hugging Face cache and the current machine's
saved login or `HF_TOKEN` for private repositories. `--local-files-only` selects
an existing cached snapshot; `--revision` selects a branch, tag or commit.
Existing local paths take precedence. Models downloaded with `--local-dir` can
be served by passing that directory directly, such as `--init data/models/Sors-0.8B`.

Complete-directory serving preserves the saved mixed precision, including FP32
decision layers. Training checkpoints retain their existing format for evaluation
and recovery; their serving path continues to accept a separate `--base-model`.
See [container deployment](docker/README.md) for the same model directory in
Docker and Podman.

## Limitations

- **Deployment uses the Sors runtime.** The service exposes `/v1/systemone`; custom decision architectures are loaded through the project's model-directory loader.
- **`score` is unimplemented.** `<|score|>` holds a token id. An ordered loss and a dataset with ordered levels are both missing, so only `choice` and its k=2 case are trained today.
- **BoolQ exceeds pure readout.** A probe fitted on frozen hidden states of the same base model tops out at AUROC 0.745; the trained run reaches 0.877. Attention LoRA moved representations there, so on that task the claim "only the format is trained" is too strong.
- **One base size measured end to end.** Everything trained is 0.6B. The zero-shot ladder says a 1.7B base starts at AUROC 0.858 on BoolQ, above where 0.6B's trained readout lands — base-model scale is the larger lever, and it is untested here past the baselines.
- **Single seed.** One class split (17 intents at seed 0), one run per configuration. The differences between adjacent rows in the trainable-surface table are within a few times the n=680 noise floor.
- **Baseline numbers need a rerun to reproduce.** `results/` is gitignored, and the AUROC and probe figures above were fitted offline from the NPZ dumps. The scripts are here; the artifacts are not.
- **Evaluation menus are 10 options.** The 256-slot capacity is exercised in training up to `--k-max`, never at full width.
- **Laptop scale.** One 8 GB GPU, `max_length` 512, batch size 8, thermal-gated.

## Development

The Python package is `sors`; imports use `from sors...`. Configure the service
with `SORS_API_KEY` and the external dataset checkout with `SORS_DATASETS_DIR`
(see [dataset setup](DATASETS.md)). Existing API-key and dataset environment
variables remain supported as fallbacks, with the SORS variables taking precedence.

Pure functions only — no GPU, no network, no test framework:

```bash
for f in tests/test_*.py; do PYTHONPATH=src .venv/bin/python "$f"; done
```

Each file is executable on its own and prints one line per case. `ThermalGuard` reads Tctl from `k10temp` before every step and every evaluation and sleeps while it sits above `--temp-max` (85 °C default); the pause count is logged as `sys/thermal_waits`, and machines without that sensor pass straight through.

| path | contents |
|---|---|
| `src/sors/core/tokens.py` | the 256 slot tokens and 3 type tokens, installed into a tokenizer |
| `src/sors/core/menu.py` | menu sampling, class splits, the dataset-agnostic `LabeledSet` |
| `src/sors/core/prompt.py` | prompt templates and the two layouts |
| `src/sors/core/batch.py` | collation, left padding, answer position pinned to the last column |
| `src/sors/core/model.py` | `SlotEmbedding`, `SlotHead`, LoRA wiring, parameter groups |
| `src/sors/training/loss.py` | slot-restricted cross-entropy |
| `src/sors/evaluation/metrics.py` | accuracy, top-5, NLL, Brier, ECE, AUROC |
| `src/sors/training/loop.py`, `src/sors/core/checkpoint.py` | training loop, evaluation, checkpoint save and load |
| `src/sors/core/cache.py` | one KV-cache prefix, N question branches |
| `src/sors/training/thermal.py` | temperature gate |
| `src/sors/data/banking77.py`, `src/sors/data/boolq.py` | dataset adapters |

## Credits

Base models from [Qwen](https://huggingface.co/Qwen/Qwen3-0.6B-Base). Datasets: [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI) and [BoolQ](https://huggingface.co/datasets/google/boolq) (Google). API shape after TypeSafe's [System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev); unaffiliated with TypeSafe AI, and the training method here is this project's own.

Related work: [kev](https://github.com/jaredpalmer/kev) trains the same class of model at 0.8B–9B and serves it behind a System One-compatible API.

The name: **SORS** stands for **State-conditioned Option Ranking System**. Latin *sors* evokes lots, chance and fate: a fitting name for a model that reads the current state and weighs the available choices. The Python package and project identifier are `sors`.

## License

[MIT](LICENSE).
