#!/usr/bin/env python
"""训练 256 槽决策头 (LoRA + D-token 嵌入).

--dataset 是用加号连起来的训练集列表, 一个 batch 里各占一份:
  banking77   Banking77, 按类留出 17 个测泛化 (seen / unseen)
  boolq       BoolQ, k=2, 逐条问句
  synth       外部资产 synth-intents-v2.5, 4096 个合成意图 × 3 条消息. 每条消息两道题, 各占一份:
              菜单题只列正确意图所在领域的意图 (k 256 即整个领域 256 个), 二元题问消息里的一个细节 (no / yes)
  synth-menu  synth 只要菜单题, 不要二元题 (不与 synth 并用)
  synth-v3    外部资产 synth-intents-v3 的五个领域 (工单、酒店文档、浏览器 agent、安全运维、编码与 CI):
              每份 state 带自己的题, 问法与选项各不相同 (2..107 项). 五个领域各占这一份的五分之一;
              写明的答案是硬标签, 没写明的题用参考模型的分布当软标签 (见 sors/data/synth_v3.py)
  synth-v5.11  外部资产 synth-intents-v5.11: 客户消息、工单与规则材料统一成「材料 + 绑定的题」；synth-v5 保留为别名.
              v5.1 改名为 v5.2 加了难题, v5.2 改名为 v5.3 加了 contrast 目标, v5.3 改名为 v5.4 加了
              JevBench 难度风格的补充题, v5.4 改名为 v5.5 给反事实材料记下来源 (contrast_of, 原题与改动题同批),
              v5.5 改名为 v5.6 给每道题标了能力类别 (--exclude-categories) 并加了只考读事实的改动组 factread,
              v5.6 改名为 v5.7 加了纯语义、带边界的评分题 scoresem, v5.7 改名为 v5.8 加了分级计算题 curric, v5.8 改名为 v5.9 给 curric 绑定加了计算过程 cot, v5.9 改名为 v5.10 把 cot 改成第一人称、引用原句、对照选项, v5.10 改名为 v5.11 让 cot 的说法按题变换;
              旧名字报错, 免得读到新数据.
              选项写成「键: 说明」, 与推理服务相同. 先按 --mix 的配比挑训练目标 (long_menu / breadth / complex /
              edge_case / long_context / ambiguous), 再领域、题型、材料各自均分; 或者给 --passes 按轮抽, 一轮每个绑定都出,
              列出的目标多出几遍. --mask-descriptions 按比例遮掉可遮的题的选项说明;
              --consistency 下材料有几种说法时, 配对的第二份读另一种说法. --fallback-rate 默认 0.5,
              随机加入题目标记的 Other / 信息不足选项；既有 other 配对每次随机选一版，一次抽取出一道题
  massive     MASSIVE 的 train 分区, 60 个语音助手意图
  contractnli / maud / legalbench / sharc / conditionalqa / mind2web / toolace / quality / reclor / logiqa2
              官方决策子任务，从 --public-manifest 声明的本地 train 文件读取；全量菜单、硬标签，
              默认按数据集各占一份，--dataset-weights 可指定配比。筛选统计和原始文件 SHA256 写进 result.json 的 split.public_decisions。
每个训练集各自组菜单, 干扰项不跨集合抽.
--datasets-dir 指定外部数据仓库根目录，默认取环境变量或同级 decidophobia-dataset；详见 DATASETS.md。
"both" 仍可用, 等于 banking77+boolq.

数据准备默认用 2 个子进程 (--data-backend process, 不与训练线程抢 GIL)，提前准备最多 2 批；
--data-workers / --data-prefetch 可调，--data-backend thread 换回进程内线程。--data-workers 0 使用同步准备。
抽样顺序与有效 batch 保持原样，worker 只做分词和组批。CUDA 上再提前 --device-prefetch 批锁页拷进显存。

--eval 是评估集列表, 与训练集无关, 默认 banking77+banking77-desc+massive+massive-desc+boolq+simple+jevbench (见 build_eval_sets).
训练中 (step 0 与每 --eval-every 步) 只跑探针: 每个评估集固定 --probe-size 道题, 固定 --probe-passes 种随机排法
(行打乱、D 码随机), 报 accuracy / agree / js. 最后一步再跑全量评估集: 部署形态 (连续编号) 下的正确率那一套,
与全量题的同一种一致性. 见 sors.training.loop.eval_record.

  PYTHONPATH=src .venv/bin/python scripts/train.py --dataset synth --grad-ckpt --steps 2000
  PYTHONPATH=src .venv/bin/python scripts/train.py --init runs/<run>/trained.safetensors --steps 0   # 只评估: 探针 + 全量
  .venv/bin/tensorboard --logdir runs

产出 (--out 目录):
  log.jsonl     每个评估点一行 (step 0 是训练前 / 加载后的基线): consistency 是探针; 最后一行另有 eval 与 consistency_full
  result.json   配置 + 类切分 + 全部评估记录
  trained.safetensors  LoRA 权重 + D 行嵌入, 元数据里记 d_ids / 训练 config / LoRA 形状 (--steps 0 时不写, 保住 --init 那份).
                旧 run 的 trained.pt 仍能给 --init
  checkpoints/step-<步数>.safetensors  训练途中的档, 格式同上, 每 --save-every 步一份 (默认跟 --eval-every 走), 最后一步不在这里
  tb/           TensorBoard 事件
"""

from __future__ import annotations

import argparse
import json
import pathlib
import hashlib
import random
import sys
import time
from dataclasses import asdict
from fractions import Fraction

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from sors.core.attention import add_attention_arguments, load_causal_lm
from sors.core.checkpoint import checkpoint_adapter, checkpoint_architecture, load_trained, save_trained
from sors.core.decision import DecisionConfig, architecture_config, decision_config
from sors.core.batch import group_contrasts
from sors.core.menu import RandomCodes, class_split, menu_k_range, with_partners
from sors.core.model import TRAINABLE, keep_layer_activations, prepare_model
from sors.core.prompt import DEFAULT_LAYOUT, LAYOUTS
from sors.core.tokens import install_context_tokens, install_d_tokens, install_type_tokens
from sors.data.public_decisions import DEFAULT_MANIFEST, PUBLIC_DATASETS, load_public_decisions
from sors.data.paths import add_datasets_argument, datasets_root
from sors.evaluation.scoring import EvalSet
from sors.training.loop import TrainConfig, train
from sors.training.prefetch import validate_data_preparation
from sors.training.loss import LOSSES
from sors.training.thermal import ThermalGuard

KNOWN = ("banking77", "boolq", "synth", "synth-menu", "synth-v3", "synth-v5.11", "synth-v5", "massive") + PUBLIC_DATASETS
RETIRED = {"synth-v5.1": "synth-v5.1 was renamed synth-v5.2 through synth-v5.11; use --dataset synth-v5.11",
           "synth-v5.2": "synth-v5.2 was renamed synth-v5.3 when the contrast goal was added, then up to synth-v5.10; "
                         "use --dataset synth-v5.11",
           "synth-v5.3": "synth-v5.3 was renamed synth-v5.4 when the JevBench-style hard items were added, then up to "
                         "synth-v5.11; use --dataset synth-v5.11",
           "synth-v5.4": "synth-v5.4 was renamed synth-v5.5 when counterfactual materials gained contrast_of, then up "
                         "to synth-v5.11; use --dataset synth-v5.11",
           "synth-v5.5": "synth-v5.5 was renamed synth-v5.6 when every question gained ability categories and the "
                         "factread groups were added, then synth-v5.11; use --dataset synth-v5.11",
           "synth-v5.6": "synth-v5.6 was renamed synth-v5.7 when the semantic score questions (scoresem) were added, "
                         "then synth-v5.11; use --dataset synth-v5.11",
           "synth-v5.7": "synth-v5.7 was renamed synth-v5.8 when the graded computation pairs (curric) were added, "
                         "then synth-v5.11; use --dataset synth-v5.11",
           "synth-v5.8": "synth-v5.8 was renamed synth-v5.9 when curric bindings gained program-written working (cot), "
                         "then synth-v5.11; use --dataset synth-v5.11",
           "synth-v5.9": "synth-v5.9 was renamed synth-v5.10 when the curric working was rewritten in the solver's own "
                         "voice, then synth-v5.11; use --dataset synth-v5.11",
           "synth-v5.10": "synth-v5.10 was renamed synth-v5.11 when the curric working's phrasing was varied per item; "
                          "use --dataset synth-v5.11"}
KNOWN_EVAL = ("banking77", "banking77-desc", "massive", "massive-desc", "boolq", "simple", "jevbench")


def _parse_list(spec: str, known: tuple[str, ...], flag: str) -> list[str]:
    names = spec.split("+")
    bad = [n for n in names if n not in known]
    if bad or len(set(names)) != len(names):
        raise SystemExit(f"{flag}: unknown or repeated {bad or names}; use + to combine {known}")
    return names


def parse_datasets(spec: str) -> list[str]:
    retired = [RETIRED[n] for n in spec.split("+") if n in RETIRED]
    if retired:
        raise SystemExit(f"--dataset: {retired[0]}")
    names = _parse_list("banking77+boolq" if spec == "both" else spec, KNOWN, "--dataset")
    names = ["synth-v5" if name == "synth-v5.11" else name for name in names]
    if len(set(names)) != len(names):
        raise SystemExit("--dataset: synth-v5.11 and synth-v5 refer to the same dataset; pick one")
    if {"synth", "synth-menu"} <= set(names):
        raise SystemExit("--dataset: synth already includes synth-menu's menu questions; pick one")
    return names


def parse_dataset_weights(spec: str | None, names: list[str]) -> dict[str, Fraction] | None:
    """Require an explicit positive weight for every selected dataset; normalize the v5 alias."""
    if spec is None:
        return None
    weights = {}
    try:
        for entry in spec.split(","):
            name, value = (part.strip() for part in entry.split("="))
            if name in RETIRED:
                raise ValueError(RETIRED[name])
            name = "synth-v5" if name == "synth-v5.11" else name
            if name in weights:
                raise ValueError(f"repeated dataset {name}")
            weight = Fraction(value)
            if weight <= 0:
                raise ValueError(f"{name} needs a positive weight")
            weights[name] = weight
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"--dataset-weights: {exc}") from exc
    if set(weights) != set(names):
        raise ValueError(f"--dataset-weights must name every selected dataset exactly once: {names}")
    return weights


def dataset_parts(n: int, names: list[str], weights: dict[str, Fraction] | None) -> list[int]:
    """Allocate whole questions deterministically, then share each dataset's quota among its samplers."""
    if weights is None:
        parts = [n // len(names)] * len(names)
        parts[0] += n - sum(parts)
        return parts
    unique = list(dict.fromkeys(names))
    total = sum(weights.values())
    exact = {name: n * weights[name] / total for name in unique}
    quota = {name: int(exact[name]) for name in unique}
    largest = sorted(unique, key=lambda name: exact[name] - quota[name], reverse=True)
    for name in largest[:n - sum(quota.values())]:
        quota[name] += 1
    parts = [0] * len(names)
    for name in unique:
        indices = [i for i, value in enumerate(names) if value == name]
        for i in indices:
            parts[i] = quota[name] // len(indices)
        parts[indices[0]] += quota[name] % len(indices)
    return parts


def build_eval_sets(args, b77_test=None, boolq_val=None) -> dict[str, EvalSet]:
    """--eval 列的评估集, 与训练集无关. 菜单全量、连续编号; 每个集合的菜单用 Random(seed + 菜单长度) 组,
    MASSIVE 那份因此与 scripts/eval-massive.py 的 k=60 逐题相同. 训练里已读过的 split 可以传进来复用.
    banking77 / massive 的菜单显示原始 label 名; -desc 那两个显示「原始名: description」(description 在
    外部数据仓库的 label-descriptions, 写法同推理服务的选项), 菜单用同一个 Random 组, 于是与不带 -desc 的逐题相同
    (消息、选项顺序、正确位置), 只换了选项的文字.
    simple 是外部数据仓库的 synth-simple-eval, 菜单写死在文件里 (不受 --k-eval / --seed 影响), 展开成每个菜单长度一个集合.
    jevbench 是 JevBench 的三档公开题 (第一次用时下载到 data/jevbench, 不进仓库), 同样写死, 展开成 jevbench_easy / _original / _hard;
    hard 的提示最长 3838 token, 批取四分之一, 一批的 token 数与 BoolQ 那档相当."""
    out = {}
    for name in _parse_list(args.eval, KNOWN_EVAL, "--eval"):
        if name == "simple":
            from sors.data.simple_eval import load_simple_eval

            for sub, exs in load_simple_eval(datasets_dir=getattr(args, "datasets_dir", None)).items():
                pos = 1 if sub == "simple_bool" else None
                out[sub] = EvalSet(exs, args.eval_batch_size, pos_class=pos)
            continue
        if name == "jevbench":
            from sors.data.jevbench import load_jevbench

            for sub, exs in load_jevbench().items():
                bs = max(1, args.eval_batch_size // 4) if sub == "jevbench_hard" else args.eval_batch_size
                out[sub] = EvalSet(exs, bs)
            continue
        if name == "banking77":
            if b77_test is None:
                from sors.data.banking77 import load_banking77

                b77_test = load_banking77(args.data_dir)[1]
            te, bs, pos = b77_test, args.eval_batch_size, None
        elif name == "banking77-desc":
            from sors.data.banking77 import load_banking77

            te, bs, pos = load_banking77(args.data_dir, labels="desc", datasets_dir=getattr(args, "datasets_dir", None))[1], args.eval_batch_size, None
        elif name in ("massive", "massive-desc"):
            from sors.data.massive import load_massive

            labels = "desc" if name == "massive-desc" else "raw"
            te, bs, pos = load_massive(partition="test", labels=labels, datasets_dir=getattr(args, "datasets_dir", None)), args.eval_batch_size, None
        else:
            if boolq_val is None:
                from sors.data.boolq import load_boolq

                boolq_val = load_boolq()[1]
            te, bs, pos = boolq_val, max(1, args.eval_batch_size // 2), 1
        classes = list(range(len(te.names)))
        k = min(args.k_eval, len(classes))
        out[name] = EvalSet(te.build_examples(classes, (k, k), random.Random(args.seed + k)), bs, pos_class=pos)
    return out


def build_data(args):
    """每个数据集给一个 sampler 和若干评估集. 返回 (sample_fn, eval_sets, split_info).
    只读数据, 不碰模型 —— tests/test_train_cli.py 直接调它."""
    datasets = parse_datasets(args.dataset)
    try:
        weights = parse_dataset_weights(getattr(args, "dataset_weights", None), datasets)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if not 0.0 <= args.random_codes <= 1.0:
        raise SystemExit(f"--random-codes is a share of training menus, 0..1; got {args.random_codes}")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise SystemExit(f"--label-smoothing is the share of the target spread over the menu, 0..1; "
                         f"got {args.label_smoothing}")
    if args.consistency < 0:
        raise SystemExit(f"--consistency is the weight of the JS term, >= 0; got {args.consistency}")
    if getattr(args, "contrast_weight", 0.0) < 0:
        raise SystemExit(f"--contrast-weight is the weight of the counterfactual pair term, >= 0; got {args.contrast_weight}")
    if args.probe_size < 1 or args.probe_passes < 2:
        raise SystemExit(f"--probe-size must be >= 1 and --probe-passes >= 2 (one arrangement has nothing to compare); "
                         f"got {args.probe_size} and {args.probe_passes}")
    if args.micro_batches < 1:
        raise SystemExit(f"--micro-batches is how many length groups a step runs, >= 1; got {args.micro_batches}")
    if not 0.0 <= args.mask_descriptions <= 1.0:
        raise SystemExit(f"--mask-descriptions is a share of maskable questions, 0..1; got {args.mask_descriptions}")
    requested_fallback = getattr(args, "fallback_rate", None)
    fallback_rate = 0.5 if requested_fallback is None else requested_fallback
    if not 0.0 <= fallback_rate <= 1.0:
        raise SystemExit(f"--fallback-rate must be a probability in 0..1; got {fallback_rate}")
    if requested_fallback is not None and "synth-v5" not in datasets:
        raise SystemExit("--fallback-rate only applies to --dataset synth-v5.11 (legacy alias: synth-v5)")
    balance = getattr(args, "contrast_balance", False)
    if "synth-v5" not in datasets and (args.mix or args.mask_descriptions or args.passes or balance):
        raise SystemExit("--mix, --passes, --mask-descriptions and --contrast-balance only apply to --dataset synth-v5.11 "
                         "(legacy alias: synth-v5)")
    if args.mix and args.passes:
        raise SystemExit("--mix draws goals by weight and --passes draws in rounds; pick one")
    erng = random.Random(args.seed + 1)
    samplers, sampler_names, eval_sets, split_info = [], [], {}, {}
    ktr = menu_k_range(args.k_min, args.k_max)
    b77 = None
    if "banking77" in datasets:
        from sors.data.banking77 import load_banking77

        b77 = load_banking77(args.data_dir)
        tr, te = b77
        split = class_split(len(tr.names), args.held_out, seed=args.seed)
        kr = (args.k_eval, args.k_eval)
        eval_sets["seen"] = EvalSet(te.build_examples(split.train, kr, erng), args.eval_batch_size)
        eval_sets["unseen"] = EvalSet(te.build_examples(split.held_out, kr, erng), args.eval_batch_size)
        samplers.append(lambda n, rng: tr.sample_examples(split.train, ktr, n, rng, k_log=args.k_log))
        sampler_names.append("banking77")
        split_info = {"train": split.train, "held_out": split.held_out,
                      "held_out_names": [tr.names[c] for c in split.held_out]}
    if "synth" in datasets or "synth-menu" in datasets:
        # 每条消息两道题: 菜单题只列正确意图所在领域的意图 (k 256 即整个领域), 二元题问消息里的一个细节.
        # 两种题各占一个 sampler, 于是一批里各一半. synth-menu 只要菜单题, 一批全是它.
        from sors.data.synth import load_synth, load_synth_binary, sample_domain_menus

        synth, synth_domains = load_synth(datasets_dir=getattr(args, "datasets_dir", None))
        samplers.append(lambda n, rng: sample_domain_menus(synth, synth_domains, ktr, n, rng))
        sampler_names.append("synth" if "synth" in datasets else "synth-menu")
        if "synth" in datasets:
            synth_bin = load_synth_binary(datasets_dir=getattr(args, "datasets_dir", None))
            samplers.append(lambda n, rng: synth_bin.sample_examples([0, 1], (2, 2), n, rng))
            sampler_names.append("synth")
        split_info["synth_classes"] = len(synth.names)
    if "synth-v3" in datasets:
        # 题自带选项, 不组菜单: 每条先挑领域 (五个领域机会均等) 再挑题, 每次换问法、重新打乱选项.
        # --k-min / --k-max 管不到它, 菜单长度就是题的选项数.
        from sors.data.synth_v3 import load_synth_v3, sample_synth_v3

        v3 = load_synth_v3(datasets_dir=getattr(args, "datasets_dir", None))
        samplers.append(lambda n, rng: sample_synth_v3(v3, n, rng))
        sampler_names.append("synth-v3")
        split_info["synth_v3_items"] = sum(len(v) for v in v3.values())
    if "synth-v5" in datasets:
        # 两种抽法. --passes: 按轮抽, 一轮把每个绑定出一遍 (列出的目标出那么多遍), 一个绑定都不漏.
        # 否则按 --mix 的配比 (可随步数分段) 挑目标, 目标内领域、题型、材料各自均分. sampler 每调一次算一步.
        from sors.data.synth_v5 import (V5Rounds, V5Sampler, V5Strata, V5TypeStrata, drop_categories, load_synth_v5,
                                        parse_mix, parse_passes, parse_type_quota)

        v5 = load_synth_v5(datasets_dir=getattr(args, "datasets_dir", None))
        excluded = sorted(set(args.exclude_categories.split(","))) if getattr(args, "exclude_categories", None) else []
        if excluded:
            try:
                v5 = drop_categories(v5, set(excluded))
            except ValueError as e:
                raise SystemExit(f"--exclude-categories: {e}") from None
            split_info["synth_v5_excluded_categories"] = excluded
        if getattr(args, "distill_traces", None):
            from sors.data.synth_v5 import attach_traces
            with open(args.distill_traces) as stream:
                v5 = attach_traces(v5, [json.loads(line) for line in stream if line.strip()])
            split_info["synth_v5_traced_items"] = sum(it.traces is not None for it in v5)
            split_info["distill_traces"] = args.distill_traces
        goals = {it.goal for it in v5}
        split_info["synth_v5_items"] = len(v5)
        split_info["synth_v5_contrast_items"] = sum(it.contrast is not None for it in v5)
        split_info["synth_v5_fallback_rate"] = fallback_rate
        if balance:
            split_info["synth_v5_contrast_balance"] = True
        try:
            pairs = getattr(args, "contrast_pairs", 0)
            type_quota = getattr(args, "type_quota", None)
            if type_quota and (not args.passes or pairs or balance):
                raise SystemExit("--type-quota needs --passes, draws each source with its edit already and replaces "
                                 "--contrast-pairs / --contrast-balance")
            if pairs and (not args.passes or balance):
                raise SystemExit("--contrast-pairs needs --passes and already balances each source with its edit "
                                 "(drop --contrast-balance)")
            if type_quota:
                quota = parse_type_quota(type_quota, {it.qtype for it in v5})
                strata = V5TypeStrata(v5, parse_passes(args.passes, goals), quota, args.mask_descriptions,
                                      fallback_rate)
                if strata.batch_size != args.batch_size:
                    raise SystemExit(f"--type-quota adds up to {strata.batch_size} questions, --batch-size is "
                                     f"{args.batch_size}")
                split_info["synth_v5_passes"] = strata.passes
                split_info["synth_v5_type_quota"] = strata.quota
                split_info["synth_v5_type_pools"] = {t: len(v) for t, v in strata.pools.items()}
                split_info["synth_v5_pair_units"] = strata.pair_units
                samplers.append(strata)
            elif pairs:
                strata = V5Strata(v5, parse_passes(args.passes, goals), pairs, args.batch_size,
                                  args.mask_descriptions, fallback_rate)
                split_info["synth_v5_passes"] = strata.passes
                split_info["synth_v5_contrast_pairs"] = pairs
                split_info["synth_v5_quota"] = strata.quota
                split_info["synth_v5_pair_units"] = len(strata.pair_units)
                samplers.append(strata)
            elif args.passes:
                rounds = V5Rounds(v5, parse_passes(args.passes, goals), args.mask_descriptions, fallback_rate,
                                  contrast_balance=balance)
                split_info["synth_v5_passes"] = rounds.passes
                split_info["synth_v5_round"] = rounds.round_size
                split_info["synth_v5_contrast_draws"] = rounds.contrast_draws
                samplers.append(rounds)
            else:
                mix = parse_mix(args.mix, goals)
                samplers.append(V5Sampler(v5, mix, args.mask_descriptions, fallback_rate, contrast_balance=balance))
                split_info["synth_v5_mix"] = mix
        except ValueError as e:
            raise SystemExit(str(e)) from None
        sampler_names.append("synth-v5")
    if "massive" in datasets:
        # MASSIVE 的 train 分区 (11514 条, 60 意图). 上下文标签是 Voice command, 不与 banking77 并池:
        # 各自全量菜单 60 项. 它的 test 分区留给 scripts/eval-massive.py.
        from sors.data.massive import load_massive

        mtr = load_massive(partition="train")
        m_classes = list(range(len(mtr.names)))
        samplers.append(lambda n, rng: mtr.sample_examples(m_classes, ktr, n, rng))
        sampler_names.append("massive")
        split_info["massive_classes"] = len(m_classes)
    if "boolq" in datasets:
        from sors.data.boolq import load_boolq

        btr, bva = load_boolq()
        samplers.append(lambda n, rng: btr.sample_examples([0, 1], (2, 2), n, rng))
        sampler_names.append("boolq")
    for name in datasets:
        if name in PUBLIC_DATASETS:
            try:
                public = load_public_decisions(name, getattr(args, "public_manifest", DEFAULT_MANIFEST))
                largest = max(len(e.options) for e in public.examples)
                if largest > args.k_max:
                    raise ValueError(f"{name}: fixed official menus need --k-max >= {largest}; got {args.k_max}")
            except ValueError as exc:
                raise SystemExit(str(exc)) from None
            samplers.append(public.sample)
            sampler_names.append(name)
            split_info.setdefault("public_decisions", {})[name] = public.summary()
    eval_sets.update(build_eval_sets(args, b77[1] if b77 else None, bva if "boolq" in datasets else None))
    if args.eval_limit:
        for k, es in eval_sets.items():
            exs = list(es.examples)
            random.Random(args.seed + 2).shuffle(exs)
            eval_sets[k] = EvalSet(exs[: args.eval_limit], es.batch_size, es.pos_class)

    random_codes = RandomCodes(args.random_codes)  # 一个 run 一个: 它记着每个码上过几次菜单
    if weights is not None:
        parts = dataset_parts(args.batch_size, sampler_names, weights)
        split_info["dataset_batch"] = {name: sum(count for source, count in zip(sampler_names, parts)
                                                 if source == name) for name in dict.fromkeys(sampler_names)}

    def sample_fn(n, rng):
        """一批按 --dataset-weights 分配，默认各 sampler 平分，再打乱. --consistency > 0 时每道题后面紧跟它的另一种排法,
        一批 2n 条. --random-codes 只作用在这里, 两种排法各换各的码:
        评估集始终按位置编号 D0, D1, ..., 与部署时调用方写的菜单同形."""
        parts = dataset_parts(n, sampler_names, weights)
        out = [ex for s, c in zip(samplers, parts) for ex in s(c, rng)]
        rng.shuffle(out)
        if getattr(args, "contrast_weight", 0.0) > 0:
            out = group_contrasts(out)  # 原题与改动题挨着, 成对损失要两道题在同一次前向里
        if args.consistency > 0:
            out = with_partners(out, rng)
        return random_codes(out, rng)

    return sample_fn, eval_sets, split_info


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train SORS — State-conditioned Option Ranking System")
    add_attention_arguments(ap)
    add_datasets_argument(ap)
    ap.add_argument("--dataset", default="synth",
                    help="训练集, banking77 / boolq / synth / synth-menu / synth-v3 / synth-v5.11 / massive 用 + 连接; "
                         "也支持 contractnli / maud / legalbench / sharc / conditionalqa / mind2web / toolace / quality / reclor / logiqa2; "
                         "synth-v5 是 synth-v5.11 的兼容别名; both = banking77+boolq")
    ap.add_argument("--public-manifest", default=str(DEFAULT_MANIFEST),
                     help="官方决策数据的本地文件清单，只读取 split=train；见数据仓库 public-decisions/README.md")
    ap.add_argument("--dataset-weights", default=None,
                    help="数据集相对抽题权重，须列出全部训练来源，如 synth-v5.11=3,sharc=1；"
                         "按最大余数法分配整题，JS配对前生效；默认沿用各sampler等份")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--init", default=None,
                    help="从这份存档 (.safetensors 或旧的 trained.pt) 加载 LoRA + D 行再开始 (或配 --steps 0 只评估)")
    ap.add_argument("--trainable", default=None, choices=sorted(TRAINABLE),
                    help="放开的范围: d-only 只训 D 行; attn 加 attention LoRA; attn-mlp 再加 MLP LoRA; "
                         "attn-mlp-linear 再给 Qwen3.5 的线性注意力层挂 LoRA (in_proj_qkv/in_proj_z/out_proj); "
                         "full 主干全参 (每层全部权重 + 最后的 norm, 词表矩阵照旧只放 D 行, --lr-lora 管主干). "
                          "decision-only 只训新增决策层与 token 行. 不给 = slots 用 attn、新决策架构用 decision-only; 配 --init 时取档里记的")
    ap.add_argument("--architecture", choices=["slots", "minimal", "structural", "candidate"], default=None,
                    help="slots 原 D 码读出 (默认); minimal 串行菜单顶部决策块; structural 独立选项与中后层支路; "
                         "candidate 逐项联合编码完整上下文/问题/选项，再做集合交互")
    ap.add_argument("--candidate-prefix-cache", choices=["auto", "on", "off", "packed"], default=None,
                    help="candidate 前缀共享：auto 在推理/编码器冻结时启用（新模型默认）；on 强制且拒绝丢弃编码器梯度；off 完整前向对照；packed 前缀算一次、全部分支同一次前向接着跑，可训（仅 Qwen3.5）。配 --init 默认继承")
    ap.add_argument("--decision-feedback", action=argparse.BooleanOptionalAction, default=None,
                    help="独立开启/关闭选项向主干回写，默认关闭；新增分支的回写投影从零起步")
    ap.add_argument("--decision-dim", type=int, default=None, help="决策支路宽度，默认128")
    ap.add_argument("--decision-heads", type=int, default=None, help="决策注意力头数，默认4")
    ap.add_argument("--decision-blocks", type=int, default=None, help="决策块数，默认2；candidate 支持0以单独测标量读出")
    ap.add_argument("--decision-layers", default=None, help="零起始主干层下标，用逗号分隔；默认按架构选择")
    ap.add_argument("--decision-option-batch-size", type=int, default=None, help="独立选项编码每批最多几项，默认32")
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, choices=LAYOUTS,
                    help="context-first: 上下文在前, 前缀可作 KV cache 共享 (默认); menu-first: 菜单在前, 对照组")
    ap.add_argument("--type-marker", action="store_true",
                    help="问句标签写成 'Question (<|bool|>):', 类型 token 随 D 行一起训")
    ap.add_argument("--context-marker", action="store_true",
                    help="state 两头包 <|context_start|> <|context_end|>, 两个 token 随 D 行一起训; 超长只截包裹里的 state")
    ap.add_argument("--loss", default="vocab", choices=LOSSES,
                    help="vocab: 分母是整个词表, 普通 token 每步被压低, 答题位置只说 D 码; "
                         "all-slots: 分母是全部 256 个 D 槽; menu: 只在菜单 k 个槽上归一 (后两种是旧版)")
    ap.add_argument("--lora-r", type=int, default=None, help="不给 = 8; 配 --init 时取档里记的")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="不给 = 16; 配 --init 时取档里记的. peft 按 alpha / r 缩放, 改 r 时一起改才可比")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr-lora", type=float, default=1e-4,
                    help="主干那一组的学习率: LoRA 权重, --trainable full 时是主干全部权重. 离开 1e-4 时写进目录名")
    ap.add_argument("--lr-embed", type=float, default=1e-3)
    ap.add_argument("--lr-decision", type=float, default=None,
                    help="新增决策层单独的学习率 (minimal / structural / candidate); 不给 = 跟主干共用 --lr-lora. "
                         "给了写进目录名")
    ap.add_argument("--max-grad-norm", type=float, default=None,
                    help="全局梯度范数上限 (全部可训参数合起来); 不给 = 不裁剪. 给了写进目录名")
    ap.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--warmup", type=int, default=100, help="线性 warmup 步数")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit"],
                    help="adamw8bit = bitsandbytes 的 8-bit AdamW (动量存 uint8), 全参大模型省显存; 只在 CUDA 上, "
                         "要 uv sync --group gpu. 给了写进目录名 (-adam8)")
    ap.add_argument("--steps", type=int, default=300, help="0 = 只评估")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data-workers", type=int, default=2,
                    help="后台 CPU 分词/组批线程数 (>= 0)，0 = 同步准备；抽样仍按原顺序")
    ap.add_argument("--data-prefetch", type=int, default=2,
                    help="提前准备的批数上限 (>= 1)，独立于 batch-size 与 micro-batches")
    ap.add_argument("--data-backend", choices=["process", "thread"], default="process",
                    help="后台准备用子进程 (默认, 不占训练进程的 GIL) 还是训练进程内的线程")
    ap.add_argument("--device-prefetch", type=int, default=1,
                    help="CUDA 上提前几批锁页、在副 stream 上拷进显存 (>= 0)；0 = 用到时才拷")
    ap.add_argument("--micro-batches", type=int, default=2,
                    help="每步的 prompt 按长度分几组各自前向, 每组只补齐到组里最长的那条 (>= 1). "
                         "每行的 logits 与整批一次前向相同, loss 照旧对整批算、反传一次; 省下的是填充的计算. "
                         "1 = 整批一次 (旧行为)")
    ap.add_argument("--micro-tokens", type=int, default=0,
                    help="> 0 时改按 token 预算分组: 每组行数 × 组里最长不超过它, 先求组数最少再求补齐最少 "
                         "(JS 配对不拆开), --micro-batches 不再起作用. 每行 logits 不变, 只改每步前向几次; 0 = 按组数")
    ap.add_argument("--accumulate-gradients", action="store_true",
                    help="每组前向完立即反传, 只留一组的激活; 一步仍只更新一次. loss 按组内 prompt 数加权, "
                         "与整批一次反传相同 (求和顺序不同, 浮点误差内)")
    ap.add_argument("--max-length", type=int, default=8192,
                    help="超长提示从左截. 实测最长: synth 256 项菜单 2880, synth-v5 256 行「键: 说明」约 4150, "
                         "JevBench hard 3838, BoolQ 1277")
    ap.add_argument("--grad-ckpt", action="store_true", help="梯度 checkpointing: 激活 5 GiB -> 0.6 GiB, 时间 +30%%")
    ap.add_argument("--grad-ckpt-keep", type=int, default=0,
                    help="开 --grad-ckpt 时让前 N 层文本层保留激活、反传不重算 (数学不变, 显存换时间). "
                         "只支持 slots 与 minimal; minimal feedback 时不能盖到第一个决策层")
    ap.add_argument("--k-min", type=int, default=None,
                    help="不给 = 全量菜单: 池子里的选项全放进去, 最多 --k-max 项. 给了才在 k-min..k-max 随机抽长度 (旧行为)")
    ap.add_argument("--k-max", type=int, default=256, help="菜单最多几项 (D 槽只有 256 个)")
    ap.add_argument("--k-log", action="store_true", help="配 --k-min: 长度按对数均匀取, 默认均匀. 只作用于 banking77")
    ap.add_argument("--k-eval", type=int, default=256,
                    help="评估菜单最多几项, 池子不够就全放: seen 60, unseen 17, --eval 的 banking77 77 / massive 60; BoolQ 恒为 2")
    ap.add_argument("--held-out", type=int, default=17, help="Banking77 留出的类数, 训练里完全不出现")
    ap.add_argument("--eval-every", type=int, default=100,
                    help="每隔几步跑一次探针: 只报一致性. 全量评估集的正确率与一致性只在最后一步跑")
    ap.add_argument("--initial-eval", action=argparse.BooleanOptionalAction, default=True,
                    help="开训前 (step 0) 先跑一次探针; --no-initial-eval 跳过 (同一个基模评一次就够). "
                         "--steps 0 时照评")
    ap.add_argument("--save-every", type=int, default=None,
                    help="每隔几步存一次档到 checkpoints/step-<步数>.safetensors. 不给 = 与 --eval-every 相同, "
                          "每个评估点存一次; 0 = 途中不存. 最后一步照旧只存 trained.safetensors")
    ap.add_argument("--save-training-state", action="store_true",
                    help="同时保存 latest.trainstate.safetensors，包含优化器/RNG；full 模式需要额外磁盘空间")
    ap.add_argument("--probe-size", type=int, default=200,
                    help="探针: 每个评估集固定抽几道题 (不够就全部), 整场训练每个评估点都是这一批")
    ap.add_argument("--probe-passes", type=int, default=5,
                    help="每道题排成几种随机的样子 (行打乱、D 码随机, 二元题也一样), 整场固定; 最后一步的全量一致性也用这个数")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--eval-limit", type=int, default=0, help="每个评估集最多用几条 (0 = 全部)")
    ap.add_argument("--eval", default="banking77+banking77-desc+massive+massive-desc+boolq+simple+jevbench",
                    help="评估集, 用 + 连接, 与 --dataset 无关: banking77 (test 3080 条, 77 类全量菜单, 原始 label 名) / "
                         "banking77-desc (同一批题, 菜单显示「原始名: description」) / "
                         "massive (test 2974 条, 60 类全量菜单, 原始 intent 名) / massive-desc (同一批题, 「原始名: description」) / "
                         "boolq (validation 3270 条) / "
                         "simple (synth-simple-eval: 消息直接说出答案, 5..255 项各 10 题 + 10 道 no/yes) / "
                         "jevbench (JevBench 公开题 easy 48 / original 72 / hard 111, 按推理服务收到请求时的样子出题)")
    ap.add_argument("--random-codes", type=float, default=0.0,
                    help="训练题里换成随机码的比例 (0..1), 二元题也换: 挑上菜单次数最少的 k 个 D 码、顺序随机, "
                         "每个码上菜单时是答案的概率都是 1/k, "
                         "整场下来 D0..D255 当答案的次数期望相同 (60 项菜单下 rate >= 0.77 才补得齐). "
                         "0 = 全部按位置 D0, D1, ... (旧行为). 评估集不受影响")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="标签平滑 (0..1): 训练目标里这一份均摊到这道题菜单的每一行, 硬标签与软标签都摊; "
                         "菜单外的 D 码和其余 token 不分. 0 = 不平滑, 全是硬标签时与旧 run 逐位相同")
    ap.add_argument("--consistency", type=float, default=0.0,
                    help="一致性项的权重 λ (>= 0): 每道训练题在同一批里再出一份, 行序重新打乱 (两项就是对调), "
                         "两份的菜单分布按描述对齐后求 Jensen-Shannon 散度, 损失 = 交叉熵 + λ · JS. "
                         "--batch-size 仍是每步几道题, 一步因此是两倍的 prompt. 0 = 不配对 (旧行为)")
    ap.add_argument("--mix", default=None,
                    help="synth-v5 各训练目标的配比, 相对权重: long_menu=4,breadth=2,complex=1,edge_case=1; "
                         "分段写成 '0: long_menu=6,...; 1000: long_menu=3,...', 到那一步换配比. "
                         "数据里有的目标每一段都要给正权重. 不给 = 各目标相同. 写进目录名 (-mix<hash>)")
    ap.add_argument("--passes", default=None,
                    help="synth-v5 按轮抽, 代替 --mix: 一轮把每个绑定出一遍, 这里列的目标出 N 遍, 如 "
                         "complex=3,edge_case=3,long_context=3 (不列的 1 遍). 轮内打乱, 一轮取完再开一轮. "
                         "一轮的题数记在 result.json 的 split.synth_v5_round (改动题连原题算两道); 要整场至少过一轮, "
                         "v5 每步分到的抽取数 × --steps 要不少于它. 写进目录名 (-pass<hash>)")
    ap.add_argument("--contrast-weight", type=float, default=0.0,
                    help="原题与改动题 (contrast_of) 的成对对数几率差损失的权重: 同一个答案在改动题上的偏好要比在原题上"
                         "高出 --contrast-margin, 两份材料共有的特征在差里抵消. 0 = 不算. 配 --accumulate-gradients 时要给 "
                         "--micro-tokens (配对整组进一次前向). 写进目录名 (-cw<w>-cm<m>)")
    ap.add_argument("--contrast-margin", type=float, default=2.0, help="成对损失要求的对数几率差间隔 m (默认 2)")
    ap.add_argument("--exclude-categories", default=None,
                    help="synth-v5 不训练带这些能力类别的绑定, 逗号分隔 (数据 categories/<domain>.json 的类别, 例如 "
                         "computation_easy,computation_hard; 另有自动的 long_context). 改动题对少了一边就单出. "
                         "写进目录名 (-xc<类别>)")
    ap.add_argument("--type-quota", default=None,
                    help="synth-v5 按题型分段抽批 (要 --passes), 例如 choice=18,bool=14,score=4: 每批每种题型固定题数, "
                         "加起来要等于 --batch-size, 数据里有的题型都要列. 各题型各自按轮出题, 原题与改动题成对出. "
                         "写进目录名 (-tq<各题数>)")
    ap.add_argument("--score-emd", type=float, default=0.0,
                    help="评分题分级距离项的权重: 每道评分题按级别比累计分布的平方差, 按整批题数平均后加进损失. "
                         "0 = 评分题只用交叉熵. 写进目录名 (-emd<w>)")
    ap.add_argument("--cot-weight", type=float, default=0.0,
                    help="计算过程的语言模型损失权重 (training.cot): 带 cot 的题 (synth-v5.11 的 curric) 另走一遍主干, "
                         "学着在 'Answer:' 之后写出程序给的计算过程; 不挂决策层, 推理永远不写. 0 = 不算. 写进目录名 (-cot<w>)")
    ap.add_argument("--distill-traces", default=None,
                    help="蒸馏式内化 (training.distill) 的计算过程文件, jsonl, 每行 {context_id, question_id, text, ask, cot}: "
                         "带计算过程的题另跑一套老师分支 (前缀 + 计算过程 + 选项), 学生读出贴老师读出. 只支持 candidate + packed")
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="学生读出贴老师读出 (1 - cos) 的权重; 0 = 不蒸馏. 写进目录名 (-dist<w>)")
    ap.add_argument("--teacher-weight", type=float, default=1.0,
                    help="蒸馏时老师分支自己的答案损失权重. 写进目录名 (-tw<w>, 默认 1 不写)")
    ap.add_argument("--contrast-qtypes", default=None,
                    help="成对损失只算这些题型的原题/改动题配对, 逗号分隔 (choice / bool / score); 不给 = 全部. "
                         "写进目录名 (-cq<题型>)")
    ap.add_argument("--contrast-pairs", type=int, default=0,
                    help="synth-v5 分层抽批 (要 --passes): 每批固定这么多对原题+改动题 (contrast_of), 其余位置按目标"
                         "固定份额 (正比于一轮里各目标单出的题数, 每个目标至少一道). 原题与改动题总是成对出. "
                         "0 = 不分层. 写进目录名 (-cp<n>)")
    ap.add_argument("--contrast-balance", action="store_true",
                    help="synth-v5 曝光平衡: 原题自己被抽到时也带上它的改动题 (contrast_of), 原题与改动题出现次数相同. "
                         "不给 = 只在抽到改动题时带原题 (原题多出一份). 写进目录名 (-cbal)")
    ap.add_argument("--mask-descriptions", type=float, default=0.0,
                    help="synth-v5 里绑定写了 maskable 的题, 以这个比例遮掉全部选项说明只留键 (0..1); "
                         "一致性配对的两份一起遮. 0 = 不遮. 写进目录名")
    ap.add_argument("--fallback-rate", type=float, default=None,
                    help="synth-v5 随机兜底增强概率 (0..1，默认 0.5)：按绑定标记加入 Other 或信息不足选项；"
                         "明确答案对照保持原答案，既有 other 配对随机选一版。一致性配对共用增强决定")
    ap.add_argument("--temp-max", type=float, default=85.0, help="CPU Tctl 超过就暂停 (°C)")
    ap.add_argument("--temp-cooldown", type=float, default=20.0, help="每次暂停多少秒")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data/banking77")
    ap.add_argument("--out", default=None, help="默认 runs/<时间戳>-<dataset>-<trainable>-<schedule>-<layout>")
    return ap


def resolve_adapter(args) -> dict:
    """LoRA 的形状 {"trainable", "lora_r", "lora_alpha"}. 给了 --init 就用档里记的, 命令行上与它矛盾的直接退出;
    没给就是命令行的值, 不写的取 attn / 8 / 16. d-only 与 full 没有 LoRA, rank 与 alpha 记 None, 与存档一致."""
    given = {"trainable": args.trainable, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
    if args.init:
        ck = checkpoint_adapter(args.init)
        bad = {k: v for k, v in given.items() if v is not None and v != ck[k]}
        if bad:
            raise SystemExit(f"{bad} contradicts {args.init}, which was trained with {ck}; drop the flags to use it")
        return ck
    trainable = given["trainable"] or ("decision-only" if getattr(args, "architecture", None) in ("minimal", "structural", "candidate") else "attn")
    if trainable in ("d-only", "full", "decision-only"):
        return {"trainable": trainable, "lora_r": None, "lora_alpha": None}
    return {"trainable": trainable, "lora_r": given["lora_r"] or 8, "lora_alpha": given["lora_alpha"] or 16}


def resolve_architecture(args) -> dict:
    """Reconstruction settings are inherited as a unit; explicit conflicts fail."""
    given = {key: getattr(args, f"decision_{key}", None)
             for key in ("dim", "heads", "blocks", "layers", "feedback", "option_batch_size")}
    given["kind"] = getattr(args, "architecture", None)
    if isinstance(given["layers"], str):
        try:
            given["layers"] = [int(v) for v in given["layers"].split(",")]
        except ValueError:
            raise SystemExit("--decision-layers needs comma-separated integer indices") from None
    explicit = {k: v for k, v in given.items() if v is not None}
    if args.init:
        record = checkpoint_architecture(args.init)
        if any(record.get(k) != v for k, v in explicit.items()):
            raise SystemExit(f"architecture flags {explicit} contradict checkpoint {record}")
        return record
    kind = given["kind"] or "slots"
    if kind == "slots":
        if explicit.keys() - {"kind"}:
            raise SystemExit("--decision-* flags require --architecture minimal, structural or candidate")
        return {"kind": "slots"}
    try:
        config = DecisionConfig(**explicit)
    except (ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from None
    return {**asdict(config), "layers": list(config.layers) if config.layers is not None else None}


def resolve_candidate_prefix_cache(args, saved_config) -> str:
    mode = getattr(args, "candidate_prefix_cache", None)
    if args.architecture != "candidate":
        if mode is not None:
            raise SystemExit("--candidate-prefix-cache requires --architecture candidate")
        return "off"
    if mode is None:
        mode = (saved_config or {}).get("candidate_prefix_cache", "off") if args.init else "auto"
    if mode not in ("auto", "on", "off", "packed"):
        raise SystemExit("candidate prefix cache mode must be auto, on, off or packed")
    return mode


def resolve_max_grad_norm(args) -> float | None:
    """--max-grad-norm 只接受正数."""
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise SystemExit(f"--max-grad-norm must be > 0; got {args.max_grad_norm}")
    return args.max_grad_norm


def resolve_lr_decision(args) -> float | None:
    """决策层单独的学习率. 不给就是 None, 决策层照旧跟主干同组; 只有带决策层的架构能给, 且须 >= 0."""
    if args.lr_decision is None:
        return None
    if args.architecture in (None, "slots"):
        raise SystemExit("--lr-decision requires --architecture minimal, structural or candidate")
    if args.lr_decision < 0:
        raise SystemExit(f"--lr-decision is a learning rate, >= 0; got {args.lr_decision}")
    return args.lr_decision


def resolve_save_every(args) -> int:
    """途中每隔几步存一次档. 不给 --save-every 就跟 --eval-every 走, 每个评估点一份; 0 = 途中不存."""
    if args.save_every is None:
        return args.eval_every
    if args.save_every < 0:
        raise SystemExit(f"--save-every is a step count, >= 0 (0 = no checkpoints along the way); got {args.save_every}")
    return args.save_every


def checkpoint_path(out: pathlib.Path, step: int) -> pathlib.Path:
    """途中那些档的位置. 步数补到 5 位, 按文件名排就是训练顺序."""
    return out / "checkpoints" / f"step-{step:05d}.safetensors"


def run_tag(args) -> str:
    """run 目录名的后缀. 旧 loss 的写法保持不变 (all-slots -> -allslots, menu 不加), 旧 run 的名字照旧能复现.
    rank / alpha 离开 8 / 16 才写, 旧 run 全是 8 / 16; --lr-lora 离开 1e-4 才写, 旧 run 全是 1e-4."""
    return ("-qtype" if args.type_marker else "") + ("-ctx" if args.context_marker else "") \
        + (f"-b{args.batch_size}" if args.batch_size != 8 else "") \
        + (f"-r{args.lora_r}" if args.lora_r not in (None, 8) else "") \
        + (f"-alpha{args.lora_alpha}" if args.lora_alpha not in (None, 16) else "") \
        + (f"-lr{args.lr_lora:g}" if args.lr_lora != 1e-4 else "") \
        + (f"-lrd{args.lr_decision:g}" if getattr(args, "lr_decision", None) is not None else "") \
        + (f"-clip{args.max_grad_norm:g}" if getattr(args, "max_grad_norm", None) is not None else "") \
        + ("-adam8" if getattr(args, "optimizer", "adamw") == "adamw8bit" else "") \
        + (f"-kfull{args.k_max}" if args.k_min is None else f"-k{args.k_min}-{args.k_max}") \
        + ("-klog" if args.k_log else "") \
        + (f"-rcodes{args.random_codes:g}" if args.random_codes > 0 else "") \
        + (f"-ls{args.label_smoothing:g}" if args.label_smoothing > 0 else "") \
        + (f"-js{args.consistency:g}" if args.consistency > 0 else "") \
        + (f"-cw{args.contrast_weight:g}-cm{args.contrast_margin:g}" if getattr(args, "contrast_weight", 0.0) > 0 else "") \
        + (f"-mask{args.mask_descriptions:g}" if args.mask_descriptions > 0 else "") \
        + (f"-fallback{0.5 if args.fallback_rate is None else args.fallback_rate:g}" if {"synth-v5", "synth-v5.11"} & set(args.dataset.split("+")) else "") \
        + (f"-mix{hashlib.sha1(args.mix.encode()).hexdigest()[:6]}" if args.mix else "") \
        + (f"-pass{hashlib.sha1(args.passes.encode()).hexdigest()[:6]}" if args.passes else "") \
        + ("-cbal" if getattr(args, "contrast_balance", False) else "") \
        + (f"-cp{args.contrast_pairs}" if getattr(args, "contrast_pairs", 0) else "") \
        + (f"-xc{'+'.join(sorted(set(args.exclude_categories.split(','))))}" if getattr(args, "exclude_categories", None) else "") \
        + (f"-tq{'.'.join(v.split('=')[1] for v in args.type_quota.split(','))}" if getattr(args, "type_quota", None) else "") \
        + (f"-emd{args.score_emd:g}" if getattr(args, "score_emd", 0.0) > 0 else "") \
        + (f"-cot{args.cot_weight:g}" if getattr(args, "cot_weight", 0.0) > 0 else "") \
        + (f"-dist{args.distill_weight:g}" if getattr(args, "distill_weight", 0.0) > 0 else "") \
        + (f"-tw{args.teacher_weight:g}" if getattr(args, "distill_weight", 0.0) > 0 and getattr(args, "teacher_weight", 1.0) != 1.0 else "") \
        + (f"-cq{args.contrast_qtypes.replace(',', '+')}" if getattr(args, "contrast_qtypes", None) else "") \
        + (f"-dweights{hashlib.sha1(args.dataset_weights.encode()).hexdigest()[:6]}" if getattr(args, "dataset_weights", None) else "") \
        + {"all-slots": "-allslots", "vocab": "-vocab"}.get(args.loss, "") \
        + (f"-{args.architecture}-decision{hashlib.sha1(json.dumps(resolve_architecture(args), sort_keys=True).encode()).hexdigest()[:6]}"
           if getattr(args, "architecture", None) in ("minimal", "structural", "candidate") else "")


def main() -> None:
    args = build_parser().parse_args()
    try:
        validate_data_preparation(args.data_workers, args.data_prefetch, args.data_backend)
        if args.device_prefetch < 0:
            raise ValueError("data_/device_prefetch must be nonnegative")
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if args.micro_tokens < 0:
        raise SystemExit(f"--micro-tokens is a padded-token budget per group, >= 0; got {args.micro_tokens}")
    if args.score_emd < 0:
        raise SystemExit(f"--score-emd is a weight, >= 0; got {args.score_emd}")
    if args.cot_weight < 0:
        raise SystemExit(f"--cot-weight is a weight, >= 0; got {args.cot_weight}")
    if args.distill_weight < 0 or args.teacher_weight < 0:
        raise SystemExit("--distill-weight and --teacher-weight are weights, >= 0")
    if args.distill_weight > 0 and not args.distill_traces:
        raise SystemExit("--distill-weight needs --distill-traces")
    if args.contrast_qtypes and not set(args.contrast_qtypes.split(",")) <= {"choice", "bool", "score"}:
        raise SystemExit(f"--contrast-qtypes takes choice / bool / score, got {args.contrast_qtypes!r}")
    if args.contrast_weight > 0 and args.accumulate_gradients and args.micro_tokens <= 0 and args.micro_batches > 1:
        raise SystemExit("--contrast-weight with --accumulate-gradients needs --micro-tokens, so each original and "
                         "edited item stay in one forward group")
    args.datasets_dir = str(datasets_root(args.datasets_dir))
    architecture = resolve_architecture(args)
    args.architecture = architecture["kind"]
    if args.candidate_prefix_cache is not None and args.architecture != "candidate":
        raise SystemExit("--candidate-prefix-cache requires --architecture candidate")
    if args.architecture != "slots" and args.loss != "menu":
        raise SystemExit("new decision architectures output a K-way distribution; pass --loss menu")
    if args.architecture in ("structural", "candidate") and args.layout != "context-first":
        raise SystemExit(f"{args.architecture} architecture requires --layout context-first")
    vars(args).update(resolve_adapter(args))  # 之后目录名、prepare_model、result.json 读的都是同一个形状
    if args.trainable == "decision-only" and args.architecture == "slots":
        raise SystemExit("--trainable decision-only requires --architecture minimal, structural or candidate")
    args.lr_decision = resolve_lr_decision(args)
    args.max_grad_norm = resolve_max_grad_norm(args)
    args.save_every = resolve_save_every(args)
    datasets = parse_datasets(args.dataset)
    if args.save_training_state and args.dataset not in ("synth-v5", "synth-v5.11"):
        raise SystemExit("--save-training-state requires --dataset synth-v5.11 (or synth-v5), matching the resume CLI contract")

    tag = run_tag(args)
    out = pathlib.Path(args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{args.dataset}-{args.trainable}-{args.lr_schedule}-{args.layout}{tag}")
    out.mkdir(parents=True, exist_ok=True)
    sample_fn, eval_sets, split_info = build_data(args)
    from sors.training.resume import sampling_provenance
    provenance = sampling_provenance(pathlib.Path(__file__).resolve().parents[1], args.datasets_dir)

    # ---- 模型 ---------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model)
    d_ids = install_d_tokens(tok)
    t_ids = install_type_tokens(tok)
    c_ids = install_context_tokens(tok)
    # 类型行与上下文行永远放开; 不带 --type-marker / --context-marker 时它们不出现在提示里, 梯度为零、原地不动
    train_ids = d_ids + t_ids + c_ids
    lm, attention = load_causal_lm(args.model, attn_implementation=args.attn_implementation,
                                   allow_kernel_download=args.allow_kernel_download)
    args.attention = asdict(attention)
    if args.architecture != "slots":
        torch.manual_seed(args.seed)  # Includes the new head and token-row initialization.
    m = prepare_model(lm, train_ids, args.lora_r, args.lora_alpha, args.lora_dropout,
                       trainable=args.trainable, grad_ckpt=args.grad_ckpt, decision=decision_config(architecture))
    if args.grad_ckpt_keep:
        try:
            keep_layer_activations(m, args.grad_ckpt_keep)
        except ValueError as error:
            raise SystemExit(f"--grad-ckpt-keep: {error}") from None
    if args.architecture != "slots":
        args.decision_layers = architecture_config(m)["layers"]
    init_cfg = load_trained(m, train_ids, args.init) if args.init else None
    args.candidate_prefix_cache = resolve_candidate_prefix_cache(args, init_cfg)
    if args.architecture == "candidate":
        m.set_candidate_prefix_cache(args.candidate_prefix_cache)

    k_pad = max([args.k_max, args.k_eval] + [len(e.options) for es in eval_sets.values() for e in es.examples])
    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, micro_batches=args.micro_batches, k_max=k_pad,
        micro_tokens=args.micro_tokens, accumulate_gradients=args.accumulate_gradients,
        data_workers=args.data_workers, data_prefetch=args.data_prefetch,
        data_backend=args.data_backend, device_prefetch=args.device_prefetch,
        max_length=args.max_length,
        lr_lora=args.lr_lora, lr_embed=args.lr_embed, lr_decision=args.lr_decision, max_grad_norm=args.max_grad_norm, weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        lr_schedule=args.lr_schedule, warmup_steps=args.warmup, layout=args.layout, type_marker=args.type_marker,
        context_marker=args.context_marker,
        loss=args.loss, label_smoothing=args.label_smoothing, consistency=args.consistency,
        contrast_weight=args.contrast_weight, contrast_margin=args.contrast_margin,
        contrast_qtypes=tuple(args.contrast_qtypes.split(",")) if args.contrast_qtypes else None,
        score_emd=args.score_emd, cot_weight=args.cot_weight,
        distill_weight=args.distill_weight, teacher_weight=args.teacher_weight,
        eval_every=args.eval_every, initial_eval=args.initial_eval, save_every=args.save_every, probe_size=args.probe_size,
        probe_passes=args.probe_passes, seed=args.seed,
        candidate_prefix_cache=args.candidate_prefix_cache,
    )
    guard = ThermalGuard(max_c=args.temp_max, cooldown_s=args.temp_cooldown)
    writer = SummaryWriter(log_dir=str(out / "tb"))
    writer.add_text("args", json.dumps(vars(args), indent=2), 0)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"dataset={'+'.join(datasets)} architecture={args.architecture} prefix_cache={args.candidate_prefix_cache} trainable={args.trainable} layout={args.layout} loss={args.loss} "
          f"random_codes={args.random_codes:g} label_smoothing={args.label_smoothing:g} "
          f"consistency={args.consistency:g} contrast_weight={args.contrast_weight:g} contrast_margin={args.contrast_margin:g} "
          f"contrast_qtypes={args.contrast_qtypes} score_emd={args.score_emd:g} cot_weight={args.cot_weight:g} "
          f"distill_weight={args.distill_weight:g} teacher_weight={args.teacher_weight:g} "
          f"data_workers={cfg.data_workers} data_prefetch={cfg.data_prefetch} data_backend={cfg.data_backend} "
          f"device_prefetch={cfg.device_prefetch} "
          f"k={menu_k_range(args.k_min, args.k_max)}{' log' if args.k_log else ''} params {n_train:,}  "
          f"init={args.init or '-'}  eval " + " ".join(f"{k}={len(v.examples)}" for k, v in eval_sets.items())
          + f"  tctl {guard.read()}  → {out}", flush=True)
    def save_checkpoint(step: int) -> None:
        path = checkpoint_path(out, step)
        path.parent.mkdir(exist_ok=True)
        save_trained(m, train_ids, cfg, path)
        print(f"step {step:5d}  saved {path}", flush=True)

    on_state = None
    if args.save_training_state:
        import transformers
        from sors.training.resume import sampling_fingerprint, save_state
        metadata = {"args": vars(args), "sampling_fingerprint": sampling_fingerprint(pathlib.Path(__file__).resolve().parents[1], args.datasets_dir),
                    "sampling_provenance": provenance,
                    "versions": {"python": sys.version, "torch": str(torch.__version__), "transformers": transformers.__version__},
                    "train_ids": train_ids, "adapter": resolve_adapter(args), "architecture": architecture_config(m),
                    "model_revision": getattr(lm.config, "_commit_hash", None)}

        def on_state(value):
            path = out / "checkpoints" / "latest.trainstate.safetensors"
            path.parent.mkdir(exist_ok=True)
            save_state({**value, "metadata": metadata}, path)
            print(f"step {value['step']:5d} saved complete training state {path}", flush=True)

    history = train(m, tok, d_ids, sample_fn, eval_sets, cfg, log_path=out / "log.jsonl", writer=writer, guard=guard,
                    on_checkpoint=save_checkpoint, on_state=on_state)
    writer.close()
    if args.steps > 0:
        save_trained(m, train_ids, cfg, out / "trained.safetensors")
    (out / "result.json").write_text(json.dumps({
        "args": vars(args),
        "trainable_params": n_train,
        "init_config": init_cfg,
        "split": split_info,
        "sampling_provenance": provenance,
        "history": history,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
