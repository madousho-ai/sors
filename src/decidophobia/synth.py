"""synth-intents 适配: datasets/synth-intents/*.jsonl -> LabeledSet. 只做训练.

4096 个虚构意图 × 3 条用户消息, 16 个领域各 256 个. 菜单里显示的是 description.
类 id 按 intent id 字母序 (与 banking77 / massive 同一约定). 同时返回每个类的领域, 供按领域留出.
每个意图还带一道 yes/no 问题, 三条消息各一个答案: load_synth_binary 把它读成与 BoolQ 同形的二元题.
规格见 datasets/synth-intents/SPEC.md.
"""

from __future__ import annotations

import json
import pathlib
import random

from decidophobia.data import LabeledSet, MenuExample, compose_menu, draw_k

DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[2] / "datasets" / "synth-intents"
CONTEXT_LABEL = "Customer message"


def _rows(data_dir) -> list[dict]:
    """全部意图, 按 intent id 字母序 —— 行号就是类 id."""
    rows = []
    for f in sorted(pathlib.Path(data_dir).glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    rows.sort(key=lambda r: r["id"])
    return rows


def load_synth(data_dir=DEFAULT_DIR) -> tuple[LabeledSet, list[str]]:
    rows = _rows(data_dir)
    queries, labels = [], []
    for c, r in enumerate(rows):
        for u in r["utterances"]:
            queries.append(u)
            labels.append(c)
    s = LabeledSet(queries=queries, labels=labels, names={c: r["description"] for c, r in enumerate(rows)},
                   context_label=CONTEXT_LABEL)
    return s, [r["domain"] for r in rows]


def load_synth_binary(data_dir=DEFAULT_DIR) -> LabeledSet:
    """二元题: 每条消息配它所属意图的问句, 答案取 answers 里对应的那一个.
    消息顺序与 load_synth 相同. 类 id 0 = no, 1 = yes, 与 boolq.NAMES 一致."""
    queries, labels, questions = [], [], []
    for r in _rows(data_dir):
        for u, a in zip(r["utterances"], r["answers"], strict=True):
            queries.append(u)
            labels.append(1 if a else 0)
            questions.append(r["question"])
    return LabeledSet(queries=queries, labels=labels, names={0: "no", 1: "yes"}, context_label=CONTEXT_LABEL,
                      questions=questions, qtype="bool")


def sample_domain_menus(s: LabeledSet, domains: list[str], k_range: tuple[int, int], n: int,
                        rng: random.Random) -> list[MenuExample]:
    """训练批的菜单题: 随机抽 n 条消息, 菜单只从正确意图所在领域里抽 (256 项即整个领域, 顺序随机).
    不同领域的意图不进同一份菜单 —— 部署时一个请求带的是一个应用自己的意图, 跨领域近义描述也就碰不到一起.
    s 与 domains 是 load_synth 的两个返回值."""
    by_domain: dict[str, list[int]] = {}
    for c, d in enumerate(domains):
        by_domain.setdefault(d, []).append(c)
    out = []
    for i in rng.sample(range(len(s.labels)), n):
        lab = s.labels[i]
        opts, gi = compose_menu(lab, by_domain[domains[lab]], draw_k(k_range, rng), rng)
        out.append(s.make_example(i, opts, gi))
    return out


def synth_eval_examples(k: int, seed: int, data_dir=DEFAULT_DIR) -> list[MenuExample]:
    """留出评估: 1024 条合成意图消息各配一个 k 项菜单, 选项全来自 512 个合成意图, 连续编号.
    train.py 的 --eval-synth (seed = --seed + k) 与 scripts/eval-invariance.py 都从这里取题, 同 seed 即同一批题."""
    s, _ = load_synth(data_dir)
    return s.build_examples(list(range(len(s.names))), (k, k), random.Random(seed))
