"""synth-intents 适配: datasets/synth-intents/*.jsonl -> LabeledSet. 只做训练.

512 个虚构意图 × 2 条用户消息, 16 个领域. 菜单里显示的是 description.
类 id 按 intent id 字母序 (与 banking77 / massive 同一约定). 同时返回每个类的领域, 供按领域留出.
规格见 datasets/synth-intents/SPEC.md.
"""

from __future__ import annotations

import json
import pathlib

from decidophobia.data import LabeledSet

DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[2] / "datasets" / "synth-intents"


def load_synth(data_dir=DEFAULT_DIR) -> tuple[LabeledSet, list[str]]:
    rows = []
    for f in sorted(pathlib.Path(data_dir).glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    rows.sort(key=lambda r: r["id"])
    queries, labels = [], []
    for c, r in enumerate(rows):
        for u in r["utterances"]:
            queries.append(u)
            labels.append(c)
    s = LabeledSet(queries=queries, labels=labels, names={c: r["description"] for c, r in enumerate(rows)},
                   context_label="Customer message")
    return s, [r["domain"] for r in rows]
