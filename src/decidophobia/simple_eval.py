"""synth-simple-eval 适配: datasets/synth-simple-eval/synth-simple-eval.jsonl -> 按菜单长度分开的评估集. 只做评估.

客户消息直接说出答案, 选项是常见名词: 5 / 10 / 20 / 40 / 60 / 100 / 255 项各 10 题, 另有 10 道 no / yes 题.
题目是固定文件 (菜单顺序、正确答案位置都已写死), 由同目录的 generate.py 生成, 设计见那里的说明.

返回 {"simple5": [...], ..., "simple255": [...], "simple_bool": [...]}, 每项一个 list[MenuExample].
选择题的类 id 是名词在全部名词里的字母序; 二元题 0 = no, 1 = yes, 与 boolq.NAMES 一致. 菜单按位置编号 D0, D1, ...
"""

from __future__ import annotations

import json
import pathlib

from decidophobia.data import MenuExample

DEFAULT_PATH = pathlib.Path(__file__).resolve().parents[2] / "datasets" / "synth-simple-eval" / "synth-simple-eval.jsonl"
SIZES = (5, 10, 20, 40, 60, 100, 255)
CONTEXT_LABEL = "Customer message"
_BOOL = {"no": 0, "yes": 1}


def load_simple_eval(path=DEFAULT_PATH) -> dict[str, list[MenuExample]]:
    with pathlib.Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    nouns = sorted({o for r in rows if r["qtype"] == "choice" for o in r["options"]})
    ids = {n: i for i, n in enumerate(nouns)}
    out: dict[str, list[MenuExample]] = {f"simple{k}": [] for k in SIZES}
    out["simple_bool"] = []
    for r in rows:
        if r["qtype"] == "choice":
            opts, name = [ids[o] for o in r["options"]], f"simple{len(r['options'])}"
        else:
            opts, name = [_BOOL[o] for o in r["options"]], "simple_bool"
        out[name].append(MenuExample(
            query=r["context"], options=opts, gold_idx=r["answer"], label=opts[r["answer"]],
            option_names=list(r["options"]), context_label=CONTEXT_LABEL, question=r["question"], qtype=r["qtype"],
        ))
    return out
