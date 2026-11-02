"""json-eval 适配: 外部数据仓库的 json-eval/json-eval-v1.jsonl (JSON 阅读能力评测集, JevBench 格式) -> 按能力分开的评估集.
只做评估, 不进训练.

每条题点名 JSON 里的一处范围, 只按那里的内容作答; provenance.surface_answer 是范围外诱饵字段给出的答案, 记成 decoy,
评估报模型被诱饵带走的比例. 切片 (provenance.slice):
  c1 定位  c2 范围隔离  c3 近似字段  c4 按条件选元素  c5 空与缺失  c6 长度与位置  c7 组合范围  c8 字符串里的 JSON
  c9 反查路径  tr 迁移 (没见过的结构)
c6 按材料长度 (provenance.size: s 约 1k / m 约 5k / l 约 20k 字符) 再分三个集合, 三者之差就是长度曲线.
每题走推理服务的同一条路 (jevbench.to_eval_example).

返回 {"json_c1": [...], ..., "json_c6_s": [...], ...}, 键按文件里首次出现的顺序, 题序同文件.
"""

from __future__ import annotations

import json
import pathlib

from sors.core.menu import MenuExample
from sors.data.jevbench import to_eval_example
from sors.data.paths import asset_path

FILE = "json-eval/json-eval-v1.jsonl"


def _set_name(r: dict) -> str:
    p = r.get("provenance", {})
    sl = p.get("slice", r.get("family"))
    return f"json_{sl}_{p['size']}" if sl == "c6" else f"json_{sl}"


def load_json_eval(path=None, *, datasets_dir=None) -> dict[str, list[MenuExample]]:
    path = path if path is not None else asset_path(FILE, datasets_dir)
    out: dict[str, list[MenuExample]] = {}
    with pathlib.Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.setdefault(_set_name(r), []).append(to_eval_example(r))
    order = [k for k in out if not k.startswith("json_c6_")] + [f"json_c6_{s}" for s in "sml" if f"json_c6_{s}" in out]
    return {k: out[k] for k in order}
