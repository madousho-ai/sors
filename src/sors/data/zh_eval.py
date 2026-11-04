"""zh-eval 适配: 外部数据仓库的 zh-eval/zh-eval-v1.jsonl (中文阅读评测集, JevBench 格式) -> 按切片分开的评估集.
只做评估, 不进训练.

切片 (provenance.slice):
  z1 基础中文  z2 中英夹杂  z3 网络用语  z4 语音转写  z5 群聊与转发  z6 语用 (字面意思是错的)  z7 中文数字与时间
  z8 规则加案例 (含长材料)  z9 跨语言  zp 中英平行
z9 按方向 (provenance.direction) 分成 zh_z9_zhstate (中文材料、英文题) 与 zh_z9_enstate (英文材料、中文题);
zp 按语言 (provenance.lang) 分成 zh_zp_zh 与 zh_zp_en, 两者是同一批题的中文版与英译, 分数之差就是语言差距.
provenance.surface_answer 是按字面 / 关键词 / 只读中文部分会选的答案, 记成 decoy.
每题走推理服务的同一条路 (jevbench.to_eval_example).

返回 {"zh_z1": [...], ...}, 键按文件里首次出现的顺序, 题序同文件.
"""

from __future__ import annotations

import json
import pathlib

from sors.core.menu import MenuExample
from sors.data.jevbench import to_eval_example
from sors.data.paths import asset_path

FILE = "zh-eval/zh-eval-v1.jsonl"


def _set_name(r: dict) -> str:
    p = r.get("provenance", {})
    sl = p["slice"]
    if sl == "z9":
        return "zh_z9_zhstate" if p["direction"] == "zh_state_en_q" else "zh_z9_enstate"
    if sl == "zp":
        return f"zh_zp_{p['lang']}"
    return f"zh_{sl}"


def load_zh_eval(path=None, *, datasets_dir=None) -> dict[str, list[MenuExample]]:
    path = path if path is not None else asset_path(FILE, datasets_dir)
    out: dict[str, list[MenuExample]] = {}
    with pathlib.Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.setdefault(_set_name(r), []).append(to_eval_example(r))
    return out
