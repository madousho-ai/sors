"""synth-intents-v6: synth-intents-v5.25 上按类挑出来的一份训练名单, 不另存材料.

数据仓库的 synth-intents-v6/ 里有:
  spec.json        选择规则: classes {类: [数据家族]}, n {类: 绑定数}, eval_n (每类留出的评测题), salt
  drop.json        不进训练的材料: 超过 16384 token 的 jsonvalue 材料, 以及 tg-spam-heldout 的留出材料
  selection.json   source (v5.25 的目录名)、source_sha256 (v5.25 每个文件)、bindings [[领域, 材料, 题], ...]

load_synth_v6 读 v5.25 (load_synth_v5, 改标、译文、类别照旧), 再只留名单里的绑定; 名单由 build_selection
从 spec 与 drop 生成 (scripts/build-synth-v6.py). 选法与 v6 阶梯实验同一套 ladder_split / ladder_pool.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import replace

from sors.data.paths import asset_path
from sors.data.synth_v5 import V5Item, ladder_pool, ladder_split, load_synth_v5

NAME = "synth-intents-v6"


def source_sha256(source_dir) -> dict[str, str]:
    """目录下每个文件 (相对路径 -> sha256), __pycache__ 不算."""
    root = pathlib.Path(source_dir)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def select_bindings(items: list[V5Item], keys) -> list[V5Item]:
    """只留 keys 里的 (领域, 材料, 题), 照 items 原来的顺序; pair、contrast 重新编号, 指向名单之外的去掉.
    名单里有 items 没有的绑定报 ValueError."""
    want = {tuple(k) for k in keys}
    keep = [i for i, it in enumerate(items) if (it.domain, it.context_id, it.question_id) in want]
    if len(keep) != len(want):
        have = {(items[i].domain, items[i].context_id, items[i].question_id) for i in keep}
        missing = sorted(want - have)
        raise ValueError(f"{len(missing)} selected bindings are not in the source, e.g. {missing[:3]}")
    new = {old: j for j, old in enumerate(keep)}
    return [replace(items[i], pair=new.get(items[i].pair), contrast=new.get(items[i].contrast)) for i in keep]


def build_selection(items: list[V5Item], spec: dict, drop: set[str] = frozenset()) -> list[tuple[str, str, str]]:
    """按 spec 生成名单: 每类先按材料组留出 eval_n 道评测题, 再按固定顺序取够 n 道; drop 里的材料连同它的改动材料去掉."""
    split = ladder_split(items, spec["classes"], eval_n=spec["eval_n"], drop=set(drop), salt=spec["salt"])
    pool, _ = ladder_pool(split, spec["n"])
    return [(it.domain, it.context_id, it.question_id) for it in pool]


def load_synth_v6(data_dir=None, *, datasets_dir=None) -> list[V5Item]:
    """v6 的训练绑定. v5.25 的文件与 selection.json 记的 sha256 对不上时报 ValueError."""
    data_dir = pathlib.Path(data_dir) if data_dir is not None else asset_path(NAME, datasets_dir)
    sel = json.loads((data_dir / "selection.json").read_text(encoding="utf-8"))
    src = data_dir.parent / sel["source"]
    got = source_sha256(src)
    bad = sorted(k for k, v in sel["source_sha256"].items() if got.get(k) != v)
    extra = sorted(set(got) - set(sel["source_sha256"]))
    if bad or extra:
        raise ValueError(f"{src} does not match the sha256 in {data_dir / 'selection.json'}: "
                         f"changed or missing {bad[:3]}, unexpected {extra[:3]}")
    return select_bindings(load_synth_v5(src), sel["bindings"])
