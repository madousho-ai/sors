"""JevBench 公开题适配: 下载到 data/jevbench/<tier>.jsonl (钉 commit + sha256) -> 按档分开的评估集. 只做评估.

JevBench (https://github.com/fstandhartinger/jevbench, MIT) 是给 Jev 类决策模型的基准, 每条题就是一次 TypeSafe
/v1/systemone 请求: state + 一道 noul / choice / score 题 (instructions + criteria), 外加标准答案 expected.
这里用它公开的三档, 取自 commit fd54ea7 的 datasets/public/. 文件不进仓库: 第一次用时下载到 data/jevbench
(data/ 整个被 .gitignore 忽略), 之后每次读都核对 sha256, 与它 datasets/manifest.json 里记的相同:
  easy      48 条  意图、明说的事实、字段抽取、选工具, 答案在文本里写明
  original  72 条  路由、回答是否达标、政策是非、意图、严重度分级、字段抽取 (36 对同义改写)
  hard     111 条  长政策、多跳、时间与数字、陷阱题等; state 最长约 3.7k token, 有的是 JSON 对象
留出的私有题不公开, 这里没有.

每条题走推理服务的同一条路 (serve.menus.to_example), 上下文标签为空 —— 与 scripts/serve.py 默认收到这个请求时
模型看到的提示逐字相同: choice 按 criteria 的键序 (JevBench 原样发出, 文件里是字母序) 写成「名字: 说明」,
noul 是 no / yes 两行, score 是分级从低到高. 菜单连续编号 D0, D1, ...
gold 是 expected 所在的那一行. labels 与菜单对不上时报 ValueError. hard 档 10 道题另有 gold_probs, 这里不用.

返回 {"jevbench_easy": [...], "jevbench_original": [...], "jevbench_hard": [...]}, 每项一个 list[MenuExample], 题序同文件.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import urllib.request

from pydantic import TypeAdapter

from sors.core.menu import MenuExample
from sors.serve.api import Question
from sors.serve.menus import YES_NO, to_example

DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "jevbench"
COMMIT = "fd54ea7dc02bbe29c6ac8f6e015a54cdcff26805"
SHA256 = {
    "easy": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "original": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "hard": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}
TIERS = tuple(SHA256)
_QUESTION = TypeAdapter(Question)


def _fetch(tier: str, data_dir: pathlib.Path) -> pathlib.Path:
    """这一档的文件, 没有就下载. 先下到 .part 再改名, 下到一半断掉不会留下一份半截的文件."""
    path = data_dir / f"{tier}.jsonl"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{COMMIT}/datasets/public/{tier}.jsonl", part)
        part.rename(path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SHA256[tier]:
        raise RuntimeError(f"{path}: sha256 {got} != {SHA256[tier]} (JevBench {COMMIT[:7]}); delete it to download again")
    return path


def _gold(r: dict) -> int:
    """expected 在菜单里的行号. 菜单的行 (choice 的 criteria 键 / no, yes / 分级序号) 必须与 labels 是同一组."""
    q, labels, expected = r["question"], r["labels"], r["expected"]
    if q["type"] == "choice":
        rows = list(q["criteria"])
        ok = sorted(labels) == sorted(rows) and expected in rows
        gold = rows.index(expected) if ok else None
    elif q["type"] == "noul":
        ok = labels == list(YES_NO) and expected in YES_NO
        gold = YES_NO.index(expected) if ok else None
    else:
        n = len(q["criteria"])
        ok = labels == [str(i) for i in range(n)] and isinstance(expected, int) and 0 <= expected < n
        gold = expected if ok else None
    if gold is None:
        raise ValueError(f"{r['id']}: labels {labels} / expected {expected!r} do not match its {q['type']} menu")
    return gold


def to_eval_example(r: dict) -> MenuExample:
    """一条 JevBench 记录 -> 评估用的菜单样本 (推理服务的提示, 加上 gold)."""
    ex = to_example(_QUESTION.validate_python(r["question"]), r["state"], "")
    gold = _gold(r)
    return MenuExample(query=ex.query, options=ex.options, gold_idx=gold, label=ex.options[gold],
                       option_names=ex.option_names, context_label=ex.context_label, question=ex.question,
                       qtype=ex.qtype)


def load_jevbench(data_dir=DEFAULT_DIR) -> dict[str, list[MenuExample]]:
    out = {}
    for t in TIERS:
        with _fetch(t, pathlib.Path(data_dir)).open(encoding="utf-8") as f:
            out[f"jevbench_{t}"] = [to_eval_example(json.loads(line)) for line in f if line.strip()]
    return out
