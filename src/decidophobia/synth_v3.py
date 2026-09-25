"""synth-intents-v3 适配: datasets/synth-intents-v3/ -> 逐题的 MenuExample. 只做训练.

一个领域是 <domain>.py (题与文本, 格式见同目录 schema.py) 加参考模型的答案 <domain>.answer.json;
有 answer.json 的都读. 每份文本被问到的每道题 (needs / texts 满足) 是一条 V3Item, 标签按来源分三种:
  stated   文本写明的答案: 硬标签 (target None). 参考模型读错也以写明的为准.
  unknown  文本对这道题一点线索都没有: 均匀分布.
  soft     其余: answer.json 里参考模型的分布 p (按选项循环移位取平均, 存 3 位小数, 读进来重新归一).
answer.json 里那道题的 hash 与当前 <domain>.py 对不上 (改了题或文本没重答) 时报错, 不拿过期的分布训练.

一条 V3Item 的 example 是规范形: 选项按 <domain>.py 里的顺序, 类 id 就是选项下标, 问句取第一种问法.
训练抽题 (sample_synth_v3) 每次换一种问法、重新打乱选项, 软标签跟着各自的选项走.
上下文标签是领域的 LABEL (Support ticket / Browser agent state ...), JSON state 缩进 2 格展开, 与参考模型答题时看到的相同.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import sys
from dataclasses import dataclass, replace

from decidophobia.data import MenuExample, reorder_menu

DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[2] / "datasets" / "synth-intents-v3"


@dataclass(frozen=True)
class V3Item:
    domain: str
    text_id: str
    question_id: str
    source: str  # stated | unknown | soft
    asks: list[str]  # 这道题的几种问法
    example: MenuExample  # 规范形: 原选项顺序, 第一种问法


def _module(path: pathlib.Path, name: str, register: bool = False):
    """按文件路径载入模块. register=True 时先登记进 sys.modules 再执行 —— 里面有 @dataclass 的模块要这样,
    dataclass 执行时会去 sys.modules 里找自己所在的模块."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def state_text(text) -> str:
    return text if isinstance(text, str) else json.dumps(text, indent=2, ensure_ascii=False)


def _load_domain(data_dir: pathlib.Path, domain: str, schema, item_hash) -> list[V3Item]:
    m = _module(data_dir / f"{domain}.py", f"synth_v3_{domain}")
    errs = schema.problems(m.DOMAIN, m.LABEL, m.QUESTIONS, m.TEXTS)
    if errs or m.DOMAIN != domain:
        raise ValueError(f"{domain}.py fails its format check ({len(errs)} problems, DOMAIN {m.DOMAIN!r}), "
                         f"e.g. {errs[:1]}; run it directly to see them all")
    answers = json.loads((data_dir / f"{domain}.answer.json").read_text(encoding="utf-8"))["answers"]
    out = []
    for t in m.TEXTS:
        for q in m.QUESTIONS:
            if not schema.applies(q, t):
                continue
            k = len(q.options)
            if q.id in t.stated:
                source, gold, target = "stated", t.stated[q.id], None
            elif q.id in t.unknown:
                source, gold, target = "unknown", 0, [1 / k] * k
            else:
                e = answers.get(t.id, {}).get(q.id)
                h = item_hash(m.LABEL, t.text, q.ask[0], q.options)
                if not e or e.get("hash") != h or len(e["p"]) != k or not sum(e["p"]):
                    raise ValueError(f"{domain}.answer.json has no current answer for {t.id} / {q.id}; "
                                     f"rerun datasets/synth-intents-v3/answer.py {domain}")
                source, target = "soft", [x / sum(e["p"]) for x in e["p"]]
                gold = max(range(k), key=target.__getitem__)
            ex = MenuExample(
                query=state_text(t.text), options=list(range(k)), gold_idx=gold, label=gold,
                option_names=list(q.options), context_label=m.LABEL, question=q.ask[0],
                qtype="bool" if q.options == schema.YES_NO else "choice", target=target,
            )
            out.append(V3Item(domain, t.id, q.id, source, list(q.ask), ex))
    return out


def load_synth_v3(data_dir=DEFAULT_DIR) -> dict[str, list[V3Item]]:
    """{领域: [V3Item]}, 领域按名字排序. <domain>.py 过不了格式检查、或 answer.json 过期, 都报 ValueError."""
    data_dir = pathlib.Path(data_dir)
    # <domain>.py 与 answer.py 里写的是 from schema import ...: 读的时候把这个目录的 schema.py 临时登记成 schema,
    # 数据目录临时放进 sys.path, 读完都还原. hash 借 answer.py 的 item_hash, 与参考模型答题时算的是同一个.
    saved_path, saved_schema = list(sys.path), sys.modules.get("schema")
    sys.path.insert(0, str(data_dir))
    try:
        schema = _module(data_dir / "schema.py", "schema", register=True)
        item_hash = _module(data_dir / "answer.py", "synth_v3_answer").item_hash
        domains = sorted(p.name[: -len(".answer.json")] for p in data_dir.glob("*.answer.json"))
        return {d: _load_domain(data_dir, d, schema, item_hash) for d in domains}
    finally:
        sys.path[:] = saved_path
        if saved_schema is None:
            sys.modules.pop("schema", None)
        else:
            sys.modules["schema"] = saved_schema


def sample_synth_v3(by_domain: dict[str, list[V3Item]], n: int, rng: random.Random) -> list[MenuExample]:
    """训练批里 v3 的那一份: 每条先随机挑领域 (各领域机会均等, 不论题多题少), 再在领域里随机挑一道题,
    换一种问法、重新打乱选项. 软标签的每一格跟着它的选项走."""
    domains = sorted(by_domain)
    out = []
    for _ in range(n):
        it = rng.choice(by_domain[rng.choice(domains)])
        ex = replace(it.example, question=rng.choice(it.asks))
        out.append(reorder_menu(ex, rng.sample(range(len(ex.options)), len(ex.options))))
    return out
