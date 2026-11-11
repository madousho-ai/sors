"""评估面板: 各来源的评估集统一成「一条 /v1/systemone 请求 + 每道题的标准答案」, 经 HTTP 打分.

面板的一行 (item):
  set       评估集名, 报告按它分行
  id        题号, 续跑按它跳过已答的
  family    来源里的细分类 (JevBench 格式的 family、Decision Index 的 benchmark、simple 的菜单长度)
  request   {state, questions}: 原样发给服务, 不含任何答案字段
  gold      {问题 id: 标准答案}: choice 是选项键, noul 是 "yes" / "no", score 是档位序号
  type      {问题 id: choice / noul / score}
  k         {问题 id: 选项数}: choice 是 criteria 的键数, noul 是 2, score 是档数

计分照 JevBench v1.5 §3 (也是数据仓库里留出评估集 README 写的规则), 每道题给一个 metric:
  choice  选中的键就是标准答案记 1; 汇总时扣掉随机水平: (正确率 − 平均 1/k) / (1 − 平均 1/k)
  noul    P(yes) ≥ 0.8 且标准答案是 yes, 或 ≤ 0.2 且是 no, 记 1; 落在 0.2..0.8 之间一律记 0
  score   1 − |期望档位 − 标准档位| / (档数 − 1)
另记 top1: 概率最大的那一项是否标准答案 (noul 以 0.5 为界), 跨题型可比, 报告里单列.
请求报错时, 它的每道题都记为答错, 进各自题型的分母.
"""

from __future__ import annotations

import collections

from sors.core.menu import LabeledSet

NOUL_BAND = (0.2, 0.8)


def _k(question: dict) -> int:
    if question["type"] == "noul":
        return 2
    return len(question["criteria"])


def _item(set_name: str, item_id: str, family, state, questions: dict, gold: dict) -> dict:
    return {"set": set_name, "id": item_id, "family": family, "request": {"state": state, "questions": questions},
            "gold": gold, "type": {q: v["type"] for q, v in questions.items()}, "k": {q: _k(v) for q, v in questions.items()}}


def item_from_task(set_name: str, task: dict) -> dict:
    """JevBench 题目格式 (state, question, labels, expected, ...) 的一道题. 题目只发 question 本身."""
    return _item(set_name, task["id"], task.get("family"), task["state"], {"q": task["question"]}, {"q": task["expected"]})


def item_from_simple(row: dict) -> dict:
    """synth-simple-eval 的一行: 菜单是 options 的原文 (没有说明), bool 题改成 noul. family 是 id 的前缀 (k5、bool 等)."""
    family = row["id"].rsplit("-", 1)[0]
    if row["qtype"] == "bool":
        question = {"type": "noul", "instructions": row["question"]}
        gold = row["options"][row["answer"]]
    else:
        question = {"type": "choice", "instructions": row["question"], "criteria": {o: None for o in row["options"]}}
        gold = row["options"][row["answer"]]
    return _item("simple", row["id"], family, row["context"], {"q": question}, {"q": gold})


def items_from_labeled(set_name: str, ls: LabeledSet, instructions: str | None = None) -> list[dict]:
    """训练内部的 LabeledSet (MASSIVE、BoolQ 等) 逐条转成请求. choice 的菜单是全部类别, 按类别名排序, 不带说明;
    bool 集用每条自己的问句. instructions 给了就覆盖 LabeledSet 的问句."""
    out = []
    names = [ls.names[c] for c in sorted(ls.names, key=lambda c: ls.names[c])]
    for i, (query, label) in enumerate(zip(ls.queries, ls.labels)):
        ask = instructions or (ls.questions[i] if ls.questions is not None else ls.question_default)
        if ls.qtype == "bool":
            question = {"type": "noul", "instructions": ask}
        else:
            question = {"type": "choice", "instructions": ask, "criteria": {n: None for n in names}}
        out.append(_item(set_name, f"{set_name}-{i:05d}", None, query, {"q": question}, {"q": ls.names[label]}))
    return out


def item_from_suite_row(set_name: str, row: dict) -> dict:
    """Decision Index 公开题集的一行 (state, questions, gold, _evaluation): 一条请求可以有多道题, 全部保留."""
    ev = row["_evaluation"]
    return _item(set_name, ev["run_id"], ev["dataset"], row["state"], row["questions"], dict(row["gold"]))


def score_question(qtype: str, k: int, answer: dict, gold) -> dict:
    """一道题的 {top1, metric, chance}. answer 是服务返回的那道题的答案对象."""
    if qtype == "choice":
        top1 = answer.get("choice") == gold
        return {"top1": top1, "metric": float(top1), "chance": 1 / k}
    if qtype == "noul":
        p = float(answer["noul"])
        lo, hi = NOUL_BAND
        decided = (p >= hi and gold == "yes") or (p <= lo and gold == "no")
        return {"top1": (p >= 0.5) == (gold == "yes"), "metric": float(decided), "chance": 0.5}
    probs = {int(level): float(p) for level, p in answer["probabilities"].items()}
    top = max(probs, key=probs.get)
    return {"top1": top == int(gold), "metric": 1 - abs(float(answer["score"]) - int(gold)) / (k - 1), "chance": None}


def summarize(results: list[dict]) -> dict[str, dict]:
    """结果行 (每行一条请求: scores 或 error) 按 set 汇总: n 条请求, errors, top1, 每种题型的 metric."""
    acc = collections.defaultdict(lambda: {"n": 0, "errors": 0, "top1": [], "choice": [], "chance": [], "noul": [], "score": []})
    for r in results:
        s = acc[r["set"]]
        s["n"] += 1
        if "error" in r:
            s["errors"] += 1
            scores = {q: {"type": t, "top1": False, "metric": 0.0, "chance": 1 / r["k"][q] if t == "choice" else None}
                      for q, t in r["type"].items()}
        else:
            scores = r["scores"]
        for sc in scores.values():
            s["top1"].append(sc["top1"])
            s[sc["type"]].append(sc["metric"])
            if sc["type"] == "choice":
                s["chance"].append(sc["chance"])
    out = {}
    for name, s in acc.items():
        row = {"n": s["n"], "errors": s["errors"], "top1": sum(s["top1"]) / len(s["top1"]) if s["top1"] else None}
        if s["choice"]:
            a, c = sum(s["choice"]) / len(s["choice"]), sum(s["chance"]) / len(s["chance"])
            row["choice"] = (a - c) / (1 - c)
        else:
            row["choice"] = None
        for t in ("noul", "score"):
            row[t] = sum(s[t]) / len(s[t]) if s[t] else None
        row["questions"] = {t: len(s[t]) for t in ("choice", "noul", "score")}
        out[name] = row
    return out
