"""一道 API 问题 -> 模型看到的菜单样本 (core.menu.MenuExample). 提示模板用训练时那一份 (core.prompt).

  choice  criteria 的每一项一行, 顺序照调用方写的, 写成「名字: 说明」; 说明是 null 就只写名字
  noul    两行 no / yes, 与训练里的二元题同形; criteria 的 false / true 各跟在 no / yes 后面
  score   分级从低到高各一行. 类型写 choice: 训练里没有带 <|score|> 的题, 期望值由 serve.api 从分布算
菜单连续编号 D0, D1, ..., 与评估集的部署形态相同. state 与 instructions 是对象或数组时展开成缩进 2 格的 JSON;
菜单一行一项, 所以选项里的换行换成空格, 对象或数组的说明写成单行 JSON.
"""

from __future__ import annotations

import json
import re

from decidophobia.core.menu import MenuExample
from decidophobia.core.prompt import state_text
from decidophobia.serve.api import Choice, Noul, Score

YES_NO = ("no", "yes")  # 与训练里二元题的两项同名同序


def _one_line(v) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return re.sub(r"\s*[\r\n]+\s*", " ", s)


def option_row(name: str, description) -> str:
    """菜单一行: 「名字: 说明」, 说明为 None 时只写名字, 压成一行. 训练数据 (synth-v5) 也用它排选项."""
    return _one_line(name) if description is None else f"{_one_line(name)}: {_one_line(description)}"


def to_example(q: Noul | Choice | Score, state, context_label: str) -> MenuExample:
    """q 在 state 上的菜单样本. 选项的类 id 就是行号, gold 只是占位 (推理没有标准答案)."""
    if q.type == "choice":
        names, qtype = [option_row(n, d) for n, d in q.criteria.items()], "choice"
    elif q.type == "noul":
        c = q.criteria
        names, qtype = [option_row(YES_NO[0], c and c.false), option_row(YES_NO[1], c and c.true)], "bool"
    else:
        names, qtype = [_one_line(level) for level in q.criteria], "choice"
    k = len(names)
    return MenuExample(query=state_text(state), options=list(range(k)), gold_idx=0, label=0, option_names=names,
                       context_label=context_label, question=state_text(q.instructions), qtype=qtype)
