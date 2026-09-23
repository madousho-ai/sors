"""把一条菜单样本渲染成提示文本. 模板是被训练的对象, 改动等于换任务.

三段: 上下文 / 问句 / 菜单 + Answer:. 两种布局:
  context-first  <label>: <query>\\n\\n | Question: <q>\\nOptions:\\n<menu>\\n\\nAnswer:
                 前缀只含上下文, N 个问题共用它的 KV cache, 每个问题只喂自己那段. 默认.
  menu-first     Question: <q>\\nOptions:\\n<menu>\\n\\n<label>: <query>\\nAnswer:
                 答案位置读上下文时已经知道选项. 对照组, 没有可共享的前缀.

默认定为 context-first 的依据 (attn LoRA + cosine, 2000 步, 17 个留出类, 10 选 1):
unseen acc 0.903 vs 0.888, NLL 0.310 vs 0.341, 9 个评估点全部领先, 后半程 sd 0.0012 vs 0.0083,
step 250 即到 0.88 (menu-first 要 1000 步). 与冻结探针 h_bare 0.880 > h_menu 0.857 同向.
唯一输的一项是 ECE 0.041 vs 0.033 (过自信 +3.8 vs +2.9 点), 标定项加进来之后再看.

split_prompt 给出两段, 分界处两边都以换行收尾, BPE 不会跨界合并.
"""

from __future__ import annotations

from decidophobia.data import MenuExample
from decidophobia.tokens import D_TOKENS

LAYOUTS = ("context-first", "menu-first")
DEFAULT_LAYOUT = "context-first"
DEFAULT_QUESTION = "Which option best describes the message?"


def _menu(ex: MenuExample) -> str:
    return "\n".join(f"{D_TOKENS[c]}. {name}" for c, name in zip(ex.slot_codes, ex.option_names))


def _question_block(ex: MenuExample, type_marker: bool) -> str:
    q = ex.question if ex.question is not None else DEFAULT_QUESTION
    label = f"Question (<|{ex.qtype}|>):" if type_marker else "Question:"
    return f"{label} {q}\nOptions:\n{_menu(ex)}"


def split_prompt(ex: MenuExample, layout: str = DEFAULT_LAYOUT, type_marker: bool = False) -> tuple[str, str]:
    """(context, question). menu-first 下 context 为空串 —— 那种布局没有可共享的前缀.
    type_marker=True 时问句标签写成 'Question (<|bool|>):', 类型 token 挂在 Question 这个锚上."""
    ctx = f"{ex.context_label}: {ex.query}"
    qb = _question_block(ex, type_marker)
    if layout == "context-first":
        return ctx + "\n\n", qb + "\n\nAnswer:"
    if layout == "menu-first":
        return "", qb + f"\n\n{ctx}\nAnswer:"
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def render_menu(ex: MenuExample, layout: str = DEFAULT_LAYOUT, type_marker: bool = False) -> str:
    """整条提示. 第 i 行 '<|D{slot_codes[i]}|>. <名字>' (不给 codes 时就是 D{i});
    以 'Answer:' 收尾, 答案 token 紧跟冒号之后, 中间无空格."""
    ctx, q = split_prompt(ex, layout, type_marker)
    return ctx + q
