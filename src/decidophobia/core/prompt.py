"""把一条菜单样本渲染成提示. 模板是被训练的对象, 改动等于换任务.

三段: 上下文 / 问句 / 菜单 + Answer:. 两种布局:
  context-first  <ctx>\\n\\n | Question: <q>\\nOptions:\\n<menu>\\n\\nAnswer:
                 前缀只含上下文, N 个问题共用它的 KV cache, 每个问题只喂自己那段. 默认.
  menu-first     Question: <q>\\nOptions:\\n<menu>\\n\\n<ctx>\\nAnswer:
                 答案位置读上下文时已经知道选项. 对照组, 没有可共享的前缀.
<ctx> 是「<label>: <query>」; 标签为空串时 <query> 原样放进去, 不带「: 」: 推理服务默认这样,
要标签的调用方自己写在 state 开头. 训练数据的标签都不为空.
context_marker=True 时 <ctx> 两头包上 <|context_start|> <|context_end|>, 标签照旧在包裹里面.

默认定为 context-first 的依据 (attn LoRA + cosine, 2000 步, 17 个留出类, 10 选 1):
unseen acc 0.903 vs 0.888, NLL 0.310 vs 0.341, 9 个评估点全部领先, 后半程 sd 0.0012 vs 0.0083,
step 250 即到 0.88 (menu-first 要 1000 步). 与冻结探针 h_bare 0.880 > h_menu 0.857 同向.
唯一输的一项是 ECE 0.041 vs 0.033 (过自信 +3.8 vs +2.9 点), 标定项加进来之后再看.

提示在这里是一串片段: 普通文字, 或者模板自己放进去的保留 token (Special: D 码、类型标记、上下文起止).
encode_prompts 只把 Special 编成 special token, 其余文字 (连同 state、问句、选项名) 里就算写着
'<|D5|>' 也切成普通 token —— 调用方的文字伪造不出菜单行、类型标记或上下文边界.
split_prompt / render_menu 是同一串片段拼成的文本, 给人看, 也给测试对照整条编码.
split_prompt 给出两段, 分界处两边都以换行收尾, BPE 不会跨界合并.
"""

from __future__ import annotations

import json

from decidophobia.core.menu import MenuExample
from decidophobia.core.tokens import CONTEXT_TOKENS, D_TOKENS

LAYOUTS = ("context-first", "menu-first")
DEFAULT_LAYOUT = "context-first"
DEFAULT_QUESTION = "Which option best describes the message?"


class Special(str):
    """模板自己放进提示的保留 token. 提示的片段里只有这种会编成 special token, 见 encode_prompts."""


def state_text(state) -> str:
    """state 在提示里的样子: 字符串原样; 对象或数组展开成缩进 2 格的 JSON, 非 ASCII 不转义.
    synth-v3 的 JSON state 训练时这样写, 推理服务收到的对象也这样写."""
    return state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)


def _menu(ex: MenuExample) -> list[str]:
    out: list[str] = []
    for i, (c, name) in enumerate(zip(ex.slot_codes, ex.option_names)):
        out += ["\n"] if i else []
        out += [Special(D_TOKENS[c]), f". {name}"]
    return out


def _question_block(ex: MenuExample, type_marker: bool) -> list[str]:
    q = ex.question if ex.question is not None else DEFAULT_QUESTION
    label = ["Question (", Special(f"<|{ex.qtype}|>"), "):"] if type_marker else ["Question:"]
    return [*label, f" {q}\nOptions:\n", *_menu(ex)]


def _context(ex: MenuExample, context_marker: bool) -> list[str]:
    ctx = f"{ex.context_label}: {ex.query}" if ex.context_label else ex.query
    return [Special(CONTEXT_TOKENS[0]), ctx, Special(CONTEXT_TOKENS[1])] if context_marker else [ctx]


def prompt_pieces(ex: MenuExample, layout: str = DEFAULT_LAYOUT, type_marker: bool = False,
                  context_marker: bool = False) -> tuple[list[str], list[str]]:
    """(context 片段, question 片段), 分法同 split_prompt; 每个片段是普通文字或 Special."""
    ctx, qb = _context(ex, context_marker), _question_block(ex, type_marker)
    if layout == "context-first":
        return [*ctx, "\n\n"], [*qb, "\n\nAnswer:"]
    if layout == "menu-first":
        return [], [*qb, "\n\n", *ctx, "\nAnswer:"]
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def split_prompt(ex: MenuExample, layout: str = DEFAULT_LAYOUT, type_marker: bool = False,
                 context_marker: bool = False) -> tuple[str, str]:
    """(context, question). menu-first 下 context 为空串 —— 那种布局没有可共享的前缀.
    type_marker=True 时问句标签写成 'Question (<|bool|>):', 类型 token 挂在 Question 这个锚上.
    context_marker=True 时 context 包在 <|context_start|> <|context_end|> 之间."""
    ctx, q = prompt_pieces(ex, layout, type_marker, context_marker)
    return "".join(ctx), "".join(q)


def render_menu(ex: MenuExample, layout: str = DEFAULT_LAYOUT, type_marker: bool = False,
                context_marker: bool = False) -> str:
    """整条提示. 第 i 行 '<|D{slot_codes[i]}|>. <名字>' (不给 codes 时就是 D{i});
    以 'Answer:' 收尾, 答案 token 紧跟冒号之后, 中间无空格."""
    ctx, q = split_prompt(ex, layout, type_marker, context_marker)
    return ctx + q


def _special_id(tokenizer, name: str) -> int:
    i = tokenizer.convert_tokens_to_ids(name)
    if i is None or i == tokenizer.unk_token_id:
        raise ValueError(f"the prompt uses {name}, which is not installed in the tokenizer (core.tokens.install_*)")
    return i


def encode_prompts(tokenizer, prompts: list[list[str]]) -> list[list[int]]:
    """每条提示的片段编成 token id. Special 直接换成它的 id; 两个 Special 之间的文字拼成一段,
    全部文字段一次批量编码, 且 split_special_tokens=True: 文字里写着的保留 token 名切成普通 token.
    文字里没有保留 token 名时, 结果与整条字符串的 tokenizer.encode 逐 id 相同 —— special token 本来就是 BPE 的硬边界.
    Special 没装进 tokenizer 就报错."""
    plans, texts, ids = [], [], {}
    for pieces in prompts:
        plan, buf = [], []
        for p in [*pieces, None]:  # None 收尾, 把最后一段文字冲出来
            if (p is None or isinstance(p, Special)) and buf:
                plan.append(("text", len(texts)))
                texts.append("".join(buf))
                buf = []
            if isinstance(p, Special):
                if p not in ids:
                    ids[p] = _special_id(tokenizer, p)
                plan.append(("id", ids[p]))
            elif p:
                buf.append(p)
        plans.append(plan)
    enc = tokenizer(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"] if texts else []
    return [[i for kind, v in plan for i in ([v] if kind == "id" else enc[v])] for plan in plans]
