"""256 个决策槽 token. 它们既是菜单里的行首标记, 也是答案位置上要预测的 token.

Qwen3 的 tokenizer 有 151669 个 token 而 vocab_size 是 151936, 空 267 行,
256 个 D-token 装进去不用 resize 嵌入, checkpoint 形状不变.
tie_word_embeddings=True 时嵌入行和输出头行是同一个向量: 输出位置上 D_k 的
分数 = h · E[D_k], 与 nn.Linear(hidden, 256) 的第 k 行等价, 区别只是这个向量
还会在菜单里作为输入出现.
"""

from __future__ import annotations

N_SLOTS = 256
D_TOKENS = [f"<|D{i}|>" for i in range(N_SLOTS)]

# 问题类型标记, 写在 "Question (<|bool|>):" 的括号里. choice = k 个无序选项; bool = choice 的 k=2;
# score = k 个有序选项、期望值当分数 —— score 需要有序损失和对应数据集, 现在只占 token 位.
TYPE_TOKENS = ["<|choice|>", "<|bool|>", "<|score|>"]
QTYPES = ("choice", "bool", "score")

# 上下文 (state) 的起止标记, 开了 context_marker 时包在 state 两头, 见 core.prompt.
CONTEXT_TOKENS = ["<|context_start|>", "<|context_end|>"]


def install_d_tokens(tokenizer) -> list[int]:
    """把 256 个 D-token 加进 tokenizer (已有则跳过), 返回它们的 id, 顺序与 D_TOKENS 一致.
    顺带把基模自带、却没标 special 的 added token (Qwen 的 <think> </think> <tool_call> 等) 改标 special:
    不标的话 split_special_tokens 切不开它们, 调用方文字里写着 <think> 就会编成模板的 <think>. id 不变."""
    from tokenizers import AddedToken

    plain = [t.content for t in getattr(tokenizer, "added_tokens_decoder", {}).values() if not t.special]
    if plain:
        tokenizer.add_tokens([AddedToken(c, special=True, normalized=False) for c in plain], special_tokens=True)
    tokenizer.add_tokens(D_TOKENS, special_tokens=True)
    return tokenizer.convert_tokens_to_ids(D_TOKENS)


def install_type_tokens(tokenizer) -> list[int]:
    """把 3 个类型 token 加进 tokenizer, 紧跟 D-token 之后. 先装 D 再装它, id 才稳定."""
    tokenizer.add_tokens(TYPE_TOKENS, special_tokens=True)
    return tokenizer.convert_tokens_to_ids(TYPE_TOKENS)


def install_context_tokens(tokenizer) -> list[int]:
    """把 2 个上下文起止 token 加进 tokenizer, 紧跟类型 token 之后. D、类型、它, 按这个顺序装 id 才稳定."""
    tokenizer.add_tokens(CONTEXT_TOKENS, special_tokens=True)
    return tokenizer.convert_tokens_to_ids(CONTEXT_TOKENS)
