"""decidophobia.core.cache 的测试: 上下文算一次 KV cache, 多个问题各自接在后面, 结果必须等于完整前向.

跑:  OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python tests/test_cache.py
"""

import sys

import torch

from decidophobia.core.cache import branch_logits, prefix_cache
from decidophobia.core.menu import MenuExample
from decidophobia.core.model import last_logits
from decidophobia.core.prompt import render_menu, split_prompt
from decidophobia.core.tokens import install_d_tokens

MODEL = "Qwen/Qwen3-0.6B-Base"


def test_branch_logits_match_full_forward_on_d_slots():
    """同一个用户句配两个不同长度的菜单 (2 选 / 4 选):
    context 前向一次得 cache, 每个问题只喂自己那段 -> 最后位置 logit
    必须与「整条提示一次前向」在 D 槽上一致 (bf16 容差).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    d_ids = install_d_tokens(tok)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    nm = lambda opts: [f"intent number {i}" for i in opts]  # noqa: E731
    q1 = MenuExample(query="I am seeing a cash withdrawal that is not mine", options=[3, 5], gold_idx=0, label=3, option_names=nm([3, 5]))
    q2 = MenuExample(query=q1.query, options=[0, 1, 2, 7], gold_idx=3, label=7, option_names=nm([0, 1, 2, 7]))

    ctx, s1 = split_prompt(q1, layout="context-first")
    _, s2 = split_prompt(q2, layout="context-first")
    # 分段编码拼起来必须等于整条编码, 否则分界处的 BPE 合并会让两条路径看到不同的 token
    for ex, seg in [(q1, s1), (q2, s2)]:
        whole = tok.encode(render_menu(ex, layout="context-first"), add_special_tokens=False)
        parts = tok.encode(ctx, add_special_tokens=False) + tok.encode(seg, add_special_tokens=False)
        assert whole == parts, (len(whole), len(parts))

    cache = prefix_cache(lm, tok, ctx)
    for ex, seg in [(q1, s1), (q2, s2)]:
        got = branch_logits(lm, tok, cache, seg)
        full = tok(render_menu(ex, layout="context-first"), return_tensors="pt").to("cuda")
        want = last_logits(lm, full["input_ids"], full["attention_mask"])
        k = len(ex.options)
        a, b = got[0, d_ids[:k]], want[0, d_ids[:k]]
        assert torch.allclose(a, b, atol=0.25, rtol=0.0), (a.tolist(), b.tolist())
        # argmax 一致这条不带容差
        assert a.argmax().item() == b.argmax().item(), (a.tolist(), b.tolist())


def test_prefix_cache_is_not_mutated_by_branches():
    """两个分支先后接在同一个 cache 上, 第二个不得看到第一个分支的 token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    install_d_tokens(tok)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    ex = MenuExample(query="hello there", options=[0, 1], gold_idx=0, label=0, option_names=["intent number 0", "intent number 1"])
    ctx, seg = split_prompt(ex, layout="context-first")
    cache = prefix_cache(lm, tok, ctx)
    n0 = cache.get_seq_length()
    first = branch_logits(lm, tok, cache, seg)
    assert cache.get_seq_length() == n0, (cache.get_seq_length(), n0)
    second = branch_logits(lm, tok, cache, seg)
    assert torch.equal(first, second)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
    print(f"\n{failed} failed")
    sys.exit(1 if failed else 0)
