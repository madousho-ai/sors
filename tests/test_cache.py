"""decidophobia.core.cache 的测试: 上下文算一次 KV cache, 多个问题的分支接在后面一次前向, 结果必须等于各自整条前向.

CPU 上的几条用两层的随机 Qwen3 (fp32), 要求逐位接近; GPU 上的几条用 0.6B (bf16), 比 D 槽的 logits.
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


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


def _alone(lm, ids):
    t = torch.tensor([ids])
    with torch.no_grad():
        return last_logits(lm, t, torch.ones_like(t))[0]


def test_branches_of_different_lengths_in_one_forward_equal_each_prompt_run_alone():
    """三个分支长短不一, 左填充到同长后一次前向. 填充夹在前缀与分支之间, attention mask 挡住它,
    每行真实 token 的位置号从前缀长度接着数 —— 于是每一行都等于「前缀 + 这个分支」单独整条前向."""
    lm = _tiny()
    prefix = [5, 9, 14, 2, 33, 7]
    branches = [[11, 3, 60], [40, 41, 42, 43, 44, 45, 46], [8]]
    got = branch_logits(lm, prefix_cache(lm, prefix), branches, pad_id=0)
    assert got.shape == (3, 64), got.shape
    for i, b in enumerate(branches):
        want = _alone(lm, prefix + b)
        assert torch.allclose(got[i], want, atol=1e-5), (i, (got[i] - want).abs().max().item())


def test_the_pad_id_does_not_change_any_branch():
    """填充位被 mask 掉, 填什么 id 都一样."""
    lm = _tiny()
    cache = prefix_cache(lm, [5, 9, 14])
    a = branch_logits(lm, cache, [[1, 2, 3, 4], [6]], pad_id=0)
    b = branch_logits(lm, cache, [[1, 2, 3, 4], [6]], pad_id=63)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max().item()


def test_the_prefix_cache_is_left_as_it_was():
    """分支在 cache 的拷贝上跑; 同一个 cache 接第二批分支, 结果与第一批相同."""
    lm = _tiny()
    cache = prefix_cache(lm, [5, 9, 14, 2])
    n0 = cache.get_seq_length()
    first = branch_logits(lm, cache, [[1, 2], [3]], pad_id=0)
    assert cache.get_seq_length() == n0, (cache.get_seq_length(), n0)
    second = branch_logits(lm, cache, [[1, 2], [3]], pad_id=0)
    assert torch.equal(first, second)


def test_an_empty_branch_or_prefix_is_refused():
    lm = _tiny()
    for bad in ([[1], []], []):
        try:
            branch_logits(lm, prefix_cache(lm, [5]), bad, pad_id=0)
        except ValueError:
            continue
        raise AssertionError(f"accepted branches {bad}")
    try:
        prefix_cache(lm, [])
    except ValueError:
        return
    raise AssertionError("accepted an empty prefix")


def test_branch_logits_match_full_forward_on_d_slots_on_the_real_model():
    """GPU, 0.6B bf16: 同一个用户句配两个不同长度的菜单 (2 选 / 4 选), 两个分支一次前向,
    D 槽上的 logits 与「整条提示一次前向」一致 (bf16 容差), argmax 相同."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    d_ids = install_d_tokens(tok)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    nm = lambda opts: [f"intent number {i}" for i in opts]  # noqa: E731
    q1 = MenuExample(query="I am seeing a cash withdrawal that is not mine", options=[3, 5], gold_idx=0, label=3,
                     option_names=nm([3, 5]))
    q2 = MenuExample(query=q1.query, options=[0, 1, 2, 7], gold_idx=3, label=7, option_names=nm([0, 1, 2, 7]))

    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    ctx = split_prompt(q1, layout="context-first")[0]
    segs = [split_prompt(ex, layout="context-first")[1] for ex in (q1, q2)]
    # 分段编码拼起来必须等于整条编码, 否则分界处的 BPE 合并会让两条路径看到不同的 token
    for ex, seg in zip((q1, q2), segs):
        assert enc(render_menu(ex, layout="context-first")) == enc(ctx) + enc(seg)

    got = branch_logits(lm, prefix_cache(lm, enc(ctx)), [enc(s) for s in segs], pad_id=tok.pad_token_id)
    for i, ex in enumerate((q1, q2)):
        full = tok(render_menu(ex, layout="context-first"), return_tensors="pt").to("cuda")
        want = last_logits(lm, full["input_ids"], full["attention_mask"])[0]
        k = len(ex.options)
        a, b = got[i, d_ids[:k]], want[d_ids[:k]]
        assert torch.allclose(a, b, atol=0.25, rtol=0.0), (a.tolist(), b.tolist())
        assert a.argmax().item() == b.argmax().item(), (a.tolist(), b.tolist())


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
