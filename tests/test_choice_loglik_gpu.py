"""scripts/choice-loglik-banking77.py 的 GPU 部分: 前缀算一次 KV cache, 选项分块接在后面 teacher-forced 打分.
结果必须等于「前缀 + 这一个选项」整条前向读出的逐 token logprob.

跑:  OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python tests/test_choice_loglik_gpu.py
fp32 下两条路径只差累加顺序, 容差 1e-3.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _runner import run  # noqa: E402

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "choice-loglik-banking77.py"
_spec = importlib.util.spec_from_file_location("choice_loglik_banking77", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MODEL = "Qwen/Qwen3-0.6B-Base"
NAMES = ["card arrival", "lost or stolen card", "balance not updated after cheque or cash deposit"]
_state: dict = {}


def _load():
    if not _state:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda").eval()
        _state.update(tok=tok, lm=lm)
    return _state["tok"], _state["lm"]


def _full_forward_lp(lm, prefix_ids, cont_ids, stop_ids):
    """参照实现: 整条 prefix+cont 前向, 在每个续写 token 的前一位置取 log_softmax.
    最后一个续写 token (终止符) 的位置取 stop 集合上的 logsumexp."""
    ids = torch.tensor([prefix_ids + cont_ids], device="cuda")
    with torch.no_grad():
        logp = torch.log_softmax(lm(input_ids=ids).logits[0].float(), -1)
    n = len(prefix_ids)
    out = [logp[n - 1 + j, t].item() for j, t in enumerate(cont_ids[:-1])]
    out.append(torch.logsumexp(logp[n - 1 + len(cont_ids) - 1, stop_ids], -1).item())
    return out


def test_chunked_branch_scores_equal_per_choice_full_forward():
    """3 个长度不同的选项 (3 / 6 / 9 个续写 token), chunk=2 -> 两块, 第二块只有 1 条, 第一块有右填充.
    每个选项每个 token 的 logprob 与整条前向差 < 1e-3, 填充位是 NaN."""
    tok, lm = _load()
    prefix = _mod.render_prefix("my new card still hasn't come", NAMES)
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    conts = [tok.encode(" " + n + "\n", add_special_tokens=False) for n in NAMES]
    stop_ids = _mod.stop_token_ids(tok)

    lp, _ = _mod.score_continuations(lm, prefix_ids, conts, stop_ids, chunk=2)
    assert lp.shape == (3, max(len(c) for c in conts)), lp.shape
    for k, c in enumerate(conts):
        want = _full_forward_lp(lm, prefix_ids, c, stop_ids)
        got = lp[k, : len(c)]
        assert np.allclose(got, want, atol=1e-3), (k, got.tolist(), want)
        assert np.isnan(lp[k, len(c):]).all(), lp[k].tolist()


def test_first_position_logits_equal_prefix_last_logits():
    """score_continuations 顺带交回前缀最后位置的全词表 logits, 即模型在 'Answer:' 之后想说什么."""
    tok, lm = _load()
    prefix_ids = tok.encode(_mod.render_prefix("hello", NAMES), add_special_tokens=False)
    conts = [tok.encode(" " + n + "\n", add_special_tokens=False) for n in NAMES]
    _, first = _mod.score_continuations(lm, prefix_ids, conts, _mod.stop_token_ids(tok), chunk=8)
    with torch.no_grad():
        want = lm(input_ids=torch.tensor([prefix_ids], device="cuda")).logits[0, -1].float()
    assert torch.allclose(first, want, atol=1e-3), (first - want).abs().max().item()


if __name__ == "__main__":
    run(globals())
