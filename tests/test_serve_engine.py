"""decidophobia.serve.engine 的测试. 一层、hidden 16 的随机 Qwen3 配真的 tokenizer, CPU fp32.

要钉住的是: 服务端的路 (state 算一次 KV cache、各问题的分支接在后面分组前向) 给每道题的菜单分布,
与评估的路 (evaluation.scoring.score_examples: 整条提示、左填充成批、一次前向) 对同一份菜单给的相同.

跑:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python tests/test_serve_engine.py
"""

import json
import pathlib
import tempfile

import torch

from _runner import run
from decidophobia.core.checkpoint import save_trained
from decidophobia.core.model import prepare_model
from decidophobia.core.prompt import render_menu
from decidophobia.core.tokens import install_d_tokens, install_type_tokens
from decidophobia.evaluation.scoring import score_examples
from decidophobia.serve.api import SystemOneRequest
from decidophobia.serve.engine import Engine, RequestTooLong, load_engine, recorded_base_model
from decidophobia.serve.menus import to_example
from decidophobia.training.loop import TrainConfig

MODEL = "Qwen/Qwen3-0.6B-Base"
STATE = {"ticket": "Help! My payouts have been failing for 3 days.", "plan": "pro"}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
                                "sales": None, "legal": "Contracts and compliance"}},
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
    "tiny": {"type": "noul", "instructions": "Ok?"},
}
_cache: dict = {}


def _tok_and_ids():
    if "tok" not in _cache:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        d_ids = install_d_tokens(tok)
        _cache["tok"], _cache["d_ids"], _cache["ids"] = tok, d_ids, d_ids + install_type_tokens(tok)
    return _cache["tok"], _cache["d_ids"], _cache["ids"]


def _tiny_cfg():
    from transformers import Qwen3Config

    return Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=8)


def _tiny_lm():
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(0)
    return Qwen3ForCausalLM(_tiny_cfg()).eval()


def _questions(qs=QUESTIONS):
    return SystemOneRequest.model_validate({"state": STATE, "model": "m", "questions": qs}).questions


def _reference(lm, questions, label, type_marker):
    """评估的路: 每道题整条提示, 走 score_examples."""
    tok, d_ids, _ = _tok_and_ids()
    exs = [to_example(q, STATE, label) for q in questions.values()]
    k = max(len(e.options) for e in exs)
    q = score_examples(lm, tok, d_ids, exs, batch_size=len(exs), k_max=k, max_length=4096,
                       layout="context-first", type_marker=type_marker)["q"]
    return {qid: row[: len(e.options)] for qid, row, e in zip(questions, q, exs)}


def _close(got, want, tol=1e-5):
    return all(len(got[k]) == len(want[k]) and max(abs(a - b) for a, b in zip(got[k], want[k])) < tol for k in want)


def test_each_question_gets_the_distribution_the_evaluation_path_gives_its_menu():
    tok, d_ids, _ = _tok_and_ids()
    lm = _tiny_lm()
    qs = _questions()
    for type_marker in (False, True):
        e = Engine(lm, tok, d_ids, context_label="State", type_marker=type_marker)
        got = e.evaluate(STATE, qs).probs
        want = _reference(lm, qs, "State", type_marker)
        assert list(got) == list(qs) and _close(got, want), (type_marker, got, want)
        assert all(abs(sum(p) - 1) < 1e-6 for p in got.values())


def test_splitting_the_questions_into_small_groups_changes_nothing():
    """每组的 cache 是 组员数 × (state + 组内最长的问题) 个 token; 预算小到一题一组, 分布照旧."""
    tok, d_ids, _ = _tok_and_ids()
    lm = _tiny_lm()
    qs = _questions()
    one = Engine(lm, tok, d_ids, max_batch_tokens=1).evaluate(STATE, qs).probs
    all_ = Engine(lm, tok, d_ids, max_batch_tokens=10**6).evaluate(STATE, qs).probs
    assert _close(one, all_), (one, all_)


def test_input_tokens_count_the_state_once_and_each_question_once():
    tok, d_ids, _ = _tok_and_ids()
    qs = _questions()
    e = Engine(_tiny_lm(), tok, d_ids, context_label="State")
    ev = e.evaluate(STATE, qs)
    p = e.prompts(STATE, qs)
    n = len(tok.encode(p["department"][0])) + sum(len(tok.encode(branch)) for _, branch in p.values())
    assert ev.input_tokens == n, (ev.input_tokens, n)


def test_prompts_are_the_state_segment_and_each_questions_segment_as_the_model_reads_them():
    """state 段所有题相同, 前向一次; 问题段各题自己的, 接在后面. 两段拼起来是训练模板 (context-first) 的整条提示,
    上下文标签与类型标记照引擎的设置. 只渲染文本, 不碰模型."""
    tok, d_ids, _ = _tok_and_ids()
    qs = _questions()
    e = Engine(_tiny_lm(), tok, d_ids, context_label="Game state", type_marker=True)
    got = e.prompts(STATE, qs)
    assert list(got) == list(qs)
    assert len({state for state, _ in got.values()}) == 1
    for qid, q in qs.items():
        ex = to_example(q, STATE, "Game state")
        assert "".join(got[qid]) == render_menu(ex, "context-first", type_marker=True), qid
    assert got["department"][0].startswith("Game state: {\n")
    assert got["department"][1].startswith("Question (<|choice|>): Which team should handle this?\nOptions:\n<|D0|>. billing")
    assert got["is_urgent"][1].startswith("Question (<|bool|>):")
    assert got["tiny"][1].endswith("\n\nAnswer:")


def test_without_a_label_the_state_segment_is_the_state_as_the_caller_wrote_it():
    """标签默认不加: 调用方要标签就自己写在 state 开头, 对象 state 照样展开成 JSON, 前面什么都没有."""
    tok, d_ids, _ = _tok_and_ids()
    e = Engine(_tiny_lm(), tok, d_ids)
    text = SystemOneRequest.model_validate({"state": "Game state: hp 3", "model": "m", "questions": QUESTIONS})
    assert {s for s, _ in e.prompts(text.state, text.questions).values()} == {"Game state: hp 3\n\n"}
    assert e.prompts(STATE, _questions())["tiny"][0].startswith('{\n  "ticket": "Help!')


def test_a_state_plus_its_longest_question_over_the_limit_is_refused_before_touching_the_model():
    tok, d_ids, _ = _tok_and_ids()
    qs = _questions()
    e = Engine(_tiny_lm(), tok, d_ids, max_tokens=40)
    try:
        e.evaluate(STATE, qs)
    except RequestTooLong as err:
        assert "40" in str(err), str(err)
        return
    raise AssertionError("a request over max_tokens was answered")


# --------------------------------------------------------------------------
# load_engine: 基模目录 + 训练存档 -> Engine
# --------------------------------------------------------------------------


def _base_dir() -> pathlib.Path:
    """一层的随机 Qwen3 连同 tokenizer (装 D 码之前的) 存成一个基模目录, 形同 HF 上的 Qwen3-0.6B-Base."""
    if "base" not in _cache:
        from transformers import AutoTokenizer

        d = pathlib.Path(tempfile.mkdtemp(prefix="serve-base-"))
        _tiny_lm().save_pretrained(d)
        AutoTokenizer.from_pretrained(MODEL).save_pretrained(d)
        _cache["base"] = d
    return _cache["base"]


def _trained(layout="context-first", type_marker=False):
    """在那个基模上 prepare_model, 把 LoRA 与 D 行打乱成非零的值 (否则 LoRA 的 B 是零, 合并与否看不出差别), 存一份档.
    返回 (档的路径, 没合并的模型)."""
    from transformers import AutoModelForCausalLM

    _, _, ids = _tok_and_ids()
    lm = AutoModelForCausalLM.from_pretrained(_base_dir(), dtype=torch.float32)
    m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0, trainable="attn")
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))
    path = pathlib.Path(tempfile.mkdtemp(prefix="serve-ckpt-")) / "trained.safetensors"
    save_trained(m, ids, TrainConfig(layout=layout, type_marker=type_marker), path)
    return path, m.eval()


def test_load_engine_merges_the_lora_and_answers_as_the_unmerged_model_would():
    """档里记的 type_marker 照搬; LoRA 合并进权重之后, 分布与训练时的旁路挂法相同 (fp32 容差)."""
    path, m = _trained(type_marker=True)
    e = load_engine(path, _base_dir(), context_label="State", device="cpu", dtype=torch.float32)
    assert e.type_marker is True
    assert not any("lora_" in n for n, _ in e.lm.named_parameters()), "LoRA was not merged"
    qs = _questions()
    got = e.evaluate(STATE, qs).probs
    want = _reference(m, qs, "State", True)
    assert _close(got, want, tol=1e-4), (got, want)


def test_load_engine_refuses_a_checkpoint_trained_with_the_menu_before_the_state():
    """menu-first 的提示没有可共享的 state 前缀, 服务端这条路算不了它."""
    path, _ = _trained(layout="menu-first")
    try:
        load_engine(path, _base_dir(), device="cpu", dtype=torch.float32)
    except ValueError as err:
        assert "menu-first" in str(err), str(err)
        return
    raise AssertionError("loaded a menu-first checkpoint")


def test_the_base_model_is_read_from_the_result_json_of_the_run_the_checkpoint_belongs_to():
    """scripts/train.py 把 --model 记在 runs/<run>/result.json; 存档本身没记. 最终档与途中的档都找得到."""
    run_dir = pathlib.Path(tempfile.mkdtemp(prefix="serve-run-"))
    (run_dir / "checkpoints").mkdir()
    (run_dir / "result.json").write_text(json.dumps({"args": {"model": "Qwen/Qwen3-1.7B-Base"}}))
    assert recorded_base_model(run_dir / "trained.safetensors") == "Qwen/Qwen3-1.7B-Base"
    assert recorded_base_model(run_dir / "checkpoints" / "step-00100.safetensors") == "Qwen/Qwen3-1.7B-Base"
    assert recorded_base_model(pathlib.Path(tempfile.mkdtemp()) / "trained.safetensors") is None


if __name__ == "__main__":
    run(globals())
