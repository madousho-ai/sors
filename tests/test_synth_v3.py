"""decidophobia.data.synth_v3 的测试: datasets/synth-intents-v3 读成逐题的 MenuExample, 以及训练时的抽题.

跑:  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tests/test_synth_v3.py
"""

import collections
import contextlib
import importlib.util
import io
import os
import sys
from unittest.mock import patch
import json
import pathlib
import random
import shutil
import tempfile

from _runner import run
from decidophobia.data.synth_v3 import DEFAULT_DIR, load_synth_v3, sample_synth_v3

DOMAINS = {"browser_agent", "coding_ci", "hotel", "sec_ops", "telecom"}


def _answers(domain):
    return json.loads((DEFAULT_DIR / f"{domain}.answer.json").read_text())["answers"]


def _find(items, text_id, question_id):
    return next(it for it in items if it.text_id == text_id and it.question_id == question_id)


def test_every_domain_with_an_answer_file_is_loaded_with_one_item_per_question_a_text_is_asked():
    """一个领域 = <domain>.py 加它的 <domain>.answer.json. 每份文本被问到的每道题 (needs / texts 满足) 各一条."""
    by = load_synth_v3()
    assert set(by) == DOMAINS, set(by)
    for d, items in by.items():
        assert items and all(it.domain == d for it in items)
        keys = [(it.text_id, it.question_id) for it in items]
        assert len(set(keys)) == len(keys), d
    assert sum(len(v) for v in by.values()) == 688


def test_a_stated_answer_is_a_hard_label_even_where_the_reference_model_disagrees():
    """写明的答案一律硬标签 (target None). sec_ops queue_a 的 queue_new_owned 写明 yes,
    关思考的 DeepSeek 在 answer.json 里给 no 0.95 —— 训练只认写明的 yes."""
    it = _find(load_synth_v3()["sec_ops"], "sec_ops_queue_a", "queue_new_owned")
    ex = it.example
    assert ex.target is None and ex.option_names == ["no", "yes"] and ex.gold_idx == 1 and ex.qtype == "bool"
    assert _answers("sec_ops")["sec_ops_queue_a"]["queue_new_owned"]["p"][0] > 0.9


def test_an_unstated_question_takes_the_reference_models_distribution_renormalised():
    """没写明的题用 answer.json 的 p (存到小数点后 3 位, 加起来不一定正好是 1, 读进来重新归一);
    gold 是概率最高的那一项."""
    by = load_synth_v3()
    it = _find(by["coding_ci"], "coding_ci_ci1", "ci_next")
    p = _answers("coding_ci")["coding_ci_ci1"]["ci_next"]["p"]
    ex = it.example
    assert all(abs(a - b / sum(p)) < 1e-9 for a, b in zip(ex.target, p)), (ex.target, p)
    assert ex.gold_idx == max(range(len(p)), key=p.__getitem__)
    soft = [it for v in by.values() for it in v if it.example.target is not None]
    assert all(abs(sum(it.example.target) - 1) < 1e-9 for it in soft)


def test_a_question_the_text_gives_no_clue_for_is_uniform():
    by = load_synth_v3()
    unknown = [it for it in by["telecom"] if it.source == "unknown"]
    assert len(unknown) == 29
    assert all(it.example.target == [1 / len(it.example.options)] * len(it.example.options) for it in unknown)


def test_label_sources_add_up():
    c = collections.Counter(it.source for v in load_synth_v3().values() for it in v)
    assert c == {"stated": 435, "soft": 224, "unknown": 29}, c


def test_an_example_carries_the_domain_label_the_rendered_state_and_the_question_options():
    """上下文标签是领域的 LABEL; JSON state 缩进 2 格展开, 与参考模型答题时看到的相同; 选项就是题里的说明."""
    by = load_synth_v3()
    it = _find(by["sec_ops"], "sec_ops_scan_a", "scan_a_urgent")
    ex = it.example
    assert ex.context_label == "Security on-call screen"
    assert ex.query.startswith("{\n  ") and json.loads(ex.query)
    assert ex.options == list(range(len(ex.option_names))) and ex.label == ex.options[ex.gold_idx]
    assert ex.question in it.asks


def test_a_stale_answer_file_is_refused():
    """<domain>.py 改了题或文本而没重跑 answer.py 时, answer.json 里那道题的 hash 对不上, 软标签已经过期: 读的时候报错."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        for f in DEFAULT_DIR.iterdir():
            if f.suffix in (".py", ".json"):
                shutil.copy(f, tmp / f.name)
        doc = json.loads((tmp / "coding_ci.answer.json").read_text())
        doc["answers"]["coding_ci_ci1"]["ci_next"]["hash"] = "000000000000"
        (tmp / "coding_ci.answer.json").write_text(json.dumps(doc))
        try:
            load_synth_v3(tmp)
        except ValueError as e:
            assert "coding_ci_ci1" in str(e) and "ci_next" in str(e), e
            return
        raise AssertionError("a stale answer.json was accepted")
    finally:
        shutil.rmtree(tmp)


def test_sampling_gives_each_domain_an_equal_share():
    """一条样本先随机挑领域再在领域里挑题: telecom 408 题、browser_agent 54 题, 各占约五分之一."""
    by = load_synth_v3()
    exs = sample_synth_v3(by, 5000, random.Random(0))
    share = collections.Counter(e.context_label for e in exs)
    assert len(share) == 5 and all(abs(v / 5000 - 0.2) < 0.03 for v in share.values()), share


def test_sampling_reshuffles_the_options_and_the_target_follows_its_row():
    """每次抽到都重新打乱选项, 答案落在不同位置; 软标签的每一格跟着它的选项走; 问法在几种之间换."""
    by = {"coding_ci": [_find(load_synth_v3()["coding_ci"], "coding_ci_ci1", "ci_next")]}
    base = by["coding_ci"][0].example
    exs = sample_synth_v3(by, 300, random.Random(1))
    assert len({e.gold_idx for e in exs}) == len(base.options)
    for e in exs:
        assert e.option_names[e.gold_idx] == base.option_names[base.gold_idx]
        for name, p in zip(e.option_names, e.target):
            assert abs(p - base.target[base.option_names.index(name)]) < 1e-12
    assert len({e.question for e in exs}) == len(by["coding_ci"][0].asks) > 1


def test_sampling_hard_labels_keeps_them_hard():
    by = {"sec_ops": [_find(load_synth_v3()["sec_ops"], "sec_ops_queue_a", "queue_new_owned")]}
    exs = sample_synth_v3(by, 50, random.Random(2))
    assert all(e.target is None and e.option_names[e.gold_idx] == "yes" for e in exs)
    assert {tuple(e.option_names) for e in exs} == {("no", "yes"), ("yes", "no")}



def _reference_endpoint(*, environment=None, command_line=None, summary=False):
    """Run the real CLI until its first HTTP request, with all network I/O intercepted."""
    class RequestObserved(BaseException):
        pass

    observed = []

    def intercept(request, timeout):
        observed.append(request.full_url)
        raise RequestObserved()

    with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
        os.environ.pop("LLM_BASE_URL", None)
        if environment is not None:
            os.environ["LLM_BASE_URL"] = environment
        spec = importlib.util.spec_from_file_location("reference_answer_test", DEFAULT_DIR / "answer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        build = module.build
        argv = [str(DEFAULT_DIR / "answer.py"), "telecom", "--threads", "1"]
        argv.append("--summary" if summary else "--restart")
        if command_line is not None:
            argv.extend(["--base-url", command_line])
        output = io.StringIO()
        refusal = None
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
                patch.object(module, "build", side_effect=lambda m, doc: build(m, doc)[:1]), \
                patch.object(module.urllib.request, "urlopen", side_effect=intercept):
            try:
                module.main()
            except RequestObserved:
                pass
            except SystemExit as exc:
                refusal = str(exc.code)
        return observed, refusal, output.getvalue()


def test_reference_answer_requires_an_explicit_endpoint_before_sending_a_request():
    observed, refusal, _ = _reference_endpoint()
    assert observed == [], "a request was sent without an explicitly configured endpoint"
    assert refusal and "LLM_BASE_URL" in refusal and "--base-url" in refusal


def test_reference_answer_sends_requests_to_the_environment_endpoint():
    observed, refusal, _ = _reference_endpoint(environment="https://environment.example/v1")
    assert observed == ["https://environment.example/v1/chat/completions"], "environment endpoint was ignored"
    assert refusal is None


def test_reference_answer_command_line_endpoint_overrides_the_environment():
    observed, refusal, _ = _reference_endpoint(
        environment="https://environment.example/v1", command_line="https://command.example/v1")
    assert observed == ["https://command.example/v1/chat/completions"]
    assert refusal is None


def test_reference_answer_offline_summary_needs_no_endpoint():
    observed, refusal, output = _reference_endpoint(summary=True)
    assert observed == [] and refusal is None
    assert output.strip(), "offline summary did not produce statistics"

if __name__ == "__main__":
    run(globals())
