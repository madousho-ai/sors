"""蒸馏式内化 (candidate, 打包前向): 老师分支 = 前缀 + 计算过程 + 选项, 学生分支 = 前缀 + 选项 (推理形式).

计算过程 (MenuExample.trace) 只是老师读的输入, 不对任何位置算下一个词的损失; 监督只落在选项读出的 H 上.
打包时计算过程是前缀的又一段子段, 老师的各选项分支挂在它下面: 全注意力里老师分支看前缀、计算过程与自己,
DeltaNet 里老师分支从计算过程末尾的状态出发. 语义与把 前缀 + 计算过程 + 选项 单独编码一遍相同.
"""

import random
from dataclasses import replace

import pytest
import torch

from test_candidate import candidate_batch, candidate_model
from test_decision import examples

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="DeltaNet kernels need CUDA")
TRACES = ["red context context red",
          None]


def traced_examples():
    return [replace(ex, trace=t) for ex, t in zip(examples(), TRACES)]


def test_collate_adds_trace_tokens_only_for_examples_that_have_a_trace():
    _, tok, d, _ = candidate_model(0)
    plain = candidate_batch(tok, d, traced_examples())
    b = candidate_batch(tok, d, traced_examples(), traces=True)
    assert "trace_input_ids" not in plain
    mask = b["trace_attention_mask"]
    assert mask[0].sum() > 0 and mask[1].sum() == 0
    text = tok.decode(b["trace_input_ids"][0][mask[0].bool()])
    assert TRACES[0] in text
    for key, value in plain.items():
        assert torch.equal(value, b[key]), key


def test_partner_reads_the_trace_written_for_its_own_wording():
    from sors.core.menu import partner
    ex = replace(examples()[0], partner_query="other wording", trace="for the first", partner_trace="for the other")
    p = partner(ex, random.Random(0))
    assert p.query == "other wording" and p.trace == "for the other"


def _teacher_reference(m, data):
    """老师分支逐个单独编码: 前缀 + 计算过程 + 选项续写, 取最后一个 token."""
    valid = data["slot_ids"] >= 0
    has = data["trace_attention_mask"].bool().any(1)
    out = []
    for r, c in valid.nonzero().tolist():
        if not has[r]:
            continue
        prefix = data["input_ids"][r][data["attention_mask"][r].bool()][:int(data["prefix_lengths"][r])]
        trace = data["trace_input_ids"][r][data["trace_attention_mask"][r].bool()]
        n = int(data["suffix_attention_mask"][r, c].sum())
        ids = torch.cat([prefix, trace, data["suffix_input_ids"][r, c, -n:]])[None]
        out.append(m._encode(ids, torch.ones_like(ids))[0, -1])
    return torch.stack(out)


def _setup():
    m, tok, d, _ = candidate_model(0, "qwen35", trainable="full")
    m.to("cuda")
    data = {k: v.cuda() for k, v in candidate_batch(tok, d, traced_examples(), traces=True).items()}
    return m, data


@cuda
def test_teacher_readout_matches_encoding_prefix_trace_and_option_on_their_own():
    from sors.core.packed import packed_candidate_hidden

    m, data = _setup()
    m.eval()
    valid = data["slot_ids"] >= 0
    with torch.no_grad():
        student, teacher = packed_candidate_hidden(m.decoder, data, valid, traces=True)
        alone = packed_candidate_hidden(m.decoder, data, valid)
        want = _teacher_reference(m, data)
    torch.testing.assert_close(student, alone)  # 学生分支不受老师那几段影响
    torch.testing.assert_close(teacher.float(), want.float(), atol=2e-3, rtol=2e-3)


@cuda
def test_teacher_gradients_match_encoding_each_teacher_branch_on_its_own(monkeypatch):
    import fla.ops.gated_delta_rule as fla_gdn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from exact_gdn import exact_gated_delta_rule
    from sors.core.packed import packed_candidate_hidden

    monkeypatch.setattr(fla_gdn, "chunk_gated_delta_rule", exact_gated_delta_rule)
    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", exact_gated_delta_rule)
    m, data = _setup()
    m.double().train()
    valid = data["slot_ids"] >= 0
    grads = []
    for packed in (False, True):
        m.zero_grad(set_to_none=True)
        h = packed_candidate_hidden(m.decoder, data, valid, traces=True)[1] if packed else _teacher_reference(m, data)
        torch.manual_seed(5)
        (h * torch.randn_like(h)).sum().backward()
        grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters()
                      if p.grad is not None and n.startswith("base.")})
    assert grads[0].keys() == grads[1].keys()
    for name, g in grads[0].items():
        assert float((grads[1][name] - g).norm() / g.norm()) < 1e-5, name


@cuda
def test_distill_loss_pulls_the_student_readout_toward_a_detached_teacher():
    from sors.training.distill import distill_outputs

    m, data = _setup()
    m.set_candidate_prefix_cache("packed")
    m.train()
    out = distill_outputs(m, data)
    assert out["student_logits"].shape[0] == 2
    assert out["teacher_logits"].shape[0] == 1  # 只有第一道题带计算过程
    out["distill"].backward()
    assert out["teacher_h"].grad is None and not out["teacher_h"].requires_grad
    assert float(out["distill"]) >= 0


def test_synth_examples_carry_the_trace_written_for_their_wording_and_question():
    from sors.data.synth_v5 import V5Item, attach_traces, item_example
    it = V5Item("d", "c1", "q1", "k", "complex", "Note", ["text A", "text B"], ["ask 1", "ask 2"],
                ["yes", "no"], None, "bool", None, 0)
    traces = [{"context_id": "c1", "question_id": "q1", "text": 1, "ask": 0, "cot": "B for ask one"},
              {"context_id": "c1", "question_id": "q1", "text": 0, "ask": 0, "cot": "A for ask one"},
              {"context_id": "zz", "question_id": "q1", "text": 0, "ask": 0, "cot": "other item"}]
    [traced] = attach_traces([it], traces)
    seen = set()
    for seed in range(40):
        ex = item_example(traced, random.Random(seed), 0.0, fallback_rate=0.0)
        want = {("text A", "ask 1"): "A for ask one", ("text B", "ask 1"): "B for ask one"}.get((ex.query, ex.question))
        assert ex.trace == want
        other = {"text A": "A for ask one", "text B": "B for ask one"}[ex.partner_query] if ex.question == "ask 1" else None
        assert ex.partner_trace == other
        seen.add(ex.trace)
    assert {"A for ask one", "B for ask one", None} <= seen


def test_group_cost_counts_the_trace_and_the_teacher_branches():
    from sors.training.loop import batch_costs
    _, tok, d, _ = candidate_model(0)
    plain = batch_costs(candidate_batch(tok, d, traced_examples(), option_branches=False), "candidate")
    b = candidate_batch(tok, d, traced_examples(), option_branches=False, traces=True)
    costs = batch_costs(b, "candidate")
    trace = int(b["trace_attention_mask"][0].sum())
    assert costs[1] == plain[1]
    assert costs[0] - plain[0] >= trace + int(b["suffix_attention_mask"][0].sum())


@cuda
def test_a_training_step_with_distillation_backpropagates_teacher_and_distill_terms():
    from sors.training.loop import TrainConfig, backward_groups, _prepare_batch

    m, tok, d, _ = candidate_model(0, "qwen35", trainable="full")
    m.to("cuda").train()
    m.set_candidate_prefix_cache("packed")
    exs = traced_examples()
    spec = (list(d), 4, "context-first", 512, False, False, "candidate", 0.0, False, True, True)
    b, _, _, costs = _prepare_batch(spec, (exs, None), tok)
    assert "trace_input_ids" in b and "option_input_ids" not in b
    b = {k: v.cuda() for k, v in b.items()}
    cfg = TrainConfig(loss="menu", micro_tokens=4096, accumulate_gradients=True, candidate_prefix_cache="packed",
                      distill_weight=1.0, teacher_weight=1.0)
    stats = {}
    backward_groups(m, cfg, exs, b, d, costs, stats=stats)
    assert stats["teacher_n"] == 1 and stats["distill"] >= 0 and stats["teacher_ce"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("base."))


def test_cli_attaches_traces_from_a_file_to_the_synth_data(tmp_path):
    import importlib.util, json, pathlib
    from sors.data.synth_v5 import load_synth_v5
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_cli_distill", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    it = next(i for i in load_synth_v5() if i.goal == "complex" and i.target is None)
    path = tmp_path / "traces.jsonl"
    rows = [{"context_id": it.context_id, "question_id": it.question_id, "text": t, "ask": a, "cot": f"trace {t} {a}"}
            for t in range(len(it.texts)) for a in range(len(it.asks))]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = cli.build_parser().parse_args(["--dataset", "synth-v5.14", "--eval", "simple", "--architecture", "candidate",
                                          "--candidate-prefix-cache", "packed", "--distill-traces", str(path),
                                          "--distill-weight", "0.5", "--teacher-weight", "2"])
    assert (args.distill_weight, args.teacher_weight) == (0.5, 2.0)
    _, _, info = cli.build_data(args)
    assert info["synth_v5_traced_items"] == 1
    assert "-dist0.5" in cli.run_tag(args)


def test_trace_keep_fraction_falls_linearly_between_the_two_marks():
    from sors.training.distill import trace_keep
    assert trace_keep(1, 100, (0.2, 0.6)) == 1.0
    assert trace_keep(20, 100, (0.2, 0.6)) == 1.0
    assert trace_keep(40, 100, (0.2, 0.6)) == pytest.approx(0.5)
    assert trace_keep(60, 100, (0.2, 0.6)) == 0.0
    assert trace_keep(99, 100, (0.2, 0.6)) == 0.0
    assert trace_keep(50, 100, None) == 1.0


def test_shortening_drops_whole_sentences_from_the_end():
    from sors.training.distill import shorten_trace
    text = "Rule is 15 days. Paid on day 20.\nSo 20 > 15. Late."
    assert shorten_trace(text, 1.0) == text
    assert shorten_trace(text, 0.0) is None
    assert shorten_trace(text, 0.7) == "Rule is 15 days. Paid on day 20."
    assert shorten_trace(text, 0.35) == "Rule is 15 days."
    assert shorten_trace(text, 0.1) is None  # 连第一句都放不下: 老师这道题不读计算过程
    assert shorten_trace(None, 0.5) is None


def test_shortened_examples_cut_both_wordings_and_leave_other_fields():
    from sors.training.distill import shorten_examples
    ex = replace(examples()[0], trace="First step here. Second step here.", partner_trace="Other first. Other second.")
    [out] = shorten_examples([ex], 0.6)
    assert out.trace == "First step here." and out.partner_trace == "Other first."
    assert replace(out, trace=ex.trace, partner_trace=ex.partner_trace) == ex
    assert shorten_examples([ex], 1.0)[0] is ex


def test_cli_reads_the_trace_fade_schedule(tmp_path):
    import importlib.util, pathlib
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_cli_fade", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base = ["--dataset", "synth-v5.14", "--architecture", "candidate", "--candidate-prefix-cache", "packed",
            "--distill-traces", "t.jsonl", "--distill-weight", "1"]
    args = cli.build_parser().parse_args(base + ["--distill-fade", "0.2,0.6"])
    assert cli.distill_fade(args) == (0.2, 0.6)
    assert "-fade0.2-0.6" in cli.run_tag(args)
    plain = cli.build_parser().parse_args(base)
    assert cli.distill_fade(plain) is None and "-fade" not in cli.run_tag(plain)
    for bad in ("0.6,0.2", "0.2", "-0.1,0.5", "0.2,1.5"):
        with pytest.raises(SystemExit):
            cli.distill_fade(cli.build_parser().parse_args(base + ["--distill-fade", bad]))
