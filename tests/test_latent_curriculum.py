"""隐式推理的课程 (Coconut 式, variant B): 带计算过程 (MenuExample.cot) 的题, 计算过程按句切成步;
第 s 阶段把前 s 句换成 s 个隐式位置, 剩下的句子仍是文字, 接在隐式位置后面, 对它们算下一个词的损失;
选项分支接在文字后面. 最后阶段 (s = k_max) 全换成隐式位置, 多出的句子并进最后一步 (不再出现).
不带计算过程的题一直用 k_max 个隐式位置, 与推理时一样.
"""

import random
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from test_candidate import candidate_batch, candidate_model
from test_decision import examples
from sors.core.menu import with_partners

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="DeltaNet kernels need CUDA")


def test_working_splits_into_sentences_and_lines():
    from sors.training.latent_curriculum import split_steps

    cot = 'The rule says "refunds take 5 days." So 3 + 2 = 5.\nThat holds! My answer is yes.'
    assert split_steps(cot) == ['The rule says "refunds take 5 days."', "So 3 + 2 = 5.", "That holds!",
                                "My answer is yes."]
    assert split_steps("  ") == []


def test_stage_climbs_from_zero_to_k_max_between_the_two_marks():
    from sors.training.latent_curriculum import latent_stage

    stages = [latent_stage(step, 100, 4, (0.2, 0.6)) for step in range(0, 101, 10)]
    assert stages == [0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4]


def test_curriculum_replaces_the_first_sentences_and_keeps_the_rest_as_text():
    from sors.training.latent_curriculum import curriculum_examples

    cot = "One. Two. Three."
    plain, worked = examples()[0], replace(examples()[1], cot=cot)
    at = lambda s: curriculum_examples([plain, worked], s, 4)
    assert [(e.latent_n, e.latent_text) for e in at(0)] == [(None, None), (0, "One. Two. Three.")]
    assert [(e.latent_n, e.latent_text) for e in at(1)] == [(None, None), (1, "Two. Three.")]
    assert [(e.latent_n, e.latent_text) for e in at(3)] == [(None, None), (3, None)]
    assert [(e.latent_n, e.latent_text) for e in at(4)] == [(None, None), (4, None)]
    long = replace(worked, cot="A. B. C. D. E. F.")
    assert (curriculum_examples([long], 4, 4)[0].latent_text, curriculum_examples([long], 4, 4)[0].latent_n) == (None, 4)
    assert curriculum_examples([long], 3, 4)[0].latent_text == "D. E. F."


def _curric(n_text):
    exs = examples()
    return [replace(exs[0], latent_n=n_text[0][0], latent_text=n_text[0][1]),
            replace(exs[1], latent_n=n_text[1][0], latent_text=n_text[1][1])]


def test_collate_adds_latent_counts_and_text_only_when_examples_carry_them():
    _, tok, d, _ = candidate_model(0)
    plain = candidate_batch(tok, d, examples())
    assert "latent_n" not in plain
    b = candidate_batch(tok, d, _curric([(1, "red context"), (None, None)]))
    assert b["latent_n"].tolist() == [1, -1]
    mask = b["latent_text_mask"]
    assert mask[0].sum() > 0 and mask[1].sum() == 0
    assert "red context" in tok.decode(b["latent_text_ids"][0][mask[0].bool()])
    for key, value in plain.items():
        assert torch.equal(value, b[key]), key


def _run(decoder, x):
    pos = torch.arange(x.shape[0], device=x.device)[None]
    return decoder(inputs_embeds=x[None], position_ids=pos, use_cache=False).last_hidden_state[0]


def _reference(m, data, default_n):
    """逐题完整编码: 隐式位置逐个算出, 接上文字 (取文字各 token 的下一个词损失), 再把每个分支单独接上去."""
    from sors.training.cot import _parts

    decoder, head = _parts(m)
    valid = data["slot_ids"] >= 0
    seqs, ce, count = {}, [], 0
    for r in sorted({r for r, _ in valid.nonzero().tolist()}):
        prefix = data["input_ids"][r][data["attention_mask"][r].bool()][:int(data["prefix_lengths"][r])]
        x = decoder.embed_tokens(prefix)
        n = int(data["latent_n"][r])
        for _ in range(default_n if n < 0 else n):
            x = torch.cat([x, _run(decoder, x)[-1:].to(x.dtype)])
        text = data["latent_text_ids"][r][data["latent_text_mask"][r].bool()]
        if len(text):
            start = len(x)
            x = torch.cat([x, decoder.embed_tokens(text)])
            h = _run(decoder, x)
            ce.append(F.cross_entropy(head(h[start - 1:-1]).float(), text, reduction="sum"))
            count += len(text)
        seqs[r] = x
    out = []
    for r, c in valid.nonzero().tolist():
        k = int(data["suffix_attention_mask"][r, c].sum())
        out.append(_run(decoder, torch.cat([seqs[r], decoder.embed_tokens(data["suffix_input_ids"][r, c, -k:])]))[-1])
    return torch.stack(out), (torch.stack(ce).sum() / count if ce else None)


def _setup(curric):
    m, tok, d, _ = candidate_model(0, "qwen35", trainable="full")
    m.to("cuda")
    exs = with_partners(_curric(curric), random.Random(0))
    data = {k: v.cuda() for k, v in candidate_batch(tok, d, exs, option_branches=False).items()}
    return m, data


def _exact(monkeypatch):
    import fla.ops.gated_delta_rule as fla_gdn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35
    from exact_gdn import exact_gated_delta_rule

    monkeypatch.setattr(fla_gdn, "chunk_gated_delta_rule", exact_gated_delta_rule)
    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", exact_gated_delta_rule)


CASES = [[(1, "red context context"), (None, None)], [(0, "context red"), (2, None)], [(3, None), (None, None)]]


@cuda
@pytest.mark.parametrize("curric", CASES)
def test_per_example_latent_counts_and_text_match_encoding_each_example_on_its_own(curric, monkeypatch):
    from sors.core.latent import latent_forward
    from sors.training.cot import _parts

    _exact(monkeypatch)
    m, data = _setup(curric)
    m.double().eval()
    valid = data["slot_ids"] >= 0
    with torch.no_grad():
        want_h, want_lm = _reference(m, data, 2)
        got_h, lm = latent_forward(m.decoder, data, valid, 2, head=_parts(m)[1])
    torch.testing.assert_close(got_h, want_h, atol=1e-9, rtol=1e-9)
    if any(t for _, t in curric):
        torch.testing.assert_close(lm[0] / lm[1], want_lm, atol=1e-9, rtol=1e-9)
    else:
        assert lm is None


@cuda
def test_latent_curriculum_gradients_match_the_reference(monkeypatch):
    from sors.core.latent import latent_forward
    from sors.training.cot import _parts

    _exact(monkeypatch)
    m, data = _setup(CASES[0])
    m.double().train()
    valid = data["slot_ids"] >= 0
    grads = []
    for packed in (False, True):
        m.zero_grad(set_to_none=True)
        if packed:
            h, (total, count) = latent_forward(m.decoder, data, valid, 2, head=_parts(m)[1])
            lm = total / count
        else:
            h, lm = _reference(m, data, 2)
        torch.manual_seed(5)
        ((h * torch.randn_like(h)).sum() + lm).backward()
        grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters()
                      if p.grad is not None and n.startswith("base.")})
    assert grads[0].keys() == grads[1].keys()
    for name, g in grads[0].items():
        assert float((grads[1][name] - g).norm() / g.norm()) < 1e-5, name


def _latent_model(n):
    from test_decision import tokenizer, tiny_backbone
    from sors.core.decision import DecisionConfig
    from sors.core.model import prepare_model

    tok, d, ids = tokenizer()
    cfg = DecisionConfig(kind="candidate", blocks=0, dim=16, heads=4, option_batch_size=2, latent_steps=n)
    return prepare_model(tiny_backbone(tok, "qwen35"), ids, 4, 8, 0., trainable="full", decision=cfg), tok, d


@cuda
def test_a_training_group_adds_the_weighted_text_loss_and_reports_it():
    from sors.training.loop import TrainConfig, backward_groups

    m, tok, d = _latent_model(2)
    m.to("cuda").train()
    exs = _curric(CASES[0])
    b = {k: v.cuda() for k, v in candidate_batch(tok, d, exs, option_branches=False).items()}
    out = []
    for weight in (0.0, 1.0):
        m.zero_grad(set_to_none=True)
        cfg = TrainConfig(batch_size=2, micro_batches=1, accumulate_gradients=True, candidate_prefix_cache="packed",
                          latent_lm_weight=weight)
        stats = {}
        backward_groups(m, cfg, exs, b, d, stats=stats)
        g = torch.cat([p.grad.flatten() for n, p in m.named_parameters() if n.startswith("base.") and p.grad is not None])
        out.append((stats.get("latent_lm"), g.clone()))
    assert out[0][0] is not None and out[0][0] == pytest.approx(out[1][0], rel=1e-4)
    assert not torch.equal(out[0][1], out[1][1])


def test_cli_reads_the_latent_curriculum(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("train_cli", Path(__file__).parents[1] / "scripts/train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = cli.build_parser().parse_args(["--architecture", "candidate", "--decision-blocks", "0",
                                          "--decision-latent-steps", "4", "--latent-curriculum", "0.1,0.6",
                                          "--latent-lm-weight", "0.5"])
    assert cli.latent_curriculum(args) == (0.1, 0.6)
    with pytest.raises(SystemExit):
        cli.latent_curriculum(cli.build_parser().parse_args(["--latent-curriculum", "0.1,0.6"]))
    with pytest.raises(SystemExit):
        cli.latent_curriculum(cli.build_parser().parse_args(
            ["--architecture", "candidate", "--decision-latent-steps", "4", "--latent-curriculum", "0.7,0.6"]))
