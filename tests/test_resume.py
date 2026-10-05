"""续跑与逐组反传的行为测试；在服务器的隔离验证目录执行。"""

import copy
import json
import importlib.util
import pathlib
import random
import tempfile
from dataclasses import replace

import torch

from _runner import run
from test_evaluate import _tiny, _uneven_questions, _Writer
from sors.core.batch import collate
from sors.core.menu import MenuExample, with_partners
from sors.core.model import last_logits
from sors.training.loop import TrainConfig, step_loss, train


def _examples():
    exs = _uneven_questions()
    exs.append(MenuExample(query="unknown", options=[0, 1, 2], gold_idx=0, label=0,
                           option_names=exs[0].option_names, target=[0.5, 0.3, 0.2]))
    exs[1] = replace(exs[1], partner_query="The customer's bank card went missing.")
    return with_partners(exs, random.Random(2))


def test_accumulation_matches_whole_batch_gradients_with_unequal_groups_and_soft_labels():
    """抓住按组数均分 loss、拆散 JS 对、每组清梯度、遗漏某组的错误。"""
    from sors.training.loop import backward_groups

    tok, ids, m = _tiny()
    exs = _examples()  # 5 对，分 3 组 => 2、2、1 对
    cfg = TrainConfig(k_max=3, loss="vocab", consistency=0.8, micro_batches=3, label_smoothing=0.1)
    b = collate(exs, tok, ids, 3)
    m.zero_grad(set_to_none=True)
    logits = last_logits(m, b["input_ids"], b["attention_mask"])
    loss, ce, js = step_loss(cfg, exs, b, logits, ids)
    loss.backward()
    expected = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    expected_ce, expected_js = ce.item(), js.item()
    del logits, loss, ce, js
    m.zero_grad(set_to_none=True)
    got_ce, got_js, _, _ = backward_groups(m, cfg, exs, b, ids)
    assert abs(got_ce - expected_ce) < 2e-5
    assert abs(got_js - expected_js) < 2e-6
    for k, p in m.named_parameters():
        if k in expected:
            torch.testing.assert_close(p.grad, expected[k], rtol=2e-4, atol=2e-5)


def test_token_budget_groups_keep_pairs_together_and_match_whole_batch_gradients():
    """抓住按 token 预算分组时拆散 JS 对或漏掉某行的实现。"""
    from sors.core.batch import token_groups
    from sors.training.loop import backward_groups

    tok, ids, m = _tiny()
    exs = _examples()
    b = collate(exs, tok, ids, 3)
    lengths = b["attention_mask"].sum(1).tolist()
    budget = 4 * max(lengths)
    cfg = TrainConfig(k_max=3, loss="vocab", consistency=0.8, micro_tokens=budget, label_smoothing=0.1)
    m.zero_grad(set_to_none=True)
    logits = last_logits(m, b["input_ids"], b["attention_mask"])
    loss, ce, js = step_loss(cfg, exs, b, logits, ids)
    loss.backward()
    expected = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    expected_ce, expected_js = ce.item(), js.item()
    del logits, loss, ce, js
    groups = token_groups(lengths, budget, unit=2)
    assert 1 < len(groups) < len(exs) // 2, groups
    seen = []
    h = m.register_forward_pre_hook(lambda mod, args, kwargs: seen.append(kwargs["input_ids"].shape[0]),
                                    with_kwargs=True)
    m.zero_grad(set_to_none=True)
    got_ce, got_js, _, _ = backward_groups(m, cfg, exs, b, ids, lengths)
    h.remove()
    assert seen == [len(g) for g in groups], (seen, groups)
    assert abs(got_ce - expected_ce) < 2e-5
    assert abs(got_js - expected_js) < 2e-6
    for k, p in m.named_parameters():
        if k in expected:
            torch.testing.assert_close(p.grad, expected[k], rtol=2e-4, atol=2e-5)


def test_each_group_finishes_backward_before_the_next_forward():
    """抓住重新退回整批保留图的实现。"""
    from sors.training.loop import backward_groups

    tok, ids, m = _tiny()
    exs = _examples()
    events = []
    h1 = m.register_forward_pre_hook(lambda *args: events.append("forward"))
    h2 = m.get_input_embeddings().rows.register_hook(lambda grad: events.append("backward"))
    cfg = TrainConfig(k_max=3, loss="vocab", consistency=1, micro_batches=3)
    backward_groups(m, cfg, exs, collate(exs, tok, ids, 3), ids)
    h1.remove()
    h2.remove()
    assert events == ["forward", "backward"] * 3, events


def test_state_file_roundtrips_tensors_rng_tuples_and_integer_optimizer_keys():
    """抓住 RNG tuple 或 optimizer 的整数 key 在 JSON 往返时被破坏。"""
    from safetensors import safe_open
    from sors.training.resume import read_state, save_state

    rng = random.Random(4)
    value = {"step": 1500, "rng": rng.getstate(), "optimizer": {0: {"exp_avg": torch.arange(3.)}},
             "master": [torch.tensor([1.0001], dtype=torch.float32)], "optional": None}
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "state.safetensors"
        save_state(value, path)
        with safe_open(str(path), framework="pt") as f:
            assert f.metadata()["format"] == "sors-training-state-v1"
        got = read_state(path)
    assert got["step"] == 1500 and got["optional"] is None
    assert got["rng"] == value["rng"] and set(got["optimizer"]) == {0}
    torch.testing.assert_close(got["master"][0], value["master"][0], rtol=0, atol=0)
    rng.setstate(got["rng"])
    assert rng.random() == random.Random(4).random()


def test_training_state_reads_sors_and_legacy_files_and_rejects_unknown_formats():
    from safetensors.torch import save_file
    from sors.training.resume import read_state

    structure = {"type": "dict", "value": [["step", 7], ["master", {"type": "tensor", "value": "weights"}]]}
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "state.safetensors"
        for kind in ("sors-training-state-v1", "decidophobia-training-state-v1"):
            save_file({"weights": torch.tensor([1.25])}, str(path),
                      metadata={"format": kind, "structure": json.dumps(structure)})
            got = read_state(path)
            assert got["step"] == 7
            torch.testing.assert_close(got["master"], torch.tensor([1.25]), atol=0, rtol=0)
        save_file({}, str(path), metadata={"format": "other-training-state-v1", "structure": "{}"})
        try:
            read_state(path)
        except ValueError:
            return
        raise AssertionError("unknown training-state format was accepted")


class _Sampler:
    """有队列和历史计数的实际 callable，用于检查重放覆盖整个采样调用。"""
    def __init__(self):
        self.queue, self.calls, self.seen = [], 0, []

    def __call__(self, n, rng):
        self.calls += 1
        out = []
        for _ in range(n):
            if not self.queue:
                self.queue = rng.sample(list(range(7)), 7)
            i = self.queue.pop()
            out.append(MenuExample(query=f"message {i}: " + "hi " * rng.randrange(1, 5),
                                   options=[0, 1, 2], gold_idx=i % 3, label=i % 3,
                                   option_names=["first", "second", "third"]))
        rng.shuffle(out)
        out = with_partners(out, rng)
        self.seen.append(out)
        return out


def _training_config():
    return TrainConfig(steps=4, batch_size=2, k_max=3, loss="vocab", consistency=0.7,
                       micro_batches=2, accumulate_gradients=True, lr_schedule="cosine", warmup_steps=1,
                       eval_every=2, log_every=2, save_every=2)


def test_full_resume_matches_uninterrupted_training_samples_parameters_and_optimizer():
    """抓住 sampler、Adam、LR、torch RNG、统计窗口恢复遗漏；停在非评估边界的第3步。"""
    from sors.training.resume import read_state, save_state

    cfg = _training_config()
    tok, ids, reference = _tiny()
    # 在真实 embedding 输出上使用 dropout，使 torch RNG 恢复错误可被观测。
    hook = lambda mod, args, out: torch.nn.functional.dropout(out, p=0.15, training=mod.training)
    reference.get_input_embeddings().register_forward_hook(hook)
    ref_sampler, ref_states = _Sampler(), []
    train(reference, tok, ids, ref_sampler, {}, cfg, on_state=lambda s: ref_states.append(copy.deepcopy(s)))
    tok, ids, first = _tiny()
    first.get_input_embeddings().register_forward_hook(hook)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "state.safetensors"
        train(first, tok, ids, _Sampler(), {}, cfg, stop_after=3, on_state=lambda s: save_state(s, path))
        saved = read_state(path)
        assert saved["step"] == 3
        torch.rand(9)
        tok, ids, resumed = _tiny()
        resumed.get_input_embeddings().register_forward_hook(hook)
        sampler, states, writer = _Sampler(), [], _Writer()
        hist = train(resumed, tok, ids, sampler, {}, cfg, resume=saved, writer=writer,
                     on_state=lambda s: states.append(copy.deepcopy(s)))
    assert sampler.seen == ref_sampler.seen
    assert [h["step"] for h in hist] == [0, 2, 4]
    assert [s for t, v, s in writer.scalars if t == "train/loss"] == [4]
    assert hist[-1]["train_accuracy_n"] == ref_states[-1]["history"][-1]["train_accuracy_n"]
    for a, b in zip(reference.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(ref_states[-1]["master"], states[-1]["master"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert ref_states[-1]["scheduler"] == states[-1]["scheduler"]
    for key, values in ref_states[-1]["optimizer"]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, states[-1]["optimizer"]["state"][key][name], rtol=0, atol=0)


def test_legacy_resume_starts_at_next_step_with_original_schedule_and_sampling_position():
    """抓住旧权重恢复时重新 warmup、从第一批抽题、重写 step0 的错误。"""
    cfg = _training_config()
    tok, ids, m = _tiny()
    sampler, states = _Sampler(), []
    history = [{"step": 0, "t": 0}, {"step": 2, "t": 12}]
    writer = _Writer()
    hist = train(m, tok, ids, sampler, {}, cfg, resume={"step": 2, "history": history}, writer=writer,
                 on_state=lambda s: states.append(copy.deepcopy(s)))
    oracle, rng = _Sampler(), random.Random(cfg.seed)
    for _ in range(4):
        oracle(cfg.batch_size, rng)
    assert sampler.seen == oracle.seen
    assert [h["step"] for h in hist] == [0, 2, 4]
    # 原日程在第2步之后的LR为0.75*基础值；第一次更新会用它。
    assert states[-1]["resume_info"]["lr_at_resume"][0] == cfg.lr_lora * 0.75
    assert states[-1]["resume_info"]["optimizer_reset_at"] == 2
    assert {s for t, v, s in writer.scalars if t.startswith("train/")} == {4}
    assert all(int(v["step"]) == 2 for v in states[-1]["optimizer"]["state"].values())


def test_tensorboard_resume_keeps_checkpoint_step_and_hides_old_future_events():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter
    from sors.training.resume import resume_writer, rollback_history

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        with SummaryWriter(str(out / "tb")) as w:
            w.add_scalar("train/loss", 1, 1500)
            w.add_scalar("train/loss", 99, 1520)
            w.add_scalar("train/loss", 98, 1580)
        original = '\n'.join(json.dumps({"step": s}) for s in (0, 1500, 1750)) + '\n'
        (out / "log.jsonl").write_text(original)
        history = rollback_history(out, 1500)
        assert [h["step"] for h in history] == [0, 1500]
        assert any(p.read_text() == original for p in out.glob("log.jsonl.before-resume-*"))
        with resume_writer(out, 1500) as w:
            w.add_scalar("train/loss", 2, 1520)
        ea = EventAccumulator(str(out / "tb"))
        ea.Reload()
        assert [(e.step, e.value) for e in ea.Scalars("train/loss")] == [(1500, 1), (1520, 2)]


def test_full_resume_preserves_fp32_master_low_bits_for_bfloat16_parameters():
    cfg = _training_config()
    tok, ids, uninterrupted = _tiny()
    uninterrupted.to(torch.bfloat16)
    train(uninterrupted, tok, ids, _Sampler(), {}, cfg)
    tok, ids, first = _tiny()
    first.to(torch.bfloat16)
    states = []
    train(first, tok, ids, _Sampler(), {}, cfg, stop_after=2, on_state=lambda s: states.append(copy.deepcopy(s)))
    saved = states[-1]
    assert any(not torch.equal(w, w.bfloat16().float()) for w in saved["master"])
    tok, ids, resumed = _tiny()
    resumed.to(torch.bfloat16)
    train(resumed, tok, ids, _Sampler(), {}, cfg, resume=saved)
    for a, b in zip(uninterrupted.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_resume_config_change_is_refused_before_sampling():
    cfg = _training_config()
    tok, ids, m = _tiny()
    sampler = _Sampler()
    from dataclasses import asdict
    saved = {"step": 2, "config": {**asdict(cfg), "batch_size": 99}}
    try:
        train(m, tok, ids, sampler, {}, cfg, resume=saved)
    except ValueError:
        assert sampler.calls == 0
        return
    raise AssertionError("changed batch_size accepted")


def test_resume_cli_reads_original_args_and_requires_explicit_legacy_reset_consent():
    from torch.utils.tensorboard import SummaryWriter
    spec = importlib.util.spec_from_file_location("resume_cli", pathlib.Path(__file__).parents[1] / "scripts/resume.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with tempfile.TemporaryDirectory() as d:
        original = {"model": "Qwen/Qwen3-1.7B-Base", "batch_size": 36, "seed": 0}
        with SummaryWriter(str(pathlib.Path(d) / "tb")) as writer:
            writer.add_text("args", json.dumps(original), 0)
        assert cli.original_args(pathlib.Path(d)) == original
        assert cli.legacy_step(pathlib.Path("step-01500.safetensors"), True) == 1500
        try:
            cli.legacy_step(pathlib.Path("step-01500.safetensors"), False)
        except ValueError:
            pass
        else:
            raise AssertionError("optimizer reset accepted without consent")


def test_rollback_archives_and_drops_a_partial_final_jsonl_record():
    from sors.training.resume import rollback_history
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        original = '{"step": 1500}\n{"step": 1750, "train_loss":'
        (out / "log.jsonl").write_text(original)
        assert rollback_history(out, 1500) == [{"step": 1500}]
        assert json.loads((out / "log.jsonl").read_text()) == {"step": 1500}
        assert any(p.read_text() == original for p in out.glob("log.jsonl.before-resume-*"))


def test_completed_native_state_can_finish_export_after_interruption():
    spec = importlib.util.spec_from_file_location("resume_cli", pathlib.Path(__file__).parents[1] / "scripts/resume.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.validate_resume_step(2000, 2000, full=True)
    for step, full in ((2001, True), (2000, False), (-1, True)):
        try:
            cli.validate_resume_step(step, 2000, full=full)
        except ValueError:
            continue
        raise AssertionError("invalid step accepted")


def test_sampling_fingerprint_tracks_the_v51_dataset():
    from sors.training.resume import sampling_fingerprint

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for name in ("scripts/train.py", "src/sors/data/synth_v5.py", "src/sors/core/menu.py",
                     "src/sors/core/prompt.py", "src/sors/serve/menus.py",
                     "src/sors/data/paths.py", "src/sors/training/resume.py",
                     "assets/synth-intents-v5.3/schema.py"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# source fixture\n")
        data = root / "assets/synth-intents-v5.3/telecom.contexts.json"
        data.write_text('{"domain":"telecom","contexts":[]}')
        before = sampling_fingerprint(root, root / "assets")
        data.write_text('{"domain":"telecom","contexts":[{"id":"telecom_changed"}]}')
        assert sampling_fingerprint(root, root / "assets") != before, "dataset edits must invalidate an old sampling fingerprint"


if __name__ == "__main__":
    run(globals())
