"""--optimizer: adamw (默认, 旧 run 不变) 与 adamw8bit (bitsandbytes, 只在 CUDA 上). 直接 python tests/test_optimizer.py."""

import copy
import importlib.util
import pathlib
from dataclasses import asdict, replace

import torch

from _runner import run
from test_evaluate import _tiny
from test_resume import _Sampler, _training_config
from sors.training.loop import TrainConfig, train

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_cli_optimizer", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_the_default_optimizer_is_plain_adamw():
    assert TrainConfig().optimizer == "adamw"
    args = _mod.build_parser().parse_args([])
    assert args.optimizer == "adamw"
    assert "adam8" not in _mod.run_tag(args)


def test_the_8bit_optimizer_is_recorded_in_the_run_tag():
    args = _mod.build_parser().parse_args(["--optimizer", "adamw8bit"])
    assert "-adam8" in _mod.run_tag(args)


def test_an_unknown_optimizer_is_refused_before_training():
    cfg = replace(_training_config(), optimizer="sgd")
    tok, ids, m = _tiny()
    sampler = _Sampler()
    try:
        train(m, tok, ids, sampler, {}, cfg)
    except ValueError as e:
        assert "sgd" in str(e)
        assert sampler.calls == 0
        return
    raise AssertionError("unknown optimizer accepted")


def test_resuming_an_old_adamw_state_with_the_8bit_optimizer_is_refused():
    """旧存档的 config 里没有 optimizer 这一项, 它们全是 AdamW 训的; 换成 8-bit 接着跑要被拦下."""
    cfg = _training_config()
    old = {k: v for k, v in asdict(cfg).items() if k != "optimizer"}
    tok, ids, m = _tiny()
    sampler = _Sampler()
    try:
        train(m, tok, ids, sampler, {}, replace(cfg, optimizer="adamw8bit"), resume={"step": 2, "config": old})
    except ValueError as e:
        assert "optimizer" in str(e)
        assert sampler.calls == 0
        return
    raise AssertionError("old AdamW state resumed with adamw8bit")


def _moments(state: dict) -> list[torch.Tensor]:
    """bitsandbytes 的 state_dict 把量化状态包在 __bnb_optimizer_quant_state__ 里."""
    out = []
    for values in state["optimizer"]["state"].values():
        inner = values.get("__bnb_optimizer_quant_state__", values)
        out += [v for k, v in inner.items() if k in ("state1", "state2")]
    return out


def test_the_8bit_optimizer_keeps_uint8_moments_and_trains_on_cuda():
    if not torch.cuda.is_available():
        print("SKIP  needs CUDA")
        return
    cfg = replace(_training_config(), optimizer="adamw8bit")
    tok, ids, m = _tiny()
    m.cuda()
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    states = []
    train(m, tok, ids, _Sampler(), {}, cfg, on_state=lambda s: states.append(copy.deepcopy(s)))
    # 小模型里只有 4144 元素的 D 码行够 bitsandbytes 的 4096 门槛, LoRA 小张量留在 fp32
    assert any(v.dtype == torch.uint8 for v in _moments(states[-1])), "no 8-bit moments"
    after = dict(m.named_parameters())
    assert any(not torch.equal(before[n], after[n]) for n in before), "no parameter moved"


def test_8bit_resume_matches_uninterrupted_training_on_cuda():
    if not torch.cuda.is_available():
        print("SKIP  needs CUDA")
        return
    import pathlib as _p
    import tempfile
    from sors.training.resume import read_state, save_state

    cfg = replace(_training_config(), optimizer="adamw8bit")
    tok, ids, reference = _tiny()
    reference.cuda()
    train(reference, tok, ids, _Sampler(), {}, cfg)
    tok, ids, first = _tiny()
    first.cuda()
    with tempfile.TemporaryDirectory() as d:
        path = _p.Path(d) / "state.safetensors"
        train(first, tok, ids, _Sampler(), {}, cfg, stop_after=3, on_state=lambda s: save_state(s, path))
        saved = read_state(path)
    tok, ids, resumed = _tiny()
    resumed.cuda()
    train(resumed, tok, ids, _Sampler(), {}, cfg, resume=saved)
    for a, b in zip(reference.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    run(globals())
