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


def test_d_code_rows_and_decision_layers_are_the_full_precision_params():
    from sors.training.loop import full_precision_params

    tok, ids, m = _tiny()
    rows = [n for n, p in m.named_parameters() if p.requires_grad and n.endswith(".rows")]
    assert rows and [id(p) for p in full_precision_params(m)] == [id(dict(m.named_parameters())[n]) for n in rows]

    class Decision(torch.nn.Module):
        decision_config = object()

        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(4, 4)
            self.base.rows = torch.nn.Parameter(torch.zeros(2, 4))
            self.blocks = torch.nn.Linear(4, 4)
            self.frozen = torch.nn.Linear(4, 4).requires_grad_(False)

    d = Decision()
    assert {id(p) for p in full_precision_params(d)} == {id(d.base.rows), id(d.blocks.weight), id(d.blocks.bias)}


def test_the_8bit_optimizer_keeps_listed_params_in_32_bit_and_the_rest_in_8_bit():
    if not torch.cuda.is_available():
        print("SKIP  needs CUDA")
        return
    import bitsandbytes as bnb
    from sors.training.loop import make_optimizer

    body = torch.nn.Parameter(torch.randn(8192, device="cuda"))
    rows = torch.nn.Parameter(torch.randn(8192, device="cuda"))
    opt = make_optimizer("adamw8bit", [{"params": [body], "lr": 1e-3}, {"params": [rows], "lr": 1e-3}], 0.0,
                         full_precision=[rows])
    body.grad, rows.grad = torch.randn_like(body), torch.randn_like(rows)
    opt.step()
    assert opt.state[body]["state1"].dtype == torch.uint8
    assert opt.state[rows]["state1"].dtype == torch.float32
    assert opt.state[rows]["state2"].dtype == torch.float32
    # 覆盖只记在这个优化器上, 进程级的单例不留痕迹, 以后别的参数拿到同一个 id 也不受影响
    assert id(rows) not in bnb.optim.GlobalOptimManager.get_instance().pid2config


def test_32_bit_d_code_rows_hold_in_a_bf16_model():
    """bf16 模型里 D 码行也是 bf16 (rows 组不经过 Fp32Master, 优化器直接拿模型上那份), 照样 32 位动量."""
    if not torch.cuda.is_available():
        print("SKIP  needs CUDA")
        return
    cfg = replace(_training_config(), optimizer="adamw8bit")
    tok, ids, m = _tiny()
    m.to(device="cuda", dtype=torch.bfloat16)
    states = []
    train(m, tok, ids, _Sampler(), {}, cfg, on_state=lambda s: states.append(copy.deepcopy(s)))
    moments = _moments(states[-1])
    assert moments and all(v.dtype == torch.float32 for v in moments), "bf16 D-code rows were quantised"


def test_the_8bit_optimizer_trains_with_32_bit_d_code_rows_on_cuda():
    if not torch.cuda.is_available():
        print("SKIP  needs CUDA")
        return
    cfg = replace(_training_config(), optimizer="adamw8bit")
    tok, ids, m = _tiny()
    m.cuda()
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    states = []
    train(m, tok, ids, _Sampler(), {}, cfg, on_state=lambda s: states.append(copy.deepcopy(s)))
    # 小模型里够 bitsandbytes 4096 门槛的只有 D 码行 (4144 元素); 它要留在 32 位, 于是不该剩任何 uint8
    moments = _moments(states[-1])
    assert moments and all(v.dtype == torch.float32 for v in moments), "D-code rows were quantised"
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
