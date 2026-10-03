#!/usr/bin/env python
"""推理服务: 完整模型目录或训练存档, 照 TypeSafe 的 System One API 回答请求.

  PYTHONPATH=src .venv/bin/python scripts/serve.py --init /path/to/Sors-0.8B
  PYTHONPATH=src .venv/bin/python scripts/serve.py --init runs/<run>/trained.safetensors
  curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
    "state": "Help! My payouts have been failing for 3 days.", "model": "<run>",
    "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'

官方 SDK 也能直接连: TYPESAFE_BASE_URL=http://127.0.0.1:8000, TYPESAFE_DEFAULT_MODEL=<服务名>.
服务名默认是存档所在的 run 目录名 (途中的档再接上 step-<步数>), --model-name 另起; 请求的 model 必须是它.
训练存档的基模默认读 run 目录里的 result.json，也可用 --base-model 指定；完整目录自带权重.
--demo 把仓库的 demos/ 挂在 /demo/ 下, 默认不挂. 手填请求: http://127.0.0.1:8000/demo/playground/,
左边表单右边 JSON, 发送后显示每项概率与模型读到的提示 (见 demos/playground/). 贪吃蛇: /demo/snake/,
每一步由服务里的模型决定往哪走 (见 demos/snake/).
训练时每条提示都以「<标签>: <内容>」开头 (Customer message / Passage / Game state ...). 服务默认不加标签, state
原样进提示, 要标签就写在 state 开头, 如 "Game state: ..."; 贪吃蛇页面就是这样写的. --context-label 给了才替每个
请求加上, 如评估对账时用 --context-label "Customer message".
端点、答案格式与请求怎么跑见 sors.serve.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import pathlib

from sors.serve.engine import recorded_base_model

DEMOS = pathlib.Path(__file__).resolve().parents[1] / "demos"


def build_parser() -> argparse.ArgumentParser:
    from sors.core.attention import add_attention_arguments

    ap = argparse.ArgumentParser(description="SORS — State-conditioned Option Ranking System")
    ap.add_argument("--init", required=True, help="完整模型目录，或 scripts/train.py 的存档；必须是 context-first 训的")
    ap.add_argument("--base-model", default=None, help="旧训练存档的基模，默认读 result.json；完整模型目录省略此项")
    ap.add_argument("--model-name", default=None,
                    help="服务名: 响应里的 model, 也是请求的 model 必须写的值. 不给 = run 目录名 (途中的档接上 step-<步数>)")
    ap.add_argument("--context-label", default="",
                    help="给了就在每个请求的 state 前面加「<标签>: 」, 如 'Customer message'. 不给 = 不加, state 原样进提示")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="state 加最长那道题的 token 上限, 超了 422. 训练时的提示最长 4096")
    ap.add_argument("--max-batch-tokens", type=int, default=16384,
                     help="一次前向的 KV cache 预算: 同组问题数 × (state + 组内最长的问题). 显存紧就调小")
    ap.add_argument("--candidate-prefix-cache", choices=["auto", "on", "off"], default=None,
                    help="覆盖 candidate 存档的共享前缀执行方式；默认继承存档，旧存档保持 off")
    ap.add_argument("--api-key", default=None, help="给了就要求 Authorization: Bearer <key>. 不给 = 读 SORS_API_KEY, 也没有就不查")
    ap.add_argument("--demo", action="store_true", help="把 demos/ 下的演示页挂在 /demo/ (手填请求在 /demo/playground/, 贪吃蛇在 /demo/snake/)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    add_attention_arguments(ap)
    ap.set_defaults(attn_implementation="sdpa", allow_kernel_download=False)
    ap.add_argument("--local-files-only", action="store_true", help="基模和 tokenizer 只从本地加载")
    ap.add_argument("--warmup", action="store_true", help="监听 HTTP 前执行真实推理并检查输出概率")
    return ap


def served_name(args) -> str:
    if args.model_name:
        return args.model_name
    p = pathlib.Path(args.init)
    if p.is_dir():
        return p.name
    return f"{p.parent.parent.name}-{p.stem}" if p.parent.name == "checkpoints" else p.parent.name


def base_model(args) -> str | None:
    if pathlib.Path(args.init).is_dir():
        if args.base_model is not None:
            raise SystemExit("a complete model directory includes its weights; omit --base-model")
        return None
    found = args.base_model or recorded_base_model(args.init)
    if not found:
        raise SystemExit(f"{args.init} is not inside a run directory with a result.json; name its base model "
                         "with --base-model")
    return found


def api_key(args) -> str | None:
    # Keep authentication enabled for deployments using the pre-rebrand variable.
    return args.api_key or os.environ.get("SORS_API_KEY") or os.environ.get("DECIDOPHOBIA_API_KEY") or None


def demo_dir(args) -> pathlib.Path | None:
    return DEMOS if args.demo else None


def warmup(engine) -> None:
    """Exercise the serving path, including cache branching, before readiness."""
    from sors.serve.api import Choice, Noul, Score

    questions = {
        "color": Choice(type="choice", instructions="Which color?", criteria={"red": None, "blue": None}),
        "mentioned": Noul(type="noul", instructions="Does the message mention a red object?"),
        "score": Score(type="score", instructions="How much detail does the message contain?",
                       criteria=["little", "some", "much"]),
    }
    for repetitions in (1, 16):
        result = engine.evaluate("red blue context. " * repetitions, questions)
        for name, expected_count in (("color", 2), ("mentioned", 2), ("score", 3)):
            probabilities = result.probs[name]
            if (len(probabilities) != expected_count
                    or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                    or abs(sum(probabilities) - 1) > 1e-4):
                raise ValueError(f"warmup returned invalid probabilities for {name}: {probabilities}")
    print("warmup: real serving requests passed", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    import torch
    import uvicorn

    from sors.serve.app import create_app
    from sors.serve.engine import load_engine

    name, base = served_name(args), base_model(args)
    engine = load_engine(args.init, base, context_label=args.context_label, device=args.device,
                         dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
                          max_tokens=args.max_tokens, max_batch_tokens=args.max_batch_tokens,
                          candidate_prefix_cache=args.candidate_prefix_cache,
                          attn_implementation=args.attn_implementation,
                          allow_kernel_download=args.allow_kernel_download,
                          local_files_only=args.local_files_only)
    if args.warmup:
        warmup(engine)
    released = datetime.date.fromtimestamp(pathlib.Path(args.init).stat().st_mtime).isoformat()
    source = base or "self-contained model weights"
    app = create_app(engine, name, api_key=api_key(args), description=f"{pathlib.Path(args.init).name} on {source}",
                     release_date=released, demo_dir=demo_dir(args))
    print(f"serving {args.init} on {source} as {name!r}, type_marker={engine.type_marker}, "
          f"context_marker={engine.context_marker}, "
          f"context label {repr(args.context_label) if args.context_label else 'none'}, auth {'on' if api_key(args) else 'off'}", flush=True)
    if args.demo:
        for page in ("playground", "snake"):
            print(f"demo: http://{args.host}:{args.port}/demo/{page}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
