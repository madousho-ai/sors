#!/usr/bin/env python
"""推理服务: 一份训好的存档, 照 TypeSafe 的 System One API 回答请求.

  PYTHONPATH=src .venv/bin/python scripts/serve.py --init runs/<run>/trained.safetensors
  curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
    "state": "Help! My payouts have been failing for 3 days.", "model": "<run>",
    "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'

官方 SDK 也能直接连: TYPESAFE_BASE_URL=http://127.0.0.1:8000, TYPESAFE_DEFAULT_MODEL=<服务名>.
服务名默认是存档所在的 run 目录名 (途中的档再接上 step-<步数>), --model-name 另起; 请求的 model 必须是它.
基模默认读 run 目录里 result.json 记的 --model; 存档不在 run 目录里时用 --base-model 给.
--demo 把仓库的 demos/ 挂在 /demo/ 下, 默认不挂. 贪吃蛇: 浏览器打开 http://127.0.0.1:8000/demo/snake/,
每一步由服务里的模型决定往哪走 (见 demos/snake/). 游戏局面用 --context-label "Game state" 更贴近训练数据.
端点、答案格式与请求怎么跑见 decidophobia.serve.
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib

from decidophobia.serve.engine import recorded_base_model

DEMOS = pathlib.Path(__file__).resolve().parents[1] / "demos"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="scripts/train.py 的存档 (.safetensors 或旧的 trained.pt), 必须是 context-first 训的")
    ap.add_argument("--base-model", default=None, help="存档训练时的基模. 不给 = run 目录里 result.json 记的那个")
    ap.add_argument("--model-name", default=None,
                    help="服务名: 响应里的 model, 也是请求的 model 必须写的值. 不给 = run 目录名 (途中的档接上 step-<步数>)")
    ap.add_argument("--context-label", default="State", help="提示里 state 前面的标签, 如 'Customer message'")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="state 加最长那道题的 token 上限, 超了 422. 训练时的提示最长 4096")
    ap.add_argument("--max-batch-tokens", type=int, default=16384,
                    help="一次前向的 KV cache 预算: 同组问题数 × (state + 组内最长的问题). 显存紧就调小")
    ap.add_argument("--api-key", default=None, help="给了就要求 Authorization: Bearer <key>. 不给 = 读 DECIDOPHOBIA_API_KEY, 也没有就不查")
    ap.add_argument("--demo", action="store_true", help="把 demos/ 下的演示页挂在 /demo/ (贪吃蛇在 /demo/snake/)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    return ap


def served_name(args) -> str:
    if args.model_name:
        return args.model_name
    p = pathlib.Path(args.init)
    return f"{p.parent.parent.name}-{p.stem}" if p.parent.name == "checkpoints" else p.parent.name


def base_model(args) -> str:
    found = args.base_model or recorded_base_model(args.init)
    if not found:
        raise SystemExit(f"{args.init} is not inside a run directory with a result.json; name its base model "
                         "with --base-model")
    return found


def api_key(args) -> str | None:
    return args.api_key or os.environ.get("DECIDOPHOBIA_API_KEY") or None


def demo_dir(args) -> pathlib.Path | None:
    return DEMOS if args.demo else None


def main() -> None:
    args = build_parser().parse_args()
    import torch
    import uvicorn

    from decidophobia.serve.app import create_app
    from decidophobia.serve.engine import load_engine

    name, base = served_name(args), base_model(args)
    engine = load_engine(args.init, base, context_label=args.context_label, device=args.device,
                         dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
                         max_tokens=args.max_tokens, max_batch_tokens=args.max_batch_tokens)
    released = datetime.date.fromtimestamp(pathlib.Path(args.init).stat().st_mtime).isoformat()
    app = create_app(engine, name, api_key=api_key(args), description=f"{pathlib.Path(args.init).name} on {base}",
                     release_date=released, demo_dir=demo_dir(args))
    print(f"serving {args.init} on {base} as {name!r}, type_marker={engine.type_marker}, "
          f"context label {args.context_label!r}, auth {'on' if api_key(args) else 'off'}", flush=True)
    if args.demo:
        print(f"demo: http://{args.host}:{args.port}/demo/snake/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
