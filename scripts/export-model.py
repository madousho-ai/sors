#!/usr/bin/env python
"""Export one complete Sors model directory for offline distribution and serving."""

import argparse
import json
from pathlib import Path

import torch

from sors.core.attention import add_attention_arguments
from sors.core.checkpoint import read_checkpoint
from sors.core.pretrained import save_pretrained
from sors.serve.engine import load_engine, recorded_base_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, type=Path, help="训练存档或已有完整模型目录")
    parser.add_argument("--base-model", help="旧训练存档的基模；默认读取 run 的 result.json")
    parser.add_argument("--out", required=True, type=Path, help="新的完整模型目录，必须尚未存在")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16",
                        help="旧存档加载基模时的精度；完整模型目录保留已保存的混合精度")
    parser.add_argument("--local-files-only", action="store_true")
    add_attention_arguments(parser)
    parser.set_defaults(attn_implementation="sdpa", allow_kernel_download=False)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"export directory already exists: {args.out}")
    if args.init.is_dir():
        if args.base_model is not None:
            parser.error("a complete model already includes its weights; omit --base-model")
        base = None
        config = json.loads((args.init / "config.json").read_text())["training_config"]
    else:
        base = args.base_model or recorded_base_model(args.init)
        if base is None:
            parser.error("the training checkpoint needs --base-model or a run result.json")
        config = read_checkpoint(args.init)["config"]
    engine = load_engine(args.init, base, device=args.device, dtype=getattr(torch, args.dtype),
                         attn_implementation=args.attn_implementation,
                         allow_kernel_download=args.allow_kernel_download, local_files_only=args.local_files_only)
    save_pretrained(engine.lm, engine.tok, config, args.out)
    size = (args.out / "model.safetensors").stat().st_size
    print(f"exported complete model: {args.out} ({size:,} weight bytes)", flush=True)


if __name__ == "__main__":
    main()
