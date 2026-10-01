# 训练 attention 后端

`scripts/train.py` 和 `scripts/resume.py` 共用按 GPU 架构选择 attention 的模型加载器。
启动日志打印请求的后端、实际加载的实现、GPU 架构和回退原因；训练参数、TensorBoard
和续跑记录保存实际选择。训练目标、微批划分、优化器与采样算法保持原设置。

## 自动选择

| 显卡架构 | `auto` 候选顺序 |
|---|---|
| Ampere / Ada：SM80、86、87、89，包括 A100、3090、3070 Ti | FA2 → SDPA |
| Hopper：SM90，包括 H100/H800 | FA3 → FA2 → SDPA |
| Blackwell：SM100，包括 B100/B200 | FA4 → FA2 → SDPA |
| Blackwell：SM120，包括 RTX 5090 | 支持 SM120 的本地 FA4 → FA2 → SDPA |
| CPU、其他架构、FP32 | SDPA |

FlashAttention 使用 FP16/BF16。FA4 本地包要求 `flash-attn-4>=4.0.0b33`；
Blackwell 的本地 FA2 要求 `flash-attn>=2.8.3.post1`，并需要匹配该显卡的二进制构建。
当前 Hub FA4 v0 的架构范围为 SM9x/10x/11x；SM120 跳过这条 Hub 路径。
非零 attention dropout 使用支持该操作的 FA2 或 SDPA。head dimension 的边界同样会检查。

自动选择会先实际导入可选扩展。扩展缺失或导入失败时记录原因并尝试下一候选。
显式指定的 FlashAttention 后端遇到不兼容硬件或缺少依赖时直接报错。
模型加载、OOM 和训练过程中的 CUDA 错误会保留并向调用者报告。

## 使用

已安装本地 FlashAttention 包时，默认自动选择：

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset synth-v5.1 --grad-ckpt
```

使用预编译 Hub 内核需要显式授权下载；该标志同时允许读取已有 Hub 缓存。
在 Transformers 5.17.0 环境中，可选依赖的兼容范围为 `kernels>=0.16,<0.17`：

```bash
# 依赖安装由操作者执行；训练脚本自身只负责加载。
uv pip install --python .venv/bin/python 'kernels>=0.16,<0.17'

HF_HUB_OFFLINE=0 PYTHONPATH=src .venv/bin/python scripts/train.py \
  --dataset synth-v5.1 --grad-ckpt \
  --attn-implementation auto --allow-kernel-download
```

手动选择支持 `sdpa`、`eager`、`flash_attention_2`、`flash_attention_3`、`flash_attention_4`。
例如在 A100 上强制验证 FA2：

```bash
HF_HUB_OFFLINE=0 PYTHONPATH=src .venv/bin/python scripts/train.py \
  --dataset synth-v5.1 --grad-ckpt \
  --attn-implementation flash_attention_2 --allow-kernel-download
```

Hub 仓库限于 Transformers 的官方 `kernels-community` attention 实现。
kernel major version 采用当前 Transformers 的固定映射，5.17.0 分别为 FA2 v3、FA3 v1、FA4 v0。
以上预编译路径显式允许 Hub 联网。全局离线模式继续受 `HF_HUB_OFFLINE` 控制，
此时 Hub 缓存须满足依赖库的离线检查；加载失败会按显式请求报错或按 auto 回退。
`--no-allow-kernel-download` 可撤销续跑参数里保存的 Hub 下载许可。

## 续跑

续跑默认沿用存档记录的实际后端；早期缺少 attention 记录的档使用 SDPA。
在同一份经校验的代码和数据上，可显式重新选择：

```bash
PYTHONPATH=src .venv/bin/python scripts/resume.py --run runs/<run> \
  --attn-implementation auto --allow-kernel-download
```

原采样指纹与环境版本校验继续生效，attention 开关保留这些保护。
指纹包含 `scripts/train.py`；脚本版本变化会触发原有的拒绝恢复规则。
旧 run 的完整恢复应使用它对应的代码快照。后端切换可能产生浮点舍入差异，
采样位置、优化器状态、学习率日程和 TensorBoard 步数仍按完整状态恢复。

## 验证

```bash
# 硬件选择、缺依赖、下载授权和 CLI 参数测试
PYTHONPATH=src .venv/bin/python tests/test_attention.py

# 在训练服务器运行真实 CUDA 对照，要求显式 FA2 成功加载
HF_HUB_OFFLINE=0 PYTHONPATH=src .venv/bin/python tests/test_attention_gpu.py \
  --attn-implementation flash_attention_2 --allow-kernel-download
```

GPU 对照覆盖 GQA、左 padding、全参、LoRA 与梯度 checkpoint；比较 SDPA 和 FlashAttention
的最后位置输出、loss 和梯度。A100 是已执行的实机验证环境；其余架构的自动选择由策略测试覆盖。
切换后端的实际速度收益取决于样本长度、padding 和微批大小，应使用目标训练负载测量。
