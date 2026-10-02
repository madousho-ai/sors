# 选项集合决策架构

固定预训练基模，通过新增决策块学习 `P(option | context, question, option-set)`。
训练、评估、权重存档、完整训练状态与 HTTP 服务使用同一架构配置。

## 选择架构

| `--architecture` | 输入与连接 | 默认训练范围 |
|---|---|---|
| `slots` | 原串行菜单、答案位置的 D 码 logits；默认路径 | `attn` |
| `minimal` | 保留原提示及其 tokenization；从选项描述末端取得表示，在主干最后几个层之后接入决策块 | `decision-only` |
| `structural` | 上下文和问题组成公共文本流；每个“问题＋选项”独立编码，权重共享；选项支路连接主干中后层 | `decision-only` |
| `candidate` | 每个完整“上下文＋问题＋单个选项＋Answer:”经过共享 Qwen，再接可选 SetBlock 和共享标量读出 | `decision-only` |

三种新架构支持 Qwen3 与 Qwen3.5 **文本主干**。`structural` 和 `candidate` 使用 `context-first` 布局。
`minimal` 的内部表示保留串行菜单的顺序影响；`structural` 和 `candidate` 在确定性推理下满足选项行置换等变性，允许浮点误差。
选项内部保留词序；选项集合层省去行位置编码，所有候选项共用参数。

minimal/structural 的每个决策块包含：

1. 选项向文本 token 发起 cross-attention，提取问题与上下文证据。
2. 选项集合 self-attention，再接共享的逐行前馈层。
3. 可选的文本向选项 cross-attention，将候选集合信息回写当前主干层输出。

这两条路径的最终共享打分器同时读取选项表示和主干末位置表示。输出在 K 个合法选项内归一化，随后映射到本次 D 码。
基模的 LM 词表输出矩阵保持在模型中，决策前向直接使用文本主干。

## Candidate 联合编码基线

每条分支完整编码 `C + Q + option_i + Answer:`，读取末位置隐藏表示 `h_i`。
`Answer:` 使用已有普通词表；D 码只用于最后的概率映射。默认关闭类型/context 特殊标记，
配合 `decision-only` 可固定原始 Qwen 表征，单独训练集合模块和读出。

`--decision-blocks 0` 对应 `s_i = wᵀ RMSNorm(h_i)`，每个候选独立计算自己的证据。
`--decision-blocks 1` 或 `2` 在这些表示上加入共享的集合 self-attention 和逐行 FFN，再用同一个标量头打分。
候选向量和残差保留主干 hidden_size；`decision-dim` 控制 SetBlock 内部注意力宽度，FFN 中间宽度为其四倍。
所有配置都使用同一个 RMSNorm＋无偏置标量头，温度固定 `T=1`，合法选项上 softmax。

candidate 的 SetBlock 位于联合编码之后，数量独立于主干层数；模型使用固定的独立分支结构。
`--decision-feedback` 与指定主干接入层的 `--decision-layers` 在该模式下会被拒绝。
公共前缀支持一次预填充后复用；`--decision-option-batch-size` 控制一次编码的候选数。
温度固定为 1，独立校准属于后续实验。

## Candidate 共享前缀

`--candidate-prefix-cache` 控制执行策略，模型结构和权重形状保持一致：

| 值 | 行为 |
|---|---|
| `auto` | 新 candidate 训练命令默认值；无梯度推理和编码过程完全冻结时共享前缀，其余情况保留完整前向 |
| `on` | 强制共享；训练时遇到可训编码参数或实际用到的新增 token 行，直接报错 |
| `off` | 完整联合前向，供数值对照或兼容旧运行 |

每次模型调用按实际 prefix token 串分组：相同 C+Q 的不同排列视图可以共用一次 prefill，
每组处理完释放缓存。每个候选块从相同快照分叉，完整复制全注意力 KV、卷积历史和 DeltaNet 递推状态。
后缀按长度分桶，避免在已缓存前缀与候选之间插入会推进递推状态的 padding。
分词边界以完整输入的 token 序列为准；跨文本分界的 BPE 合并 token 留在后缀，保持两条路径输入完全一致。

冻结编码器采用 `eval()` 和 `no_grad()`，并在调用结束后恢复原模式；SetBlock 和 scorer 正常训练。
`--grad-ckpt` 此时只重算决策头，昂贵的前缀提取保留在重算区间外。LoRA/full 训练的 `auto` 模式保留完整编码器梯度。
类型/context 特殊标记引用可训练的新增 token 行时，也走完整前向。

执行策略写入训练配置和权重存档；`--init` 默认继承，可用显式参数覆盖。旧 candidate 存档缺少该字段时按 `off` 加载。
续跑按完整状态的配置恢复。服务同样可使用 `--candidate-prefix-cache auto|on|off` 覆盖存档设置。
直接调用 Python `train()` 的旧配置默认保持 `off`；显式设置 `TrainConfig(candidate_prefix_cache="auto", ...)` 启用。

共享模式的服务 token 计数为实际 prefill 加各后缀；KV 批量预算预留原始快照、分支缓存和分叉临时空间。
单个分支超过批量预算时独立执行，沿用原服务规则；attention 激活与内核工作区另占显存。
BF16 的缓存/完整计算会有舍入差异，几乎同分的候选可能交换首选；需要严格对照时保留 `off` 结果。

## 回写是独立开关

minimal/structural 使用 `--decision-feedback` 开启、`--no-decision-feedback` 关闭。新建模型默认关闭。
开启后，回写输出投影零初始化，训练起点保留主干原始激活；投影从第一步即可接收梯度。
回写修改真实 decoder 层的输出，后续主干层和最终读出都会消费这个结果。

## 训练命令

下面展示架构开关。比较实验需要自行统一数据曝光量、学习率、批大小和主干训练范围。

```bash
# 最小改动，回写关闭
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --model Qwen/Qwen3.5-0.8B-Base --dataset synth-v5.1 \
  --architecture minimal --no-decision-feedback \
  --loss menu --consistency 1 --trainable decision-only \
  --grad-ckpt --save-training-state --steps 2000

# 结构版，回写开启
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --model Qwen/Qwen3.5-0.8B-Base --dataset synth-v5.1 \
  --architecture structural --decision-feedback \
  --loss menu --consistency 1 --trainable decision-only \
  --grad-ckpt --save-training-state --steps 2000

# Candidate 联合编码，先测纯标量读出；把 0 改成 1 或 2 可对照集合交互
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --model Qwen/Qwen3.5-0.8B-Base --dataset synth-v5.1 \
  --architecture candidate --decision-blocks 0 --candidate-prefix-cache auto \
  --loss menu --consistency 1 --trainable decision-only \
  --grad-ckpt --save-training-state --steps 2000
```

`decision-only` 冻结原始主干权重，训练新增决策参数与新增 token 行。`--lr-lora` 控制决策参数学习率，
`--lr-embed` 控制 token 行；结构版输入使用的类型/context 标记同样属于新增行。
`full` 继续采用现有的主干全参范围；原始词表矩阵冻结。原有 `d-only` / `attn` / `attn-mlp` 范围也保留，
新增决策参数在这些模式下始终可训。LoRA 的投影名称沿用现有配置。

三种新架构要求显式 `--loss menu`。硬标签、软标签、标签平滑、语义对齐 JS 与分组梯度累积共用现有实现。
新架构的 `m_answer` 和 `top1_in_menu_rate` 由合法输出约束保证接近 1，`m_offmenu` 为 0；这些值表示输出协议的性质。
决策质量应比较 accuracy、NLL、Brier、ECE 和扰动一致性。

## 结构参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--decision-blocks` | 2 | candidate 集合层数，允许 0；minimal/structural 块数至少 1，至多等于主干层数 |
| `--decision-dim` | 128 | 决策支路宽度／candidate 集合注意力内部宽度，至少为 2 |
| `--decision-heads` | 4 | 注意力头数，须整除支路宽度 |
| `--decision-layers` | 自动选择 | minimal/structural 的零起始主干层下标，如 `11,23`，严格递增且数量等于块数 |
| `--decision-option-batch-size` | 32 | structural/candidate 独立选项编码的分批上限 |

设主干共 L 层、新增 B 块：minimal 默认接在 `L-B .. L-1`；structural 第 j 块（j 从 1 开始）
接在 `floor(j*(L-1)/B)`。存档记录解析后的实际层下标。

新增层默认使用 fp32，与低精度主干之间显式转换；主干低精度可训参数继续使用现有 fp32 master 更新。
关闭回写的 minimal/structural 使用主干原生逐层 checkpointing，并为每个决策块单独设置检查点。
主干前向仅收集选定层的原始输出，保留梯度连接；各决策块随后依次更新选项状态，最终查询继续使用主干归一化后的输出。
关闭回写的 structural 还为每批独立选项编码设置检查点，批次间保留输入和末位置读出。
较短的选项块在反向整块重算一次；当 `token数 × hidden_size × 主干层数` 超过 128 Mi 个元素时，保留内部逐层检查点控制重算显存。
该策略按调用上下文和模型归属隔离，公共长文本继续逐层检查点。所有选项块的裁剪宽度一次性读回，保持原有行序和分块边界。
此时主干状态仅由文本输入决定，决策块依次读取相同的层输出，后置计算保持同一打分函数和梯度连接。
开启回写的 minimal/structural 以及可训练 candidate 编码器继续使用整条耦合前向重算；冻结 candidate 编码器时只重算决策头。
原 slots 路径继续使用原有的逐层 checkpointing。检查点划分保持模型参数、架构元数据和打分公式兼容。

## 长度、批处理与推理

- minimal 保留原提示编码。截断可以缩短上下文；涉及候选项的截断会直接报错。
- structural 的 `max_length` 分别约束公共文本流与每个选项分支。选项分支须完整保留。
- candidate 的 `max_length` 约束每条完整联合分支；超长即报错，保留 C、Q、选项与读出位置的完整语义。
- 新架构服务逐题计算，独立选项按配置和服务 token 预算分批。整个请求先校验长度，通过后再调用模型。
- minimal/structural 服务走完整前向；candidate 按共享前缀执行策略运行；旧 slots 服务保留原共享 KV cache 路径。
- structural 的提示预览展示公共文本流和各独立选项分支；`input_tokens` 统计实际编码流，包含重复的问题文本。
- candidate 的提示预览展示公共前缀及各分支后缀；每条真实输入是两者相接。`input_tokens` 按实际执行方式计数。

## 存档与续跑

权重档的 `architecture` metadata 保存架构、支路尺寸、实际接入层与回写开关；`adapter` 保存主干训练范围。
缺少 `architecture` 的历史权重按 slots 重建。架构冲突和缺失的决策权重在加载阶段报错。

```bash
# 从存档继承架构和回写配置
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --model Qwen/Qwen3.5-0.8B-Base --dataset synth-v5.1 \
  --init runs/<run>/trained.safetensors --loss menu --steps 0

# --save-training-state 保存的完整状态可由原续跑入口恢复
PYTHONPATH=src .venv/bin/python scripts/resume.py --run runs/<run>
```

`--save-training-state` 是显式开关，保持原训练命令的存档行为。它在保存边界和最后一步原子替换
`checkpoints/latest.trainstate.safetensors`，保存参数、master、优化器、调度器、RNG、采样位置和架构元数据。
完整状态保存开关和 resume CLI 的数据版本核验范围为 synth-v5.1 及兼容别名 synth-v5；
其余数据集使用权重存档。原有源码/数据指纹保护继续生效。
`--init` 用于载入权重开始新一次运行；完整训练状态通过 resume 入口续接。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_decision.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_minimal_checkpoint.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_structural_checkpoint.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_decision_integration.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_candidate.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_candidate_integration.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_candidate_prefix.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_candidate_prefix_integration.py
```

测试使用小型真实 Qwen3/Qwen3.5 模型，覆盖非零回写后的换序等变性、跨层回写效果、梯度、混合精度、
完整前向重算、padding/分组、256 项菜单、真实 tokenizer、存档恢复、完整续步对照和服务公共评分路径。
minimal 的独立回归还对照原耦合前向，检查输出、全部梯度、SGD 单步更新，以及主干逐层反向重算的执行顺序。
structural 的独立回归覆盖上述数值对照、双精度 AdamW 连续两步更新，并检查选项批次保留的激活、LoRA dropout 随机状态、异常清理和并发重算隔离。
这些检查验证实现行为；任务成绩、正式模型显存与吞吐由后续训练实验测量。

服务器 CUDA 检查：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_decision_gpu.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_candidate_gpu.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python tests/test_candidate_prefix_gpu.py
```

A100 已通过 Qwen3/Qwen3.5 × minimal/structural × 回写开/关的 8 组小模型检查：bf16 主干、fp32 新层、
非零回写、重算梯度对照和保存重载。该检查使用现有环境的 attention 实现，Qwen3.5 卷积使用 PyTorch 参考路径。
candidate 的 Qwen3/Qwen3.5 × 0/1/2 SetBlock 共 6 组 A100 小模型检查同样通过，覆盖混合精度、
重算梯度、选项与 D 码换序、精确重载；这组检查的最大概率换序误差为 `2.98e-8`。

共享前缀另做了真实 Qwen3.5-0.8B-Base 的 BF16 数值检查：revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`，A100，FA2＋causal-conv1d v2＋FLA。
使用同一份固定初始化的决策头比较两种执行路径，73/1685/7821-token 前缀均只 prefill 一次。
处理 token 数分别从 340/6788/31332 降到 118/1730/7866；最大候选概率差分别为 0.001762/0.000999/0.002679，首选相同。
隐藏表示相对 L2 误差分别为 0.01345/0.01232/0.01440。该检查验证执行数值与复用次数，任务准确率和正式吞吐另行测量。
GPU 验证脚本支持 `--real-model`、`--revision`，以及 `--fa2-kernel-path` / `--conv-kernel-path` 指定已有本地内核，
用于离线缓存缺少 Hub 版本引用或完整 snapshot 元数据的环境。
