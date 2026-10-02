# SORS

**State-conditioned Option Ranking System · 基于当前状态的候选排序系统**

在一块笔记本显卡上就能训的 Jev 式决策模型。

[![License: MIT](https://img.shields.io/badge/license-MIT-0a0a0a.svg?style=for-the-badge&labelColor=000000)](LICENSE)
[![Base: Qwen3-0.6B-Base](https://img.shields.io/badge/BASE-Qwen3--0.6B--Base-0a0a0a.svg?style=for-the-badge&labelColor=000000)](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
[![Python 3.13](https://img.shields.io/badge/PYTHON-3.13-0a0a0a.svg?style=for-the-badge&labelColor=000000)](.python-version)
[![Trainable: 0.43%](https://img.shields.io/badge/TRAINABLE-0.43%25-0a0a0a.svg?style=for-the-badge&labelColor=000000)](#工作原理)

[English](README.md) | **简体中文**

SORS 是 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 决策模型的开源实现，基模用千问。你交给它一段 state 和一组带类型的问题，它在你定义的那些选项上给出概率分布，此外什么都不产生。

前提是答案本来就在基模里。基模缺的是把答案说成软件能直接消费的形状 —— 它的输出是一个字符串，概率摊在 151936 个词表条目上。**所以这里训练的只有输出格式这一件事。** 知识留在从头到尾不吃梯度的权重里：动的是 2.56 M 个参数，占整个模型的 0.43%。

## 要点

- 256 个匿名决策槽住在千问空闲的词表行里。嵌入不用 resize，也不加新的 head —— 槽与选项名的绑定由每次请求给出。
- 训练的只有输出格式：259 行嵌入加一个 attention 上的 LoRA。596 M 参数的骨干全程冻结。
- Banking77 的 77 个意图里有 17 个完全退出训练。模型在评估时第一次读到它们的名字，得分 **0.921**，训练见过的那些是 0.948。
- 每段 state 一份 KV cache 前缀，每个问题接一个分支 —— 同一段 state 上的 N 个问题只付一次这段 state 的前向。
- 8 GB 笔记本显卡上 2000 步 14 分钟，还带一道给容易过热的机器用的温度闸。
- 零样本基线脚本会把隐状态和 logits 落盘，探针、温度标定和消融都能离线拟合。

## 上手

需要 Python 3.13 和 [uv](https://docs.astral.sh/uv/)。默认配方在 6 GB 显存的 CUDA 卡上就能跑。

```bash
git clone <this repo> sors && cd sors
uv sync
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset banking77 --steps 2000
```

Banking77 自己从钉住的上游 commit 下载到 `data/banking77`，md5 和行数双重校验。每 20 步打一行进度，每 100 步做一次完整评估：

```text
step  1980  loss 0.0304  845s  tctl 66°C
{"step": 2000, "train_loss": 0.0299, "t": 852.1, "tctl_c": 65.8, "thermal_waits": 10,
 "seen":   {"accuracy": 0.948, "nll": 0.176, "ece": 0.009, "n": 2400},
 "unseen": {"accuracy": 0.921, "nll": 0.296, "ece": 0.029, "n": 680}}
```

`seen` 是训练用的那 60 个意图下留出的**样本**。`unseen` 是 17 个训练里完全没出现过的意图 —— 样本没出现，名字也没出现。这两列之间的差距就是全部重点：0.921 说明匹配这件事由基模完成，训练提供的是作答格式。

`.venv/bin/tensorboard --logdir runs` 实时看曲线。

```bash
# 两个数据集一起训，同一组槽服务两种问题类型
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset both --steps 2000

# 只训读出 —— 骨干逐比特未变，只有那 259 行嵌入在动
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset banking77 --trainable d-only --steps 2000

# 拿一份 checkpoint 去评另一个任务，不训练
PYTHONPATH=src .venv/bin/python scripts/train.py --dataset boolq --init runs/<run>/trained.pt --steps 0
```

## 工作原理

一次请求变成一条提示。答案 token 紧跟在 `Answer:` 之后；一次前向，读那个位置的 logits，在这条问题的槽 id 上做 softmax：

```text
Passage: <state>

Question: <问句>
Options:
<|D0|>. no
<|D1|>. yes

Answer:
```

**256 个槽。** 千问的 tokenizer 占了嵌入矩阵声明的 151936 行里的 151669 行，空着 267 行。256 个槽 token `<|D0|>` … `<|D255|>` 加 3 个类型标记装进这条尾巴，矩阵形状不变，任何 checkpoint 都不用 resize。槽与选项的绑定每条训练样本重新抽一次，同一个意图每次出现都落在不同的槽上 —— 模型学到的是「在给出的选项里指出匹配的那个」，而这正是调用方逐请求定义选项时必须成立的性质。

**一个参数，两端共用。** 千问的 `tie_word_embeddings` 成立，菜单行里读进来的向量和答案位置上打分的向量是同一行。给第 k 个槽打分就是 `h · E[D_k]`，代数上等价于 `nn.Linear(hidden, 256)` 的第 k 行。`SlotEmbedding` 做冻结的查表，再把槽位置盖成自己那个 259 × 1024 的参数；`SlotHead` 算冻结的 `h @ Wᵀ`，再把那 259 列换成 `h @ rowsᵀ`，引用的是同一个张量。AdamW 因此只给 259 行存动量，而非整张 151936 行 —— 早先那版把整张矩阵标成可训、再用梯度 hook 清零，训练效果相同，代价是 1.2 GiB 的优化器状态，其中 99.8% 永远是零。

**损失**是只在这条样本的 k 个槽上做 softmax，交叉熵到正确的那个。其余约 15 万个词表条目不进分母：训练形状与推理形状一致 —— 调用方给 k 个选项，答案必须是其中之一。

**问题隔离与 state 前缀。** `context-first`（默认）把 state 放在菜单之前，同一段 state 的 N 个问题因此共用一份 KV cache 前缀 —— `cache.py` 把 state 跑一次，每个问题接一个分支，每个分支看得到 state 和它自己那段。`menu-first` 是对照组，9 个评估点全部落后（留出类 accuracy 0.888 对 0.903，NLL 0.341 对 0.310），也没有可共享的前缀。

**问题类型。** `<|choice|>` 是 k 个无序选项，`<|bool|>` 是 choice 取 k=2，`<|score|>` 是 k 个有序档位、期望值当分数。带 `--type-marker` 时问句标签写成 `Question (<|bool|>):`，这几行跟着槽一起训。`<|score|>` 目前只占着 id，见[已知限制](#已知限制)。

## 结果

默认配方是 `--trainable attn --lr-schedule cosine --layout context-first`，下面几张表用的就是它：Qwen3-0.6B-Base，2000 步，attention 上 r=8 的 LoRA，cosine 学习率配 100 步 warmup，评估菜单 10 选 1。

**Banking77** —— 77 个意图，17 个完全退出训练：

| | accuracy | NLL | Brier | ECE |
|---|---|---|---|---|
| 见过的意图（n=2400） | 0.948 | 0.176 | 0.081 | 0.009 |
| **留出意图（n=680）** | **0.921** | 0.296 | 0.126 | 0.029 |
| 留出意图，训练之前 | 0.206 | 2.800 | 0.942 | 0.241 |

10 选 1 的随机水平是 0.100。

**BoolQ** —— yes/no 作为 2 选 1 的菜单，validation n=3270，多数类 0.622：

| | accuracy | AUROC | Brier | ECE |
|---|---|---|---|---|
| 训练 1500 步 | 0.805 | 0.877 | 0.277 | 0.032 |
| `--dataset both`，2000 步 | 0.786 | 0.857 | 0.298 | 0.036 |
| Banking77 的 checkpoint，零 BoolQ 训练 | 0.612 | 0.645 | 0.502 | 0.168 |

最后一行是一份只在客服意图上训过的 checkpoint，配 `--steps 0` 直接指向阅读理解。AUROC 0.645 说明槽位这套机制跨过了任务边界；accuracy 略低于 0.622 的多数类基线，说明任务本身仍然得训。

**放开范围** —— 模型要动多少：

| `--trainable` | 动的部分 | 见过的意图 | 留出意图 | 峰值显存 |
|---|---|---|---|---|
| `d-only` | 259 行嵌入；骨干冻结 | 0.812 | 0.813 | 4.1 GiB |
| `attn`（默认） | 上面这些 + `q/k/v/o` 上的 LoRA | 0.934 | 0.871 | 5.0 GiB |
| `attn-mlp` | 上面这些 + `gate/up/down` 上的 LoRA | 0.945 | 0.878 | 6.0 GiB |

`d-only` 在骨干逐比特未变的前提下，对没见过的意图拿到 0.813 —— 这个数字是纯读出。路由住在 attention 里，所以放开它值 +0.058，在它之上再放开 MLP 值 +0.007。这三条跑在布局开关之前，用的是 `menu-first` 加恒定学习率，因此只在它们三者之间横向比较。

**墙钟时间**，一块 RTX 3070 Ti Laptop（8 GB），含温度暂停：Banking77 2000 步 14.2 分钟，BoolQ 1500 步 16.5 分钟，`both` 2000 步 23.0 分钟。

## 训练

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --help
```

要紧的几个旋钮：`--dataset banking77 | boolq | both`、`--trainable d-only | attn | attn-mlp`、`--layout context-first | menu-first`、`--type-marker`、`--held-out 17`、`--k-min 2 --k-max 10`（训练菜单长度，每条样本重抽）、`--k-eval 10`。

默认值：LoRA r=8、alpha=16、dropout=0.05；LoRA 用 `1e-4`，嵌入行用 `1e-3`（它们从零起步，需要更快的钟）；cosine 衰减配 100 步 warmup。恒定学习率下留出曲线在评估之间摆动 ±3 个点，大于 n=680 的噪声地板。

每次运行写出 `runs/<时间戳>-<dataset>-<trainable>-<schedule>-<layout>/`：

| 文件 | 内容 |
|---|---|
| `log.jsonl` | 每次评估一行；step 0 是训练前、或 `--init` 加载之后的基线 |
| `result.json` | 参数、类切分、全部评估记录、峰值显存 |
| `trained.pt` | LoRA 权重加那 259 行嵌入 —— 基模照 `--model` 重新加载 |
| `tb/` | TensorBoard：`train/loss`、`train/lr_*`、`eval/<set>/<metric>`、`sys/tctl_c` |

`--init runs/<run>/trained.pt` 从 checkpoint 续；配 `--steps 0` 就是只评估。

## 评估

每次评估报 accuracy、top-5、NLL、Brier、ECE 和槽分布上的平均置信度；二元集另加 AUROC、预测正类率和二元 Brier。`n_saturated` 数的是正确类概率跌破 NLL 下限的条目，让那个指标的失真程度可审计。

`scripts/baseline-boolq.py` 和 `scripts/baseline-banking77.py` 在同一套部署形状下量没训练过的读出。两个脚本独立，不从包里 import 任何东西。

BoolQ validation，n=3270，多数类 0.622：

| 模型 | accuracy | Brier | ECE | AUROC |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.650 | 0.213 | 0.058 | 0.709 |
| Qwen3-0.6B（instruct） | 0.657 | 0.255 | 0.206 | 0.721 |
| Qwen3-1.7B-Base | 0.785 | 0.149 | 0.020 | 0.858 |
| Qwen3-1.7B（instruct） | 0.754 | 0.231 | 0.225 | 0.845 |

instruct 微调把判别力留在原处，把标定弄糟 —— 0.6B 上 ECE 0.206 对 0.058。基模因此是这里的起点。

Banking77 test，n=3080，77 选 1 菜单，Qwen3-1.7B-Base：两套 permutation seed 下 accuracy 0.260 / 0.269，top-5 0.443 / 0.470，45% 的概率质量落在 77 个码之外。随机水平是 0.013。知识本来就在，缺的是读出。

每个基线写出一份 JSON（指标）加一份 NPZ（隐状态、槽 logit、全词表 logsumexp、top-k），探针、温度标定和消融因此可以离线拟合，无需再跑一次前向。Banking77 那份给每个意图绑一个两字母码，带空格和不带空格都是单 token，并跑两套 permutation seed 来量出偏好噪声的地板。

## 已知限制

- **没有服务端。** 这是一个训练与测量的仓库。没有 `/v1/systemone` 端点，没有 SDK，没有发布权重。`cache.py` 里有服务端需要的前缀共享通路，上面还什么都没接。
- **`score` 没实现。** `<|score|>` 只占着一个 token id。有序损失和带有序档位的数据集两样都缺，所以今天训的只有 `choice` 及其 k=2 的情形。
- **BoolQ 超出了纯读出。** 同一基模冻结隐状态上拟合的探针天花板是 AUROC 0.745，训练后的运行到了 0.877。attention LoRA 在那里动了表征，所以「只训格式」这句话在这个任务上说得太满。
- **端到端只量过一个基模尺寸。** 训过的全是 0.6B。零样本阶梯显示 1.7B 基模在 BoolQ 上起点就是 AUROC 0.858，高于 0.6B 训练后读出的落点 —— 基模规模是更大的那根杠杆，而它在这里除了基线之外没被测过。
- **单种子。** 一次类切分（seed 0 下的 17 个意图），每种配置一次运行。放开范围那张表里相邻两行的差距，只有 n=680 噪声地板的几倍。
- **基线数字需要重跑才能复现。** `results/` 在 `.gitignore` 里，上面那些 AUROC 和探针数字是从 NPZ 落盘离线拟合出来的。脚本在仓库里，产物不在。
- **评估菜单是 10 个选项。** 256 个槽的容量在训练里最多用到 `--k-max`，从未按满宽跑过。
- **笔记本规模。** 一块 8 GB 显卡，`max_length` 512，batch size 8，还带温度闸。

## 开发

Python 包名为 `sors`，导入使用 `from sors...`。服务鉴权通过 `SORS_API_KEY`
配置，外部数据仓库通过 `SORS_DATASETS_DIR` 配置，详见[数据配置](DATASETS.md)。
旧版 API key 和数据路径环境变量继续作为回退项，SORS 变量优先。

只覆盖纯函数 —— 不用 GPU、不联网、不依赖测试框架：

```bash
for f in tests/test_*.py; do PYTHONPATH=src .venv/bin/python "$f"; done
```

每个文件单独可执行，一个用例打印一行。`ThermalGuard` 每步之前和每次评估之前读一次 `k10temp` 的 Tctl，高于 `--temp-max`（默认 85 °C）就睡着等，暂停次数记在 `sys/thermal_waits` 里，没有这个传感器的机器直接放行。

| 路径 | 内容 |
|---|---|
| `src/sors/core/tokens.py` | 256 个槽 token 与 3 个类型 token，装进 tokenizer |
| `src/sors/core/menu.py` | 菜单采样、类切分、与数据集无关的 `LabeledSet` |
| `src/sors/core/prompt.py` | 提示模板与两种布局 |
| `src/sors/core/batch.py` | 批量张量化，左填充，答案位置钉在最后一列 |
| `src/sors/core/model.py` | `SlotEmbedding`、`SlotHead`、LoRA 接线、参数分组 |
| `src/sors/training/loss.py` | 限制在槽上的交叉熵 |
| `src/sors/evaluation/metrics.py` | accuracy、top-5、NLL、Brier、ECE、AUROC |
| `src/sors/training/loop.py`、`src/sors/core/checkpoint.py` | 训练循环、评估、存档与读档 |
| `src/sors/core/cache.py` | 一份 KV cache 前缀接 N 个问题分支 |
| `src/sors/training/thermal.py` | 温度闸 |
| `src/sors/data/banking77.py`、`src/sors/data/boolq.py` | 数据集适配 |

## 致谢

基模来自 [Qwen](https://huggingface.co/Qwen/Qwen3-0.6B-Base)。数据集：[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets)（PolyAI）与 [BoolQ](https://huggingface.co/datasets/google/boolq)（Google）。接口形状参照 TypeSafe 的 [System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev)；与 TypeSafe AI 无隶属关系，训练方法是本项目自己的。

相关工作：[kev](https://github.com/jaredpalmer/kev) 在 0.8B–9B 上训练同一类模型，并用兼容 System One 的 API 提供服务。

名字的由来：**SORS** 展开为 **State-conditioned Option Ranking System**，即「基于当前状态的候选排序系统」。拉丁语 *sors* 带有抽签、机缘与命运的意象，呼应模型读取局势、衡量各个选择的职责。Python 包名和项目标识统一使用 `sors`。

## 许可

[MIT](LICENSE)。
