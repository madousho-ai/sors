# SORS

**State-conditioned Option Ranking System**

[![License: MIT](https://img.shields.io/badge/license-MIT-0a0a0a.svg?style=for-the-badge&labelColor=000000)](LICENSE)

[English](README.md) | **简体中文**

给定一段状态、一组问题和调用方自己定义的选项，SORS 只输出这些选项上的概率分布。

**模型下载：** [🤗 SakuraYuyuko/Sors-2B](https://huggingface.co/SakuraYuyuko/Sors-2B)

![SORS 架构](docs/sors-architecture.svg)

SORS 对每个选项单独打分。每道题按选项拆成若干分支：共享前缀（先状态、后问题），后面接 `Option: <|D1|> 选项文字` 和 `Answer:`。
前缀只编码一次，所有分支共用，所以一个菜单的开销是读一遍状态，再加每个选项一小段后缀。全量微调的 Qwen3.5 主干按因果顺序读每个分支，
在 `Answer:` 处取最后一层的隐状态，由一个标量头给出这个选项的分数；再在这道题自己的选项上做 softmax，得到概率。
选项的名称和数量都由调用方在请求里决定，一个菜单最多 255 项，换一组选项不用重新训练。

每个分支都看不到别的选项，所以打乱菜单只会打乱分数的顺序，一个选项的分数也不受菜单上其他选项影响。
标量头的初始值是语言模型对下一个词的对数几率：是 `<|D1|>`（这个选项合适）还是 `<|D2|>`（不合适）。
训练先让主干对每个选项单独学会这个下一词判断，再打开标量头，在菜单上训练 softmax，同时保留逐选项的那一项。
有歧义的题训练目标是一个概率分布，不是单一答案；一半的题隐藏选项说明，一半的菜单加一个「以上都不是」选项。

## 跑分

Sors-2B 只用我们自己合成的数据集 synth-intents-v5.17 训练（26 个领域共 370,639 道题，数据集整理完成后开源），
下表里的评估集都不参与训练。合成数据由我们从零编写，不包含任何现有数据集的内容，生成时刻意避开了 Banking77 的银行业务
和 MASSIVE 的语音助手场景，并有自动检查拦截这些领域的用词。我们拿训练材料（167,759 段不同的文本）逐条比对了下表的全部评估集：
归一化后完全相同的文本为 0；与 MASSIVE、BoolQ、JevBench 共享连续 8 个词的条目为 0；
Banking77 的 3,080 条里有 5 条与训练材料共享 8 个词，都是「I don't want to be charged for」这类日常短语。

| 评估集 | 题数 | [Sors-2B](https://huggingface.co/SakuraYuyuko/Sors-2B) |
|---|---:|---:|
| Banking77 | 3,080 | 67.0 |
| Banking77 + 描述 | 3,080 | 79.7 |
| MASSIVE | 2,974 | 68.6 |
| MASSIVE + 描述 | 2,974 | 80.4 |
| BoolQ | 3,270 | 81.1 |
| Public JevBench easy | 48 | 100.0 |
| Public JevBench original | 72 | 97.2 |
| Public JevBench hard | 111 | 66.7 |
| Public JevBench 合计 | 231 | 83.1 |

数值是准确率（%）。

- **Banking77 和 MASSIVE 每道题都在完整的意图菜单上作答**：Banking77 一次给出全部 77 个意图，MASSIVE 一次给出全部 60 个意图，
  没有先用检索或其他模型挑出 top-K 候选再让模型选。「+ 描述」是在每个意图名后面附一句说明。
- 选项顺序：同一批题把选项随机打乱成 5 种顺序分别作答，Sors-2B 的平均准确率与上表相差不超过 0.1 个点，
  5 种顺序选中同一选项的比例在 Banking77 和 MASSIVE 上是 99.7–99.8%，BoolQ 上是全部。
- Public JevBench 是 JevBench 的公开题集，用 JevBench 官方 CLI 和它的 `typesafe` adapter 评测；三档分别只有 48 / 72 / 111 题，几个点以内的差距属于噪声；「合计」是全部 231 题的准确率（192 题答对）。JevBench 榜单的成绩出来后会更新到这里。
- 这是 2026-10-29 发布的权重。2026-10-08 发布、用 synth-intents-v5.3 训练的权重仍可通过 Hugging Face revision `v5.3-2026-10-08` 获取。

## 部署

### 直接运行

需要 Python 3.13、[uv](https://docs.astral.sh/uv/) 和一块 CUDA GPU。

```bash
uv sync
PYTHONPATH=src .venv/bin/python scripts/serve.py \
  --init SakuraYuyuko/Sors-2B --warmup --port 8000
```

`--init` 写 HF 仓库名时，第一次启动会把模型下载进标准 HF 缓存（`~/.cache/huggingface`），之后直接从缓存加载；
私有仓库读取本机的 `hf auth login` 或 `HF_TOKEN`。离线环境加 `--local-files-only` 只读缓存，`--revision` 可以指定分支、tag 或 commit。
`--init` 也接受本地模型目录。
设置 `SORS_API_KEY` 或 `--api-key` 后，请求需要带 `Authorization: Bearer <key>`。加上 `--demo` 会在 `/demo/playground/` 开一个试用页。

### 请求

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Sors-2B",
    "state": "I need to cancel my subscription before it renews.",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
          "Cancel a subscription": null,
          "Reset a password": null,
          "Track a shipment": null
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does the customer need this done right away?"
      }
    }
  }'
```

`model` 必须和服务名一致。服务名默认取 HF 仓库名的最后一段或模型目录名，这里是 `Sors-2B`，可以用 `--model-name` 改。一个请求里可以放多道问题，共用同一段 `state`。

返回（概率数值为示意）：

```json
{
  "model": "Sors-2B",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "Cancel a subscription",
      "probabilities": {"Cancel a subscription": 0.99, "Reset a password": 0.005, "Track a shipment": 0.005},
      "confidence": 0.985
    },
    "urgent": {"type": "noul", "noul": 0.71}
  },
  "usage": {"input_tokens": 118, "output_tokens": 2}
}
```

问题有三种类型：`choice` 在调用方给的选项里选一个（2–255 项，每个选项可以附一段说明）；`noul` 是是非题，返回答「是」的概率；
`score` 是有序等级。`state`、`instructions` 和选项说明都可以是字符串或 JSON。

### Docker

同一个模型目录可以挂进容器运行，构建与启动方法见 [docker/README.md](docker/README.md)。

## 测试

```bash
uv sync
SORS_DATASETS_DIR=../decidophobia-dataset HF_HUB_OFFLINE=1 .venv/bin/pytest
```

默认用 pytest-xdist 占满所有核并行跑；单独调一条测试时加 `-n0` 串行。
`tests/*_gpu.py` 默认跳过，加 `--gpu` 才跑，需要 CUDA。

## License

代码使用 [MIT](LICENSE)。模型权重沿用 Qwen 的 Apache-2.0。API 形状参照 TypeSafe 的
[System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev)，本项目与 TypeSafe AI 无关联。
