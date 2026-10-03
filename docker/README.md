# 离线 GPU 推理镜像：Docker / Podman

公共 base 安装共享环境；runtime 的四个 target 各自准备对应 GPU 的内核。
两份 Dockerfile 使用标准多阶段构建，Docker 和 Podman/Buildah 都通过 `build --target` 使用。
镜像目标平台是 **Linux x86_64、Python 3.13、PyTorch 2.14.0、CUDA 13.0**。

## 构建

在仓库根目录执行。`ENGINE` 可设为 `docker` 或 `podman`；base 与 runtime 使用同一引擎、同一用户的镜像存储。
构建会联网拉取基础镜像、安装依赖和下载内核，模型资源在下文单独提供。

```bash
ENGINE=podman

$ENGINE build -f docker/Dockerfile.base -t localhost/sors-base:cu130 .
$ENGINE build -f docker/Dockerfile.runtime --target sm8x -t localhost/sors:sm8x .
```

| target | GPU 范围 | 固定 attention | 额外资源 |
|---|---|---|---|
| `sm8x` | SM80/86/87/89：A100、RTX 30/40 系列等 | Hub FA2 v3 | 卷积 Hub v2 |
| `sm90` | SM90：H100/H200 | Hub FA3 v1 | 卷积 Hub v2 |
| `sm100` | SM100：B100/B200 | Hub FA4 v0 | `gpu-fa4` 依赖组、卷积 Hub v2 |
| `sm120` | SM120：RTX 5090 / RTX PRO 6000 Blackwell | Hub FA2 v3 | 卷积 Hub v2 |

`sm120` 使用明确包含 SM120 的 FA2 构建，沿用现有加载器对 Hub FA4 的兼容性限制。
表中的范围来自项目选择策略和固定内核的构建元数据；每个目标 GPU 上仍需执行下文的真实推理验收。

例如构建 Hopper 版本：

```bash
$ENGINE build -f docker/Dockerfile.runtime --target sm90 -t localhost/sors:sm90 .
```

共享基础镜像名称可通过 `--build-arg BASE_IMAGE=...` 指定。基础 Python 和 uv 镜像固定 digest，
Python 包由 `uv.lock` 固定；系统编译工具从基础镜像对应的 Debian 软件源安装。

`kernels.toml` 固定每个 Hub 仓库的完整 commit SHA 和匹配当前环境的构建目录。
构建工具校验上游 SHA256，复制真实文件并移除临时 Hub 缓存，最终写入 `/opt/kernels/bundle.json`。
这个过程仅下载所选 target 的两个内核，准备时可以使用无 GPU 的构建机器。

## 准备模型

推理需要与训练一致的**基模及 tokenizer**，加一份 SORS **推理存档**。
公共运行镜像接受以下目录：

```text
model-payload/
  base/                  # Hugging Face config、tokenizer、完整权重或全部分片
  trained.safetensors    # scripts/train.py 导出的推理存档
```

`base/` 使用固定 revision 的完整本地快照，复制时展开指向 Hugging Face 缓存的符号链接。
`trained.safetensors` 可以取中途导出的推理 checkpoint；续跑用的 `latest.trainstate.safetensors` 另作训练用途。
准备目录时记录基模 revision、存档 SHA256 和对应代码 commit。容器默认明确使用 `/models/base`，
服务名称为 `sors`；也支持原有 `--base-model`、`--init` 和 `--model-name` 参数。

开发时将该目录只读挂载；正式交付时可用第三份 Dockerfile 将它封装进镜像：

```bash
REPO="$PWD"
$ENGINE build -f "$REPO/docker/Dockerfile.model" \
  --build-arg RUNTIME_IMAGE=localhost/sors:sm8x \
  -t localhost/sors-submission:sm8x /absolute/path/model-payload
```

这次构建的 context 是单独的模型目录。仓库的 `.dockerignore` 采用允许清单，runtime 构建只接收代码、
锁文件和容器配置。运行记录、训练数据、开发环境及凭据保留在宿主机。

## 运行

宿主需要支持 CUDA 13.0 的 NVIDIA 驱动，并配置 NVIDIA Container Toolkit：
Docker 使用 GPU runtime；Podman 使用生成好的 NVIDIA CDI 配置。
镜像保留 C 编译器和 Python headers，供 FLA/Triton 在目标 GPU 上即时编译。

### Docker

```bash
docker run --rm --gpus all --read-only \
  --tmpfs /tmp:rw,exec,nosuid,size=2g \
  -p 127.0.0.1:8000:8000 \
  -v /absolute/path/model-payload:/models:ro \
  localhost/sors:sm8x
```

### Podman

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  --read-only --tmpfs /tmp:rw,exec,nosuid,size=2g \
  -p 127.0.0.1:8000:8000 \
  -v /absolute/path/model-payload:/models:ro \
  localhost/sors:sm8x
```

模型已封装进 submission 镜像时，使用该镜像并省略模型挂载。
`/tmp/sors` 存放 HF modules、Triton、CUDA 和 Torch extension 缓存；tmpfs 计入容器内存，
较大工作负载可改用独立可写 volume 并根据验证结果设置容量。

启动入口会校验环境、GPU SM 和内核文件校验值，输出完整内核版本清单。
服务强制本地加载并关闭 Hub 下载，通过两个不同长度 state、三种问题类型的真实推理预热后才监听 HTTP。
缺失资源、错误 GPU 或预热失败会使进程退出。训练 CLI 的现有下载默认值保持原样。

## 禁网验收

将运行命令的网络改为 `--network none`，移除 `-p` 并用 `-d --name sors-offline` 启动。
此时在容器内通过 loopback 访问服务，例如：

```bash
$ENGINE exec sors-offline python -c '
import json, urllib.request
payload = {"model": "sors", "state": "The object is red.", "questions": {
    "color": {"type": "choice", "instructions": "Which color?", "criteria": {"red": None, "blue": None}},
    "red": {"type": "noul", "instructions": "Is the object red?"}
}}
request = urllib.request.Request("http://127.0.0.1:8000/v1/systemone",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=120) as response:
    print(response.read().decode())
'
$ENGINE logs sors-offline
$ENGINE stop sors-offline
```

设置 API key 后，请求同时携带对应 Bearer 凭据。
评测程序也可以与服务放在同一网络命名空间中，仅通过 loopback 通信。
正式提交应覆盖实际存档、不同问题长度、重复请求和最大目标输入长度，保存启动清单和结果。
宿主驱动、GPU 型号以及基础镜像和最终镜像的 digest 一并列入提交说明。

## 本地回归测试

```bash
PYTHONPATH=src .venv/bin/python tests/test_container_kernels.py
PYTHONPATH=src .venv/bin/python tests/test_container_entrypoint.py
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python tests/test_container_serve.py
PYTHONPATH=src .venv/bin/python tests/test_attention.py
OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python tests/test_serve_cli.py
```

前三项分别检查构建资源选择和校验、离线启动参数、真实本地模型的存档恢复与概率输出。

镜像内可用合成 tiny 模型验证真实 GPU 路径，整个测试可在禁网、只读根目录下执行：

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  --network none --read-only --tmpfs /tmp:rw,exec,nosuid,size=2g \
  -v "$PWD/tests:/tests:ro" -e PYTHONPATH=/app/src:/app/docker:/tests \
  -e OMP_NUM_THREADS=2 --entrypoint python localhost/sors:sm8x /tests/test_container_gpu.py
```

该测试覆盖 Qwen3 slots、Qwen3.5 candidate 的不同输入长度、重复请求和与 SDPA 的概率对照，
同时确认 attention 和卷积内核从 `/opt/kernels` 加载；模型和 tokenizer 均在本地临时生成。

本轮实测环境为 Podman 5.8.4、RTX 3070 Ti Laptop（SM86）、驱动 610.57.04。
base、sm8x 和包含 tiny 模型的最终镜像均已成功构建；禁网、只读条件下的 FA2 前后向、
Qwen3.5 本地卷积/FLA、真实服务入口及三种 HTTP 问题类型通过。
本机 Docker 命令是 Podman 兼容入口；独立 Docker Engine 和其他 GPU profile 的实机验收尚待执行。
正式模型仍按前述提交验收步骤验证，构建和 GPU 验证的具体结果以执行日志为准。
