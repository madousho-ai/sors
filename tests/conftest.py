"""pytest 共用配置.

- 子进程的导入路径: pyproject 里的 pythonpath 只改本进程的 sys.path, 测试里起的
  `python -m sors...` 子进程要靠环境变量 PYTHONPATH 找到 src/.
- xdist 并行时每个 worker 分到的 CPU 线程: 不设的话 16 个 worker 各开满核的 torch
  线程池, 互相抢核反而更慢.
- tests/*_gpu.py 自动打上 gpu 标记, 默认跳过, 加 --gpu 才跑; 跑时没有 CUDA 直接判失败.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH")) if p)

_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0"))
if _workers > 1:
    _threads = str(max(1, (os.cpu_count() or 1) // _workers))
    for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, _threads)


def pytest_addoption(parser):
    group = parser.getgroup("gpu", "tests/*_gpu.py 的 CUDA 检查")
    group.addoption("--gpu", action="store_true", help="跑 gpu 标记的测试 (要 CUDA, 部分要联网下权重或内核)")
    group.addoption("--attn-implementation", default="auto", help="test_attention_gpu / 真实模型前缀对照用的 attention")
    group.addoption("--allow-kernel-download", action="store_true", help="允许从 kernels-community 下载预编译内核")
    group.addoption("--real-model", help="test_candidate_prefix_gpu: 用本地缓存的真实 Qwen 权重做前缀对照")
    group.addoption("--revision", help="--real-model 的 revision")
    group.addoption("--fa2-kernel-path", help="--real-model: 载入本地已构建的 FA2 kernel 项目")
    group.addoption("--conv-kernel-path", help="--real-model: 载入本地已构建的 causal-conv1d kernel 项目")


def pytest_collection_modifyitems(config, items):
    skip = None if config.getoption("--gpu") else pytest.mark.skip(reason="GPU 检查, 加 --gpu 才跑")
    for item in items:
        if item.path.name.endswith("_gpu.py"):
            item.add_marker(pytest.mark.gpu)
            if skip:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def require_cuda(request):
    if request.node.get_closest_marker("gpu"):
        import torch
        if not torch.cuda.is_available():
            pytest.fail("CUDA is required; run the gpu tests on a GPU machine")


@pytest.fixture
def gpu_options(request):
    """--attn-implementation 等命令行参数, 字段名与原先 argparse 的 args 相同."""
    return request.config.option
