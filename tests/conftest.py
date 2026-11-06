"""pytest 共用配置.

- 子进程的导入路径: pyproject 里的 pythonpath 只改本进程的 sys.path, 测试里起的
  `python -m sors...` 子进程要靠环境变量 PYTHONPATH 找到 src/.
- xdist 并行时每个 worker 分到的 CPU 线程: 不设的话 16 个 worker 各开满核的 torch
  线程池, 互相抢核反而更慢.
- tests/*_gpu.py 自动打上 gpu 标记, 默认跳过, 加 --gpu 才跑; 跑时没有 CUDA 直接判失败.
- 真实 synth-v5 数据集的读取结果缓存到磁盘: 读一遍 1.5 GB 要两分多钟, 其中大半是逐条格式检查.
  第一次照常读并写缓存, 之后几秒读回. 数据文件 (大小、修改时间)、jevbench 文件或读取代码一变就重读.
  缓存目录 SORS_TEST_CACHE, 默认 ~/.cache/sors-tests; 设 SORS_TEST_CACHE=off 关掉.
  同时持有这份数据集的测试数有上限 (每个占 5~7 GB, worker 一起读会 OOM), 见 _take_slot.
- 训练循环的数据准备默认开 forkserver 子进程, 每次 train() 起两个子进程各自重新 import torch,
  玩具模型跑 3 步要 8 秒. 测试里换成线程 (结果逐位相同); 专测子进程的 test_prefetch*.py 不换.
"""

import ast
import ctypes
import fcntl
import functools
import gc
import hashlib
import os
import pickle
import tempfile
import time
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


def _sources(start: Path) -> list[Path]:
    """start 及它直接、间接 import 的 sors 模块文件."""
    seen, todo = set(), [start]
    while todo:
        path = todo.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                if name.split(".")[0] == "sors":
                    base = ROOT / "src" / Path(*name.split("."))
                    todo += [base.with_suffix(".py"), base / "__init__.py"]
    return sorted(seen)


def _files_key(h, directory: Path, pattern: str = "*") -> None:
    if directory.is_dir():
        for p in sorted(directory.rglob(pattern)):
            if p.is_file():
                st = p.stat()
                h.update(f"{p.relative_to(directory)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())


_slots = []  # 本进程当前测试占着的数据集槽位 (打开的锁文件), 测试结束时放掉


def _take_slot(directory: Path) -> None:
    """每个持有数据集的测试要占 5~7 GB; 并行的 worker 一起读会 OOM. 限制同时读的测试数,
    SORS_TEST_DATASET_SLOTS 指定, 默认按每个槽位 9 GB 用总内存折算."""
    if _slots:
        return
    n = int(os.environ.get("SORS_TEST_DATASET_SLOTS", "0"))
    if n <= 0:
        total_kb = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemTotal:"))
        n = max(1, total_kb // (9 << 20))
    files = [open(directory / f"slot-{i}.lock", "w") for i in range(n)]
    while True:
        for f in files:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            _slots.append(f)
            for g in files:
                if g is not f:
                    g.close()
            return
        time.sleep(0.5)


@pytest.fixture(autouse=True)
def _release_dataset_slot():
    yield
    if _slots:
        # 数据集释放后堆仍挂在进程上, 不还给系统的话跑过这类测试的 worker 各占着 2~4 GB
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
    while _slots:
        _slots.pop().close()


def _cache_synth_v5():
    cache = os.environ.get("SORS_TEST_CACHE", str(Path.home() / ".cache" / "sors-tests"))
    if cache == "off":
        return
    import sors.data.synth_v5 as v5
    from sors.data.paths import asset_path, datasets_root

    original = v5.load_synth_v5
    sources = _sources(Path(v5.__file__).resolve())

    @functools.wraps(original)
    def load(data_dir=None, *, datasets_dir=None):
        try:
            path = (Path(data_dir) if data_dir is not None else asset_path("synth-intents-v5.24", datasets_dir)).resolve()
            real = path.is_relative_to(datasets_root()) and not path.is_relative_to(Path(tempfile.gettempdir()).resolve())
        except Exception:
            real = False
        if not real:  # 测试自己造的小数据集: 照常读
            return original(data_dir, datasets_dir=datasets_dir)
        h = hashlib.sha256(f"{path}\0{os.environ.get('JEVBENCH_DIR', '')}\n".encode())
        _files_key(h, path)
        _files_key(h, Path(os.environ.get("JEVBENCH_DIR", ROOT / "data" / "jevbench")).expanduser(), "*.jsonl")
        for p in sources:
            h.update(p.read_bytes())
        directory = Path(cache)
        directory.mkdir(parents=True, exist_ok=True)
        _take_slot(directory)
        stem = f"synth-v5-{path.name}"
        target = directory / f"{stem}-{h.hexdigest()[:24]}.pkl"
        with open(directory / f"{stem}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # 并行的 worker 等第一个读完, 不各读一遍
            if not target.exists():
                items = original(data_dir, datasets_dir=datasets_dir)
                tmp = target.with_suffix(f".{os.getpid()}.tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, target)
                for old in directory.glob(f"{stem}-*.pkl"):
                    if old != target:
                        old.unlink(missing_ok=True)
                return items
        with open(target, "rb") as f:
            return pickle.load(f)

    v5.load_synth_v5 = load


_cache_synth_v5()


@pytest.fixture(autouse=True)
def _thread_data_backend(request, monkeypatch):
    if request.node.path.name.startswith("test_prefetch"):
        return
    import sors.training.loop as loop
    original = loop.prefetch_batches

    def prefetch(*args, backend="thread", **kwargs):
        return original(*args, backend="thread" if backend == "process" else backend, **kwargs)

    monkeypatch.setattr(loop, "prefetch_batches", prefetch)


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
