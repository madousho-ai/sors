"""数据集原始包的按需下载: 盘上没有就下载, 下载回来的和盘上已有的都核对 md5.

每个数据集的读取函数在第一次用到它时调 fetch, 所以只下要用的那几个. 下载先写到同目录的 .part,
md5 对上才改名成正式文件名; 下到一半断掉或下回来的不是那个文件, 都不留任何东西, 下次再用时重新下.
盘上已有的文件 md5 不对就报错, 不自己重下 —— 那多半是手动放进去的另一个版本, 删掉它才会重新下载.
"""

from __future__ import annotations

import hashlib
import pathlib
import urllib.request


def _md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(path, url: str, md5: str) -> pathlib.Path:
    """path 处的原始包, 没有就从 url 下载. 返回 path."""
    path = pathlib.Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        try:
            urllib.request.urlretrieve(url, part)
            got = _md5(part)
            if got != md5:
                raise RuntimeError(f"{url}: downloaded md5 {got} != {md5}")
            part.rename(path)
        finally:
            part.unlink(missing_ok=True)
    got = _md5(path)
    if got != md5:
        raise RuntimeError(f"{path}: md5 {got} != {md5}; delete it to download again")
    return path
