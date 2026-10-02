"""sors.data.download.fetch 的测试: 原始包缺了就下载, 下载与盘上的文件都核对 md5.

跑:  PYTHONPATH=src .venv/bin/python tests/test_download.py
不联网: urllib.request.urlretrieve 换成往目标路径写字节的假函数.
"""

import hashlib
import pathlib
import tempfile
import urllib.request

from _runner import run
from sors.data.download import fetch

BODY = b"the archive"
MD5 = hashlib.md5(BODY).hexdigest()
URL = "https://example.org/pkg/archive.tgz"


def _patched(fake):
    """urlretrieve 换成 fake 的上下文; 退出时换回来."""

    class _Ctx:
        def __enter__(self):
            self.real, urllib.request.urlretrieve = urllib.request.urlretrieve, fake

        def __exit__(self, *exc):
            urllib.request.urlretrieve = self.real

    return _Ctx()


def _writer(body, calls):
    def fake(url, dest):
        calls.append((url, pathlib.Path(dest)))
        pathlib.Path(dest).write_bytes(body)

    return fake


def test_a_missing_file_is_downloaded_into_place_and_its_directory_made():
    calls = []
    with tempfile.TemporaryDirectory() as d, _patched(_writer(BODY, calls)):
        path = pathlib.Path(d) / "sub" / "archive.tgz"
        assert fetch(path, URL, MD5) == path
        assert path.read_bytes() == BODY
        assert [u for u, _ in calls] == [URL]
        assert calls[0][1] != path, "先下到旁边的临时文件, 下完再改名"
        assert sorted(p.name for p in path.parent.iterdir()) == ["archive.tgz"], "临时文件改名后不留"


def test_a_file_already_on_disk_is_not_downloaded_again():
    calls = []
    with tempfile.TemporaryDirectory() as d, _patched(_writer(BODY, calls)):
        path = pathlib.Path(d) / "archive.tgz"
        path.write_bytes(BODY)
        assert fetch(path, URL, MD5) == path
    assert calls == []


def test_a_file_on_disk_with_the_wrong_md5_is_refused():
    calls = []
    with tempfile.TemporaryDirectory() as d, _patched(_writer(BODY, calls)):
        path = pathlib.Path(d) / "archive.tgz"
        path.write_bytes(b"something else")
        try:
            fetch(path, URL, MD5)
        except RuntimeError as e:
            assert "md5" in str(e) and str(path) in str(e), e
        else:
            raise AssertionError("wrong md5 was accepted")
    assert calls == [], "盘上已有的文件不对时照实报错, 不自作主张重下"


def test_a_download_with_the_wrong_md5_is_refused_and_leaves_nothing_behind():
    """下回来的不是那个文件 (比如一页报错的 HTML): 报错, 而且不落盘, 下次再用时重新下载."""
    with tempfile.TemporaryDirectory() as d, _patched(_writer(b"<html>not found</html>", [])):
        path = pathlib.Path(d) / "archive.tgz"
        try:
            fetch(path, URL, MD5)
        except RuntimeError as e:
            assert "md5" in str(e) and URL in str(e), e
        else:
            raise AssertionError("wrong md5 was accepted")
        assert list(pathlib.Path(d).iterdir()) == []


def test_a_download_that_breaks_off_leaves_nothing_behind():
    def fake(url, dest):
        pathlib.Path(dest).write_bytes(BODY[:3])
        raise ConnectionResetError("peer went away")

    with tempfile.TemporaryDirectory() as d, _patched(fake):
        path = pathlib.Path(d) / "archive.tgz"
        try:
            fetch(path, URL, MD5)
        except ConnectionResetError:
            pass
        else:
            raise AssertionError("the error was swallowed")
        assert list(pathlib.Path(d).iterdir()) == []


if __name__ == "__main__":
    run(globals())
