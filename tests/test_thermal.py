"""sors.training.thermal 的测试.

跑:  PYTHONPATH=src .venv/bin/python tests/test_thermal.py
"""

import sys

from sors.training.thermal import ThermalGuard


def test_guard_sleeps_while_hot_and_returns_count():
    """读数 85, 85, 70, 阈值 80: 睡两次, 第三次读到 70 放行, 返回 2."""
    reads = iter([85.0, 85.0, 70.0])
    slept = []
    g = ThermalGuard(read=lambda: next(reads), sleep=slept.append, max_c=80.0, cooldown_s=5.0)
    n = g.wait()
    assert n == 2, n
    assert slept == [5.0, 5.0], slept


def test_guard_passes_immediately_when_cool():
    slept = []
    g = ThermalGuard(read=lambda: 60.0, sleep=slept.append, max_c=80.0, cooldown_s=5.0)
    assert g.wait() == 0 and slept == []


def test_guard_is_disabled_when_reader_gives_none():
    """没有传感器 (read 返回 None) 时不阻塞, 也不报错."""
    slept = []
    g = ThermalGuard(read=lambda: None, sleep=slept.append, max_c=80.0, cooldown_s=5.0)
    assert g.wait() == 0 and slept == []


def test_default_reader_returns_float_or_none():
    """真机上的默认读数: 有 k10temp 就是个合理的摄氏度, 没有就是 None."""
    g = ThermalGuard(max_c=80.0, cooldown_s=5.0)
    t = g.read()
    assert t is None or 10.0 < t < 120.0, t


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{failed} failed")
    sys.exit(1 if failed else 0)
