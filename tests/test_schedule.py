"""decidophobia.training.schedule 的测试.

跑:  PYTHONPATH=src .venv/bin/python tests/test_schedule.py
"""

import sys

from decidophobia.training.schedule import lr_scale


def test_cosine_warmup_ramps_then_decays_to_zero():
    """warmup 100, total 1000:
    step 0 -> 0.0; step 50 -> 0.5; step 100 -> 1.0
    step 550 (cosine 中点) -> 0.5; step 1000 -> 0.0
    """
    f = lambda s: lr_scale(s, warmup=100, total=1000, kind="cosine")  # noqa: E731
    assert abs(f(0) - 0.0) < 1e-12
    assert abs(f(50) - 0.5) < 1e-12
    assert abs(f(100) - 1.0) < 1e-12
    assert abs(f(550) - 0.5) < 1e-9, f(550)
    assert abs(f(1000) - 0.0) < 1e-12


def test_constant_is_one_everywhere():
    for s in [0, 1, 500, 1000]:
        assert lr_scale(s, warmup=100, total=1000, kind="constant") == 1.0


def test_zero_warmup_starts_at_full_lr():
    assert lr_scale(0, warmup=0, total=1000, kind="cosine") == 1.0


def test_beyond_total_stays_at_zero():
    assert lr_scale(1500, warmup=100, total=1000, kind="cosine") == 0.0


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
