"""温度闸. 这台笔记本的 CPU 散热余量小: 一次 20 秒的 smoke run 就从 54°C 到 78°C,
之前一次 CPU 满载的分析把机器热死机过. 训练循环每步之前问一次, 超阈值就等.

读数走 k10temp 的 Tctl (AMD 的控制温度, 风扇曲线看的就是它).
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable

_HWMON = pathlib.Path("/sys/class/hwmon")


def _find_tctl() -> pathlib.Path | None:
    for d in sorted(_HWMON.glob("hwmon*")):
        try:
            if (d / "name").read_text().strip() == "k10temp":
                return d / "temp1_input"
        except OSError:
            continue
    return None


def read_tctl() -> float | None:
    p = _find_tctl()
    if p is None:
        return None
    try:
        return int(p.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


class ThermalGuard:
    def __init__(
        self,
        max_c: float,
        cooldown_s: float,
        read: Callable[[], float | None] = read_tctl,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.max_c = max_c
        self.cooldown_s = cooldown_s
        self.read = read
        self.sleep = sleep

    def wait(self) -> int:
        """温度高于 max_c 就睡 cooldown_s 再读, 直到降下来. 返回睡了几次. 没传感器时直接放行."""
        n = 0
        while True:
            t = self.read()
            if t is None or t <= self.max_c:
                return n
            self.sleep(self.cooldown_s)
            n += 1
