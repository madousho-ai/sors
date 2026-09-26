"""意图数据集 (Banking77 / MASSIVE) 的菜单上显示什么. 类 id 与题目不受影响, 只换选项的文字.

  raw   原始 label 名, 一字不改: 'Refund_not_showing_up', 'reverted_card_payment?', 'iot_hue_lightchange'
  desc  datasets/label-descriptions/<数据集>.json 里的 description: 照 TypeSafe (jev) 文档里人写选项描述的样子,
        首字母大写的话题名词短语, 不带句号, 不出现 I/my/you 这类人称 (如 'Card payment still showing as pending');
        菜单上只有这句话、没有标签名, 所以每句自成一体, 写明与近邻类的分界 (只照 train 写, 见那两个文件的提交说明)
"""

from __future__ import annotations

import json
import pathlib

LABEL_STYLES = ("raw", "desc")
DESC_DIR = pathlib.Path(__file__).resolve().parents[3] / "datasets" / "label-descriptions"


def label_names(dataset: str, raw_names: list[str], style: str) -> dict[int, str]:
    """类 id (raw_names 的下标) -> 菜单上显示的文字. desc 缺了哪个类就报错, 不静默退回原名."""
    if style == "raw":
        return dict(enumerate(raw_names))
    if style == "desc":
        desc = json.loads((DESC_DIR / f"{dataset}.json").read_text(encoding="utf-8"))
        missing = [n for n in raw_names if n not in desc]
        if missing:
            raise ValueError(f"{DESC_DIR / f'{dataset}.json'} has no description for {missing}")
        return {i: desc[n] for i, n in enumerate(raw_names)}
    raise ValueError(f"unknown label style {style!r}; expected one of {LABEL_STYLES}")
