"""sors.core.vision.rope_positions: 一段 token (文字里夹着图片占位符) 的 M-RoPE 三维位置.

文字 token 三个坐标相同, 每个 +1; 一张图的占位符按合并后的网格排行列, 整张图走完位置只前进 max(高, 宽) 格.
期望值是照 Qwen3.5 的规则手算的; 另有一条拿 transformers 的 get_rope_index 当参照, 对随机排布逐位比.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from sors.core.vision import rope_positions

IMG = 99


def test_text_only_counts_up_from_start_on_all_three_axes():
    pos, nxt = rope_positions([5, 6, 7], [], IMG, start=4)
    assert pos.tolist() == [[4, 5, 6]] * 3
    assert nxt == 7


def test_image_tokens_lie_on_the_merged_grid_and_advance_by_its_longer_side():
    # 网格 (1, 4, 6) 合并 2x2 后是 2 行 3 列 = 6 个占位符; 图从位置 2 起, 走完前进 max(2, 3) = 3 格
    ids = [5, 5, *[IMG] * 6, 5]
    pos, nxt = rope_positions(ids, [(1, 4, 6)], IMG)
    assert pos.tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5],
        [0, 1, 2, 2, 2, 3, 3, 3, 5],
        [0, 1, 2, 3, 4, 2, 3, 4, 5],
    ]
    assert nxt == 6


def test_adjacent_images_each_take_their_own_grid():
    ids = [*[IMG] * 4, *[IMG] * 2]  # (1,4,4) -> 2x2, 接着 (1,2,4) -> 1x2
    pos, nxt = rope_positions(ids, [(1, 4, 4), (1, 2, 4)], IMG, start=10)
    assert pos.tolist() == [
        [10, 10, 10, 10, 12, 12],
        [10, 10, 11, 11, 12, 12],
        [10, 11, 10, 11, 12, 13],
    ]
    assert nxt == 14


@pytest.mark.parametrize("ids, grids", [
    ([IMG] * 6, []),  # 占位符比图多
    ([5], [(1, 4, 6)]),  # 图比占位符多
    ([*[IMG] * 5, 5], [(1, 4, 6)]),  # 一张图的占位符不够数
])
def test_placeholders_that_disagree_with_the_grids_are_rejected(ids, grids):
    with pytest.raises(ValueError):
        rope_positions(ids, grids, IMG)


def test_matches_transformers_get_rope_index_on_random_layouts():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    owner = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    owner.get_vision_position_ids = Qwen3_5Model.get_vision_position_ids.__get__(owner)
    rng = random.Random(3)
    for _ in range(20):
        ids, grids = [], []
        for _ in range(rng.randint(1, 4)):
            # 参照实现按连续的同类 token 分段, 两张图之间至少隔一个文字 token (实际提示里是 vision_end / vision_start)
            ids += [rng.randint(0, 50) for _ in range(rng.randint(1, 5))]
            if rng.random() < 0.7:
                g = (1, 2 * rng.randint(1, 4), 2 * rng.randint(1, 4))
                grids.append(g)
                ids += [IMG] * (g[1] * g[2] // 4)
        if not ids:
            continue
        t = torch.tensor([ids])
        want, _ = Qwen3_5Model.get_rope_index(owner, t, (t == IMG).int(),
                                              image_grid_thw=torch.tensor(grids) if grids else None)
        got, nxt = rope_positions(ids, grids, IMG)
        assert got.tolist() == want[:, 0].tolist()
        assert nxt == int(want.max()) + 1
