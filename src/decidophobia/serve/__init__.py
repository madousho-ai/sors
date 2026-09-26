"""推理服务: 照 TypeSafe 的 System One API (POST /v1/systemone, GET /v1/models) 回答请求.

api      请求体的校验, 以及把模型的菜单分布写成线上格式的答案
menus    一道 API 问题 -> 模型看到的菜单样本, 模板与训练时相同
engine   载入训好的存档; state 算一次 KV cache, 各问题的分支分组接在后面前向
"""
