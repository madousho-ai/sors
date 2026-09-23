"""decidophobia.model 的测试. 要加载 0.6B, 走 GPU.

跑:  OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python tests/test_model.py
"""

import torch

from _runner import run
from decidophobia.batch import collate
from decidophobia.data import MenuExample
from decidophobia.loss import slot_cross_entropy
from decidophobia.model import SlotEmbedding, SlotHead, last_logits, prepare_model, trainable_param_groups
from decidophobia.tokens import install_d_tokens, install_type_tokens

MODEL = "Qwen/Qwen3-0.6B-Base"


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    d_ids = install_d_tokens(tok)
    t_ids = install_type_tokens(tok)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    return tok, d_ids + t_ids, lm


def test_trainable_params_are_lora_and_the_259_rows_only():
    """可训参数: LoRA 张量 + 一个 259×1024 的 rows. 整张嵌入矩阵冻死, 总量 < 3M (原先 158M)."""
    tok, ids, lm = _load()
    m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    bad = [n for n in names if "lora_" not in n and not n.endswith(".rows")]
    assert not bad, bad[:5]
    rows = [n for n in names if n.endswith(".rows")]
    assert len(rows) == 1, rows
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_train < 3_000_000, n_train
    assert isinstance(m.get_input_embeddings(), SlotEmbedding)
    assert m.get_input_embeddings().rows.shape == (len(ids), 1024)


def test_gradient_reaches_rows_and_base_embedding_stays_untouched():
    """一步反传: rows 里出现在提示里的行有梯度 (D0/D1 在菜单, <|choice|> 在 Question 行);
    基模嵌入矩阵 requires_grad=False, grad 为 None."""
    tok, ids, lm = _load()
    m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    ex = MenuExample(query="I lost my card", options=[0, 1], gold_idx=0, label=0,
                     option_names=["card lost", "change pin"], qtype="choice")
    b = collate([ex], tok, ids[:256], k_max=2, type_marker=True)
    b = {k: v.to("cuda") for k, v in b.items()}
    logits = last_logits(m, b["input_ids"], b["attention_mask"])
    slot_cross_entropy(logits, b["slot_ids"], b["gold"]).backward()
    emb = m.get_input_embeddings()
    g = emb.rows.grad
    assert g is not None
    assert g[0].abs().max() > 0 and g[1].abs().max() > 0, "D0/D1 在菜单里, 该有梯度"
    assert g[256].abs().max() > 0, "<|choice|> 在 Question 行里, 该有梯度"
    assert g[2:256].abs().max() == 0 and g[257:].abs().max() == 0, "没出现的行梯度为零"
    assert emb.base.weight.grad is None and not emb.base.weight.requires_grad


def test_split_embedding_equals_patched_full_matrix_on_both_ends():
    """拆开的实现必须逐位等于「把 259 行贴回整张矩阵」这个参考实现:
    输入端 embed(ids) 与 head 端 h @ Wᵀ 都要对上. 两端共用同一个 rows."""
    tok, ids, lm = _load()
    m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    emb = m.get_input_embeddings()
    with torch.no_grad():
        emb.rows.normal_(0, 0.05)  # 离开初始化, 让贴回去的行和原行明显不同
        W = emb.base.weight.detach().clone()
        W[ids] = emb.rows.detach().to(W.dtype)
        # 输入端: 普通 token + D 槽 + 类型 token 混着查
        probe = torch.tensor([[13, ids[0], ids[5], ids[256], 100, ids[258]]], device="cuda")
        assert torch.equal(emb(probe), W[probe])
        # 输出端: D 列必须精确等于 h @ rowsᵀ, 其余列精确等于冻结 head 的原输出.
        # 不拿 h @ Wᵀ 整体当参照: nn.Linear 和裸 matmul 在 bf16 下 cuBLAS 可能选不同 kernel,
        # 归约顺序不同会差 1 ulp, 那是数值抖动, 与实现无关.
        h = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16)
        head = m.get_output_embeddings()
        assert isinstance(head, SlotHead)
        out = head(h)
        want_d = (h @ emb.rows.T.to(h.dtype)).to(out.dtype)
        assert torch.equal(out[:, ids], want_d)
        keep = torch.ones(out.shape[1], dtype=torch.bool, device="cuda")
        keep[ids] = False
        assert torch.equal(out[:, keep], head.base(h)[:, keep])


def test_trainable_param_groups_split_lora_from_rows():
    tok, ids, lm = _load()
    m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    groups = trainable_param_groups(m, lr_lora=1e-4, lr_embed=1e-3)
    assert [g["lr"] for g in groups] == [1e-4, 1e-3]
    assert len(groups[1]["params"]) == 1 and groups[1]["params"][0].shape == (len(ids), 1024)
    assert len(groups[0]["params"]) > 0


def test_trainable_d_only_has_no_lora():
    tok, ids, lm = _load()
    m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0, trainable="d-only")
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names == ["model.embed_tokens.rows"], names


def test_trainable_attn_mlp_covers_mlp_projections():
    tok, ids, lm = _load()
    m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0, trainable="attn-mlp")
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any("gate_proj" in n and "lora_" in n for n in names), names[:5]
    assert any("q_proj" in n and "lora_" in n for n in names), names[:5]


def test_grad_checkpointing_gives_same_loss_and_gradients():
    """grad_ckpt=True 反传时重算前向, 只省激活显存; loss 与梯度必须与不开时一致 (bf16 容差)."""
    import random

    from transformers import AutoModelForCausalLM

    tok, ids, _ = _load()
    ex = MenuExample(query="I lost my card and need a replacement", options=[0, 1, 2], gold_idx=1, label=1,
                     option_names=["card lost", "get physical card", "change pin"])
    b = collate([ex], tok, ids[:256], k_max=3)
    b = {k: v.to("cuda") for k, v in b.items()}
    got = {}
    for ckpt in (False, True):
        torch.manual_seed(0)
        lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
        m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0, grad_ckpt=ckpt)
        with torch.no_grad():  # 让 LoRA B 非零, 否则 LoRA 梯度恒零, 比不出东西
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.02)
        m.train()
        loss = slot_cross_entropy(last_logits(m, b["input_ids"], b["attention_mask"]), b["slot_ids"], b["gold"])
        loss.backward()
        lora_g = next(p.grad for n, p in m.named_parameters() if "layers.0.self_attn.q_proj.lora_A" in n)
        got[ckpt] = (loss.item(), lora_g.clone(), m.get_input_embeddings().rows.grad[:3].clone())
        del m, lm
        torch.cuda.empty_cache()
    assert abs(got[False][0] - got[True][0]) < 1e-3, (got[False][0], got[True][0])
    assert torch.allclose(got[False][1], got[True][1], rtol=0.05, atol=1e-4), (got[False][1].abs().max(), (got[False][1] - got[True][1]).abs().max())
    assert torch.allclose(got[False][2], got[True][2], rtol=0.05, atol=1e-4)
    assert got[True][1].abs().max() > 0, "checkpointing 下 LoRA 梯度不得为零"


def test_train_loss_all_slots_moves_offmenu_d_rows_and_menu_leaves_them():
    """train() 跑一步, 菜单只有 D0/D1. cfg.loss='all-slots' 时 D2..D255 在分母里, 这些行必须动;
    cfg.loss='menu' 时它们不在提示里也不在分母里, 梯度为零、weight_decay=0, 必须原样不动."""
    import random

    from decidophobia.train import TrainConfig, train

    ex = MenuExample(query="I lost my card", options=[0, 1], gold_idx=0, label=0,
                     option_names=["card lost", "change pin"])
    moved = {}
    for kind in ("menu", "all-slots"):
        tok, ids, lm = _load()
        m = prepare_model(lm, ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
        rows = m.get_input_embeddings().rows
        before = rows.detach()[2:256].clone()
        cfg = TrainConfig(steps=1, batch_size=1, k_max=2, loss=kind, eval_every=1, log_every=1)
        train(m, tok, ids[:256], lambda n, rng: [ex] * n, {}, cfg)
        moved[kind] = (rows.detach()[2:256] - before).abs().max().item()
        del m, lm
        torch.cuda.empty_cache()
    assert moved["menu"] == 0, moved
    assert moved["all-slots"] > 0, moved


if __name__ == "__main__":
    run(globals())
