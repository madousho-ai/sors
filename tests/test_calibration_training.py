"""Training calibration shares the bulk metric readback and existing JSON/TensorBoard writes."""

import json

import pytest
import torch

from test_evaluate import _Writer, _tiny
from test_sync_reduction import Operations
from sors.core.batch import collate
from sors.core.menu import MenuExample
from sors.core.model import last_logits
from sors.evaluation.scoring import EvalSet
from sors.training.loop import TrainConfig, _metric_tensor, _prepare_batch, train
from sors.training.loss import gather_slot_logits


def test_training_confidence_is_packed_on_device_with_the_existing_metrics():
    logits = torch.tensor([[0.6, 0.3, 0.1], [0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.6, 0.4, 0.0]],
                          dtype=torch.float64).log().requires_grad_(True)
    slots = torch.tensor([[0, 1, 2], [0, 1, -1], [0, 1, -1], [0, 1, -1]])
    target = torch.tensor([[1., 0., 0.]] * 4)
    with Operations() as operations:
        got = _metric_tensor(logits.new_tensor(0.25), None, logits, slots, target,
                             noul_mask=torch.tensor([False, True, True, True]))
    assert got[4:].tolist() == pytest.approx([2.8, 4, 2, 3])
    assert not got.requires_grad
    assert operations.counts["aten._local_scalar_dense.default"] == 0
    assert operations.counts["bool_index"] == 0


def _mixed_questions():
    return [
        MenuExample("short", [0, 1, 2], 0, 0, ["a", "b", "c"]),
        MenuExample("a longer yes/no question", [1, 0], 1, 0, ["yes", "no"], qtype="bool"),
        MenuExample("middle length", [2, 0, 1], 2, 1, ["high", "low", "middle"], qtype="score"),
    ]


def test_boolean_row_mask_is_prepared_on_cpu_for_the_existing_batch_transfer():
    tok, ids, _ = _tiny()
    batch, _, _, _ = _prepare_batch((ids, 3, "context-first", 512, False, False, "slots"),
                                    (_mixed_questions(), None), tok)
    assert batch["noul_mask"].device.type == "cpu"
    assert batch["noul_mask"].dtype == torch.bool
    assert batch["noul_mask"].tolist() == [False, True, False]


@pytest.mark.parametrize("accumulate", [False, True])
def test_training_logs_batch_confidence_over_the_whole_window_without_extra_forwards(tmp_path, accumulate):
    tok, ids, model = _tiny()
    examples = _mixed_questions()
    batch = collate(examples, tok, ids, k_max=3)
    with torch.no_grad():
        q = gather_slot_logits(last_logits(model, batch["input_ids"], batch["attention_mask"]), batch["slot_ids"]).softmax(-1)
    mean = q.max(-1).values.double().mean().item()
    decisive = float(q[1].max() >= 0.8)
    forwards = []
    model.get_input_embeddings().register_forward_pre_hook(lambda mod, args: forwards.append(len(args[0])))
    writer, path = _Writer(), tmp_path / "log.jsonl"
    cfg = TrainConfig(steps=2, batch_size=3, k_max=3, loss="menu", eval_every=2, log_every=2, micro_batches=2,
                      accumulate_gradients=accumulate, lr_lora=0.0, lr_embed=0.0, data_workers=0)
    hist = train(model, tok, ids, lambda n, rng: list(examples), {}, cfg, log_path=path, writer=writer)
    values = {(tag, step): value for tag, value, step in writer.scalars}
    assert values["train/conf_mean", 2] == pytest.approx(mean)
    assert values["train/noul_decisive_rate", 2] == decisive
    assert sum(forwards) == 6 and len(forwards) == 4
    assert [(r["step"], r["train_conf_n"], r["train_noul_n"]) for r in hist] == [(0, 0, 0), (2, 6, 2)]
    assert hist[0]["train_conf_mean"] is None and hist[0]["train_noul_decisive_rate"] is None
    assert hist[-1]["train_conf_mean"] == pytest.approx(mean)
    assert hist[-1]["train_noul_decisive_rate"] == decisive
    assert [json.loads(line) for line in path.read_text().splitlines()] == hist


def test_eval_calibration_is_written_to_tensorboard_and_jsonl_for_all_eval_paths(tmp_path):
    tok, ids, model = _tiny()
    writer, path = _Writer(), tmp_path / "log.jsonl"
    hist = train(model, tok, ids, lambda n, rng: [], {"mixed": EvalSet(_mixed_questions(), batch_size=3)},
                 TrainConfig(steps=0, k_max=3, probe_passes=1), log_path=path, writer=writer)
    assert hist[0]["train_conf_mean"] is None
    values = {tag: value for tag, value, _ in writer.scalars}
    record = json.loads(path.read_text())
    for group in ("consistency", "eval", "consistency_full"):
        for suffix in ("conf_mean", "accuracy_minus_confidence", "choice/ece", "noul/decisive_rate", "score/nrps",
                       "score/interval90_coverage", "score/interval90_width"):
            node = record[group]["mixed"]["calib"]
            for key in suffix.split("/"):
                node = node[key]
            assert values[f"{group}/mixed/calib/{suffix}"] == node
