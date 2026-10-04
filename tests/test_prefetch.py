"""Threaded CPU preparation: overlap, bounded FIFO delivery and exact resume."""

import copy
import importlib.util
import random
import threading
from dataclasses import replace
from unittest.mock import patch

import torch

from _runner import run
from test_evaluate import _tiny, _Writer
from test_resume import _Sampler, _training_config
from sors.training import loop


def _prefetch():
    assert importlib.util.find_spec("sors.training.prefetch") is not None, "threaded batch preparation is missing"
    from sors.training.prefetch import prefetch_batches
    return prefetch_batches


def _cfg(workers=2, capacity=2, **changes):
    cfg = replace(_training_config(), **changes)
    cfg.data_workers, cfg.data_prefetch = workers, capacity
    return cfg


def test_thread_backend_prepares_batches_off_the_training_thread():
    """A synchronous collate in the training loop must fail this test."""
    tok, ids, model = _tiny()
    calls, original = [], loop.collate

    def observed(*args, **kwargs):
        calls.append(threading.current_thread())
        return original(*args, **kwargs)

    with patch.object(loop, "collate", observed):
        loop.train(model, tok, ids, _Sampler(), {}, _cfg(data_backend="thread"))
    assert calls and all(t is not threading.current_thread() for t in calls), "collate still runs on the training thread"
    assert all(not t.is_alive() for t in calls), "data workers survived training"


def test_two_workers_overlap_but_deliver_original_order_and_isolate_tokenizers():
    """Completion-order delivery, one worker, or a shared tokenizer breaks this."""
    prefetch = _prefetch()
    second_done = threading.Event()
    both_started = threading.Barrier(2)
    workers, tokens, completed, sampled_on = set(), [], [], []
    original = {"owner": None, "vocabulary": [1, 2, 3]}

    def source():
        for value in range(2):
            sampled_on.append(threading.current_thread())
            yield value

    def prepare(value, tok):
        worker = threading.current_thread()
        workers.add(worker)
        tokens.append(tok)
        tok["owner"] = worker.name
        both_started.wait(timeout=5)
        assert tok["owner"] == worker.name, "tokenizer shared across workers"
        if value == 0:
            assert second_done.wait(5), "second batch could not finish while the first was blocked"
        completed.append(value)
        if value == 1:
            second_done.set()
        return value * 10

    with prefetch(source(), prepare, original, workers=2, capacity=2) as batches:
        assert list(batches) == [0, 10]
    assert completed == [1, 0] and len(workers) == 2
    assert all(t is threading.current_thread() for t in sampled_on)
    assert tokens[0] is not tokens[1] and tokens[0] is not original
    assert original == {"owner": None, "vocabulary": [1, 2, 3]}
    assert all(not t.is_alive() for t in workers)


def test_prefetch_is_bounded_and_workers_are_joined_on_early_exit():
    prefetch = _prefetch()
    sampled, workers = [], set()

    def source():
        for i in range(100):
            sampled.append(i)
            yield i

    def prepare(i, tok):
        workers.add(threading.current_thread())
        return i

    with prefetch(source(), prepare, {}, workers=2, capacity=2) as batches:
        assert next(batches) == 0
        assert 1 < len(sampled) <= 3, "unbounded or absent look-ahead"
    assert all(not t.is_alive() for t in workers)
    assert len(sampled) <= 3, "sampling continued after consumer exit"


def test_preparation_and_sampling_errors_arrive_in_batch_order():
    prefetch = _prefetch()
    for fail_in_source in (False, True):
        error = ValueError("bad second batch")
        workers = set()

        def source():
            yield 0
            if fail_in_source:
                raise error
            yield 1
            yield 2

        def prepare(i, tok):
            workers.add(threading.current_thread())
            if i == 1:
                raise error
            return i

        try:
            with prefetch(source(), prepare, {}, workers=2, capacity=2) as batches:
                assert next(batches) == 0, "a future failure hid a valid earlier batch"
                next(batches)
        except ValueError as caught:
            assert caught is error
        else:
            raise AssertionError("background failure was swallowed")
        assert all(not t.is_alive() for t in workers)


def test_consumer_failure_joins_workers_and_stops_source():
    prefetch = _prefetch()
    workers, sampled = set(), []
    error = RuntimeError("training failed")

    def source():
        for i in range(20):
            sampled.append(i)
            yield i

    def prepare(i, tok):
        workers.add(threading.current_thread())
        return i

    try:
        with prefetch(source(), prepare, {}, workers=2, capacity=2) as batches:
            next(batches)
            raise error
    except RuntimeError as caught:
        assert caught is error
    assert all(not t.is_alive() for t in workers)
    assert len(sampled) <= 3


def test_zero_workers_is_lazy_synchronous_and_preserves_tokenizer_identity():
    prefetch = _prefetch()
    original, sampled, prepared = {}, [], []

    def source():
        for i in range(5):
            sampled.append(i)
            yield i

    def prepare(i, tok):
        assert threading.current_thread() is threading.main_thread()
        assert tok is original
        prepared.append(i)
        return i

    with prefetch(source(), prepare, original, workers=0, capacity=2) as batches:
        assert sampled == prepared == []
        assert next(batches) == 0
        assert sampled == prepared == [0]
    assert sampled == prepared == [0]


def test_invalid_prefetch_settings_fail_before_sampling():
    prefetch = _prefetch()
    for workers, capacity in ((-1, 2), (2, 0), (0, 0)):
        sampled = []

        def source():
            sampled.append(True)
            yield 0

        try:
            with prefetch(source(), lambda i, tok: i, {}, workers=workers, capacity=capacity) as batches:
                list(batches)
        except ValueError:
            assert sampled == []
        else:
            raise AssertionError("invalid worker/queue limits accepted")


def test_empty_source_does_no_preparation():
    prefetch = _prefetch()

    def prepare(i, tok):
        raise AssertionError("prepared an empty input")

    for workers in (0, 2):
        with prefetch([], prepare, {}, workers=workers, capacity=2) as batches:
            assert list(batches) == []


def test_threaded_collation_matches_serial_tensors_for_every_architecture():
    prefetch = _prefetch()
    tok, ids, _ = _tiny()
    examples = _Sampler()(3, random.Random(7))
    examples[0] = replace(examples[0], target=[0.6, 0.3, 0.1], codes=[19, 3, 125])
    examples[1] = replace(examples[1], codes=[202, 12, 58])
    for architecture in ("slots", "minimal", "structural", "candidate"):
        def prepare(exs, tokenizer):
            return loop.collate(exs, tokenizer, ids, 5, max_length=512, architecture=architecture)

        expected = prepare(examples, tok)
        with prefetch([examples] * 4, prepare, tok, workers=2, capacity=2) as batches:
            for actual in batches:
                assert actual.keys() == expected.keys()
                for key, value in actual.items():
                    assert value.device.type == "cpu"
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_prefetched_checkpoint_records_consumed_rng_and_resumes_exactly():
    """Saving the producer's advanced RNG skips data; this also checks Adam and dropout."""
    cfg = _cfg(steps=6, eval_every=3, save_every=2)
    tok, ids, reference = _tiny()
    dropout = lambda mod, args, out: torch.nn.functional.dropout(out, p=0.15, training=mod.training)
    reference.get_input_embeddings().register_forward_hook(dropout)
    ref_sampler, ref_states, ref_writer = _Sampler(), [], _Writer()
    loop.train(reference, tok, ids, ref_sampler, {}, _cfg(workers=0, steps=6, eval_every=3, save_every=2),
               writer=ref_writer, on_state=lambda s: ref_states.append(copy.deepcopy(s)))

    tok, ids, first = _tiny()
    first.get_input_embeddings().register_forward_hook(dropout)
    sampler, states, ahead_at_save, writer = _Sampler(), [], [], _Writer()

    def save(value):
        states.append(copy.deepcopy(value))
        ahead_at_save.append(sampler.calls)

    loop.train(first, tok, ids, sampler, {}, cfg, stop_after=5, writer=writer, on_state=save)
    assert ahead_at_save[0] > states[0]["step"], "checkpoint fixture never had prefetched future samples"
    assert states[0]["rng"] == ref_states[0]["rng"], "saved prefetched RNG instead of consumed RNG"

    saved = states[0]  # Complete step 2; batches 3+ were already prepared in the abandoned run.
    tok, ids, resumed = _tiny()
    resumed.get_input_embeddings().register_forward_hook(dropout)
    resumed_sampler, final = _Sampler(), []
    loop.train(resumed, tok, ids, resumed_sampler, {}, _cfg(workers=1, capacity=3, steps=6, eval_every=3, save_every=2),
               resume=saved, on_state=lambda s: final.append(copy.deepcopy(s)))
    assert resumed_sampler.seen == ref_sampler.seen
    for a, b in zip(reference.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(ref_states[-1]["master"], final[-1]["master"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert ref_states[-1]["scheduler"] == final[-1]["scheduler"]
    assert ref_states[-1]["rng"] == final[-1]["rng"]
    assert torch.equal(ref_states[-1]["torch_rng"], final[-1]["torch_rng"])
    for key, values in ref_states[-1]["optimizer"]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, final[-1]["optimizer"]["state"][key][name], rtol=0, atol=0)


def _pid_prepare(item, tok):
    """Module-level so process workers can unpickle it."""
    import os
    if item == "boom":
        raise ValueError("bad item")
    return item, os.getpid(), tok["name"]


def _gone(pid, timeout=5.0):
    import os
    import time
    deadline = time.monotonic() + timeout
    while os.path.exists(f"/proc/{pid}") and time.monotonic() < deadline:
        time.sleep(0.05)
    return not os.path.exists(f"/proc/{pid}")


def test_process_workers_prepare_outside_the_training_process_in_source_order():
    """A thread pool, or completion-order delivery, fails this."""
    import os
    prefetch = _prefetch()
    sampled_on = []

    def source():
        for i in range(6):
            sampled_on.append(threading.current_thread())
            yield i

    with prefetch(source(), _pid_prepare, {"name": "tok"}, workers=2, capacity=3, backend="process") as batches:
        got = list(batches)
    assert [g[0] for g in got] == list(range(6))
    assert all(pid != os.getpid() for _, pid, _ in got), "prepared inside the training process"
    assert all(name == "tok" for *_, name in got), "worker did not receive the tokenizer"
    assert all(t is threading.current_thread() for t in sampled_on), "sampling left the training thread"
    assert all(_gone(pid) for pid in {pid for _, pid, _ in got}), "worker processes survived"


def test_process_worker_errors_arrive_at_their_batch_position_and_early_exit_stops_sampling():
    prefetch = _prefetch()
    sampled = []

    def source():
        for item in (0, "boom", 2, 3, 4, 5, 6, 7):
            sampled.append(item)
            yield item

    try:
        with prefetch(source(), _pid_prepare, {"name": "tok"}, workers=2, capacity=2, backend="process") as batches:
            first, pid, _ = next(batches)
            assert first == 0, "a later failure hid an earlier batch"
            next(batches)
    except ValueError as caught:
        assert "bad item" in str(caught)
    else:
        raise AssertionError("worker failure was swallowed")
    assert len(sampled) <= 4, "sampling ran past the bounded look-ahead"
    assert _gone(pid), "worker process survived the failure"


def test_unknown_backend_fails_before_sampling():
    prefetch = _prefetch()
    sampled = []

    def source():
        sampled.append(True)
        yield 0

    try:
        with prefetch(source(), _pid_prepare, {}, workers=1, capacity=1, backend="fibers") as batches:
            list(batches)
    except ValueError:
        assert sampled == []
    else:
        raise AssertionError("unknown backend accepted")


def test_training_defaults_to_process_workers_and_matches_synchronous_training_bitwise():
    """Thread default, or any change to batches, parameters or RNG, fails this."""
    assert loop.TrainConfig().data_backend == "process"
    results = []
    for workers in (0, 2):
        tok, ids, model = _tiny()
        sampler, states = _Sampler(), []
        loop.train(model, tok, ids, sampler, {}, _cfg(workers=workers, capacity=3, steps=4),
                   on_state=lambda s: states.append(copy.deepcopy(s)))
        results.append((sampler.seen, [p.detach().clone() for p in model.parameters()], states[-1]))
    (seen_a, params_a, state_a), (seen_b, params_b, state_b) = results
    assert seen_a == seen_b
    for a, b in zip(params_a, params_b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert state_a["rng"] == state_b["rng"] and state_a["running"] == state_b["running"]


def _device_prefetch():
    from sors.training.prefetch import device_prefetch
    return device_prefetch


def test_device_prefetch_on_cpu_moves_lazily_and_keeps_extras():
    feed = _device_prefetch()
    pulled = []

    def source():
        for i in range(3):
            pulled.append(i)
            yield {"x": torch.full((2, 3), i)}, f"extra{i}"

    out = feed(source(), torch.device("cpu"), depth=2)
    batch, extra = next(out)
    assert pulled == [0], "CPU training read ahead with nothing to overlap"
    assert extra == "extra0" and torch.equal(batch["x"], torch.zeros(2, 3, dtype=torch.long))
    assert [e for _, e in out] == ["extra1", "extra2"]


def test_device_prefetch_copies_ahead_onto_the_gpu_with_identical_values():
    if not torch.cuda.is_available():
        print("  (skipped: no CUDA)")
        return
    feed = _device_prefetch()
    pulled = []
    expected = [torch.randint(0, 1000, (64, 2048)) for _ in range(5)]

    def source():
        for i, x in enumerate(expected):
            pulled.append(i)
            yield {"input_ids": x, "mask": x > 500}, i

    dev = torch.device("cuda")
    for depth in (0, 1, 3):
        pulled.clear()
        seen = []
        for batch, i in feed(source(), dev, depth=depth):
            assert len(pulled) == min(5, i + 1 + depth), f"depth {depth}: read {len(pulled)} at batch {i}"
            assert batch["input_ids"].device.type == "cuda"
            # Overwrite on the consumer stream: a stream-unsafe feeder would corrupt the next batches.
            ref = batch["input_ids"].sum().item()
            batch["input_ids"].zero_()
            assert ref == expected[i].sum().item()
            assert torch.equal(batch["mask"].cpu(), expected[i] > 500)
            seen.append(i)
        assert seen == list(range(5))


if __name__ == "__main__":
    run(globals())
