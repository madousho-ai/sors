"""Bounded, ordered CPU preparation; sampling stays on the consuming thread.

Two worker backends prepare batches ahead of the training loop:
  process  forkserver worker processes, each holding its own tokenizer. Preparation
           is mostly Python, so worker threads would contend for the GIL with the
           training thread that launches GPU kernels; processes do not.
  thread   worker threads in the training process (the original backend).
device_prefetch then copies prepared batches onto the GPU ahead of use.
"""

from __future__ import annotations

import copy
import multiprocessing
import pickle
import threading
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from queue import SimpleQueue

import torch

BACKENDS = ("process", "thread")


def validate_data_preparation(workers: int, capacity: int, backend: str = "thread") -> None:
    if workers < 0 or capacity < 1:
        raise ValueError("data_workers must be nonnegative and data_prefetch must be positive")
    if backend not in BACKENDS:
        raise ValueError(f"data_backend must be one of {BACKENDS}, got {backend!r}")


_worker = threading.local()  # process backend: this worker's prepare and tokenizer


def _process_initialize(prepare, tokenizer):
    torch.set_num_threads(1)  # many workers; small integer tensor ops only
    _worker.prepare, _worker.tokenizer = prepare, tokenizer


def _process_work(item):
    with torch.no_grad():
        return _worker.prepare(item, _worker.tokenizer)


def _thread_pool(tokenizer, count):
    tokenizers = SimpleQueue()
    # Clone on the caller before workers start; evaluation can then keep using
    # its own tokenizer while future batches are prepared on isolated copies.
    for _ in range(count):
        tokenizers.put(copy.deepcopy(tokenizer))
    local = threading.local()

    def initialize():
        local.tokenizer = tokenizers.get()

    pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="sors-data", initializer=initialize)
    return pool, local


@contextmanager
def prefetch_batches(source, prepare, tokenizer, *, workers: int = 2, capacity: int = 2, backend: str = "thread"):
    """Yield an iterator of prepare(item, worker_tokenizer), in source order.

    The caller advances source serially, preserving stateful samplers. At most
    capacity future batches are retained in addition to the consumed batch.
    Each worker owns a tokenizer copy, isolating HF tokenizer configuration
    from other workers and from evaluation on the training thread. CPU-only
    preparation must treat examples as read-only and use no global RNG.
    The process backend needs a picklable prepare (module-level function or
    functools.partial of one); items and results cross process boundaries.

    workers=0 is the original lazy, synchronous path. Errors are delivered at
    their batch position. Exiting cancels queued work and joins all workers.
    """
    validate_data_preparation(workers, capacity, backend)
    source = iter(source)
    if workers == 0:
        yield map(lambda item: prepare(item, tokenizer), source)
        return

    count = min(workers, capacity)
    if backend == "process":
        try:
            pickle.dumps(prepare)
        except Exception as error:
            raise ValueError("the process data backend needs a picklable prepare function") from error
        pool = ProcessPoolExecutor(count, mp_context=multiprocessing.get_context("forkserver"),
                                   initializer=_process_initialize, initargs=(prepare, tokenizer))
        submit = lambda item: pool.submit(_process_work, item)
    else:
        pool, local = _thread_pool(tokenizer, count)

        def work(item):
            with torch.no_grad(), torch.device("cpu"):
                return prepare(item, local.tokenizer)

        submit = lambda item: pool.submit(work, item)
    pending = deque()
    exhausted = False

    def fill():
        nonlocal exhausted
        while not exhausted and len(pending) < capacity:
            try:
                item = next(source)
            except StopIteration:
                exhausted = True
            except Exception as error:
                # A failure in a later sample must surface after earlier batches.
                failed = Future()
                failed.set_exception(error)
                pending.append(failed)
                exhausted = True
            else:
                pending.append(submit(item))

    def consume():
        fill()
        while pending:
            result = pending.popleft().result()
            fill()
            yield result

    batches = consume()
    try:
        yield batches
    finally:
        batches.close()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        pending.clear()


def device_prefetch(source, device: torch.device, depth: int = 1):
    """Yield (batch on device, *extras) from source items (cpu batch dict, *extras).

    On CUDA, the next depth batches are pinned and copied on a side stream while
    the current step runs; the consumer stream waits only for its own batch.
    Values are identical to a plain .to(device). On other devices batches move
    lazily, one at a time.
    """
    source = iter(source)
    if device.type != "cuda":
        for batch, *extras in source:
            yield ({k: v.to(device) for k, v in batch.items()}, *extras)
        return

    stream = torch.cuda.Stream(device)
    pending = deque()

    def issue():
        try:
            batch, *extras = next(source)
        except StopIteration:
            return False
        with torch.cuda.stream(stream):
            moved = {k: v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
        pending.append((moved, extras))
        return True

    while len(pending) < depth + 1 and issue():
        pass
    while pending:
        moved, extras = pending.popleft()
        current = torch.cuda.current_stream(device)
        current.wait_stream(stream)
        for value in moved.values():
            value.record_stream(current)  # memory allocated on the side stream, used on this one
        yield (moved, *extras)
        while len(pending) < depth + 1 and issue():
            pass
