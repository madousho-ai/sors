"""Bounded, ordered CPU preparation; sampling stays on the consuming thread."""

from __future__ import annotations

import copy
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from queue import SimpleQueue

import torch


def validate_data_preparation(workers: int, capacity: int) -> None:
    if workers < 0 or capacity < 1:
        raise ValueError("data_workers must be nonnegative and data_prefetch must be positive")


@contextmanager
def prefetch_batches(source, prepare, tokenizer, *, workers: int = 2, capacity: int = 2):
    """Yield an iterator of prepare(item, worker_tokenizer), in source order.

    The caller advances source serially, preserving stateful samplers. At most
    capacity future batches are retained in addition to the consumed batch.
    Each worker owns a tokenizer clone, isolating HF tokenizer configuration
    from other workers and from evaluation on the training thread. CPU-only
    preparation must treat examples as read-only and use no global RNG.

    workers=0 is the original lazy, synchronous path. Errors are delivered at
    their batch position. Exiting cancels queued work and joins active workers.
    """
    validate_data_preparation(workers, capacity)
    source = iter(source)
    if workers == 0:
        yield map(lambda item: prepare(item, tokenizer), source)
        return

    count = min(workers, capacity)
    tokenizers = SimpleQueue()
    # Clone on the caller before workers start; evaluation can then keep using
    # its own tokenizer while future batches are prepared on isolated copies.
    for _ in range(count):
        tokenizers.put(copy.deepcopy(tokenizer))
    local = threading.local()

    def initialize():
        local.tokenizer = tokenizers.get()

    def work(item):
        with torch.no_grad(), torch.device("cpu"):
            return prepare(item, local.tokenizer)

    pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="sors-data", initializer=initialize)
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
                pending.append(pool.submit(work, item))

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
