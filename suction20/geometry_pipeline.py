"""Two in-flight microbatches, with CPU geometry on one background executor.

GPU preparation, scoring, loss, backward and optimizer work stay on the caller
thread. The executor is only given a detached CPU job. Each iterator contains
one accumulation window, so a later window never uses pre-update parameters.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Callable

import torch


@dataclass
class GeometryCompletion:
    payload: Any
    geometry_result: Any = None
    timing: dict | None = None


def _run_geometry(job):
    started = time.perf_counter()
    with torch.profiler.record_function("geometry.cpu_job"):
        result = job()
    ended = time.perf_counter()
    return result, {"geometry_job_wall_s": ended - started,
                    "geometry_started_monotonic": started,
                    "geometry_ended_monotonic": ended}


class GeometryPipeline:
    """Bound graph residency to two; native geometry manages its own threads."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="contact-geometry")

    def iter_update(self, count: int, prepare: Callable):
        """Yield ready work in order; the caller finishes/backwards each item.

        ``prepare(index)`` runs on the caller and returns ``(payload, cpu_job)``.
        Fill two slots once, then refill a slot only after the caller has
        finished the yielded work. The caller must release each payload before
        requesting the next result to retain the two-graph memory bound.
        """
        pending = deque()
        next_index = 0
        try:
            while next_index < count or pending:
                while next_index < count and len(pending) < 2:
                    payload, job = prepare(next_index)
                    pending.append((payload, self.executor.submit(_run_geometry, job)))
                    next_index += 1
                    del payload, job
                payload, future = pending.popleft()
                waiting = time.perf_counter()
                result, timing = future.result()
                timing["geometry_wait_host_s"] = time.perf_counter() - waiting
                yield GeometryCompletion(payload, result, timing)
                del payload, future, result, timing
        finally:
            # On a caller error, queued jobs need not run. Running native work
            # is joined by close(); no optimizer update crosses this boundary.
            for _, future in pending:
                future.cancel()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
