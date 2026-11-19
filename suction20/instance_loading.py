"""Ordered, bounded preparation in lightweight persistent spawn workers.

The coordinator owns sampling, augmentation draws and checkpoint state. Workers
receive one validated manifest record at a time, never the parent dataset's
cache, manifest index or model. The same iterator may span optimizer updates.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
import copy
import multiprocessing as mp
from numbers import Integral
import os
from pathlib import Path
import queue
import threading
import traceback
from types import SimpleNamespace


_SPAWN_ENVIRONMENT_LOCK = threading.Lock()
_THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
_WORKER_SPEC = None


def _positive_int(value, name, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@contextmanager
def _spawn_environment(threads):
    """Children inherit thread limits before importing NumPy/Torch in __main__."""
    with _SPAWN_ENVIRONMENT_LOCK:
        previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
        try:
            for name in _THREAD_VARIABLES:
                os.environ[name] = str(threads)
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _worker_initialize(spec, threads):
    global _WORKER_SPEC
    _WORKER_SPEC = spec
    for name in _THREAD_VARIABLES:
        os.environ[name] = str(threads)
    # CPU tensors can be used by helpers, but workers never initialize CUDA.
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def _reader_spec(dataset):
    from .instance_data import InstanceDataset, InstanceAttemptDataset
    from .instance_selection import SourceEvaluationSelection
    base = dataset
    while isinstance(base, SourceEvaluationSelection):
        base = base.dataset
    if isinstance(base, InstanceDataset):
        return {'kind': 'source', 'root': str(base.root.resolve()),
                'training_queries': base.training_queries, 'require_labels': base.require_labels}
    if isinstance(base, InstanceAttemptDataset):
        return {'kind': 'target', 'root': str(base.root.resolve())}
    raise TypeError('Spawn preparation requires InstanceDataset or InstanceAttemptDataset '
                    '(optionally wrapped by SourceEvaluationSelection); use workers=0 for custom datasets')


def _prepare_observation(item, rgb, spec):
    if rgb is not None:
        if spec['kind'] == 'source':
            from .instance_photometry import apply_source_rgb
            observation = apply_source_rgb(item['observation'], rgb)
        else:
            from .instance_training import _apply_rgb
            observation = _apply_rgb(item['observation'], rgb)
        item = dict(item, observation=observation)
    if spec['kind'] == 'source':
        if spec.get('route') == 'a':
            from models.instance_route_a import prepare_source_geometry
            item = dict(item, observation=prepare_source_geometry(
                item['observation'], **spec.get('geometry_options', {})))
        elif spec.get('route') == 'b':
            from models.instance_route_b import prepare_source_inputs_cpu
            # Prepared context/texture arrays remain NumPy through IPC. Phase
            # caches contain only pose identity, never a guessed future step.
            item = dict(item, observation=prepare_source_inputs_cpu(
                item['observation'], sample_image_channels=spec.get('compact_source', False),
                **spec.get('geometry_options', {})))
    return item


def _worker_prepare(payload):
    record, rgb = payload
    spec = _WORKER_SPEC
    if spec is None:
        raise RuntimeError('Instance preparation worker was not initialized')
    from .instance_data import InstanceDataset, InstanceAttemptDataset
    # Reuse the existing, validated per-record reader without reparsing the full
    # manifest in every child or serializing a warm parent cache.
    reader = SimpleNamespace(root=Path(spec['root']), records=[record],
                             training_queries=spec.get('training_queries'),
                             require_labels=spec.get('require_labels', True))
    cls = InstanceDataset if spec['kind'] == 'source' else InstanceAttemptDataset
    item = _prepare_observation(cls.load_uncached(reader, 0), rgb, spec)
    if spec.get('compact_source') and spec['kind'] == 'source':
        if spec['route'] == 'a':
            from models.instance_route_a import compact_prepared_source_item
        else:
            from models.instance_route_b import compact_prepared_source_item
        item = compact_prepared_source_item(item)
    return item


def _worker_loop(spec, threads, requests, results):
    """Own a private result pipe, so abrupt death cannot strand a partial frame.

    Shared multiprocessing result queues can retain another process's write
    descriptor after a worker dies mid-pickle transfer. With one pipe per child
    and the parent's child-end descriptors closed, recv instead receives EOF.
    """
    try:
        _worker_initialize(spec, threads)
        while True:
            try:
                payload = requests.recv()
            except EOFError:
                return
            if payload is None:
                return
            try:
                item = _worker_prepare(payload)
                results.send(('ok', item))
                del item
            except Exception as error:
                detail = traceback.format_exc()
                try:
                    results.send(('error', error, detail))
                except (TypeError, AttributeError):
                    results.send(('error', RuntimeError(str(error)), detail))
    finally:
        requests.close()
        results.close()


class InstancePreparationPool:
    """Persistent spawn pool with a bounded, ordered ``map((index, rgb), ...)``.

    At most ``workers * prefetch`` records/results are in flight. One replacement
    ticket is submitted before yielding a prepared item, allowing CPU work to
    overlap the consumer's GPU computation. The producer may be infinite and is
    advanced only for submitted work. A coordinator can track its yielded
    tickets in a deque, remove one on consumption, and save that small pending
    deque with its sampler/RNG state; no prepared array cache needs checkpointing.

    ``close()`` cancels pending work, terminates children and joins them. Consume
    an iterator from one coordinator thread; concurrent maps are rejected.
    """
    def __init__(self, dataset, *, workers, prefetch=2, route=None,
                 geometry_options=None, thread_limit=1, compact_source=False):
        self.workers = _positive_int(workers, 'workers', allow_zero=True)
        self.prefetch = _positive_int(prefetch, 'prefetch')
        self.thread_limit = _positive_int(thread_limit, 'thread_limit')
        if route not in (None, 'a', 'b'):
            raise ValueError('preparation route must be a, b or None')
        self.dataset = dataset
        self.capacity = self.workers * self.prefetch if self.workers else 1
        self.submitted = self.completed = 0
        self.closed = False
        self._active = False
        self._pool = None
        self._processes = ()
        self._connections = []
        self._available_workers = queue.SimpleQueue()
        if self.workers:
            spec = _reader_spec(dataset)
        else:
            try:
                spec = _reader_spec(dataset)
            except TypeError:
                spec = {'kind': 'custom'}
        if compact_source and route not in ('a', 'b'):
            raise ValueError('Compact source training requires an explicit model route')
        spec.update(route=route, geometry_options=copy.deepcopy(geometry_options or {}),
                    compact_source=bool(compact_source))
        self.reader_spec = spec
        if self.workers:
            # These lightweight coordinator threads only send/receive private
            # pipes; all file reads, augmentation and geometry run in children.
            self._pool = ThreadPoolExecutor(max_workers=self.workers)
            context = mp.get_context('spawn')
            try:
                with _spawn_environment(self.thread_limit):
                    for index in range(self.workers):
                        child_read, parent_write = context.Pipe(duplex=False)
                        parent_read, child_write = context.Pipe(duplex=False)
                        process = context.Process(target=_worker_loop,
                            args=(spec, self.thread_limit, child_read, child_write), daemon=True)
                        try:
                            process.start()
                        except BaseException:
                            parent_read.close()
                            parent_write.close()
                            raise
                        finally:
                            child_read.close()
                            child_write.close()
                        self._processes += (process,)
                        self._connections.append((parent_write, parent_read))
                        self._available_workers.put(index)
            except BaseException:
                self.close()
                raise

    def _ticket(self, ticket):
        if not isinstance(ticket, (tuple, list)) or len(ticket) != 2:
            raise ValueError('preparation ticket must be (dataset_index, rgb_parameters_or_None)')
        index, rgb = ticket
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.dataset):
            raise IndexError('preparation dataset index is out of range')
        return int(index), rgb

    def map(self, tickets):
        if self.closed:
            raise RuntimeError('Instance preparation pool is closed')
        if self._active:
            raise RuntimeError('Only one preparation iterator may be active per pool')
        self._active = True
        return self._iterate(tickets)

    def _iterate(self, tickets):
        pending = deque()
        try:
            if not self.workers:
                read = getattr(self.dataset, 'load_uncached', self.dataset.__getitem__)
                for ticket in tickets:
                    if self.closed:
                        raise RuntimeError('Instance preparation pool is closed')
                    index, rgb = self._ticket(ticket)
                    self.submitted += 1
                    result = _prepare_observation(read(index), rgb, self.reader_spec)
                    self.completed += 1
                    yield result
                return
            iterator = iter(tickets)
            exhausted = False
            while len(pending) < self.capacity:
                try:
                    ticket = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(self._submit(ticket))
            while pending:
                if self.closed:
                    raise RuntimeError('Instance preparation pool is closed')
                result = self._result(pending.popleft())
                if not exhausted:
                    try:
                        ticket = next(iterator)
                    except StopIteration:
                        exhausted = True
                    else:
                        pending.append(self._submit(ticket))
                self.completed += 1
                yield result
        except BaseException:
            self.close()
            raise
        finally:
            self._active = False

    def _submit(self, ticket):
        index, rgb = self._ticket(ticket)
        # The selected wrapper's records already have its subset order. Only
        # this small row and deterministic augmentation parameters cross IPC.
        payload = (copy.deepcopy(self.dataset.records[index]), rgb)
        self.submitted += 1
        return self._pool.submit(self._request, payload)

    def _request(self, payload):
        index = self._available_workers.get()
        try:
            process = self._processes[index]
            if self.closed or not process.is_alive():
                raise RuntimeError('An instance preparation worker exited before returning its record')
            requests, results = self._connections[index]
            try:
                requests.send(payload)
                response = results.recv()
            except (EOFError, BrokenPipeError, ConnectionError, OSError) as error:
                raise RuntimeError('An instance preparation worker exited before returning its record') from error
            if response[0] == 'error':
                error, detail = response[1:]
                if hasattr(error, 'add_note'):
                    error.add_note('Instance preparation worker traceback:\n' + detail)
                raise error
            return response[1]
        finally:
            self._available_workers.put(index)

    def _result(self, future):
        # Private child pipes let EOF interrupt even a partially transmitted
        # large NumPy result, rather than stranding a shared-queue receiver.
        while True:
            if self.closed or any(process.exitcode is not None for process in self._processes):
                raise RuntimeError('An instance preparation worker exited before returning its record')
            try:
                return future.result(timeout=1.)
            except TimeoutError:
                if self.closed or any(process.exitcode is not None for process in self._processes):
                    raise RuntimeError('An instance preparation worker exited before returning its record')

    @property
    def worker_pids(self):
        """Actual child PIDs for bounded resource measurements, not identity."""
        return tuple(process.pid for process in self._processes)

    def close(self):
        if not self.closed:
            self.closed = True
            if self._pool is not None:
                pool = self._pool
                self._pool = None
                for process in self._processes:
                    if process.is_alive():
                        process.terminate()
                pool.shutdown(wait=True, cancel_futures=True)
                for process in self._processes:
                    process.join()
                for requests, results in self._connections:
                    requests.close()
                    results.close()

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Instance preparation pool is closed')
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
