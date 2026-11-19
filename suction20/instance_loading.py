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
_WORKER_STATIC_CACHE = None


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
    global _WORKER_SPEC, _WORKER_STATIC_CACHE
    _WORKER_SPEC = spec
    from .instance_preparation_cache import StaticGeometryCache
    _WORKER_STATIC_CACHE = StaticGeometryCache(spec.get('static_cache_bytes', 0))
    from .instance_selection_cache import reset_reader_cache_stats
    from .instance_static_store import reset_static_store_stats
    reset_reader_cache_stats()
    reset_static_store_stats()
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


def _read_or_prepare_static(item, spec, key, prepare):
    root = spec.get('static_geometry_root')
    if root:
        from .instance_static_store import read_static_geometry
        source = Path(item['record']['observation_npz'])
        if not source.is_absolute():
            source = Path(spec['root']) / source
        value = read_static_geometry(root, source, key=key, route=spec['route'])
        if value is not None:
            return value
    return prepare()


def _prepare_observation(item, rgb, spec, *, static_cache=None, static_key=None):
    route = spec.get('route')
    compact = bool(spec.get('compact_source')) and spec['kind'] == 'source'
    options = spec.get('geometry_options', {})
    static = None
    if compact and route == 'b' and static_cache is not None:
        from models.instance_route_b import prepare_source_static_inputs
        static = static_cache.get_or_prepare(static_key, lambda: _read_or_prepare_static(
            item, spec, static_key, lambda: prepare_source_static_inputs(
                item['observation'], sample_image_channels=True, **options)))
    if rgb is not None:
        if spec['kind'] == 'source':
            from .instance_photometry import apply_source_rgb
            observation = apply_source_rgb(item['observation'], rgb,
                                           prepare_support=not (compact and route == 'b'))
        else:
            from .instance_training import _apply_rgb
            observation = _apply_rgb(item['observation'], rgb)
        item = dict(item, observation=observation)
    if spec['kind'] == 'source':
        if route == 'a':
            from models.instance_route_a import prepare_source_geometry, prepare_source_static_geometry
            if compact and static_cache is not None:
                # RGB-only augmentation preserves geometry and validity. The
                # cached object excludes colors; bind this ticket's RGB below.
                key = (static_key, rgb is not None)
                static = static_cache.get_or_prepare(key, lambda: _read_or_prepare_static(
                    item, spec, key, lambda: prepare_source_static_geometry(item['observation'], **options)))
            item = dict(item, observation=prepare_source_geometry(
                item['observation'], static_geometry=static, **options))
        elif route == 'b':
            from models.instance_route_b import prepare_source_inputs_cpu
            # Prepared context/texture arrays remain NumPy through IPC. Phase
            # caches contain only pose identity, never a guessed future step.
            item = dict(item, observation=prepare_source_inputs_cpu(
                item['observation'], sample_image_channels=spec.get('compact_source', False),
                static_inputs=static, **options))
    return item


def _read_record(record, spec):
    """Use the selected record identically in serial and spawned execution."""
    from .instance_data import InstanceDataset, InstanceAttemptDataset
    # Reuse the existing, validated per-record reader without reparsing the full
    # manifest in every child or serializing a warm parent cache.
    reader = SimpleNamespace(root=Path(spec['root']), records=[record],
                             training_queries=spec.get('training_queries'),
                             require_labels=spec.get('require_labels', True))
    if spec['kind'] == 'source':
        return InstanceDataset.load_uncached(reader, 0,
            _compact_source=bool(spec.get('compact_source')),
            source_route=spec.get('route') if spec.get('compact_source') else None)
    return InstanceAttemptDataset.load_uncached(reader, 0)


def _worker_prepare(payload):
    record, rgb = payload
    spec = _WORKER_SPEC
    if spec is None:
        raise RuntimeError('Instance preparation worker was not initialized')
    static_key = None
    if spec.get('compact_source') and spec['kind'] == 'source':
        from .instance_preparation_cache import source_geometry_key, unchanged_source_key
        static_key = source_geometry_key(record, spec)
    item = _read_record(record, spec)
    if spec.get('compact_source') and spec['kind'] == 'source':
        static_key = unchanged_source_key(record, spec, static_key)
    item = _prepare_observation(item, rgb, spec,
                                static_cache=_WORKER_STATIC_CACHE if static_key is not None else None,
                                static_key=static_key)
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
                from .instance_selection_cache import reader_cache_stats
                from .instance_static_store import static_store_stats
                results.send(('ok', item, dict(_WORKER_STATIC_CACHE.stats(),
                    reader=reader_cache_stats(), geometry_store=static_store_stats())))
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
                 geometry_options=None, thread_limit=1, compact_source=False,
                 static_cache_bytes=None):
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
        self._cache_stats = {}
        if static_cache_bytes is None:
            static_cache_bytes = int(os.environ.get('SUCTION_INSTANCE_STATIC_CACHE_MB', '64')) * 1024 * 1024
        from .instance_preparation_cache import StaticGeometryCache
        self._static_cache = StaticGeometryCache(static_cache_bytes)
        from .instance_selection_cache import reader_cache_stats
        from .instance_static_store import static_store_stats
        self._serial_reader_stats = dict.fromkeys(reader_cache_stats(), 0)
        self._serial_store_stats = dict.fromkeys(static_store_stats(), 0)
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
                    compact_source=bool(compact_source), static_cache_bytes=static_cache_bytes)
        spec['static_geometry_root'] = (os.environ.get(f'SUCTION_INSTANCE_GEOMETRY_CACHE_{route.upper()}')
            if compact_source and spec['kind'] == 'source' else None)
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
                from .instance_selection_cache import reader_cache_stats
                from .instance_static_store import static_store_stats
                read = getattr(self.dataset, 'load_uncached', self.dataset.__getitem__)
                for ticket in tickets:
                    if self.closed:
                        raise RuntimeError('Instance preparation pool is closed')
                    index, rgb = self._ticket(ticket)
                    self.submitted += 1
                    reader_before, store_before = reader_cache_stats(), static_store_stats()
                    compact = self.reader_spec.get('compact_source') and self.reader_spec['kind'] == 'source'
                    if compact:
                        from .instance_preparation_cache import source_geometry_key, unchanged_source_key
                        static_key = source_geometry_key(self.dataset.records[index], self.reader_spec)
                        item = _read_record(self.dataset.records[index], self.reader_spec)
                        static_key = unchanged_source_key(self.dataset.records[index], self.reader_spec, static_key)
                        result = _prepare_observation(item, rgb, self.reader_spec,
                            static_cache=self._static_cache if static_key is not None else None, static_key=static_key)
                        if self.reader_spec['route'] == 'a':
                            from models.instance_route_a import compact_prepared_source_item
                        else:
                            from models.instance_route_b import compact_prepared_source_item
                        result = compact_prepared_source_item(result)
                    else:
                        result = _prepare_observation(read(index), rgb, self.reader_spec)
                    for name, value in reader_cache_stats().items():
                        self._serial_reader_stats[name] += value - reader_before[name]
                    for name, value in static_store_stats().items():
                        self._serial_store_stats[name] += value - store_before[name]
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
            if len(response) > 2:
                self._cache_stats[index] = response[2]
            return response[1]
        finally:
            self._available_workers.put(index)

    def preparation_cache_stats(self):
        """Completed worker diagnostics; cache residency is never resume state."""
        from .instance_selection_cache import reader_cache_stats
        from .instance_static_store import static_store_stats
        if self.workers:
            rows = list(self._cache_stats.values())
        else:
            rows = [dict(self._static_cache.stats(),
                reader=self._serial_reader_stats, geometry_store=self._serial_store_stats)]
        result = {name: sum(row[name] for row in rows)
                  for name in ('hits', 'misses', 'evictions', 'entries', 'retained_bytes', 'max_bytes')}
        for section, names in (('reader', reader_cache_stats()), ('geometry_store', static_store_stats())):
            result[section] = {name: sum(row.get(section, {}).get(name, 0) for row in rows) for name in names}
        return result

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
