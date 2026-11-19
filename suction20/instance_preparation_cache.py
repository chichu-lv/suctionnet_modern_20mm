"""Bounded, disposable worker caches for immutable source geometry.

Keys describe a source file and preparation settings; they are not content
checksums. Every draw still uses the validated reader and its own RGB transform.
No RNG, learned features, optimizer state, or augmented images belong here.
"""
from collections import OrderedDict
import os
from pathlib import Path
import sys

import numpy as np


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def source_geometry_key(record, spec):
    """Use explicit versions/settings and file metadata, without scanning bytes."""
    path = Path(record['observation_npz'])
    if not path.is_absolute():
        path = Path(spec['root']) / path
    path = path.resolve()
    stat = path.stat()
    return ('instance-static-geometry-v1', spec['route'], str(path),
            stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, record.get('sample_id'),
            os.environ.get('SUCTION_A_CONTACT_BACKEND', 'auto') if spec['route'] == 'a' else None,
            spec.get('training_queries'), _freeze(spec.get('geometry_options', {})),
            _freeze(record.get('observation_identity')), _freeze(record.get('query_identity')))


def unchanged_source_key(record, spec, before):
    """Never bind arrays read before a replacement to the replacement's key."""
    try:
        return before if before == source_geometry_key(record, spec) else None
    except OSError:
        # A successfully read open file may have been removed in the meantime.
        # This draw can proceed, but none of its geometry is retained.
        return None


def static_bytes(value, seen=None):
    """Conservative retained-size accounting, including SciPy index storage.

    Array views may be charged twice; this intentionally errs toward earlier
    eviction. The charge includes both Python containers and native tree nodes.
    """
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + value.nbytes + (static_bytes(value.base, seen) if value.base is not None else 0)
    if isinstance(value, memoryview):
        return size + value.nbytes + static_bytes(value.obj, seen)
    if isinstance(value, dict):
        return size + sum(static_bytes(k, seen) + static_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return size + sum(static_bytes(v, seen) for v in value)
    if type(value).__module__.startswith('scipy.spatial') and hasattr(value, 'size'):
        return (size + int(value.size) * 128 + static_bytes(value.data, seen)
                + static_bytes(value.indices, seen))
    if (type(value).__module__ == 'suction20.instance_static_store'
            and type(value).__name__ == 'DeferredNeighborIndex'):
        # Reserve tree storage before an optional arbitrary-pose query creates
        # it. Canonical training keeps the index deferred, but the LRU budget
        # must also cover a serial consumer materializing its private trees.
        return size + 256 * len(value.points) + static_bytes(vars(value), seen)
    if hasattr(value, '__dict__'):
        return size + static_bytes(vars(value), seen)
    return size


class StaticGeometryCache:
    """Per-worker LRU bounded by both retained bytes and entry count.

    A miss is always valid, including after restart or assignment to another
    worker. Oversized entries are returned to the caller but never retained.
    """
    def __init__(self, max_bytes=64 * 1024 * 1024, max_entries=16):
        for name, value in (('max_bytes', max_bytes), ('max_entries', max_entries)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get_or_prepare(self, key, prepare):
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses += 1
        value = prepare()
        if not self.max_bytes or not self.max_entries:
            return value
        size = static_bytes(value)
        if size > self.max_bytes:
            return value
        while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
            _, (_, removed_size) = self.entries.popitem(last=False)
            self.bytes -= removed_size
            self.evictions += 1
        self.entries[key] = (value, size)
        self.bytes += size
        return value

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    entries=len(self.entries), retained_bytes=self.bytes,
                    max_bytes=self.max_bytes, max_entries=self.max_entries)
