"""Exact CPU FPS backends, selected once and injected per model instance."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np


def numpy_fps_positions(points, count, first_position):
    """Original NumPy loop, including greedy order when count equals M."""
    chosen = np.zeros(len(points), bool)
    distance = np.full(len(points), np.inf)
    current = first_position
    output = []
    for _ in range(count):
        output.append(current)
        chosen[current] = True
        distance = np.minimum(distance, np.sum((points - points[current]) ** 2, axis=1))
        distance[chosen] = -1
        current = int(np.argmax(distance))
    return output


def _compiled():
    try:
        module = importlib.import_module('models._fps_cpu')
    except ImportError as error:
        raise RuntimeError('compiled FPS requires building models._fps_cpu for this Python; '
                           'install the project with a C compiler and Python headers') from error
    if module.FPS_ABI_VERSION != 1:
        raise RuntimeError('Unsupported compiled FPS ABI')
    return module


def fps_backend_identity(name):
    """Startup-only source/binary/build identity; never silently substitute FPS."""
    result = dict(backend=name, abi_version=1, python_cache_tag=sys.implementation.cache_tag)
    if name == 'compiled':
        module = _compiled()
        return dict(result, compile_flags=module.COMPILE_FLAGS.split(), compiler=module.COMPILER,
                    source_sha256=module.SOURCE_SHA256,
                    binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    if name == 'numpy':
        return dict(result, compile_flags=[], compiler=None, binary_sha256=None,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raise ValueError(f'Unsupported FPS backend: {name}')


def _fps_indices(positions_fn, xyz, indices, count, first, presorted):
    rows = np.asarray(indices, dtype=np.int64)
    if not presorted:
        rows = np.sort(rows)
    count = min(int(count), len(rows))
    if count <= 0:
        return rows[:0]
    points = np.ascontiguousarray(np.asarray(xyz)[rows], dtype=np.float64)
    start = 0 if first is None else int(np.searchsorted(rows, first))
    if start >= len(rows) or (first is not None and rows[start] != first):
        raise ValueError('FPS first raw index must belong to the candidate rows')
    return rows[np.asarray(positions_fn(points, count, start), dtype=np.int64)]


def fps_indices(xyz, indices, count, first=None, *, presorted=False):
    """Compatibility NumPy reference; ties resolve to the lowest raw index."""
    return _fps_indices(numpy_fps_positions, xyz, indices, count, first, presorted)


def make_fps_backend(name):
    """Return an independent callable; no mutable module-level backend switch."""
    if name == 'numpy':
        positions_fn = numpy_fps_positions
    elif name == 'compiled':
        positions_fn = _compiled().fps_positions
    else:
        raise ValueError(f'Unsupported FPS backend: {name}')

    def select(xyz, indices, count, first=None, *, presorted=False):
        return _fps_indices(positions_fn, xyz, indices, count, first, presorted)
    return select
