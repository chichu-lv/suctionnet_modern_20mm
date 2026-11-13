"""Native contact-query adapter; random rolls and pose frames stay in the caller.

The extension reuses the existing full-point grid. Two candidate workers write
disjoint output rows; the process affinity can reduce this to one. This is not a
DataLoader worker setting and does not change the number of training samples.
"""
from __future__ import annotations

import math
import os

import numpy as np


def native_contact_query(index, points, axes, tangent_u, tangent_v, instances,
                         batch, selected, segments, self_ignore, result, *, detailed=False):
    if getattr(index, 'backend', None) != 'uniform_grid_cpu':
        raise ValueError('native_cpu contact query requires uniform_grid_cpu')
    native = index._native
    if getattr(native, 'CONTACT_ABI_VERSION', None) != 1:
        raise RuntimeError('native_cpu contact query requires rebuilding models._spatial_index_cpu')

    # Keep NumPy's pose normalization/roll and covering-ball arithmetic. These
    # small arrays are prepared once, rather than once per candidate in Python.
    centers = np.empty((len(points), 4, 3), dtype=np.float64)
    radii = np.empty(4, dtype=np.float64)
    centers[:, 0], radii[0] = points, math.hypot(.022, .008)
    for zone, (z_min, z_max, radius) in enumerate(segments, start=1):
        centers[:, zone] = points + ((z_min + z_max) / 2) * axes
        radii[zone] = math.hypot(radius, (z_max - z_min) / 2)
    inputs = {
        'points': points, 'axes': axes, 'tangent_u': tangent_u, 'tangent_v': tangent_v,
        'centers': centers, 'radii': radii, 'segments': segments,
        'self_ignore': float(self_ignore),
        'instances': np.ascontiguousarray(instances, dtype=np.int64),
        'batch': np.ascontiguousarray(batch, dtype=np.int64),
        'selected': np.ascontiguousarray(selected, dtype=np.int64),
        'scene_instance': np.ascontiguousarray(index.instance, dtype=np.int64),
        'scene_normals': np.ascontiguousarray(index.normals, dtype=np.float32),
        'scene_normal_valid': np.ascontiguousarray(index.normal_valid, dtype=np.bool_),
    }
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    workers = min(2, max(1, available))
    return native.contact_query(index._grids, inputs, result, workers, bool(detailed))
