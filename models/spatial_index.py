"""Scene-local complete raw-point indices with explicit backend and precision.

XYZ and observation buffers must not be mutated while an index is in use. Native
handles pin the XYZ exporter; every returned ball owns a separate immutable buffer.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


def _compiled_grid():
    try:
        module = importlib.import_module('models._spatial_index_cpu')
    except ImportError as error:
        raise RuntimeError('uniform_grid_cpu requires building models._spatial_index_cpu '
                           'for this Python with a C++ compiler and Python headers') from error
    if module.GRID_ABI_VERSION != 1:
        raise RuntimeError('Unsupported native grid ABI')
    return module


def spatial_index_backend_identity(name='uniform_grid_cpu'):
    """Explicit native ABI/build metadata; no binary scan or silent fallback."""
    if name != 'uniform_grid_cpu':
        raise ValueError(f'Unsupported native spatial index backend: {name}')
    module = _compiled_grid()
    return dict(backend=name, abi_version=module.GRID_ABI_VERSION,
                python_cache_tag=sys.implementation.cache_tag,
                compile_flags=module.COMPILE_FLAGS.split(), compiler=module.COMPILER,
                link_flags=getattr(module, 'LINK_FLAGS', '').split(),
                source_version=getattr(module, 'SOURCE_VERSION', 'legacy_abi1'),
                binary_filename=Path(module.__file__).name)


@dataclass
class SpatialIndex:
    """KD-tree reference plus common factory for both input entry points."""
    xyz: np.ndarray
    instance: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray
    frame_rows: list
    trees: list
    backend = 'kdtree'

    @classmethod
    def from_cpu_geometry(cls, geometry, *, backend='kdtree', cell_size_m=.02, precision='float64'):
        started = time.perf_counter()
        if precision not in ('float32', 'float64'):
            raise ValueError('geometry precision must be float32 or float64')
        if precision != 'float64' and backend != 'uniform_grid_cpu':
            raise ValueError('float32 geometry requires uniform_grid_cpu')
        xyz = geometry.xyz_cam_m if precision == 'float32' else geometry.xyz64
        if backend == 'uniform_grid_cpu':
            return CpuGridIndex._build(xyz, geometry.instance_id_input, geometry.normal_obs,
                                       geometry.normal_obs_valid, geometry.point_offsets,
                                       cell_size_m, started, precision=precision)
        if backend != 'kdtree':
            raise ValueError(f'Unsupported spatial index backend: {backend}')
        rows = geometry.frame_rows
        return cls._build_kdtree(xyz, geometry.instance_id_input, geometry.normal_obs,
                                geometry.normal_obs_valid, rows, started)

    @classmethod
    def from_inputs(cls, inputs, *, backend='kdtree', cell_size_m=.02, precision='float64'):
        started = time.perf_counter()
        if precision not in ('float32', 'float64'):
            raise ValueError('geometry precision must be float32 or float64')
        if precision != 'float64' and backend != 'uniform_grid_cpu':
            raise ValueError('float32 geometry requires uniform_grid_cpu')
        dtype = np.float32 if precision == 'float32' else np.float64
        xyz = np.ascontiguousarray(inputs['xyz_cam_m'].detach().float().cpu().numpy(), dtype=dtype)
        batch = inputs['batch_index'].detach().cpu().numpy()
        instance = inputs['instance_id_input'].detach().cpu().numpy()
        normals = inputs['normal_obs'].detach().float().cpu().numpy()
        valid = inputs['normal_obs_valid'].detach().cpu().numpy()
        frames = len(inputs['g_cam'])
        if backend == 'uniform_grid_cpu':
            # Packed model batches are contiguous by frame. Do not reorder raw ids.
            if len(batch) and (batch.min() < 0 or batch.max() >= frames or np.any(batch[1:] < batch[:-1])):
                raise ValueError('uniform_grid_cpu requires contiguous packed frames')
            offsets = np.concatenate(([0], np.bincount(batch, minlength=frames).cumsum())).astype(np.int64)
            return CpuGridIndex._build(xyz, instance, normals, valid, offsets, cell_size_m, started,
                                       precision=precision)
        if backend != 'kdtree':
            raise ValueError(f'Unsupported spatial index backend: {backend}')
        rows = [np.flatnonzero(batch == b) for b in range(frames)]
        return cls._build_kdtree(xyz, instance, normals, valid, rows, started)

    @classmethod
    def _build_kdtree(cls, xyz, instance, normals, valid, rows, started):
        prepared = time.perf_counter()
        result = cls(xyz, instance, normals, valid, rows,
                     [cKDTree(xyz[r]) if len(r) else None for r in rows])
        result.setup_timings = {'geometry_cpu_setup_s': prepared-started,
                                'tree_build_s': time.perf_counter()-prepared,
                                'spatial_index_backend': result.backend}
        return result

    def frame_has_points(self, frame):
        return self.trees[frame] is not None

    def ball(self, frame, center, radius):
        tree = self.trees[frame]
        if tree is None:
            return np.empty(0, np.int64)
        local = tree.query_ball_point(center, radius+1e-8)
        return np.sort(self.frame_rows[frame][local])

    def balls(self, frame, centers, radii, *, workers=1):
        tree = self.trees[frame]
        if tree is None:
            return [np.empty(0, np.int64) for _ in range(len(centers))]
        local = tree.query_ball_point(centers, radii+1e-8, eps=0.0,
                                     workers=workers, return_sorted=True)
        return [self.frame_rows[frame][rows] for rows in local]


class CpuGridIndex:
    backend = 'uniform_grid_cpu'

    @classmethod
    def from_cpu_geometry(cls, geometry, *, cell_size_m=.02, precision='float64'):
        return SpatialIndex.from_cpu_geometry(geometry, backend=cls.backend, cell_size_m=cell_size_m,
                                              precision=precision)

    @classmethod
    def from_inputs(cls, inputs, *, cell_size_m=.02, precision='float64'):
        return SpatialIndex.from_inputs(inputs, backend=cls.backend, cell_size_m=cell_size_m, precision=precision)

    @classmethod
    def _build(cls, xyz, instance, normals, valid, offsets, cell_size_m, started, *, precision='float64'):
        if not np.isfinite(cell_size_m) or cell_size_m <= 0:
            raise ValueError('grid cell size must be positive and finite')
        result = cls()
        result.precision = precision
        dtype = np.float32 if precision == 'float32' else np.float64
        result.xyz = np.ascontiguousarray(xyz, dtype=dtype)
        result.instance, result.normals, result.normal_valid = instance, normals, valid
        result.frame_offsets = np.asarray(offsets, dtype=np.int64)
        if (result.frame_offsets.ndim != 1 or not len(result.frame_offsets)
                or result.frame_offsets[0] != 0 or result.frame_offsets[-1] != len(xyz)
                or np.any(result.frame_offsets[1:] < result.frame_offsets[:-1])):
            raise ValueError('grid frame offsets must partition the packed points')
        result.cell_size_m = float(cell_size_m)
        result._native = _compiled_grid()
        prepared = time.perf_counter()
        if precision == 'float32':
            if getattr(result._native, 'GEOMETRY_FLOAT32_VERSION', None) != 1:
                raise RuntimeError('FP32 geometry requires rebuilding models._spatial_index_cpu')
            create = result._native.create_float32
        else:
            create = result._native.create
        result._grids = [create(result.xyz, int(start), int(stop), result.cell_size_m)
                         for start, stop in zip(result.frame_offsets[:-1], result.frame_offsets[1:])]
        result.setup_timings = {'geometry_cpu_setup_s': prepared-started,
                                'index_build_s': time.perf_counter()-prepared,
                                'spatial_index_backend': result.backend}
        return result

    @property
    def xyz64(self):
        return np.ascontiguousarray(self.xyz, dtype=np.float64)

    def frame_has_points(self, frame):
        return bool(self.frame_offsets[frame+1] > self.frame_offsets[frame])

    def ball(self, frame, center, radius):
        center = np.asarray(center, dtype=self.xyz.dtype)
        if center.shape != (3,):
            raise ValueError('ball center must have shape [3]')
        buffer = self._native.ball(self._grids[frame], *center, float(radius))
        return np.frombuffer(buffer, dtype=np.int64)

    def balls(self, frame, centers, radii, *, workers=1):
        # Compatibility only: this backend remains scalar and single-threaded.
        return [self.ball(frame, center, radius) for center, radius in zip(centers, radii)]
