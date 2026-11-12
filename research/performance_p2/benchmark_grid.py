"""Compare exact native CSR-grid queries with the current production KDTree path.

Run from the repository root:
    python -m research.performance_p2.benchmark_grid --output-dir /tmp/p2-grid

The repository's compiled FPS extension must already be installed. This script
builds only the temporary grid library, entirely inside --output-dir.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes as ct
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

import numpy as np
import torch

from models.contact_grouping import ContactGrouping, SpatialIndex, ZONE_NAMES
from models.geometry_ops import fps_backend_identity, make_fps_backend
from .fixture import CANDIDATE_COUNT, FROZEN, POINT_COUNT, make_fixture

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

REFERENCE_HEAD = '1e28a2faa9991478d706e4d188081aa1a482ff9b'
CELL_SIZE_M = .02
COMPILE_FLAGS = ['-shared', '-fPIC', '-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off']
PD = ct.POINTER(ct.c_double)
PI = ct.POINTER(ct.c_int64)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory for the temporary grid library and newly measured JSON.')
    parser.add_argument('--repeats', type=int, choices=(1, 2), default=2,
                        help='Repeat the same fixed input once or twice; default: 2.')
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'),
                        help='C++ compiler command; default: CXX or c++.')
    return parser.parse_args(argv)


def current_head():
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None


def build_grid_library(output_dir, cxx):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'grid_index.so'
    source = Path(__file__).with_name('grid_index.cpp')
    command = [*shlex.split(cxx), *COMPILE_FLAGS, str(source), '-o', str(output)]
    subprocess.run(command, check=True)
    lib = ct.CDLL(str(output))
    lib.grid_create.argtypes = [PD, ct.c_int64, ct.c_double]
    lib.grid_create.restype = ct.c_void_p
    lib.grid_destroy.argtypes = [ct.c_void_p]
    lib.grid_ball.argtypes = [ct.c_void_p, PD, ct.c_double, ct.POINTER(PI)]
    lib.grid_ball.restype = ct.c_int64
    lib.grid_last_error.restype = ct.c_char_p
    return lib, command


class NativeGrid:
    """Single-frame, single-query-thread adapter for this research fixture."""

    def __init__(self, lib, inputs):
        start = time.perf_counter()
        self.lib = lib
        self.xyz = inputs['xyz_cam_m'].detach().float().cpu().numpy().astype(np.float64)
        batch = inputs['batch_index'].detach().cpu().numpy()
        if not np.all(batch == 0):
            raise ValueError('This research adapter supports the fixed one-frame fixture only.')
        self.frame_rows = [np.arange(len(batch), dtype=np.int64)]
        self.instance = inputs['instance_id_input'].detach().cpu().numpy()
        self.normals = inputs['normal_obs'].detach().float().cpu().numpy()
        self.normal_valid = inputs['normal_obs_valid'].detach().cpu().numpy()
        ready = time.perf_counter()
        self.handle = lib.grid_create(self.xyz.ctypes.data_as(PD), len(self.xyz), CELL_SIZE_M)
        if not self.handle:
            raise RuntimeError(lib.grid_last_error().decode())
        # Production query checks only tree presence here, then invokes ball().
        self.trees = [True]
        self.setup_timings = {'geometry_cpu_setup_s': ready - start,
                              'tree_build_s': time.perf_counter() - ready}

    def ball(self, frame, center, radius):
        if frame != 0:
            raise ValueError('This research adapter has one frame.')
        center = np.asarray(center, dtype=np.float64)
        out = PI()
        size = self.lib.grid_ball(self.handle, center.ctypes.data_as(PD),
                                  float(radius) + 1e-8, ct.byref(out))
        if size < 0:
            raise RuntimeError(self.lib.grid_last_error().decode())
        # Native scratch is reused by the next query: keep one independent array.
        return np.ctypeslib.as_array(out, shape=(size,)).copy() if size else np.empty(0, np.int64)

    def close(self):
        if self.handle:
            self.lib.grid_destroy(self.handle)
            self.handle = None


def compare_grouping(expected, actual):
    for name in ZONE_NAMES:
        for key in ('indices', 'mask', 'geometry', 'counts'):
            if not np.array_equal(expected[name][key], actual[name][key]):
                raise AssertionError(f'Grouping differs: {name}.{key}')
    if not np.array_equal(expected['lip_sector_counts'], actual['lip_sector_counts']):
        raise AssertionError('Grouping differs: lip_sector_counts')


def covering_balls(args, segments):
    """The same four covering spheres used by the original measurement."""
    points = args[0].numpy().astype(np.float64)
    axes = args[1].numpy().astype(np.float64)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    centers = np.empty((len(points), 4, 3), np.float64)
    radii = np.empty((len(points), 4), np.float64)
    centers[:, 0], radii[:, 0] = points, math.hypot(.022, .008)
    for j, (lo, hi, radius) in enumerate(segments, start=1):
        centers[:, j] = points + ((lo + hi) / 2) * axes
        radii[:, j] = math.hypot(radius, (hi - lo) / 2)
    return centers.reshape(-1, 3), radii.reshape(-1)


def add_summary(report):
    fields = ('build_including_geometry_seconds', 'standalone_ball_queries_seconds',
              'query_wall_seconds', 'build_plus_query_seconds')
    medians = {name: {field: statistics.median(r[field] for r in report['runs'] if r['index'] == name)
                      for field in fields} for name in ('kdtree', 'uniform_grid')}
    report['summary'] = {'medians': medians, 'speedup': {
        field: medians['kdtree'][field] / medians['uniform_grid'][field] for field in fields}}


def main(argv=None):
    options = parse_args(argv)
    output_dir = options.output_dir.expanduser().resolve()
    # Reuse the production extension; never compile a separate FPS implementation.
    fps_fn = make_fps_backend('compiled')
    fps_identity = fps_backend_identity('compiled')
    lib, build_command = build_grid_library(output_dir, options.cxx)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    group = ContactGrouping(FROZEN, fps_fn=fps_fn, query_chunk_size=None,
                            query_workers=1, detailed_cpu_timing=True)
    inputs, args = make_fixture()
    # Production query itself draws rolls. Restore this state before each path,
    # then compare their after-states as well as their geometric outputs.
    rng_before = torch.get_rng_state().clone()
    rng_after_reference = None
    centers, radii = covering_balls(args, group.segments)
    report = {
        'reference_head': REFERENCE_HEAD, 'current_head': current_head(),
        'scope': 'synthetic CPU only; fixed 840000-point/512-pose fixture; no CUDA or training',
        'runner_interface': 'direct ContactGrouping.query(random_roll=True)',
        'cell_size_m': CELL_SIZE_M, 'raw_points': POINT_COUNT,
        'poses': CANDIDATE_COUNT, 'ball_queries': len(centers),
        'threads': {'index': 1, 'torch': 1, 'blas': 1 if threadpool_limits else None},
        'blas_thread_control': 'threadpoolctl' if threadpool_limits else 'environment; not enforced by runner',
        'blas_environment': {key: os.environ.get(key) for key in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'compile_command': build_command, 'fps_identity': fps_identity, 'runs': [],
    }
    limits = threadpool_limits(limits=1, user_api='blas') if threadpool_limits else nullcontext()
    with limits:
        baseline = None
        for repeat in range(options.repeats):
            start = time.perf_counter()
            kd = SpatialIndex.from_inputs(inputs)
            kd_build = time.perf_counter() - start
            start = time.perf_counter()
            grid = NativeGrid(lib, inputs)
            grid_build = time.perf_counter() - start
            try:
                start = time.perf_counter()
                kd_balls = [kd.ball(0, c, r) for c, r in zip(centers, radii)]
                kd_balls_s = time.perf_counter() - start
                start = time.perf_counter()
                grid_balls = [grid.ball(0, c, r) for c, r in zip(centers, radii)]
                grid_balls_s = time.perf_counter() - start
                for ball_id, (expected, actual) in enumerate(zip(kd_balls, grid_balls)):
                    if not np.array_equal(expected, actual):
                        raise AssertionError(f'Neighborhood differs: ball {ball_id}')
                returned = sum(map(len, kd_balls))
                del kd_balls, grid_balls
                for name, index, build, balls_s in (
                    ('kdtree', kd, kd_build, kd_balls_s),
                    ('uniform_grid', grid, grid_build, grid_balls_s),
                ):
                    torch.set_rng_state(rng_before)
                    start = time.perf_counter()
                    out = group.query(index, *args, random_roll=True)
                    elapsed = time.perf_counter() - start
                    stats = dict(group.last_query_timings)
                    rng_after = torch.get_rng_state()
                    if baseline is None:
                        baseline = out
                        rng_after_reference = rng_after.clone()
                    else:
                        compare_grouping(baseline, out)
                    if not torch.equal(rng_after_reference, rng_after):
                        raise AssertionError('Production query RNG after-state differs.')
                    row = {
                        'repeat': repeat, 'index': name,
                        'build_including_geometry_seconds': build,
                        'standalone_ball_queries_seconds': balls_s,
                        'query_wall_seconds': elapsed, 'build_plus_query_seconds': build + elapsed,
                        'all_neighborhoods_equal': True, 'all_grouping_outputs_equal': True,
                        'rng_after_matches_reference': True, 'standalone_returned_points': returned,
                        'phase_timings': stats,
                    }
                    report['runs'].append(row)
                    print(json.dumps(row), flush=True)
            finally:
                grid.close()
    add_summary(report)
    result_path = output_dir / 'geometry_grid_results.json'
    result_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {result_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
