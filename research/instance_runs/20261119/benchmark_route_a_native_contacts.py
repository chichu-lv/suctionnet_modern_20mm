"""Bounded synthetic A reference/native contact benchmark; no model or CUDA."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from models import instance_route_a_geometry as geometry


def fixture(support_count, query_count, seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-.07, .07, (support_count, 2))
    height = .004 * np.sin(45*xy[:, 0]) + .002 * np.cos(55*xy[:, 1])
    points = np.column_stack((xy, height)) + [.1, -.2, .65]
    source_ids = np.arange(support_count, dtype=np.int64)
    components = np.zeros(support_count, np.int64)
    normals = np.column_stack((-.18*np.cos(45*xy[:, 0]), .11*np.sin(55*xy[:, 1]),
                               np.ones(support_count)))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    encoded = dict(points_np=points, normals_np=normals, components_np=components,
                   source_ids_np=source_ids, raw_feature_valid_np=rng.random(support_count) > .05,
                   normal_valid_np=np.ones(support_count, bool),
                   rgb_valid_np=rng.random(support_count) > .03,
                   gravity_np=np.array([0., .6, -.8]),
                   source_index=geometry.NeighborIndex(points, source_ids, components))
    selected = rng.choice(support_count, size=query_count, replace=False)
    return encoded, points[selected], normals[selected], components[selected], np.ones(query_count, bool)


def compare(actual, reference):
    maximum = {}
    for idx, name in enumerate(('indices', 'mask', 'descriptor', 'counts', 'summary', 'angular_support')):
        left, right = actual[idx], reference[idx]
        if idx in (0, 1, 3, 5):
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, rtol=2e-13, atol=2e-13)
            maximum[name] = float(np.max(np.abs(left-right), initial=0.))
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--support-count', type=int, default=32000)
    parser.add_argument('--query-count', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.query_count <= args.support_count <= 100000 or not 1 <= args.rounds <= 20:
        parser.error('require 1 <= queries <= support <= 100000 and 1..20 rounds')
    inputs = fixture(args.support_count, args.query_count, seed=20261119)
    reference = geometry.reference_contact_geometry(*inputs)
    native = geometry.native_contact_geometry(*inputs, workers=args.workers)
    if native is None:
        raise RuntimeError('build the optional Route-A contact extension before benchmarking')
    errors = compare(native, reference)
    times = {'reference_ms': [], 'native_ms': []}
    for turn in range(args.rounds):
        order = ('reference', 'native') if turn % 2 == 0 else ('native', 'reference')
        for name in order:
            started = time.perf_counter()
            value = (geometry.reference_contact_geometry(*inputs) if name == 'reference'
                     else geometry.native_contact_geometry(*inputs, workers=args.workers))
            times[name+'_ms'].append((time.perf_counter()-started)*1000)
            compare(value, reference)
    # Count reference corrections in an untimed call without changing the real
    # benchmark's dispatch or suppressing any boundary fallback.
    fallback_rows = []
    original = geometry.reference_contact_geometry
    def count_reference(*values):
        fallback_rows.append(len(values[1]))
        return original(*values)
    geometry.reference_contact_geometry = count_reference
    try:
        compare(geometry.native_contact_geometry(*inputs, workers=args.workers), reference)
    finally:
        geometry.reference_contact_geometry = original
    module = geometry._spatial_index_cpu
    result = {
        'scope': 'Synthetic CPU contact preparation only. Current A FP64 reference versus native '
                 'three-region grid/query/filter/FPS/summary/descriptor engine. Native timing includes '
                 'building its CSR grid and allocating outputs. Reference reuses its existing cKDTree. '
                 'Both exclude reader, RGB augmentation, backbone selection/association, IPC, H2D and network. '
                 'This is not a training-throughput or L20 measurement.',
        'seed': 20261119, 'support_count': args.support_count, 'query_count': args.query_count,
        'workers': args.workers, 'warmup_calls_each': 1, 'alternating_rounds': args.rounds,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'machine': platform.machine(),
                        'thread_env': {key: os.environ.get(key) for key in
                                       ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                        'native_route_a_abi': module.ROUTE_A_CONTACT_ABI_VERSION,
                        'native_route_a_source': module.ROUTE_A_CONTACT_SOURCE_VERSION,
                        'compiler': module.COMPILER, 'compile_flags': module.COMPILE_FLAGS,
                        'fps_axes': list(geometry._FPS_AXES)},
        'sampling_masks_counts_angular_support_exact': True,
        'fp64_max_absolute_error': errors,
        'fp64_comparison_rtol': 2e-13, 'fp64_comparison_atol': 2e-13,
        'boundary_reference_rows': sum(fallback_rows),
        **times,
        'reference_median_ms': statistics.median(times['reference_ms']),
        'native_median_ms': statistics.median(times['native_ms']),
    }
    result['contact_only_speedup'] = result['reference_median_ms']/result['native_median_ms']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
