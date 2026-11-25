"""Bounded CPU A preparation: original, native cold, and static-geometry RGB bind."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from models import instance_route_a as current
from models import instance_route_a_geometry as geometry

BASELINE = '4264e4d14319fcf230bb43a43d27ef582da94ae8'


def baseline_module():
    source = subprocess.run(['git', 'show', f'{BASELINE}:models/instance_route_a.py'],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    module = types.ModuleType('models._route_a_static_benchmark_baseline')
    module.__package__ = 'models'
    exec(compile(source, f'<{BASELINE}:models/instance_route_a.py>', 'exec'), module.__dict__)
    return module


def observation(support_count, query_count):
    rng = np.random.default_rng(20261119)
    xy = rng.uniform(-.07, .07, (support_count, 2))
    height = .004*np.sin(45*xy[:, 0]) + .002*np.cos(55*xy[:, 1])
    points = np.column_stack((xy, height)) + [.1, -.2, .65]
    normals = np.column_stack((-.18*np.cos(45*xy[:, 0]), .11*np.sin(55*xy[:, 1]),
                               np.ones(support_count)))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    selected = rng.choice(support_count, size=query_count, replace=False)
    return {'support_points': points, 'support_normals': normals,
            'support_rgb': rng.random(points.shape), 'support_normal_valid': np.ones(support_count, bool),
            'support_rgb_valid': rng.random(support_count) > .03,
            'support_components': np.zeros(support_count, np.int64),
            'support_source_ids': rng.permutation(support_count).astype(np.int64)*7+11,
            'centroid': points.mean(0), 'gravity': np.array([0., .6, -.8]),
            'query_points': points[selected], 'query_normals': normals[selected],
            'query_ids': selected.astype(np.int64)}


def compare_tree(actual, expected, path, errors):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray) and actual.shape == expected.shape and actual.dtype == expected.dtype, path
        if expected.dtype.kind in 'biu':
            np.testing.assert_array_equal(actual, expected, err_msg=path)
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13, equal_nan=True, err_msg=path)
            finite = np.isfinite(actual) & np.isfinite(expected)
            difference = float(np.max(np.abs(actual[finite]-expected[finite]), initial=0.))
            errors[path] = max(errors.get(path, 0.), difference)
        return
    if isinstance(expected, geometry.NeighborIndex):
        for name in ('points', 'source_ids', 'components'):
            compare_tree(getattr(actual, name), getattr(expected, name), path+'/'+name, errors)
        assert actual.groups.keys() == expected.groups.keys()
        for component in expected.groups:
            np.testing.assert_array_equal(actual.groups[component][0], expected.groups[component][0])
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for name in expected:
            compare_tree(actual[name], expected[name], path+'/'+str(name), errors)
        return
    if isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for idx, (left, right) in enumerate(zip(actual, expected)):
            compare_tree(left, right, path+'/'+str(idx), errors)
        return
    if isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13, equal_nan=True, err_msg=path)
    else:
        assert actual == expected, path


def compare_prepared(actual, expected, draw, errors):
    left, right = actual['_route_a_geometry'], expected['_route_a_geometry']
    for name in ('settings', 'points', 'normals', 'geometry', 'chunks'):
        compare_tree(left[name], right[name], name, errors)
    for name in left['source_references']:
        assert left['source_references'][name] is draw[name]
    # The current draw's colors must be rebound, including masked RGB rows.
    np.testing.assert_array_equal(left['geometry']['rgb_np'], right['geometry']['rgb_np'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--support-count', type=int, default=32000)
    parser.add_argument('--query-count', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.query_count <= args.support_count <= 100000 or not 1 <= args.rounds <= 20:
        parser.error('require 1 <= queries <= support <= 100000 and 1..20 rounds')
    if getattr(geometry._spatial_index_cpu, 'ROUTE_A_CONTACT_ABI_VERSION', None) != 1:
        raise RuntimeError('build the optional Route-A contact extension before benchmarking')
    previous = baseline_module()
    base = observation(args.support_count, args.query_count)
    settings = {'backbone_points': 2048, 'min_support_points': 100, 'query_chunk_size': 32}
    started = time.perf_counter()
    static = current.prepare_source_static_geometry(base, **settings)
    static_creation_ms = (time.perf_counter()-started)*1000
    assert 'rgb_np' not in static['geometry'] and 'source_references' not in static
    reference = previous.prepare_source_geometry(base, **settings)
    errors = {}
    compare_prepared(current.prepare_source_geometry(base, **settings), reference, base, errors)
    compare_prepared(current.prepare_source_geometry(base, static_geometry=static, **settings),
                     reference, base, errors)
    timings = {'baseline_ms': [], 'native_cold_ms': [], 'static_rgb_bind_ms': []}
    rng = np.random.default_rng(20261120)
    names = list(timings)
    last_colors = reference['_route_a_geometry']['geometry']['rgb_np']
    for turn in range(args.rounds):
        # Already augmented synthetic colors: no reader, photometry, or random
        # draw cost is included in any timed preparation call.
        draw = dict(base, support_rgb=rng.random(base['support_rgb'].shape))
        order = names[turn % 3:] + names[:turn % 3]
        results = {}
        for name in order:
            started = time.perf_counter()
            if name == 'baseline_ms':
                prepared = previous.prepare_source_geometry(draw, **settings)
            elif name == 'native_cold_ms':
                prepared = current.prepare_source_geometry(draw, **settings)
            else:
                prepared = current.prepare_source_geometry(draw, static_geometry=static, **settings)
            timings[name].append((time.perf_counter()-started)*1000)
            results[name] = prepared
        for name in names:
            compare_prepared(results[name], results['baseline_ms'], draw, errors)
        colors = results['static_rgb_bind_ms']['_route_a_geometry']['geometry']['rgb_np']
        assert not np.array_equal(colors, last_colors)
        assert results['static_rgb_bind_ms']['_route_a_geometry']['chunks'] is static['chunks']
        last_colors = colors
    # An untimed guard proves the warm path does not build geometry, run a
    # contact query, or compute any learned features; only current RGB is bound.
    functions = ('_instance_geometry', '_query_geometry', 'prepare_source_static_geometry')
    original = {name: getattr(current, name) for name in functions}
    def forbidden(*args, **kwargs):
        raise AssertionError('warm RGB binding rebuilt static geometry')
    for name in functions:
        setattr(current, name, forbidden)
    try:
        warm = current.prepare_source_geometry(draw, static_geometry=static, **settings)
        compare_prepared(warm, results['baseline_ms'], draw, errors)
    finally:
        for name, value in original.items():
            setattr(current, name, value)
    native = geometry._spatial_index_cpu
    result = {
        'scope': 'Synthetic CPU source preparation only, using identical geometry and fresh RGB each '
                 'round. Baseline loads the exact previous A model module but uses the same current '
                 'unchanged neighbor/FPS reference primitives. Native cold recomputes all static geometry; '
                 'warm binds new RGB to an already computed static geometry object. Timing excludes file '
                 'reading, augmentation, IPC, cache lookup/storage, H2D and all learned network work. '
                 'Real training cache hit rate is unknown; warm-bind latency is conditional on a valid hit '
                 'and is not an end-to-end throughput result.',
        'baseline_git_commit': BASELINE, 'seed': 20261119, 'rgb_seed': 20261120,
        'support_count': args.support_count, 'query_count': args.query_count,
        'settings': settings, 'rounds': args.rounds, 'order': 'rotating baseline/cold/warm',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'thread_env': {key: os.environ.get(key) for key in
                                       ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                        'native_route_a_abi': getattr(native, 'ROUTE_A_CONTACT_ABI_VERSION', None),
                        'native_route_a_source': getattr(native, 'ROUTE_A_CONTACT_SOURCE_VERSION', None)},
        'initial_static_creation_ms': static_creation_ms,
        'all_discrete_arrays_exact': True, 'all_rgb_arrays_exact_per_draw': True,
        'warm_path_static_geometry_recompute_forbidden': True,
        'comparison_rtol': 2e-13, 'comparison_atol': 2e-13,
        'max_fp64_absolute_error': max(errors.values(), default=0.),
        'nonzero_fp64_absolute_error_by_array': {key: value for key, value in errors.items() if value},
        **timings,
        'medians_ms': {name: statistics.median(values) for name, values in timings.items()},
    }
    result['native_cold_prepare_speedup'] = result['medians_ms']['baseline_ms']/result['medians_ms']['native_cold_ms']
    result['warm_bind_relative_to_baseline'] = result['medians_ms']['baseline_ms']/result['medians_ms']['static_rgb_bind_ms']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
