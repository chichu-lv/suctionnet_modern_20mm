"""Paired CPU cold geometry against the previous native A implementation.

Build both native kernels with identical compiler flags in temporary directories.
No data reader, static-cache hit, IPC, GPU or learned model work is timed.
"""
from __future__ import annotations

import argparse
import cProfile
import importlib.util
import json
import os
from pathlib import Path
import platform
import pstats
import shlex
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models import instance_route_a as current
from models import instance_route_a_geometry as geometry
from benchmark_route_a_static_geometry import observation
import numpy as np

BASELINE = '4809d6b39ecf11dc31c0ae14c08acdb316be2e46'
FLAGS = ['-shared', '-fPIC', '-O3', '-fno-fast-math', '-ffp-contract=off', '-pthread']


def snapshot(path):
    return subprocess.run(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def native_module(directory, name, baseline):
    directory.mkdir()
    for filename in ('spatial_index_cpu.cpp', 'instance_route_a_contact.h'):
        path = 'models/csrc/' + filename
        text = snapshot(path) if baseline else (ROOT / path).read_text(encoding='utf-8')
        (directory / filename).write_text(text, encoding='utf-8')
    output = directory / ('_spatial_index_cpu' + sysconfig.get_config_var('EXT_SUFFIX'))
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    command = compiler + FLAGS + ['-I', sysconfig.get_path('include'),
        str(directory / 'spatial_index_cpu.cpp'), '-o', str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location(name + '._spatial_index_cpu', output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def python_module(name, source):
    module = types.ModuleType(name)
    module.__package__ = 'models'
    sys.modules[name] = module
    exec(compile(source, f'<{BASELINE}:{name}>', 'exec'), module.__dict__)
    return module


def assert_geometry_equal(actual, expected, path='geometry'):
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif type(expected).__name__ == 'NeighborIndex':
        for name in ('points', 'source_ids', 'components'):
            assert_geometry_equal(getattr(actual, name), getattr(expected, name), path + '/' + name)
        assert actual.groups.keys() == expected.groups.keys(), path
        for component in expected.groups:
            np.testing.assert_array_equal(actual.groups[component][0], expected.groups[component][0])
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for name in expected:
            assert_geometry_equal(actual[name], expected[name], path + '/' + str(name))
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_geometry_equal(left, right, path + '/' + str(index))
    else:
        assert actual == expected, path


def python_profile(function):
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (file, line, name), (_, calls, own, cumulative, _) in stats.stats.items():
        rows.append({'file': Path(file).name, 'line': line, 'function': name, 'calls': calls,
                     'self_ms': own * 1000, 'cumulative_ms': cumulative * 1000})
    return sorted(rows, key=lambda row: row['self_ms'], reverse=True)[:20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--support-count', type=int, default=32000)
    parser.add_argument('--query-count', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.query_count <= args.support_count <= 100000 or not 1 <= args.rounds <= 20:
        parser.error('require 1 <= queries <= support <= 100000 and 1..20 rounds')
    os.environ['SUCTION_A_CONTACT_BACKEND'] = 'native'
    source = observation(args.support_count, args.query_count)
    settings = dict(backbone_points=2048, min_support_points=100, query_chunk_size=128)
    with tempfile.TemporaryDirectory(prefix='route-a-cold-') as temporary:
        root = Path(temporary)
        old_native = native_module(root / 'baseline', '_cold_baseline', True)
        new_native = native_module(root / 'current', '_cold_current', False)
        old_geometry = python_module('models._cold_baseline_geometry',
                                     snapshot('models/instance_route_a_geometry.py'))
        old_geometry._spatial_index_cpu = old_native
        old_source = snapshot('models/instance_route_a.py').replace(
            'from .instance_route_a_geometry import', 'from ._cold_baseline_geometry import')
        previous = python_module('models._cold_baseline_model', old_source)
        geometry._spatial_index_cpu = new_native
        functions = {
            'baseline_prepare_ms': lambda: previous.prepare_source_static_geometry(source, **settings),
            'current_prepare_ms': lambda: current.prepare_source_static_geometry(source, **settings),
        }
        expected, actual = (function() for function in functions.values())
        assert_geometry_equal(actual, expected)
        encoded = actual['geometry']
        valid, normals = current._normalize_queries(source['query_points'], source['query_normals'])
        components = np.zeros(len(valid), np.int64)
        contact_args = (encoded, source['query_points'], normals, components, valid)
        functions.update(
            baseline_contact_ms=lambda: old_geometry.native_contact_geometry(*contact_args),
            current_contact_ms=lambda: geometry.native_contact_geometry(*contact_args))
        for left, right in zip(functions['baseline_contact_ms'](), functions['current_contact_ms']()):
            np.testing.assert_array_equal(left, right)
        timings = {name: [] for name in functions}
        names = list(functions)
        for turn in range(args.rounds):
            order = names[turn % len(names):] + names[:turn % len(names)]
            results = {}
            for name in order:
                started = time.perf_counter()
                results[name] = functions[name]()
                timings[name].append((time.perf_counter() - started) * 1000)
            assert_geometry_equal(results['current_prepare_ms'], results['baseline_prepare_ms'])
            for left, right in zip(results['baseline_contact_ms'], results['current_contact_ms']):
                np.testing.assert_array_equal(left, right)
        profile = {}
        geometry.native_contact_geometry(*contact_args, profile=profile)
        result = {
            'scope': 'Synthetic CPU cold geometry only. Native extensions for baseline/current are '
                     'built with identical flags. All point/descriptor/count/summary and prepared '
                     'geometry arrays are compared exactly. The same fixed RGB is used. No input '
                     'reading, augmentation, static-cache hits, IPC, H2D or learned network is timed. '
                     'Optional phase profile is a separate untimed call and excludes grid creation '
                     'and ambiguous-row reference correction; summed worker work is not elapsed '
                     'wall time with multiple threads. This is not an L20 training throughput result.',
            'baseline_git_commit': BASELINE, 'seed': 20261119,
            'support_count': args.support_count, 'query_count': args.query_count,
            'settings': settings, 'rounds': args.rounds, 'order': 'rotating four variants',
            'all_contact_arrays_exact': True, 'all_static_geometry_arrays_exact': True,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                'compiler': new_native.COMPILER, 'compiler_flags': FLAGS,
                'baseline_native_source': old_native.ROUTE_A_CONTACT_SOURCE_VERSION,
                'current_native_source': new_native.ROUTE_A_CONTACT_SOURCE_VERSION,
                'thread_env': {key: os.environ.get(key) for key in
                              ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}},
            'current_native_stage_work_ms': profile,
            'python_profiles': {name: python_profile(functions[name])
                                for name in ('baseline_prepare_ms', 'current_prepare_ms')},
            **timings, 'medians_ms': {name: statistics.median(values) for name, values in timings.items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('medians_ms', 'current_native_stage_work_ms',
          'all_contact_arrays_exact', 'all_static_geometry_arrays_exact')}, indent=2))


if __name__ == '__main__':
    main()
