"""Compare current P2 geometry with native contact queries on one fixed fixture.

Run from the repository root after building the production CPU extensions:
    python -m research.performance_p3.benchmark_contact \
        --output docs/performance/P3_CPU_RESULTS.json

Both paths use the same uniform_grid_cpu index and compiled FPS. Native contact
uses its fixed two-worker implementation. Each path runs one warmup and two
measured queries; no worker sweep, CUDA work, optimizer updates or real data.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import scipy
import torch

from models.contact_grouping import ContactGrouping, SpatialIndex, ZONE_NAMES
from models.geometry_ops import fps_backend_identity, make_fps_backend
from models.spatial_index import spatial_index_backend_identity
from research.performance_p2.fixture import CANDIDATE_COUNT, FROZEN, POINT_COUNT, make_fixture

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    threadpool_info = threadpool_limits = None

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_WORKERS = 2
WARMUPS = 1
REPEATS = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='JSON path for the measured results.')
    parser.add_argument('--github-base', '--source-head', dest='github_base', default='0bda78b',
                        help='GitHub baseline revision label; recorded separately from the local snapshot commit.')
    return parser.parse_args(argv)


def source_identity(github_base):
    paths = [REPO_ROOT / name for name in (
        'models/contact_grouping.py', 'models/contact_ops.py', 'models/geometry_ops.py', 'models/spatial_index.py',
        'models/suction20net.py', 'setup.py', 'research/performance_p2/fixture.py',
        'research/performance_p3/benchmark_contact.py')]
    paths.extend(p for p in (REPO_ROOT / 'models/csrc').iterdir()
                 if p.is_file() and p.suffix in ('.c', '.cc', '.cpp', '.h', '.hpp'))
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
        current_head = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        current_head = None
    return {'github_base': github_base, 'local_snapshot_head': current_head,
            'head_note': 'The local snapshot commit is distinct from the supplied GitHub baseline; file sizes are descriptive metadata only.',
            'source_files': {str(p.relative_to(REPO_ROOT)): {'size_bytes': p.stat().st_size}
                             for p in sorted(paths)}}


def compare_queries(expected, actual):
    for name in ZONE_NAMES:
        for key in ('indices', 'mask', 'geometry', 'counts'):
            np.testing.assert_array_equal(actual[name][key], expected[name][key],
                                          err_msg=f'{name}.{key}')
    np.testing.assert_array_equal(actual['lip_sector_counts'], expected['lip_sector_counts'],
                                  err_msg='lip_sector_counts')


def main(argv=None):
    options = parse_args(argv)
    contact_module = importlib.import_module('models._spatial_index_cpu')
    if contact_module.CONTACT_ABI_VERSION != 1 or not callable(contact_module.contact_query):
        raise RuntimeError('Build the production native contact extension with CONTACT_ABI_VERSION=1.')
    fps_fn = make_fps_backend('compiled')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    common = dict(fps_fn=fps_fn, query_chunk_size=None, query_workers=1,
                  detailed_cpu_timing=True,
                  spatial_index={'backend': 'uniform_grid_cpu', 'cell_size_m': .02})
    p2 = ContactGrouping(FROZEN, **common)
    # Query has no learned parameters; constructing its alternate wrapper must
    # not alter the fixed roll state inherited from the existing fixture runner.
    with torch.random.fork_rng(devices=[]):
        native = ContactGrouping(FROZEN, contact_query_backend='native_cpu', **common)
    inputs, args = make_fixture()
    index = SpatialIndex.from_inputs(inputs, backend='uniform_grid_cpu', cell_size_m=.02)
    rng_before = torch.get_rng_state().clone()
    rng_after_reference = None
    reference = None
    report = {
        'schema_version': 'suction20.p3-cpu-contact-prototype.v1',
        'scope': 'synthetic CPU only; no CUDA work, training updates or real frames',
        'device': 'cpu', 'optimizer_updates': 0, 'real_frames': False,
        'points': POINT_COUNT, 'candidates': CANDIDATE_COUNT,
        'index_backend': 'uniform_grid_cpu', 'cell_size_m': .02,
        'fps_backend': 'compiled', 'native_contact_workers': NATIVE_WORKERS,
        'warmups_per_path': WARMUPS, 'measured_repeats_per_path': REPEATS,
        'timing_scope': 'ContactGrouping.query wall time; shared index build, fixture, RNG restore and comparisons excluded',
        'native_phase_semantics': 'native_*_worker_s are summed worker phase durations, not additive query wall time',
        'source': source_identity(options.github_base),
        'extensions': {
            'fps': fps_backend_identity('compiled'),
            'grid': spatial_index_backend_identity('uniform_grid_cpu'),
            'contact': {'abi_version': contact_module.CONTACT_ABI_VERSION,
                        'link_flags': getattr(contact_module, 'LINK_FLAGS', '').split(),
                        'source_version': getattr(contact_module, 'SOURCE_VERSION', 'legacy_abi1'),
                        'binary_filename': Path(contact_module.__file__).name},
        },
        'environment': {
            'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
            'torch_cuda_build': torch.version.cuda, 'cpu_count': os.cpu_count(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'torch_threads': torch.get_num_threads(),
            'blas_threads_enforced': 1 if threadpool_limits else None,
            'blas_environment': {key: os.environ.get(key) for key in
                                 ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        },
        'warmups': [], 'runs': [],
    }
    limits = threadpool_limits(limits=1, user_api='blas') if threadpool_limits else nullcontext()
    with limits:
        report['environment']['threadpools_during_measurement'] = threadpool_info() if threadpool_info else None
        for iteration in range(WARMUPS + REPEATS):
            warmup = iteration < WARMUPS
            for name, group in (('p2_python', p2), ('native_cpu', native)):
                torch.set_rng_state(rng_before)
                started = time.perf_counter()
                output = group.query(index, *args, random_roll=True)
                elapsed = time.perf_counter() - started
                after = torch.get_rng_state()
                if reference is None:
                    reference = output
                    rng_after_reference = after.clone()
                else:
                    compare_queries(reference, output)
                if not torch.equal(rng_after_reference, after):
                    raise AssertionError('Torch RNG after-state differs between contact paths.')
                phases = dict(group.last_query_timings)
                if name == 'native_cpu' and 'native_query_wall_s' not in phases:
                    raise AssertionError('Native contact path did not report its native query timing.')
                row = {'path': name, 'repeat': iteration if warmup else iteration-WARMUPS,
                       'wall_seconds': elapsed, 'complete_query_equal': True,
                       'rng_after_matches_reference': True, 'phase_timings': phases}
                report['warmups' if warmup else 'runs'].append(row)
                print(json.dumps({'warmup': warmup, **row}), flush=True)
    medians = {name: statistics.median(row['wall_seconds'] for row in report['runs'] if row['path'] == name)
               for name in ('p2_python', 'native_cpu')}
    report['summary'] = {
        'median_wall_seconds': medians,
        'query_speedup': medians['p2_python'] / medians['native_cpu'],
        'query_time_reduction_percent': 100 * (1 - medians['native_cpu'] / medians['p2_python']),
        'all_complete_queries_equal': True, 'all_rng_after_states_equal': True,
    }
    report['zone_counts'] = {name: {'mean': float(reference[name]['counts'].mean()),
                                   'max': int(reference[name]['counts'].max())} for name in ZONE_NAMES}
    output_path = options.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {output_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
