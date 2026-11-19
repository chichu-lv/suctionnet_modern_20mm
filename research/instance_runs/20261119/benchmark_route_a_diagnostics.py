"""Compare Route A's CPU diagnostic reductions on bounded synthetic arrays.

The support counts span a previously documented real-data range, but the arrays
are synthetic. This measures neither live data loading nor GPU training speed.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from models.instance_route_a_geometry import association_diagnostics


SOURCE_FILE = 'models/instance_route_a_geometry.py'
DEFAULT_BASELINE = '59e05c0934bf07ea5c6fb47818c61966b3c8bb02'
SUPPORT_COUNTS = [6923, 32768, 65536, 143048] * 8


def baseline_diagnostics(revision):
    """Use the actual baseline functions without importing its whole model."""
    revision = subprocess.check_output(
        ['git', 'rev-parse', '--verify', '--end-of-options', f'{revision}^{{commit}}'],
        cwd=REPOSITORY, text=True, timeout=30).strip()
    source = subprocess.check_output(
        ['git', 'show', f'{revision}:{SOURCE_FILE}'], cwd=REPOSITORY, text=True, timeout=30)
    names = {'distance_summary', 'association_diagnostics'}
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError('Baseline must define distance_summary and association_diagnostics')
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 f'{revision}:{SOURCE_FILE}', 'exec'), namespace)
    return namespace['association_diagnostics']


def synthetic_associations(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for count in SUPPORT_COUNTS:
        nearest = rng.uniform(0., .02, count)
        third = nearest + rng.uniform(0., .01, count)
        nearest[::73] = np.nan
        third[::53] = np.nan
        rows.append({'d1': nearest, 'd3': third,
                     'valid': np.isfinite(nearest) & (nearest <= .01),
                     'component_missing': np.isnan(nearest),
                     'counts': rng.integers(0, 4, count)})
    return rows


def run_benchmark(*, baseline_ref, repeats, seed):
    functions = {'baseline': baseline_diagnostics(baseline_ref),
                 'current': association_diagnostics}
    rows = synthetic_associations(seed)
    for index, row in enumerate(rows):
        if functions['baseline'](row) != functions['current'](row):
            raise AssertionError(f'Diagnostic field mismatch for synthetic instance {index}')
    # One complete warmup for each implementation precedes timed rounds.
    for function in functions.values():
        for row in rows:
            function(row)
    measurements = []
    for repeat in range(repeats):
        order = ['baseline', 'current'] if repeat % 2 == 0 else ['current', 'baseline']
        record = {'round': repeat + 1, 'order': order}
        for name in order:
            started = time.perf_counter()
            for row in rows:
                functions[name](row)
            record[name + '_ms'] = (time.perf_counter() - started) * 1000.
        measurements.append(record)
    return {
        'schema': 'route_a_diagnostics_cpu_v1',
        'scope': ('CPU diagnostic functions on synthetic association arrays only; '
                  'not live source data, CUDA execution, or end-to-end training throughput'),
        'baseline_ref': baseline_ref,
        'source_file': SOURCE_FILE,
        'current_source': 'Imported from the working tree at invocation',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'machine': platform.machine()},
        'seed': seed,
        'instances_per_round': len(rows),
        'support_counts': SUPPORT_COUNTS,
        'support_count_range': [min(SUPPORT_COUNTS), max(SUPPORT_COUNTS)],
        'support_range_reference': 'research/instance_runs/20261119/route-a-geometry.md',
        'distance_dtype': 'float64',
        'repeats': repeats,
        'all_diagnostic_fields_exactly_equal': True,
        'measurements': measurements,
        'summary': {name: {
            'median_ms_per_32_instances': statistics.median(row[name + '_ms'] for row in measurements),
            'min_ms': min(row[name + '_ms'] for row in measurements),
            'max_ms': max(row[name + '_ms'] for row in measurements),
        } for name in functions},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default=DEFAULT_BASELINE)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    result = run_benchmark(baseline_ref=args.baseline_ref, repeats=args.repeats, seed=args.seed)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
