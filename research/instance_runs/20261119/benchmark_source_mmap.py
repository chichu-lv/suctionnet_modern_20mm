#!/usr/bin/env python3
"""Paired CPU NPZ/NPY source-read diagnostic using one synthetic observation.

Files live in a temporary local directory. Repeated reads have a warm OS cache;
this does not estimate first-touch NVMe latency or end-to-end GPU throughput.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from scripts.build_instance_source_cache import _archive_plan, _convert_archive
from suction20.instance_data import SOURCE_TRAIN_ARRAYS, _load_observation, save_observation
from suction20.instance_geometry import prepare_instance
from suction20.instance_selection_cache import (SOURCE_CACHE_FORMAT, reader_cache_stats,
    reset_reader_cache_stats, source_cache_location, source_file_record)


def _summary(values):
    return {'runs': values, 'median': statistics.median(values)}


def _touch(value):
    # Force every selected numerical array's pages to be read once. This
    # separately reports the downstream memory-access cost that mmap can defer.
    for array in value.values():
        if isinstance(array, np.ndarray) and array.dtype.kind in 'biufc':
            array.sum()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--storage-only', action='store_true',
                        help='Compare dense/sparse NPY allocation once, without repeating read timings')
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    names = ('SUCTION_INSTANCE_SELECTION_CACHE', 'SUCTION_INSTANCE_SOURCE_CACHE_A',
             'SUCTION_INSTANCE_SOURCE_CACHE_B')
    previous = {name: os.environ.pop(name, None) for name in names}
    try:
        height, width = 720, 1280
        mask = np.zeros((height, width), bool)
        mask[270:430, 540:740] = True
        depth = np.zeros((height, width), np.float64)
        depth[mask] = .6
        rgb = np.zeros((height, width, 3), np.float32)
        rgb[mask] = np.random.default_rng(7).uniform(0, 1, (int(mask.sum()), 3))
        intrinsics = np.array([[920., 0., 640.], [0., 920., 360.], [0., 0., 1.]])
        observation = prepare_instance(rgb, depth, mask, intrinsics, [0., 0., -1.])
        results = {}
        with tempfile.TemporaryDirectory(prefix='instance-source-mmap-') as directory:
            root = Path(directory)
            source_root = root / 'source'
            source_root.mkdir()
            path = source_root / 'observation.npz'
            save_observation(path, observation)
            compressed_bytes = path.stat().st_size
            for route in ('a', 'b'):
                mirror = root / route
                mirror.mkdir()
                needed = SOURCE_TRAIN_ARRAYS[route] | {'metadata_json'}
                members, selected, payload, reserve = _archive_plan(path, needed)
                task = {'source': source_file_record(path),
                        'destination': str(source_cache_location(mirror, path, source_root)),
                        'members': members, 'selected': selected, 'reserved_bytes': reserve,
                        'route': route, 'temporary_root': str(mirror / '.temporary')}
                stored_size = _convert_archive(task)
                if args.storage_only:
                    sparse_task = dict(task, destination=str(root / (route + '-sparse')),
                                       sparse=True)
                    sparse_size = _convert_archive(sparse_task)
                    for key in selected:
                        dense_value = np.load(Path(task['destination']) / (key + '.npy'), allow_pickle=False)
                        sparse_value = np.load(Path(sparse_task['destination']) / (key + '.npy'),
                                               allow_pickle=False, mmap_mode='c')
                        np.testing.assert_array_equal(
                            np.ascontiguousarray(dense_value).view(np.uint8).reshape(-1),
                            np.ascontiguousarray(sparse_value).view(np.uint8).reshape(-1))
                    results[route] = {'dense_npy': stored_size, 'sparse_npy_64k': sparse_size,
                                      'all_array_bytes_equal': True}
                    continue
                (mirror / 'cache.json').write_text(json.dumps({'format': SOURCE_CACHE_FORMAT,
                    'route': route, 'source_root': str(source_root), 'complete': True}))
                env = 'SUCTION_INSTANCE_SOURCE_CACHE_' + route.upper()
                timings = {mode: {'validated_reader_ms': [], 'reader_and_touch_ms': []}
                           for mode in ('npz', 'npy')}
                # One untimed warmup for both formats, then alternating order.
                for mode in ('npz', 'npy'):
                    if mode == 'npy':
                        os.environ[env] = str(mirror)
                    else:
                        os.environ.pop(env, None)
                    _touch(_load_observation(path, needed_arrays=needed, cache_route=route))
                reset_reader_cache_stats()
                for repeat in range(args.repeats):
                    outputs = {}
                    for mode in (('npz', 'npy') if repeat % 2 == 0 else ('npy', 'npz')):
                        if mode == 'npy':
                            os.environ[env] = str(mirror)
                        else:
                            os.environ.pop(env, None)
                        begin = time.perf_counter()
                        output = _load_observation(path, needed_arrays=needed, cache_route=route)
                        timings[mode]['validated_reader_ms'].append(1000 * (time.perf_counter() - begin))
                        _touch(output)
                        timings[mode]['reader_and_touch_ms'].append(1000 * (time.perf_counter() - begin))
                        outputs[mode] = output
                    assert outputs['npz'].keys() == outputs['npy'].keys()
                    for key, value in outputs['npz'].items():
                        if isinstance(value, np.ndarray):
                            np.testing.assert_array_equal(value, outputs['npy'][key])
                        else:
                            assert value == outputs['npy'][key]
                results[route] = {'timings': {mode: {key: _summary(values) for key, values in result.items()}
                                             for mode, result in timings.items()},
                                  'reader_counters': reader_cache_stats(),
                                  'selected_members': selected, 'npy_and_receipt_bytes': stored_size['logical_bytes'],
                                  'npy_allocated_bytes': stored_size['allocated_bytes'],
                                  'all_selected_arrays_and_metadata_equal': True}
                os.environ.pop(env, None)
        result = {'scope': ('Synthetic local-file storage comparison; filesystem-specific allocation. No throughput timing.'
                           if args.storage_only else 'Synthetic CPU local files, warm OS cache, identical canonical checks. '
                           'No actual training dataset, cold-disk, RGB augmentation, IPC, or GPU.'),
                  'fixture': {'image_hw': [height, width], 'support': int(mask.sum()),
                              'queries': len(observation['query_points']), 'repeats': args.repeats,
                              'compressed_bytes': compressed_bytes, 'numpy': np.__version__},
                  'routes': results,
                  'notes': ['Original manifests remain unchanged in the production builder.',
                            'NPY mirrors are larger than compressed NPZ; a finite disk budget is required.',
                            'reader_and_touch_ms forces all selected numerical pages to be consumed.',
                            'Actual worker throughput, cache coverage and storage latency require L20 measurement.',
                            'Sparse NPY uses raw zero-byte holes; negative zero and NaN payloads remain unchanged.',
                            'Storage savings depend on the filesystem; the builder retains a conservative logical budget.']}
        text = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text, end='')
        return result
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == '__main__':
    main()
