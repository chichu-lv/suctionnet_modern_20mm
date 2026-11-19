#!/usr/bin/env python3
"""Bounded synthetic source-reader diagnostic; no GPU or production data.

Use the production observation builder and compressed NPZ representation in
memory. Timings exclude disk I/O and are not L20 training throughput results.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    sys.path.insert(0, str(args.repo))
    # This process uses BytesIO, not a path in the optional selection mirror.
    os.environ.pop('SUCTION_INSTANCE_SELECTION_CACHE', None)
    import numpy as np
    from suction20.instance_data import (_COMPACT_SOURCE_UNUSED_ARRAYS, _load_observation,
                                        SOURCE_TRAIN_ARRAYS)
    from suction20.instance_geometry import prepare_instance
    from suction20.metadata import without_legacy_checksums

    height, width = 720, 1280
    rgb = np.zeros((height, width, 3), np.float32)
    mask = np.zeros((height, width), bool)
    mask[270:430, 540:740] = True
    rng = np.random.default_rng(7)
    rgb[mask] = rng.uniform(0, 1, (mask.sum(), 3)).astype(np.float32)
    depth = np.zeros((height, width), np.float64)
    depth[mask] = .6
    intrinsics = np.array([[920., 0., 640.], [0., 920., 360.], [0., 0., 1.]])
    observation = prepare_instance(rgb, depth, mask, intrinsics, [0., 0., -1.])
    arrays = {key: value for key, value in observation.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in observation.items() if key not in arrays}
    text = json.dumps(metadata, sort_keys=True, allow_nan=False)
    arrays['metadata_json'] = np.asarray(text)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    compressed = buffer.getvalue()
    excluded = _COMPACT_SOURCE_UNUSED_ARRAYS
    selections = {'full': set(arrays), 'selective': set(arrays) - excluded,
                  **{f'route_{route}': (set(arrays) & needed) | {'metadata_json'}
                     for route, needed in SOURCE_TRAIN_ARRAYS.items()}}
    per_member = {key: [] for key in arrays}
    member_reads = {mode: [] for mode in selections}
    validated_reads = {mode: [] for mode in selections}
    metadata_times = []

    for repeat in range(args.repeats):
        loaded = {}
        # Rotate execution order, using exactly the same in-memory source.
        modes = list(selections)
        start_mode = repeat % len(modes)
        modes = modes[start_mode:] + modes[:start_mode]
        for mode in modes:
            start = time.perf_counter()
            with np.load(io.BytesIO(compressed), allow_pickle=False) as archive:
                values = {}
                for key in archive.files:
                    if key not in selections[mode]:
                        continue
                    before = time.perf_counter()
                    values[key] = archive[key]
                    if mode == 'full':
                        per_member[key].append(1000 * (time.perf_counter() - before))
            member_reads[mode].append(1000 * (time.perf_counter() - start))
            loaded[mode] = values
        for mode in modes:
            for key, value in loaded[mode].items():
                np.testing.assert_array_equal(value, loaded['full'][key])
        before = time.perf_counter()
        without_legacy_checksums(json.loads(str(loaded['full']['metadata_json'])))
        metadata_times.append(1000 * (time.perf_counter() - before))
        checked = {}
        for mode in modes:
            before = time.perf_counter()
            options = ({'exclude_arrays': excluded} if mode == 'selective' else
                       {'needed_arrays': SOURCE_TRAIN_ARRAYS[mode[-1]]} if mode.startswith('route_') else {})
            checked[mode] = _load_observation(io.BytesIO(compressed), **options)
            validated_reads[mode].append(1000 * (time.perf_counter() - before))
        for mode in modes:
            omitted = set(arrays) - selections[mode]
            assert checked[mode].keys() == checked['full'].keys() - omitted
            for key, value in checked[mode].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, checked['full'][key])
                else:
                    assert value == checked['full'][key]

    def summary(values):
        return {'runs': values, 'median': statistics.median(values)}

    output = {
        'scope': 'Synthetic CPU, in-memory compressed NPZ. No disk I/O, labels, GPU, or production training.',
        'fixture': {'image_hw': [height, width], 'support': int(mask.sum()),
                    'queries': len(observation['query_points']), 'repeats': args.repeats,
                    'random_seed': 7, 'compressed_bytes': len(compressed),
                    'all_array_bytes': sum(value.nbytes for value in arrays.values()),
                    'metadata_json_chars': len(text), 'numpy_version': np.__version__},
        'omitted_bytes': sum(arrays[key].nbytes for key in excluded),
        'route_member_protocol': {
            mode: {'read_fields': sorted(selected),
                   'omitted_fields': sorted(set(arrays) - selected),
                   'read_array_bytes': sum(arrays[key].nbytes for key in selected),
                   'omitted_array_bytes': sum(arrays[key].nbytes for key in arrays if key not in selected)}
            for mode, selected in selections.items()},
        'array_member_read_ms': {mode: summary(values) for mode, values in member_reads.items()},
        'validated_observation_read_ms': {mode: summary(values) for mode, values in validated_reads.items()},
        'json_decode_and_cleanup_ms': summary(metadata_times),
        'omitted_fields': {key: {'bytes': arrays[key].nbytes,
                                'member_read_ms': summary(per_member[key])}
                           for key in sorted(excluded)},
        'all_retained_arrays_and_metadata_equal': True,
        'notes': [
            'The original four-field selective mode is retained as a baseline.',
            'Both routes still read organized XYZ/normal maps for canonical source checks.',
            'Route A releases these validation-only maps before augmentation and geometry preparation.',
            'Route B still reads support_points to preserve the original trainer support-count gate;'
            ' its private worker observation retains that count, not the array.',
            'These are decode/validation timings, not steady-state worker IPC or GPU throughput.',
        ],
    }
    report = json.dumps(output, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()
