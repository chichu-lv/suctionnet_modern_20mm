"""Cold preparation versus reopening a precomputed geometry arena for each draw.

This bounded synthetic CPU probe uses warm OS page cache. Every stored-path
draw opens and validates the files again; no in-process geometry LRU is used.
NPZ reading, RGB augmentation, IPC and GPU work are outside the timed region.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from suction20.instance_preparation_cache import source_geometry_key, static_bytes
from suction20 import instance_static_store as store


def exact_values(actual, expected, path='prepared'):
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True, msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            exact_values(actual[key], expected[key], path + '/' + str(key))
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact_values(left, right, path + '/' + str(index))
    elif type(expected).__name__ in ('NeighborIndex', 'DeferredNeighborIndex'):
        for name in ('points', 'source_ids', 'components'):
            exact_values(getattr(actual, name), getattr(expected, name), path + '/' + name)
    else:
        assert actual == expected, path


def assert_deferred(static, route):
    if route == 'a':
        for name in ('source_index', 'backbone_index'):
            index = static['geometry'][name]
            assert isinstance(index, store.DeferredNeighborIndex) and index._index is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('a', 'b'), required=True)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.rounds <= 20:
        parser.error('--rounds must be in [3,20]')
    torch.set_num_threads(1)
    if args.route == 'a':
        os.environ['SUCTION_A_CONTACT_BACKEND'] = 'native'
        os.environ['SUCTION_A_CONTACT_UNION'] = '0'
        from benchmark_route_a_static_geometry import observation
        from models import instance_route_a as module
        base = observation(32000, 128)
        options = dict(backbone_points=2048, min_support_points=100, query_chunk_size=128)
        prepare_static = lambda obs: module.prepare_source_static_geometry(obs, **options)
        prepare = lambda obs, static: module.prepare_source_geometry(obs, static_geometry=static, **options)
        colors = 'support_rgb'
    else:
        from benchmark_route_b_static_roi import observation
        from models import instance_route_b as module
        base = observation()
        options = dict(min_support_points=100)
        prepare_static = lambda obs: module.prepare_source_static_inputs(obs, **options)
        prepare = lambda obs, static: module.prepare_source_inputs_cpu(
            obs, static_inputs=static, sample_image_channels=True, **options)
        colors = 'rgb'
    exposures = [dict(base, **{colors: base[colors] * base[colors].dtype.type(.45 + .5 * i / args.rounds)})
                 for i in range(args.rounds)]
    for name in ('support_points', 'query_points', 'query_normals', colors):
        assert np.isfinite(base[name]).all()
    durations = {name: [] for name in ('cold_total_ms', 'cold_prepare_bind_ms', 'cold_compact_ms',
                                      'stored_total_ms', 'stored_read_validate_ms',
                                      'stored_bind_ms', 'stored_compact_ms')}
    with tempfile.TemporaryDirectory(prefix='geometry-store-') as temporary:
        work = Path(temporary)
        # Only a small provenance stub is needed: this probe does not execute
        # the dataset reader. Integration tests exercise full manifest/NPZ IO.
        source_path = work / 'synthetic-source.json'
        source_path.write_text(json.dumps(dict(route=args.route, seed=20261119, supports=32000, queries=128)))
        record = dict(observation_npz=str(source_path), sample_id='geometry-store/32000/128')
        spec = dict(root=str(work), route=args.route, training_queries=128, geometry_options=options)
        key = source_geometry_key(record, spec)
        if args.route == 'a':
            key = (key, True)  # RGB-only exposure changes preserve geometry/validity.
        cache = work / 'geometry'
        location = store.geometry_location(cache, source_path)
        location.parent.mkdir(parents=True)
        started = time.perf_counter()
        initial_static = prepare_static(base)
        offline_prepare_ms = (time.perf_counter() - started) * 1000
        # A second color exposure must leave the entire static packet exactly
        # unchanged, including masks/geometry and every stored input array.
        exact_values(prepare_static(exposures[-1]), initial_static, 'static_without_colors')
        started = time.perf_counter()
        entry_bytes = store.write_static_packet(location, initial_static, key=key, route=args.route)
        offline_write_validate_ms = (time.perf_counter() - started) * 1000
        cache_definition = dict(format=store.FORMAT, route=args.route, version=store.preparation_version(args.route))
        (cache / 'cache.json').write_text(json.dumps(dict(definition=cache_definition)))
        file_bytes = {path.name: path.stat().st_size for path in location.iterdir()}
        del initial_static
        store.reset_static_store_stats()
        reopened = []
        original_load = store.np.load

        def tracked_load(path, *positional, **keywords):
            arena = original_load(path, *positional, **keywords)
            if Path(path).name == 'arrays.npy':
                assert keywords.get('mmap_mode') == 'c' and isinstance(arena, np.memmap)
                reopened.append(arena)
            return arena

        def cold(draw):
            started = time.perf_counter()
            prepared = prepare(draw, None)
            bound = time.perf_counter()
            item = module.compact_prepared_source_item(dict(observation=prepared, record=record, labels={}))
            end = time.perf_counter()
            return item, dict(cold_total_ms=(end-started)*1000,
                             cold_prepare_bind_ms=(bound-started)*1000,
                             cold_compact_ms=(end-bound)*1000)

        def stored(draw):
            started = time.perf_counter()
            static = store.read_static_geometry(cache, source_path, key=key, route=args.route)
            read = time.perf_counter()
            assert static is not None
            prepared = prepare(draw, static)
            bound = time.perf_counter()
            item = module.compact_prepared_source_item(dict(observation=prepared, record=record, labels={}))
            end = time.perf_counter()
            assert_deferred(static, args.route)
            return item, dict(stored_total_ms=(end-started)*1000,
                             stored_read_validate_ms=(read-started)*1000,
                             stored_bind_ms=(bound-read)*1000,
                             stored_compact_ms=(end-bound)*1000), static

        store.np.load = tracked_load
        try:
            # First draw reopens the just-written arena. OS pages are already
            # warm from writing, but no Python prepared object or LRU exists.
            first, first_timing, first_static = stored(exposures[0])
            assert len(reopened) == 1
            exact_values(first, cold(exposures[0])[0])
            retained_charge = static_bytes(first_static)
            del first, first_static
            last_colors = None
            for turn, draw in enumerate(exposures):
                jobs = ('cold', 'stored') if turn % 2 == 0 else ('stored', 'cold')
                pair = {}
                for name in jobs:
                    if name == 'cold':
                        pair[name], timing = cold(draw)
                    else:
                        pair[name], timing, static = stored(draw)
                        assert_deferred(static, args.route)
                        del static
                    for key_name, value in timing.items():
                        durations[key_name].append(value)
                exact_values(pair['stored'], pair['cold'])
                wire = pair['stored']['observation']
                dynamic = (wire['_route_a_geometry']['geometry']['rgb_np'] if args.route == 'a'
                           else wire['_route_b_training_inputs']['encoded']['context'])
                if last_colors is not None:
                    assert not np.array_equal(dynamic, last_colors)
                last_colors = dynamic.copy()
                del pair
            assert len(reopened) == args.rounds + 1
            assert len({id(arena) for arena in reopened}) == len(reopened)
            del wire, dynamic
            # Closing only after all prepared observations are gone avoids
            # invalidating their shared NumPy owners during the measurement.
            for arena in reopened:
                arena._mmap.close()
        finally:
            store.np.load = original_load
        stats = store.static_store_stats()
        assert stats == dict(hits=args.rounds+1, missing=0, stale=0, invalid=0)
        result = dict(
            route=args.route, seed=20261119, rounds=args.rounds, support_count=32000, query_count=128,
            scope='Synthetic CPU geometry preparation and compact packing only. Stored path reopens '
                  'cache.json, packet.json and the mmap arena and performs semantic validation on every '
                  'draw. OS file pages are warm; storage-cold performance is not measured. No in-process '
                  'LRU exists in this probe. NPZ reading, RGB augmentation, IPC, learned network and GPU '
                  'work are excluded. Fresh exposure arrays are constructed before timing.',
            source_identity_scope='Small synthetic provenance stub with normal path/stat/settings key; '
                                  'full dataset/manifest reader is covered by separate integration tests.',
            settings=options, order='alternating cold/stored',
            all_prepared_values_exact_every_exposure=True, fresh_rgb_changes_preserved=True,
            static_packet_independent_of_rgb_values=True,
            a_deferred_indices_unmaterialized=(True if args.route == 'a' else None),
            independently_reopened_arenas=len(reopened), static_store_stats=stats,
            first_reopened_draw_ms=first_timing,
            offline_prepare_ms=offline_prepare_ms, offline_write_validate_ms=offline_write_validate_ms,
            entry_bytes=entry_bytes, entry_files_bytes=file_bytes,
            cache_definition_bytes=(cache / 'cache.json').stat().st_size,
            conservative_memory_cache_charge_bytes=retained_charge,
            preparation_version=store.preparation_version(args.route),
            environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                torch_threads=torch.get_num_threads(), thread_env={name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}),
            timing={name: dict(median_ms=statistics.median(values), rounds_ms=values)
                    for name, values in durations.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(route=args.route, entry_bytes=entry_bytes,
        timing={name: value['median_ms'] for name, value in result['timing'].items()},
        all_prepared_values_exact=True, reopened_arenas=len(reopened)), indent=2))


if __name__ == '__main__':
    main()
