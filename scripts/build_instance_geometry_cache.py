#!/usr/bin/env python3
"""Precompute immutable source geometry without changing training manifests.

Only entries published within --max-bytes are retained. Training opens this
cache read-only; uncovered or stale records still use ordinary preparation.
No teacher, RGB augmentation draw, phase, network weights or checkpoint changes.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suction20.instance_data import InstanceDataset, _atomic_text, source_output_lock
from suction20.instance_loading import _read_record, _spawn_environment
from suction20.instance_preparation_cache import source_geometry_key, unchanged_source_key
from suction20.instance_static_store import (FORMAT, geometry_location, preparation_version,
    read_static_packet, write_static_packet)


def _file_stat(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _cache_key(record, spec, rgb_augmentation):
    key = source_geometry_key(record, spec)
    return (key, rgb_augmentation) if spec['route'] == 'a' else key


def _source_path(record, spec):
    path = Path(record['observation_npz'])
    return (path if path.is_absolute() else Path(spec['root']) / path).resolve()


def _prepare_entry(task):
    record, spec, rgb_augmentation, staging, number = task
    import torch
    torch.set_num_threads(1)
    before = source_geometry_key(record, spec)
    item = _read_record(record, spec)
    if unchanged_source_key(record, spec, before) is None:
        raise ValueError('Source changed while reading static geometry input')
    observation = item['observation']
    if spec['route'] == 'a':
        from models.instance_route_a import prepare_source_static_geometry
        if rgb_augmentation:
            from suction20.instance_photometry import apply_source_rgb
            # Validity follows exactly the augmented-source contract. The
            # placeholder colors are discarded by static preparation; no RNG
            # is sampled and actual training still draws its own augmentation.
            observation = apply_source_rgb(observation, (np.ones(3), (0, 1, 2), 0, False))
        value = prepare_source_static_geometry(observation, **spec['geometry_options'])
        key = (before, rgb_augmentation)
    else:
        from models.instance_route_b_preparation import prepare_source_static_inputs
        value = prepare_source_static_inputs(observation, sample_image_channels=True,
                                             **spec['geometry_options'])
        key = before
    temporary = Path(staging) / str(number)
    size = write_static_packet(temporary, value, key=key, route=spec['route'])
    if unchanged_source_key(record, spec, before) is None:
        raise ValueError('Source changed during static geometry preparation')
    return dict(temporary=str(temporary), source=str(_source_path(record, spec)),
                size_bytes=size, key=key, record=record)


def _prepared(tasks, workers):
    if workers == 1:
        yield from map(_prepare_entry, tasks)
        return
    with _spawn_environment(1):
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
            iterator, pending = iter(tasks), deque()
            for _ in range(workers):
                task = next(iterator, None)
                if task is not None:
                    pending.append(executor.submit(_prepare_entry, task))
            try:
                while pending:
                    result = pending.popleft().result()
                    yield result
                    task = next(iterator, None)
                    if task is not None:
                        pending.append(executor.submit(_prepare_entry, task))
            finally:
                for future in pending:
                    future.cancel()


def build_geometry_cache(manifest, config, output, *, max_bytes, workers=1,
                         max_records=None, resume=False):
    for name, value in (('max_bytes', max_bytes), ('workers', workers)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if max_records is not None and (type(max_records) is not int or max_records < 1):
        raise ValueError('max_records must be a positive integer')
    route = config['route']
    if route not in ('a', 'b'):
        raise ValueError('Expected route a or b')
    queries = config['pretrain']['queries']
    if type(queries) is not int or queries < 1:
        raise ValueError('pretrain queries must be a positive integer')
    options = {'min_support_points': config['model'].get('min_support_points', 100)}
    if route == 'a':
        options.update(backbone_points=config['model'].get('backbone_points', 2048),
                       query_chunk_size=config['model'].get('query_chunk_size', 32))
    if any(type(value) is not int or value < 1 for value in options.values()):
        raise ValueError('Geometry options must be positive')
    rgb_augmentation = config.get('rgb_augmentation', True)
    if not isinstance(rgb_augmentation, bool):
        raise ValueError('rgb_augmentation must be boolean')
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output == manifest.parent:
        raise ValueError('Geometry cache requires a separate output directory')
    dataset = InstanceDataset(manifest, split='train', training_queries=queries, cache_size=0)
    spec = dict(kind='source', root=str(dataset.root.resolve()), route=route,
                training_queries=queries, require_labels=True, compact_source=True,
                geometry_options=options)
    definition = dict(format=FORMAT, route=route, version=preparation_version(route),
        source_manifest=_file_stat(manifest), geometry_options=options, training_queries=queries,
        rgb_augmentation=rgb_augmentation,
        contact_backend=os.environ.get('SUCTION_A_CONTACT_BACKEND', 'auto') if route == 'a' else None)
    records = dataset.records[:max_records] if max_records is not None else dataset.records
    sources = [_source_path(record, spec) for record in records]
    if len(set(sources)) != len(sources):
        raise ValueError('A geometry cache requires one manifest record per observation archive')
    if any(path.is_relative_to(output) for path in sources):
        raise ValueError('Geometry cache output contains an authoritative source archive')
    with source_output_lock(output):
        state = output / 'build_state.json'
        if state.exists():
            if not resume:
                raise FileExistsError('Geometry cache already exists; use --resume or a new output')
            previous = json.loads(state.read_text())
            if previous['definition'] != definition:
                raise ValueError('Geometry source/settings/version changed; use a new output directory')
            if len(records) < previous.get('records_selected', 0):
                raise ValueError('Resume cannot shrink the previously selected geometry records')
            old_staging = previous.get('staging')
            if old_staging:
                if Path(old_staging).name != old_staging or not old_staging.startswith('geometry-build-'):
                    raise ValueError('Invalid geometry staging ownership record')
                old_path = output / old_staging
                if old_path.exists():
                    shutil.rmtree(old_path)
        elif any(path.name != '.instance-build.lock' for path in output.iterdir()):
            raise FileExistsError('Geometry cache output is not empty')
        else:
            _atomic_text(state, json.dumps(dict(definition=definition), indent=2) + '\n')

        staging = Path(tempfile.mkdtemp(prefix='geometry-build-', dir=output))
        _atomic_text(state, json.dumps(dict(definition=definition, records_selected=len(records),
                                            staging=staging.name), indent=2) + '\n')
        used, reused, built, budget_exhausted = 0, 0, 0, False
        tasks, previous_sizes = [], {}
        try:
            # Inspect only the explicitly selected manifest entries. No content
            # digest or arbitrary dataset-tree scan is used for resume.
            for number, (record, source) in enumerate(zip(records, sources)):
                location = geometry_location(output, source)
                if location.exists():
                    if {path.name for path in location.iterdir()} - {'arrays.npy', 'packet.json'}:
                        raise ValueError('Unexpected files in geometry entry; use a new output')
                    previous_sizes[str(source)] = sum(path.stat().st_size for path in location.iterdir())
                    used += previous_sizes[str(source)]
                key = _cache_key(record, spec, rgb_augmentation)
                try:
                    cached = read_static_packet(location, key=key, route=route) if resume else None
                except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, EOFError, OverflowError):
                    cached = None
                if cached is not None:
                    reused += 1
                    del cached
                else:
                    tasks.append((record, spec, rgb_augmentation, str(staging), number))
            if used > max_bytes:
                raise ValueError('Existing selected geometry exceeds --max-bytes; choose a sufficient budget')
            iterator = _prepared(tasks, workers)
            try:
                for result in iterator:
                    replaced_size = previous_sizes.get(result['source'], 0)
                    if used - replaced_size + result['size_bytes'] > max_bytes:
                        budget_exhausted = True
                        break
                    if _cache_key(result['record'], spec, rgb_augmentation) != result['key']:
                        raise ValueError('Source changed before publishing geometry')
                    destination = geometry_location(output, result['source'])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        # Only this format's two disposable files may be
                        # replaced under the exclusively owned cache root.
                        if {path.name for path in destination.iterdir()} - {'arrays.npy', 'packet.json'}:
                            raise ValueError('Unexpected files in geometry entry; use a new output')
                        old = staging / ('old-' + str(built))
                        destination.rename(old)
                    Path(result['temporary']).rename(destination)
                    used += result['size_bytes'] - replaced_size
                    built += 1
            finally:
                iterator.close()
            if _file_stat(manifest) != definition['source_manifest']:
                raise ValueError('Source manifest changed; geometry cache was not published')
            result = dict(route=route, output=str(output), manifest=str(manifest),
                records_total=len(dataset), records_selected=len(records), built=built, reused=reused,
                entries_available=built + reused, published_bytes=used, max_bytes=max_bytes,
                budget_exhausted=budget_exhausted, complete=built + reused == len(dataset),
                source_manifest_unchanged=True)
            _atomic_text(output / 'cache.json', json.dumps(dict(definition=definition, result=result), indent=2) + '\n')
            _atomic_text(state, json.dumps(dict(definition=definition, records_selected=len(records)), indent=2) + '\n')
            return result
        finally:
            shutil.rmtree(staging)


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-bytes', type=int, required=True,
                        help='Published geometry byte limit; in-flight staging needs additional headroom')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-records', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    result = build_geometry_cache(args.manifest, yaml.safe_load(args.config.read_text()), args.output,
        max_bytes=args.max_bytes, workers=args.workers, max_records=args.max_records, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
