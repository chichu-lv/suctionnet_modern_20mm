#!/usr/bin/env python3
"""Build an explicit, independent source-training NPY/mmap mirror.

The original manifest and archives remain authoritative and unchanged. Set the
printed route-specific environment variable only after publication succeeds.
All production query/label validation still runs on every training read.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suction20.instance_data import (InstanceDataset, SOURCE_TRAIN_ARRAYS,
    _atomic_text, source_output_lock)
from suction20.instance_loading import _spawn_environment
from suction20.instance_selection_cache import (SOURCE_CACHE_FORMAT,
    reader_cache_stats, reset_reader_cache_stats, source_cache_location, source_file_record)
from suction20.metadata import without_legacy_checksums

SPARSE_BLOCK_BYTES = 65536


def _save_sparse_npy(path, value):
    """Standard NPY with holes only for all-zero raw byte blocks.

    Filesystems may allocate these holes normally; mmap/read semantics are the
    same. Raw byte checks preserve negative zero, NaNs and their payload bits.
    """
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError('Object arrays are not permitted in source mirrors')
    raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
    with Path(path).open('wb') as stream:
        np.lib.format.write_array_header_2_0(stream, {
            'descr': np.lib.format.dtype_to_descr(value.dtype),
            'fortran_order': False, 'shape': value.shape})
        offset = stream.tell()
        for begin in range(0, len(raw), SPARSE_BLOCK_BYTES):
            block = raw[begin:begin + SPARSE_BLOCK_BYTES]
            if np.any(block):
                stream.seek(offset + begin)
                stream.write(memoryview(block))
        stream.truncate(offset + len(raw))


def _allocated_bytes(files):
    blocks = [getattr(Path(path).stat(), 'st_blocks', None) for path in files]
    return None if any(value is None for value in blocks) else sum(blocks) * 512


@contextmanager
def _cache_environment(route, root):
    names = ('SUCTION_INSTANCE_SELECTION_CACHE', 'SUCTION_INSTANCE_SOURCE_CACHE_A',
             'SUCTION_INSTANCE_SOURCE_CACHE_B')
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ['SUCTION_INSTANCE_SOURCE_CACHE_' + route.upper()] = str(root)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _archive_plan(source, needed):
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        members = [entry.filename[:-4] for entry in entries if entry.filename.endswith('.npy')]
        if len(members) != len(entries) or len(set(members)) != len(members):
            raise ValueError(f'Unsupported NPZ member inventory: {source}')
        if any(Path(name).name != name or name in ('', '.', '..') for name in members):
            raise ValueError(f'Unsafe NPZ member name: {source}')
        selected = set(members) if needed is None else set(members) & set(needed)
        selected = set(without_legacy_checksums(dict.fromkeys(selected)))
        info = {entry.filename[:-4]: entry.file_size for entry in entries}
        payload = sum(info[name] for name in selected)
        # Reserve receipt/header space before parallel tasks write anything.
        # The converter enforces this bound on its complete temporary entry.
        # Each separate NPY/receipt may consume a full allocation block even
        # for scalars. Reserve 8 KiB per file as well as metadata expansion.
        reserve = payload + max(65536, (len(selected) + 4) * 8192,
                                2 * info.get('metadata_json', 0))
        return members, sorted(selected), payload, reserve


def _convert_archive(task):
    source, destination = Path(task['source']['path']), Path(task['destination'])
    if source_file_record(source) != task['source']:
        raise ValueError(f'Source archive changed during NPY conversion: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(task['temporary_root'])
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='entry-', dir=temporary_root))
    try:
        arrays, total = {}, 0
        with np.load(source, allow_pickle=False) as archive:
            if list(archive.files) != task['members']:
                raise ValueError('Source archive members changed during conversion')
            for key in task['selected']:
                value = archive[key]
                if key == 'metadata_json':
                    metadata = without_legacy_checksums(json.loads(str(value)))
                    if set(metadata) & set(task['members']) - {'metadata_json'}:
                        raise ValueError('array/metadata key collision')
                    value = np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False,
                                                   allow_nan=False))
                file = temporary / (key + '.npy')
                if task.get('sparse', False):
                    _save_sparse_npy(file, value)
                else:
                    np.save(file, value, allow_pickle=False)
                stat = file.stat()
                arrays[key] = {'shape': list(value.shape), 'dtype': value.dtype.str,
                               'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
                total += stat.st_size
        if source_file_record(source) != task['source']:
            raise ValueError(f'Source archive changed during NPY conversion: {source}')
        receipt = {'format': SOURCE_CACHE_FORMAT, 'route': task['route'],
                   'source': task['source'], 'members': task['members'], 'arrays': arrays}
        receipt_path = temporary / 'members.json'
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        total += receipt_path.stat().st_size
        allocated = _allocated_bytes([temporary / (key + '.npy') for key in arrays] + [receipt_path])
        if total > task['reserved_bytes'] or (allocated is not None and allocated > task['reserved_bytes']):
            raise ValueError('NPY entry exceeds its reserved disk budget; output not published')
        if destination.exists():
            raise FileExistsError(f'NPY entry already exists: {destination}')
        os.replace(temporary, destination)
        return {'logical_bytes': total, 'allocated_bytes': allocated}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _convert_tasks(tasks, workers):
    if workers == 1:
        yield from map(_convert_archive, tasks)
        return
    with _spawn_environment(1):
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn')) as executor:
            iterator, pending = iter(tasks), deque()
            for _ in range(2 * workers):
                task = next(iterator, None)
                if task is not None:
                    pending.append(executor.submit(_convert_archive, task))
            while pending:
                yield pending.popleft().result()
                task = next(iterator, None)
                if task is not None:
                    pending.append(executor.submit(_convert_archive, task))


def _existing_entry_bytes(task):
    destination = Path(task['destination'])
    if not destination.exists():
        return None
    receipt_path = destination / 'members.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('format') != SOURCE_CACHE_FORMAT or receipt.get('route') != task['route'] or
            receipt.get('source') != task['source'] or receipt.get('members') != task['members'] or
            set(receipt.get('arrays', {})) != set(task['selected'])):
        raise ValueError('Existing NPY entry differs from source/route; choose a new output directory')
    total = receipt_path.stat().st_size
    for key, spec in receipt['arrays'].items():
        file = destination / (key + '.npy')
        stat = file.stat()
        value = np.load(file, allow_pickle=False, mmap_mode='r')
        if (stat.st_size != spec['size_bytes'] or stat.st_mtime_ns != spec['mtime_ns'] or
                list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']):
            raise ValueError('Existing NPY entry changed; choose a new output directory')
        total += stat.st_size
    allocated = _allocated_bytes([destination / (key + '.npy') for key in receipt['arrays']] + [receipt_path])
    return {'logical_bytes': total, 'allocated_bytes': allocated}


def build_instance_source_cache(manifest, output, *, route, max_bytes, workers=1,
                                resume=False, training_queries=128, max_records=None, sparse=False):
    """Publish a finite manifest selection; never fill a cache while training."""
    if route not in SOURCE_TRAIN_ARRAYS:
        raise ValueError('route must be a or b')
    if not isinstance(sparse, bool):
        raise ValueError('sparse must be boolean')
    for name, value in (('workers', workers), ('max_bytes', max_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if max_records is not None and (isinstance(max_records, bool) or
            not isinstance(max_records, int) or max_records < 1):
        raise ValueError('max_records must be a positive integer')
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    source = InstanceDataset(manifest, split='train', training_queries=training_queries, cache_size=0)
    records = source.records if max_records is None else source.records[:max_records]
    if not records:
        raise ValueError('No source training records selected')
    archives = {}
    for row in records:
        for field, needed in (('observation_npz', SOURCE_TRAIN_ARRAYS[route] | {'metadata_json'}),
                              ('label_cache', None)):
            path = Path(row[field])
            path = (path if path.is_absolute() else manifest.parent / path).resolve()
            if path.is_relative_to(output):
                raise ValueError('Cache output contains an authoritative input; choose an independent directory')
            if path not in archives or needed is None:
                archives[path] = needed
    source_root = Path(os.path.commonpath([str(path.parent) for path in archives]))
    definition = {'format': SOURCE_CACHE_FORMAT, 'route': route,
                  'manifest': source_file_record(manifest), 'training_queries': training_queries,
                  'max_records': max_records, 'source_root': str(source_root),
                  'selected_records': len(records), 'observation_members': sorted(SOURCE_TRAIN_ARRAYS[route]),
                  'sparse': sparse, 'sparse_block_bytes': SPARSE_BLOCK_BYTES if sparse else None}
    with source_output_lock(output):
        state_path, config_path, data = output / 'build_state.json', output / 'cache.json', output / 'data'
        if state_path.exists():
            if not resume:
                raise FileExistsError('Cache output exists; choose a new directory or pass --resume')
            if json.loads(state_path.read_text()).get('definition') != definition:
                raise ValueError('Source manifest or cache definition changed; choose a new output directory')
        else:
            if any(path.name != '.instance-build.lock' for path in output.iterdir()):
                raise FileExistsError('Cache output is not empty')
            _atomic_text(state_path, json.dumps({'definition': definition, 'complete': False}, indent=2) + '\n')
        temporary_root = data / '.temporary'
        if temporary_root.exists():
            shutil.rmtree(temporary_root)  # Only this builder's unpublished crash leftovers.
        tasks, reserved = [], 0
        for path, needed in archives.items():
            before = source_file_record(path)
            members, selected, payload, reserve = _archive_plan(path, needed)
            if source_file_record(path) != before:
                raise ValueError('Source archive changed during member planning')
            tasks.append({'source': before, 'destination': str(source_cache_location(data, path, source_root)),
                          'members': members, 'selected': selected, 'reserved_bytes': reserve,
                          'payload_bytes': payload, 'route': route, 'temporary_root': str(temporary_root),
                          'sparse': sparse})
            reserved += reserve
        # The upper bound includes per-entry metadata and bounded temporary
        # conversions. Fail before creating array files if disk budget is small.
        metadata_reserve = 65536 + 2 * len(json.dumps(definition).encode())
        if reserved + metadata_reserve > max_bytes:
            raise ValueError(f'NPY cache needs up to {reserved + metadata_reserve} reserved bytes; '
                             f'--max-bytes is {max_bytes}. No arrays were converted.')
        pending, sizes, reused = [], [], 0
        for task in tasks:
            existing = _existing_entry_bytes(task)
            if existing is None:
                pending.append(task)
            else:
                reused += 1
                sizes.append(existing)
        sizes.extend(_convert_tasks(pending, workers))
        total = sum(value['logical_bytes'] for value in sizes)
        allocated = (None if any(value['allocated_bytes'] is None for value in sizes) else
                     sum(value['allocated_bytes'] for value in sizes))
        config = dict(definition, complete=True)
        _atomic_text(data / 'cache.json', json.dumps(config, indent=2) + '\n')
        # Validate the exact runtime path before exposing output/cache.json.
        # This reads mapped arrays, keeps all canonical checks, and does not
        # generate another manifest or change checkpoint data identities.
        reset_reader_cache_stats()
        with _cache_environment(route, data):
            for index in range(len(records)):
                source.load_uncached(index, _compact_source=True, source_route=route)
        stats = reader_cache_stats()
        if stats['source_hits'] != 2 * len(records) or stats['source_fallbacks'] or stats['npz_reads']:
            raise ValueError('Cache validation fell back to NPZ; output was not published')
        for task in tasks:
            if source_file_record(task['source']['path']) != task['source']:
                raise ValueError('Source archive changed before cache publication')
        if source_file_record(manifest) != definition['manifest']:
            raise ValueError('Source manifest changed before cache publication')
        config['data_directory'] = 'data'
        result = {'cache_root': str(output), 'route': route, 'records': len(records),
                  'archives': len(tasks), 'converted': len(pending), 'reused': reused,
                  'logical_bytes': total, 'allocated_bytes': allocated,
                  'byte_accounting': 'NPY arrays and per-archive receipts; root metadata is reserved separately',
                  'sparse': sparse, 'budget_basis': 'logical bytes plus conservative metadata reserves',
                  'reserved_bytes': reserved + metadata_reserve,
                  'max_bytes': max_bytes, 'validation_reads': stats,
                  'environment': {'SUCTION_INSTANCE_SOURCE_CACHE_' + route.upper(): str(output)}}
        _atomic_text(config_path, json.dumps(config, indent=2, allow_nan=False) + '\n')
        _atomic_text(state_path, json.dumps({'definition': definition, 'complete': True,
                                           'result': result}, indent=2, allow_nan=False) + '\n')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route', choices=('a', 'b'), required=True)
    parser.add_argument('--max-bytes', type=int, required=True,
                        help='Conservative logical-byte budget, including metadata reserves; also enforced with --sparse')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--training-queries', type=int, default=128)
    parser.add_argument('--max-records', type=int, help='Explicit bounded prefix for a pilot build')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sparse', action='store_true',
                        help='Write standard NPY zero-byte holes; allocated-byte savings depend on the filesystem')
    args = parser.parse_args(argv)
    result = build_instance_source_cache(args.manifest, args.output, route=args.route,
        max_bytes=args.max_bytes, workers=args.workers, training_queries=args.training_queries,
        max_records=args.max_records, resume=args.resume, sparse=args.sparse)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
