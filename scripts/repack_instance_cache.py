#!/usr/bin/env python3
"""Copy an instance manifest's archives to uncompressed NPZ without relabeling.

Inputs are read-only. Publication is atomic and resumable in a separate output
directory. File path/size/mtime record provenance, not content authenticity.
Uncompressed NPZ avoids repeated deflate work but still performs reads/copies;
it is not a memory-mapped array format or a guarantee of RAM residency.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suction20.instance_data import (InstanceDataset, _atomic_text,
    _publish_source_manifest, source_output_lock)
from suction20.instance_loading import _spawn_environment
from suction20.metadata import without_legacy_checksums


REPACK_SCHEMA = 'instance_npz_repack_v1'


def _file_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _stored_archive(path):
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            return bool(entries) and all(entry.compress_type == zipfile.ZIP_STORED for entry in entries)
    except (OSError, zipfile.BadZipFile):
        return False


def _copy_archive(task):
    source, destination, provenance = task
    source, destination = Path(source), Path(destination)
    if _file_record(source) != provenance:
        raise ValueError(f'Source archive changed during repacking: {source}')
    with np.load(source, allow_pickle=False) as archive:
        arrays = without_legacy_checksums({name: archive[name] for name in archive.files})
    if 'metadata_json' in arrays:
        metadata = without_legacy_checksums(json.loads(str(arrays['metadata_json'])))
        arrays['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name+'.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if _file_record(source) != provenance:
            raise ValueError(f'Source archive changed during repacking: {source}')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'source': provenance, 'destination': _file_record(destination)}


def _copy_tasks(tasks, workers):
    if workers == 1:
        yield from map(_copy_archive, tasks)
        return
    # Bounded submissions avoid one Future per archive for large manifests.
    with _spawn_environment(1):
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn')) as executor:
            iterator = iter(tasks)
            pending = deque()
            for _ in range(2 * workers):
                try:
                    task = next(iterator)
                except StopIteration:
                    break
                pending.append(executor.submit(_copy_archive, task))
            while pending:
                result = pending.popleft().result()
                try:
                    task = next(iterator)
                except StopIteration:
                    pass
                else:
                    pending.append(executor.submit(_copy_archive, task))
                yield result


def _journal_entries(path):
    completed = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    break  # A crash may leave one incomplete final journal line.
                entry = json.loads(line)
                completed[entry['source']['path']] = entry
    return completed


def repack_instance_cache(manifest, output, *, workers=1, resume=False):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output == manifest.parent:
        raise ValueError('Repacking requires a separate output directory; the source is read-only')
    summary_path = manifest.with_suffix('.summary.json')
    summary = without_legacy_checksums(json.loads(summary_path.read_text()))
    source = InstanceDataset(manifest, require_labels=bool(summary.get('labels_computed', True)), cache_size=0)
    definition = {'schema': REPACK_SCHEMA, 'source_manifest': _file_record(manifest),
                  'source_summary': _file_record(summary_path), 'format': 'npz_stored',
                  'destination_manifest': manifest.name}
    with source_output_lock(output):
        state_path, journal = output/'repack_state.json', output/'repack_progress.jsonl'
        if state_path.exists():
            if not resume:
                raise FileExistsError('Repack output already exists; choose a new directory or use --resume')
            previous = json.loads(state_path.read_text())
            if previous.get('definition') != definition:
                raise ValueError('Resume repack source path/stat or format differs; choose a new output directory')
        else:
            if any(path.name != '.instance-build.lock' for path in output.iterdir()):
                raise FileExistsError('Repack output is not empty and has no compatible state')
            _atomic_text(state_path, json.dumps({'definition': definition, 'complete': False}, indent=2)+'\n')

        # Paths in the new manifest are relative to its own directory, including
        # archives whose original manifest used absolute or ../ shard references.
        archives, rows = {}, []
        for original in source.records:
            row = copy.deepcopy(original)
            for key in ('observation_npz', 'query_cache', 'label_cache'):
                if row.get(key) is None:
                    continue
                value = Path(row[key])
                path = (value if value.is_absolute() else manifest.parent/value).resolve()
                if path.is_relative_to(output):
                    raise ValueError('Output contains an input archive; choose an independent output directory')
                if path not in archives:
                    relative = Path('archives')/f'{len(archives):09d}.npz'
                    archives[path] = (relative, _file_record(path))
                row[key] = archives[path][0].as_posix()
            rows.append(row)

        completed = _journal_entries(journal) if resume else {}
        pending, reused = [], 0
        for path, (relative, provenance) in archives.items():
            destination = output/relative
            old = completed.get(str(path))
            if old and old.get('source') != provenance:
                raise ValueError(f'Source archive path/stat changed since the last repack: {path}')
            if (old and destination.is_file() and old.get('destination') == _file_record(destination)
                    and _stored_archive(destination)):
                reused += 1
            else:
                pending.append((str(path), str(destination), provenance))

        # Rewrite only complete parsed entries before appending, so an interrupted
        # final line cannot be concatenated with the next valid JSON record.
        _atomic_text(journal, ''.join(json.dumps(entry, sort_keys=True)+'\n' for entry in completed.values()))
        with journal.open('a', encoding='utf-8') as stream:
            for result in _copy_tasks(pending, workers):
                stream.write(json.dumps(result, sort_keys=True)+'\n')
                stream.flush()

        # Validate the real reader, full raw query geometry and all label values.
        # Use each row's declared label prefix, so train-128 archives never need
        # invented labels for the unused part of their 256-query observation.
        for row in rows:
            reader = SimpleNamespace(root=output, records=[row],
                require_labels=source.require_labels,
                training_queries=int(row['labelled_queries']) if row.get('labelled_queries') is not None else None)
            InstanceDataset.load_uncached(reader, 0)
        if (_file_record(manifest) != definition['source_manifest'] or
                _file_record(summary_path) != definition['source_summary']):
            raise ValueError('Source manifest/summary changed during repacking; output was not published')
        new_summary = copy.deepcopy(summary)
        new_summary.update(manifest=manifest.name, storage=dict(format='npz_stored', schema=REPACK_SCHEMA,
            source_manifest=str(manifest), source_archive_count=len(archives),
            source_bytes=sum(value[1]['size_bytes'] for value in archives.values()),
            output_bytes=sum((output/value[0]).stat().st_size for value in archives.values())))
        result_manifest = output/manifest.name
        _publish_source_manifest(result_manifest, rows, new_summary)
        result = {'manifest': str(result_manifest), 'summary': str(result_manifest.with_suffix('.summary.json')),
                  'archives': len(archives), 'converted': len(pending), 'reused': reused,
                  'storage': new_summary['storage']}
        _atomic_text(state_path, json.dumps({'definition': definition, 'complete': True,
                                           'result': result}, indent=2, allow_nan=False)+'\n')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='New independently writable cache directory, including a sufficiently sized RAM disk')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    result = repack_instance_cache(args.manifest, args.output, workers=args.workers, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
