"""Read-only NPY mirrors; normal observation/label checks remain with callers.

Mirrors avoid repeatedly inflating/constructing NPZ members. They never change
manifest paths or sampling identities, and stale/missing entries use the source.
"""
from collections import Counter
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from .metadata import without_legacy_checksums

SOURCE_CACHE_FORMAT = 'instance-source-npy-v2'
_STATS = Counter()


def reader_cache_stats():
    """Process-local cumulative counters, suitable for worker result telemetry."""
    names = ('source_requests', 'source_hits', 'source_fallbacks',
             'selection_requests', 'selection_hits', 'selection_fallbacks', 'npz_reads')
    return {name: int(_STATS[name]) for name in names}


def reset_reader_cache_stats():
    _STATS.clear()


def record_reader_cache(kind, hit=None):
    if kind == 'npz':
        _STATS['npz_reads'] += 1
    else:
        _STATS[kind + '_requests'] += 1
        _STATS[kind + ('_hits' if hit else '_fallbacks')] += 1


def source_file_record(path):
    """File metadata for replacement invalidation, not a content certificate."""
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'device': stat.st_dev, 'inode': stat.st_ino}


@lru_cache(maxsize=4)
def _cache_config(root, size, mtime_ns, device, inode):
    return json.loads((Path(root) / 'cache.json').read_text())


def _configuration(root):
    path = Path(root) / 'cache.json'
    stat = path.stat()
    return _cache_config(str(Path(root).resolve()), stat.st_size, stat.st_mtime_ns,
                         stat.st_dev, stat.st_ino)


def source_cache_location(root, source, source_root):
    """Retain the full relative path, including .npz, to avoid basename aliases."""
    return Path(root) / 'arrays' / Path(source).resolve().relative_to(Path(source_root))


def read_cached_arrays(path, root, *, exclude_arrays=(), needed_arrays=None, source_route=None):
    """Return selected arrays and the full member inventory, or fall back.

    The original inventory preserves collision checks even for omitted members.
    Arrays use private copy-on-write mappings; consumers cannot alter the mirror.
    """
    try:
        config = _configuration(root)
        source = Path(os.path.abspath(path))
        if config['format'] == 'instance-selection-npy-v1':
            if source_route is not None or not source.is_relative_to(Path(config['source_root'])):
                return None
            location = Path(root) / source.parent.name / source.stem
            receipt = json.loads((location / 'members.json').read_text())
            stat = source.stat()
            identity = [str(source), stat.st_size, stat.st_mtime_ns]
        elif config['format'] == SOURCE_CACHE_FORMAT:
            if not config.get('complete') or config.get('route') != source_route:
                return None
            source = source.resolve()
            data_directory = Path(config.get('data_directory', '.'))
            if data_directory.is_absolute() or '..' in data_directory.parts:
                return None
            location = source_cache_location(Path(root) / data_directory, source, config['source_root'])
            receipt = json.loads((location / 'members.json').read_text())
            identity = source_file_record(source)
            if receipt.get('format') != SOURCE_CACHE_FORMAT or receipt.get('route') != source_route:
                return None
        else:
            return None
        if receipt['source'] != identity:
            return None
        members = tuple(receipt['members'])
        if (len(set(members)) != len(members) or any(not isinstance(key, str) or
                Path(key).name != key or key in ('', '.', '..') for key in members)):
            return None
        selected = [key for key in members if key not in exclude_arrays
                    and (needed_arrays is None or key in needed_arrays)]
        if config['format'] == SOURCE_CACHE_FORMAT:
            selected = list(without_legacy_checksums(dict.fromkeys(selected)))
        arrays = {}
        for key in selected:
            file = location / (key + '.npy')
            value = np.load(file, allow_pickle=False, mmap_mode='c')
            if config['format'] == SOURCE_CACHE_FORMAT:
                spec = receipt['arrays'][key]
                stat = file.stat()
                if (list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype'] or
                        stat.st_size != spec['size_bytes'] or stat.st_mtime_ns != spec['mtime_ns']):
                    return None
            arrays[key] = value
        # Do not expose mappings built while an authoritative file was replaced.
        if config['format'] == SOURCE_CACHE_FORMAT and source_file_record(source) != identity:
            return None
        return arrays, members
    except (ValueError, KeyError, TypeError, OSError, EOFError):
        return None
