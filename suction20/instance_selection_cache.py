"""Load optional H20 NVMe selection mirrors; authoritative NPZ validation remains with the caller."""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def _source_root(root):
    config = json.loads((Path(root)/'cache.json').read_text())
    if config['format'] != 'instance-selection-npy-v1':
        raise ValueError('Unsupported local cache')
    return Path(config['source_root'])


def read_cached_arrays(path, root, *, exclude_arrays=()):
    """Return selected arrays and the full member inventory, or fall back.

    The inventory lets the caller retain metadata collision checks without
    opening arrays unused by the compact source-training path.
    """
    source = Path(os.path.abspath(path))
    try:
        if not source.is_relative_to(_source_root(root)):
            return None
        location = Path(root) / source.parent.name / source.stem
        receipt = json.loads((location/'members.json').read_text())
        stat = source.stat()
        if receipt['source'] != [str(source), stat.st_size, stat.st_mtime_ns]:
            return None
        members = tuple(receipt['members'])
        arrays = {key: np.load(location/(key+'.npy'), allow_pickle=False, mmap_mode='c')
                  for key in members if key not in exclude_arrays}
        return arrays, members
    except (ValueError, KeyError, TypeError, OSError, EOFError):
        return None
