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


def read_cached_arrays(path, root):
    source = Path(os.path.abspath(path))
    try:
        if not source.is_relative_to(_source_root(root)):
            return None
        location = Path(root) / source.parent.name / source.stem
        receipt = json.loads((location/'members.json').read_text())
        stat = source.stat()
        if receipt['source'] != [str(source), stat.st_size, stat.st_mtime_ns]:
            return None
        return {key: np.load(location/(key+'.npy'), allow_pickle=False, mmap_mode='c')
                for key in receipt['members']}
    except (ValueError, KeyError, TypeError, OSError, EOFError):
        return None
