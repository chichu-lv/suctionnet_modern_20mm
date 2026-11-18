#!/usr/bin/env python3
"""Merge independently built instance shards with complete coverage checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suction20.instance_data import merge_source_manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='Exclusive publication directory; existing frame coverage cannot be reduced')
    parser.add_argument('--split', choices=('train', 'test'), required=True)
    parser.add_argument('--allow-partial', action='store_true',
                        help='Explicitly permit a smoke/shard subset; formal training still requires full coverage')
    args = parser.parse_args(argv)
    manifest, summary = merge_source_manifests(args.manifests, args.output,
        split=args.split, allow_partial=args.allow_partial)
    print(json.dumps({**summary, 'manifest': str(manifest.resolve())}, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
