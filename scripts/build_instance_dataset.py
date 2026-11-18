#!/usr/bin/env python3
"""Build source single-instance caches from official RGB-D and dense objects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suction20.instance_data import build_source_instances


def _range(value):
    """Comma-separated IDs or inclusive ranges, e.g. 0-99 or 0,8,16."""
    result = set()
    for part in value.split(','):
        if '-' in part:
            start, end = map(int, part.split('-', 1))
            if end < start:
                raise argparse.ArgumentTypeError('range end precedes start')
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path,
                        help='Official Dataset root containing scenes/ and dense_point_clouds/')
    parser.add_argument('--label-root', type=Path,
                        help='Frozen R20 release containing config_frozen.json and seal_r20/poisson/')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--camera', choices=('realsense', 'kinect'), default='realsense')
    parser.add_argument('--split', choices=('train', 'test', 'all'), default='all')
    parser.add_argument('--scenes', type=_range, help='Optional explicit shard; summary marks it partial')
    parser.add_argument('--frames', type=_range, help='Optional explicit views; complete default is 0-255')
    parser.add_argument('--train-queries', type=int, default=128)
    parser.add_argument('--instance-mask-root', type=Path,
                        help='Required row-index masks if a scene repeats the same object class')
    parser.add_argument('--without-labels', action='store_true', help='Unlabeled observation caches only')
    parser.add_argument('--depth-min-m', type=float, default=.2)
    parser.add_argument('--depth-max-m', type=float, default=1.2)
    args = parser.parse_args(argv)
    if not args.without_labels and args.label_root is None:
        parser.error('--label-root is required unless --without-labels is set')
    path, summary = build_source_instances(args.dataset_root, args.label_root, args.output,
        split=args.split, camera=args.camera, scenes=args.scenes, frames=args.frames,
        with_labels=not args.without_labels, train_queries=args.train_queries,
        instance_mask_root=args.instance_mask_root, depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m)
    print(json.dumps({'manifest': str(path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
