#!/usr/bin/env python3
"""Build source single-instance caches from official RGB-D and dense objects."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suction20.instance_data import (build_source_instances, _merge_source_manifests,
    source_scene_ids, source_build_output, source_output_lock)


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
    parser.add_argument('--scenes', type=_range,
                        help='Explicit scene IDs; writes a scoped subdirectory below output/shards/')
    parser.add_argument('--frames', type=_range,
                        help='Explicit views; writes a scoped subdirectory, complete default is 0-255')
    parser.add_argument('--train-queries', type=int, default=128)
    parser.add_argument('--instance-mask-root', type=Path,
                        help='Required row-index masks if a scene repeats the same object class')
    parser.add_argument('--without-labels', action='store_true', help='Unlabeled observation caches only')
    parser.add_argument('--depth-min-m', type=float, default=.2)
    parser.add_argument('--depth-max-m', type=float, default=1.2)
    parser.add_argument('--min-support-points', type=int, default=100)
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel scene shards, each owning a separate output directory; train/test split required')
    args = parser.parse_args(argv)
    if not args.without_labels and args.label_root is None:
        parser.error('--label-root is required unless --without-labels is set')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.workers > 1 and args.split == 'all':
        parser.error('parallel builds require --split train or --split test; run the two splits separately')
    options = dict(split=args.split, camera=args.camera, frames=args.frames,
        with_labels=not args.without_labels, train_queries=args.train_queries,
        instance_mask_root=args.instance_mask_root, depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m, min_support_points=args.min_support_points)
    if args.workers == 1:
        path, summary = build_source_instances(args.dataset_root, args.label_root, args.output,
            scenes=args.scenes, **options)
    else:
        scenes = list(args.scenes if args.scenes is not None else source_scene_ids(args.split))
        if not scenes:
            parser.error('parallel builds require at least one scene')
        workers = min(args.workers, len(scenes))
        destination = source_build_output(args.output, split=args.split, camera=args.camera,
            scenes=args.scenes, frames=args.frames)
        # A concurrent invocation of the same request must fail before writing
        # any worker output. Independent explicit requests use different roots.
        with source_output_lock(destination):
            # Spawn keeps inherited spatial-index/library state out of workers.
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs = [pool.submit(build_source_instances, args.dataset_root, args.label_root,
                            destination/'shards'/f'part_{i:04d}', scenes=scenes[i::workers], **options)
                        for i in range(workers)]
                manifests = [job.result()[0] for job in jobs]
            partial = (set(scenes) != set(source_scene_ids(args.split)) or
                       (args.frames is not None and set(args.frames) != set(range(256))))
            path, summary = _merge_source_manifests(manifests, destination,
                split=args.split, allow_partial=partial)
    print(json.dumps({**summary, 'manifest': str(path.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
