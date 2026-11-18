#!/usr/bin/env python3
"""Run native instance inference from an unlabeled cache or organized RGB-D."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suction20.instance_data import InstanceDataset, load_observation
from suction20.instance_inference import (INFERENCE_REPORT_SCHEMA, PUBLIC_EVALUATION,
    infer_instance, load_instance_checkpoint, load_raw_observation, resolve_ranking)
from suction20.instance_training import file_digest


def _stream_report(path, metadata, predictions):
    """Atomically stream a JSON report; a model error cannot become no_pose."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            header = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write(header[:-1] + ',\n  "predictions": [')
            count, no_pose = 0, 0
            for prediction in predictions:
                if count:
                    stream.write(',')
                stream.write('\n' + json.dumps(prediction, ensure_ascii=False, allow_nan=False))
                count += 1
                no_pose += prediction['status'] == 'no_pose'
            stream.write('\n  ],\n  "summary": ')
            stream.write(json.dumps({'instances': count, 'no_pose_instances': no_pose}, allow_nan=False))
            stream.write('\n}\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--observation', type=Path, help='Prepared observation NPZ; no teacher labels required')
    source.add_argument('--manifest', type=Path, help='Prepared source instance manifest; label files are never opened')
    source.add_argument('--rgb', type=Path, help='Raw RGB image or HxWx3 .npy; requires the raw calibration options below')
    parser.add_argument('--output', required=True, type=Path, help='Atomic JSON report with all instance Top-K predictions')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backbone-root', type=Path, help='Relocated pinned SpConv source for route A only')
    parser.add_argument('--ranking', choices=['auto', 'quality', 'task', 'mixed'], default='auto',
                        help='auto: source q_inst; finetuned task probability')
    parser.add_argument('--alpha', type=float, help='Only for mixed: P_task * max(q_inst,1e-6)^alpha; default .5')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--split', choices=['train', 'test'], help='Optional filter for --manifest')
    parser.add_argument('--max-instances', type=int, help='Explicit manifest subset for smoke runs; recorded in report')
    parser.add_argument('--depth', type=Path, help='Native HxW image or .npy in declared raw depth units')
    parser.add_argument('--mask', type=Path, help='Native binary mask or integer predicted-instance IDs')
    parser.add_argument('--intrinsics', type=Path, help='JSON 3x3 K or {fx,fy,cx,cy}; rectified pinhole camera')
    parser.add_argument('--depth-scale-m', type=float, help='Metres per stored depth unit; explicitly required for raw input')
    parser.add_argument('--gravity', nargs=3, type=float, help='Calibrated gravity vector in camera coordinates')
    parser.add_argument('--instance-id', type=int, help='Positive target ID for an integer instance mask')
    parser.add_argument('--max-queries', type=int, default=256, help='Raw input only; canonical nested prefix, 1..256')
    parser.add_argument('--depth-min-m', type=float, default=.2, help='Raw valid-depth lower bound')
    parser.add_argument('--depth-max-m', type=float, default=1.2, help='Raw valid-depth upper bound')
    args = parser.parse_args(argv)
    raw_options = (args.depth, args.mask, args.intrinsics, args.depth_scale_m, args.gravity, args.instance_id)
    if args.rgb and any(value is None for value in raw_options[:5]):
        parser.error('--rgb requires --depth, --mask, --intrinsics, --depth-scale-m and --gravity')
    if not args.rgb and any(value is not None for value in raw_options):
        parser.error('Raw calibration options require --rgb')
    if not args.rgb and (args.max_queries != 256 or args.depth_min_m != .2 or args.depth_max_m != 1.2):
        parser.error('Prepared observations keep their saved preprocessing identity; raw options require --rgb')
    if not args.manifest and (args.split is not None or args.max_instances is not None):
        parser.error('--split and --max-instances require --manifest')
    if args.max_instances is not None and args.max_instances < 1:
        parser.error('--max-instances must be positive')
    if not 1 <= args.top_k <= 256:
        parser.error('--top-k must be in [1,256]')

    model, checkpoint = load_instance_checkpoint(args.checkpoint, device=args.device,
                                                 backbone_root=args.backbone_root)
    ranking, alpha = resolve_ranking(checkpoint['stage'], args.ranking, args.alpha)
    if args.manifest:
        dataset = InstanceDataset(args.manifest, split=args.split, require_labels=False)
        limit = len(dataset) if args.max_instances is None else min(len(dataset), args.max_instances)
        if not limit:
            raise ValueError('The selected manifest contains no instances')
        inputs = {'mode': 'manifest', 'manifest': str(args.manifest),
                  'manifest_sha256': file_digest(args.manifest), 'split': args.split,
                  'available_instances': len(dataset), 'selected_instances': limit,
                  'partial_inference': limit < len(dataset), 'labels_read': False}
        items = (dataset[i] for i in range(limit))
    elif args.observation:
        observation = load_observation(args.observation)
        inputs = {'mode': 'observation', 'observation': str(args.observation),
                  'observation_sha256': file_digest(args.observation), 'labels_read': False}
        items = ({'observation': observation, 'record': {}},)
    else:
        observation = load_raw_observation(rgb_path=args.rgb, depth_path=args.depth, mask_path=args.mask,
            intrinsics_path=args.intrinsics, depth_scale_m=args.depth_scale_m, gravity=args.gravity,
            instance_id=args.instance_id, max_queries=args.max_queries,
            depth_min_m=args.depth_min_m, depth_max_m=args.depth_max_m)
        inputs = {'mode': 'raw', 'instance_id': args.instance_id,
                  'depth_scale_m': args.depth_scale_m, 'gravity': args.gravity,
                  'max_queries': args.max_queries, 'depth_min_m': args.depth_min_m,
                  'depth_max_m': args.depth_max_m,
                  'assets': {name: {'path': str(path), 'sha256': file_digest(path)}
                             for name, path in [('rgb', args.rgb), ('depth', args.depth),
                                                ('mask', args.mask), ('intrinsics', args.intrinsics)]},
                  'labels_read': False}
        items = ({'observation': observation, 'record': {}},)

    def predictions():
        for item in items:
            result = infer_instance(model, item['observation'], stage=checkpoint['stage'],
                                    ranking=args.ranking, alpha=args.alpha, top_k=args.top_k)
            record = item['record']
            result['input_record'] = {key: record[key] for key in
                ('sample_id', 'instance_uid', 'scene_id', 'frame_id', 'instance_row',
                 'camera', 'split', 'test_subset', 'observation_sha256') if key in record}
            yield result

    metadata = {'schema': INFERENCE_REPORT_SCHEMA, 'evaluation_protocol': 'native_instance_inference',
                'checkpoint': checkpoint, 'checkpoint_selection_policy': checkpoint['selection_policy'],
                'public_evaluation': PUBLIC_EVALUATION, 'inputs': inputs,
                'inference': {'ranking': ranking, 'alpha': alpha, 'top_k': args.top_k,
                              'device': args.device, 'source_task_head_is_untrained': checkpoint['stage'] == 'pretrain'}}
    _stream_report(args.output, metadata, predictions())
    print(f'Saved native instance predictions to {args.output}')


if __name__ == '__main__':
    main()
