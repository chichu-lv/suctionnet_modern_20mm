#!/usr/bin/env python3
"""Train a single-instance route; source best explicitly uses test selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suction20.instance_data import InstanceDataset, InstanceAttemptDataset
from suction20.instance_config import apply_model_runtime_overrides
from suction20.instance_selection import select_source_evaluation, validate_selection_config
from suction20.instance_training import (InstanceTrainer, check_source_splits,
    create_instance_model, load_instance_config, seed_all)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', choices=['pretrain', 'finetune'], default='pretrain')
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--selection-manifest', required=True, type=Path,
                        help='Source test manifest containing every configured selection frame; target val for finetune')
    parser.add_argument('--target-manifest', type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backbone-root', type=Path, help='Pinned external SpConv backbone directory for A')
    parser.add_argument('--query-chunk-size', type=int,
                        help='Query batching: A query_chunk_size / B query_chunk; resume must match its saved config')
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--resume', type=Path)
    start.add_argument('--init-checkpoint', type=Path, help='Same-route pretrain checkpoint; finetune only')
    parser.add_argument('--allow-partial-manifests', action='store_true',
                        help='Explicit smoke/shard run, recorded as partial and not a formal complete result')
    parser.add_argument('--stop-after', type=int,
                        help='Pause at this successful update; preserves original budget identity for resume')
    args = parser.parse_args(argv)
    config = load_instance_config(args.config)
    try:
        config = apply_model_runtime_overrides(config, backbone_root=args.backbone_root,
                                               query_chunk_size=args.query_chunk_size)
    except ValueError as error:
        parser.error(str(error))
    if config['route'] == 'a' and config['model'].get('backbone_root') is None:
        parser.error('Route A requires --backbone-root or model.backbone_root in the config')
    if args.stage == 'pretrain' and (args.target_manifest or args.init_checkpoint):
        parser.error('Source pretrain starts from scratch or strict resume; target data is not allowed')
    if args.stage == 'finetune' and (not args.target_manifest or not (args.init_checkpoint or args.resume)):
        parser.error('Finetune requires --target-manifest and --init-checkpoint or --resume')
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    if not args.resume and (args.run_dir/'last.pt').exists():
        parser.error('Run directory already has a checkpoint; use --resume or a new directory')
    seed_all(int(config['seed']))
    source = InstanceDataset(args.source_manifest, split='train',
                             training_queries=int(config['pretrain']['queries']))
    check_source_splits(source, 'train', allow_partial=args.allow_partial_manifests)
    if args.stage == 'pretrain':
        selection = InstanceDataset(args.selection_manifest, split='test')
        selection = select_source_evaluation(selection, validate_selection_config(config['pretrain']),
                                             allow_partial=args.allow_partial_manifests)
        target = None
    else:
        target = InstanceAttemptDataset(args.target_manifest, split='train')
        selection = InstanceAttemptDataset(args.selection_manifest, split='val')
    model = create_instance_model(config, args.device)
    trainer = InstanceTrainer(model, config, source, selection, args.run_dir,
                              stage=args.stage, target_dataset=target,
                              allow_partial=args.allow_partial_manifests)
    if args.resume:
        trainer.resume(args.resume)
    if args.init_checkpoint:
        trainer.initialize_finetune(args.init_checkpoint)
    trainer.run(stop_after=args.stop_after)


if __name__ == '__main__':
    main()
