#!/usr/bin/env python3
"""Evaluate native instance/recorded-action metrics, with checkpoint provenance.

Public paper AP and the proposed radius-only protocol remain deferred. Source
evaluation requires full canonical query labels, even for a training split.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suction20.instance_data import InstanceAttemptDataset, InstanceDataset
from suction20.instance_inference import load_instance_checkpoint
from suction20.instance_training import (atomic_json, check_source_splits,
    evaluate_instance_attempts, evaluate_instance_source, check_data_contract)


def _check_source_contract(dataset, metadata):
    checkpoint_data = metadata['checkpoint_identity'].get('source', {}).get('data_identity')
    if not isinstance(checkpoint_data, dict):
        raise ValueError('Checkpoint is missing the source preprocessing/teacher identity')
    check_data_contract(checkpoint_data, dataset.identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--domain', choices=['source', 'target'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backbone-root', type=Path, help='Pinned external SpConv backbone directory; route A only')
    parser.add_argument('--query-chunk-size', type=int,
                        help='Override query batching only: A query_chunk_size / B query_chunk')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test',
                        help='Source: train/test with full query labels; target: train/val/test')
    parser.add_argument('--allow-partial-manifests', action='store_true',
                        help='Explicit source smoke/shard evaluation; never a formal complete test result')
    args = parser.parse_args(argv)
    if args.domain == 'source' and args.split == 'val':
        parser.error('The source protocol has 100 train scenes and 90 test scenes, with no val split')
    if args.domain == 'target' and args.allow_partial_manifests:
        parser.error('--allow-partial-manifests describes source scene coverage only')
    if args.query_chunk_size is not None and args.query_chunk_size < 1:
        parser.error('--query-chunk-size must be positive')

    runtime_options = {'backbone_root': args.backbone_root}
    if args.query_chunk_size is not None:
        runtime_options['query_chunk_size'] = args.query_chunk_size
    model, metadata = load_instance_checkpoint(args.checkpoint, device=args.device,
                                              **runtime_options)
    if args.backbone_root is not None and metadata['route'] != 'a':
        raise ValueError('--backbone-root is only valid for route A')
    if args.domain == 'source':
        dataset = InstanceDataset(args.manifest, split=args.split, require_labels=True)
        coverage = check_source_splits(dataset, args.split, allow_partial=args.allow_partial_manifests)
        _check_source_contract(dataset, metadata)
        metrics, rows = evaluate_instance_source(model, dataset)
        protocol = 'native_INST20_not_public_AP'
        ranking = 'q_inst'
    else:
        if metadata['stage'] != 'finetune':
            raise ValueError('Target action probabilities require a finetuned checkpoint; the source task head is untrained')
        dataset = InstanceAttemptDataset(args.manifest, split=args.split)
        if not len(dataset):
            raise ValueError(f'No target observations in requested split: {args.split}')
        metrics, rows = evaluate_instance_attempts(model, dataset)
        coverage = {'split': args.split, 'grouped_observations': len(dataset)}
        protocol = 'native_recorded_attempt_success_v1'
        ranking = 'task_probability_at_recorded_pose'

    policy = deepcopy(metadata['selection_policy'])
    source_policy = policy if metadata['stage'] == 'pretrain' else (
        metadata.get('initialization') or {}).get('selection_policy')
    report = {
        'schema': 'instance_native_evaluation_v1',
        'domain': args.domain, 'split': args.split, 'evaluation_protocol': protocol,
        'evaluation_scope': ('partial_manifest_smoke' if args.allow_partial_manifests else 'complete_manifest'),
        'ranking': ranking, 'checkpoint': deepcopy(metadata),
        'checkpoint_selection_policy': policy,
        'checkpoint_selection_role': metadata['checkpoint_selection_role'],
        'test_selected': metadata['checkpoint_selection_role'] == 'test_selected',
        'source_run_uses_test_selection': (source_policy['selection_role'] == 'test_selected'
                                           if source_policy is not None else None),
        'source_pretrain_selection_scope': source_policy.get('selection_scope') if source_policy else None,
        'independent_test_estimate': False,
        'public_evaluation': {'status': 'deferred', 'radius_only_protocol': None,
                              'paper_comparison_ready': False},
        'dataset_identity': deepcopy(dataset.identity), 'coverage': coverage,
        'metrics': metrics, 'records': rows,
    }
    atomic_json(args.output, report)
    print(json.dumps({'output': str(args.output), 'domain': args.domain, 'split': args.split,
        'evaluation_protocol': protocol, 'test_selected': report['test_selected'],
        'independent_test_estimate': False, 'metrics': metrics}, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
