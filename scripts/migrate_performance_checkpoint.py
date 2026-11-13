#!/usr/bin/env python3
"""Explicit, one-way conversion of an approved paused original, P1 or P2 run.

Pause every trainer using this checkout, wait for worker exit/final backup, install
and verify the target build, then run on its owning host with the frozen profile:
  python scripts/migrate_performance_checkpoint.py --config PROFILE.json \
    --performance-config configs/performance/cd009-p3.json --run-dir RUN \
    --control-dir CONTROL/attempts/TOKEN --expected-source-sha256 SHA
Use --source-root/--label-root/--backbone-root exactly as the supervisor does.
No optimizer is constructed and no model, RNG, sampler or training counter changes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suction20.performance import (SCHEMA_VERSION_V2, SCHEMA_VERSION_V3, approved_migration_source,
                                  attach_performance, build_runtime_identity)
from suction20.pretrain_budget import resolve_config, resolve_budget, validate_resume_progress
from suction20.pretrain_runtime import (CheckpointStore, RunLock, atomic_json, file_digest,
    fsync_directory, read_json, verify_performance_migration)
from suction20.training import load_config, validate_resume_config, write_checkpoint_state


def migrate(config, run_dir, performance_path, control_dir, expected_source_sha256):
    import torch
    from suction20.run_supervisor import identity_alive
    run = Path(run_dir).resolve()
    control = Path(control_dir)
    worker, status = read_json(control / 'worker.json'), read_json(control / 'status.json')
    if (worker.get('exit_code') != 75 or status.get('phase') != 'paused' or
            worker.get('token') != status.get('attempt_id') or status.get('run') != str(run) or
            status.get('experiment') != config.get('experiment') or
            any(not worker.get(k) or identity_alive(worker[k]) for k in ('worker_identity', 'trainer_identity'))):
        raise ValueError('Migration requires positively exited paused worker and trainer')
    guard = RunLock(run)
    try:
        store = CheckpointStore(run)
        if not store.manifest:
            raise ValueError('Migration requires committed current checkpoint')
        current = store.manifest['current']
        source_path = store.verify(current)  # Never silently choose predecessor.
        if current['sha256'] != expected_source_sha256 or current.get('pending_validation') is not None or current.get('attempt_id') != status.get('attempt_id'):
            raise ValueError('Source checksum differs or validation transaction is pending')
        state = torch.load(source_path, map_location='cpu', weights_only=False)
        if state.get('stage') != 'public_pretrain' or state.get('experiment') != config.get('experiment'):
            raise ValueError('Migration requires the same public pretraining experiment')
        previous = state['runtime_state']
        source_identity = previous['identity']
        source_kind = approved_migration_source(source_identity)
        prior_migration = previous.get('performance_migration')
        if source_kind in ('p1', 'p2'):
            # A run started directly on P2 (such as E2) has no older receipt.
            if ((source_kind == 'p1' and not prior_migration) or
                    current.get('analysis', {}).get('performance_migration') != prior_migration):
                raise ValueError('Performance source requires matching checkpoint/manifest migration receipt')
            if prior_migration:
                verify_performance_migration(run, prior_migration, source_identity)
        elif prior_migration or current.get('analysis', {}).get('performance_migration'):
            raise ValueError('Original performance source must not already have a migration')
        if previous.get('pending_validation') is not None or state['global_step'] != current['global_step'] or status.get('global_step') != state['global_step']:
            raise ValueError('Source checkpoint/status transaction mismatch')
        config = resolve_config(attach_performance(config, performance_path))
        if source_kind == 'p1' and config['runtime']['performance']['schema_version'] != SCHEMA_VERSION_V2:
            raise ValueError('Approved P1 source requires a v2 performance target')
        if source_kind == 'p2' and config['runtime']['performance']['schema_version'] != SCHEMA_VERSION_V3:
            raise ValueError('Approved P2 source requires a v3 performance target')
        source_data = json.loads((Path(config['data']['label_root']) / 'config_frozen.json').read_bytes())
        target_identity = build_runtime_identity(config, source_data)
        validate_resume_config(state['training_config'], config)
        validate_resume_progress(state, resolve_budget(config))
        changed = sorted(k for k in set(source_identity) | set(target_identity) if source_identity.get(k) != target_identity.get(k))
        if set(changed) - {'code_revision', 'worktree_sha256', 'performance'} or 'performance' not in changed:
            raise ValueError('Migration changes scientific/data/dependency/math identity')
        analysis = copy.deepcopy(current['analysis'])
        if analysis.get('identity') != source_identity or analysis.get('ledger') != previous['ledger']:
            raise ValueError('Source analysis differs from checkpoint identity/ledger')
        migration_id = 'performance-' + uuid.uuid4().hex
        directory = run / 'migrations' / migration_id
        directory.mkdir(parents=True, exist_ok=False)
        os.link(source_path, directory / 'source.pt')
        (directory / 'source-manifest.json').write_bytes(store.path.read_bytes())
        (directory / 'performance-config.json').write_bytes(Path(performance_path).read_bytes())
        files = {}
        for path in directory.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            files[path.name] = dict(size_bytes=path.stat().st_size, sha256=file_digest(path))
        receipt = dict(schema_version='suction20.performance-migration.v1', migration_id=migration_id,
            source_identity=source_identity, target_identity=target_identity, global_step=state['global_step'],
            allowed_changes=changed, files=files, source_attempt_id=current['attempt_id'])
        atomic_json(directory / 'receipt.json', receipt)
        fsync_directory(directory.parent)
        reference = dict(path=str((directory / 'receipt.json').relative_to(run)), sha256=file_digest(directory / 'receipt.json'))
        verify_performance_migration(run, reference, target_identity)
        previous['identity'] = target_identity
        previous['performance_migration'] = reference
        state['training_config'] = config
        state['code_revision'] = target_identity['code_revision']
        state['code_worktree_sha256'] = target_identity['worktree_sha256']
        analysis.update(identity=target_identity, attempt_id=migration_id, performance_migration=reference)
        store.current = current
        best = store.manifest.get('best')
        entry = store.publish(lambda path: write_checkpoint_state(path, state), step=state['global_step'],
            attempt_id=migration_id, pending=None, best_step=best['global_step'] if best else None,
            retain=False, promote_best=False, analysis=analysis)
        return dict(checkpoint=entry, receipt=reference)
    finally:
        guard.close()


def main(argv=None):
    from suction20.runner import override_paths
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'performance-config', 'run-dir', 'control-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--expected-source-sha256', '--expected-source-checkpoint-sha256', required=True)
    for name in ('source-root', 'label-root', 'backbone-root'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    config = override_paths(load_config(args.config), args)
    config.setdefault('runtime', {})['raw_config_sha256'] = file_digest(args.config)
    result = migrate(config, args.run_dir, args.performance_config, args.control_dir, args.expected_source_sha256)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
