"""Durable pretraining transactions and the local supervisor wire protocol.

The manifest is the only commit point. Checkpoint blobs and attempt logs are
immutable once referenced; last.pt/best.pt are repairable compatibility aliases.
This module deliberately has no model, data, or CUDA dependencies.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid


def normalize_gpu_uuid(value):
    """Torch exposes bare UUIDs; NVML and CUDA_VISIBLE_DEVICES use GPU- UUIDs."""
    return "GPU-" + str(uuid.UUID(str(value).removeprefix("GPU-")))


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_digest(path, length=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining = length
        while remaining is None or remaining > 0:
            block = stream.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not block:
                if remaining:
                    raise ValueError(f'Truncated committed file: {path}')
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())



def verify_performance_migration(run_dir, reference, identity, *, _visited=None):
    """Verify the immutable receipt and archived inputs without loading tensors."""
    root = Path(run_dir).resolve()
    relative = Path(reference['path'])
    if relative.is_absolute() or len(relative.parts) != 3 or relative.parts[0] != 'migrations' or relative.name != 'receipt.json':
        raise ValueError('Invalid performance migration receipt path')
    receipt_path = (root / relative).resolve()
    visited = set() if _visited is None else _visited
    if receipt_path in visited:
        raise ValueError('Cyclic performance migration archive')
    visited.add(receipt_path)
    if root not in receipt_path.parents or file_digest(receipt_path) != reference['sha256']:
        raise ValueError('Performance migration receipt checksum/path mismatch')
    receipt = read_json(receipt_path)
    if receipt.get('schema_version') != 'suction20.performance-migration.v1' or receipt.get('target_identity') != identity:
        raise ValueError('Performance migration target identity mismatch')
    from .performance import approved_migration_source, load_performance_config, SCHEMA_VERSION_V2
    source = receipt['source_identity']
    source_kind = approved_migration_source(source)
    changed = sorted(key for key in set(source) | set(identity) if source.get(key) != identity.get(key))
    if changed != receipt.get('allowed_changes') or set(changed) - {'code_revision', 'worktree_sha256', 'performance'} or 'performance' not in changed or not identity.get('performance'):
        raise ValueError('Performance migration changed scientific/environment identity')
    required = {'source.pt', 'source-manifest.json', 'performance-config.json'}
    if set(receipt.get('files', {})) != required:
        raise ValueError('Incomplete performance migration archive')
    for name, item in receipt['files'].items():
        path = receipt_path.parent / name
        if path.resolve().parent != receipt_path.parent or path.stat().st_size != item['size_bytes'] or file_digest(path) != item['sha256']:
            raise ValueError('Performance migration archive checksum mismatch')
    if receipt['files']['performance-config.json']['sha256'] != identity['performance']['config_sha256']:
        raise ValueError('Performance migration sidecar identity mismatch')
    settings = load_performance_config(receipt_path.parent / 'performance-config.json')
    if settings['schema_version'] != identity['performance']['schema_version']:
        raise ValueError('Performance migration sidecar schema mismatch')
    if source_kind == 'p1' and settings['schema_version'] != SCHEMA_VERSION_V2:
        raise ValueError('Approved P1 source requires a v2 performance target')
    manifest = read_json(receipt_path.parent / 'source-manifest.json')
    current = manifest['current']
    if (current['sha256'] != receipt['files']['source.pt']['sha256'] or
            current['global_step'] != receipt['global_step'] or current['attempt_id'] != receipt['source_attempt_id'] or
            current.get('pending_validation') is not None or current.get('analysis', {}).get('identity') != source):
        raise ValueError('Performance migration source manifest mismatch')
    previous = current.get('analysis', {}).get('performance_migration')
    if source_kind == 'p1':
        if not previous:
            raise ValueError('P1 source requires its prior performance migration receipt')
        verify_performance_migration(root, previous, source, _visited=visited)
    elif previous:
        raise ValueError('Original performance source must not already have a migration')
    return receipt


class RunLock:
    """Local defense in depth; the coordinator owns cross-host exclusion."""
    def __init__(self, run_dir):
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.stream = (path / '.trainer.lock').open('a')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError('Another trainer owns this run directory') from None

    def close(self):
        self.stream.close()


class CheckpointStore:
    def __init__(self, run_dir):
        self.root = Path(run_dir)
        self.blobs = self.root / 'checkpoints'
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'checkpoint-manifest.json'
        self.manifest = read_json(self.path) if self.path.exists() else None
        if self.manifest and self.manifest.get('schema_version') != 1:
            raise ValueError('Unsupported checkpoint manifest')
        self.current = None
        self.recovery = None

    def verify(self, entry):
        path = self.root / entry['path']
        if path.parent.resolve() != self.blobs.resolve():
            raise ValueError('Checkpoint path is outside checkpoint store')
        if path.stat().st_size != entry['size_bytes'] or file_digest(path) != entry['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {path}')
        return path

    def recover(self, requested):
        if not self.manifest:
            raise ValueError('Resume requires a committed checkpoint manifest')
        current = self.manifest['current']
        # A caller must ask for the current chain, never silently load another run.
        requested = Path(requested).absolute()
        allowed = {self.root.absolute() / 'last.pt', (self.root / current['path']).absolute()}
        if requested not in allowed:
            raise ValueError('Resume must name this run last.pt or current committed checkpoint')
        try:
            path = self.verify(current)
            self.current = current
        except (ValueError, OSError) as error:
            predecessor = self.manifest.get('predecessor')
            if not predecessor:
                raise ValueError('Current checkpoint invalid and no verified predecessor') from error
            path = self.verify(predecessor)
            self.current = predecessor
            self.recovery = {'reason': str(error), 'abandoned_checkpoint': current,
                             'resumed_checkpoint': predecessor,
                             'rollback_steps': current['global_step'] - predecessor['global_step']}
        return path

    def _alias(self, name, entry):
        temporary = self.root / f'.{name}.{uuid.uuid4().hex}.tmp'
        try:
            os.link(self.root / entry['path'], temporary)
            os.replace(temporary, self.root / name)
        finally:
            temporary.unlink(missing_ok=True)

    def publish(self, writer, *, step, attempt_id, pending, best_step=None, retain=False, promote_best=False, analysis=None):
        """writer(path) must fsync a complete checkpoint before returning."""
        generation = uuid.uuid4().hex
        path = self.blobs / f'{step:06d}-{generation}.pt'
        writer(path)
        entry = {'path': str(path.relative_to(self.root)), 'sha256': file_digest(path),
                 'size_bytes': path.stat().st_size, 'global_step': step,
                 'generation': generation, 'attempt_id': attempt_id,
                 'pending_validation': copy.deepcopy(pending)}
        if analysis is not None:
            entry['analysis'] = copy.deepcopy(analysis)
            # Public aliases must share only the frozen publication snapshot,
            # never caller-owned lists updated by subsequent validations.
            analysis = entry['analysis']
            entry.update(effective_samples_completed=analysis['counters']['effective_samples_completed'],
                         budget=analysis['budget'], runtime_profile=analysis['identity'].get('runtime_profile'),
                         contract_version=analysis['budget']['contract_version'],
                         completed_validation_nodes=analysis['completed_validation_nodes'],
                         last_completed_validation=analysis['last_completed_validation'])
        # Never retain progress that was discarded during predecessor recovery.
        nodes = {key: value for key, value in (self.manifest or {}).get('nodes', {}).items()
                 if int(key) <= step}
        if retain:
            nodes[str(step)] = entry
        best = None
        if best_step is not None:
            if promote_best:
                if best_step != step or pending:
                    raise ValueError('Best promotion requires completed validation at this step')
                best = entry
            else:
                previous_best = (self.manifest or {}).get('best')
                if previous_best and previous_best['global_step'] == best_step:
                    best = previous_best
                else:
                    best = nodes.get(str(best_step))
            if best is None:
                raise ValueError('Best checkpoint missing from committed chain')
            self.verify(best)
        manifest = {'schema_version': 1, 'current': entry, 'predecessor': self.current,
                    'best': best, 'nodes': nodes, 'published_unix': time.time()}
        atomic_json(self.path, manifest)  # sole commit point, aliases follow
        self.manifest, self.current = manifest, entry
        self._alias('last.pt', entry)
        if best:
            self._alias('best.pt', best)
        elif (self.root / 'best.pt').exists():
            (self.root / 'best.pt').unlink()
        fsync_directory(self.root)
        # Only unreferenced blobs are disposable. Predecessor and all evidence
        # nodes remain pinned, including pending validation's source generation.
        pinned = {item['path'] for item in [entry, manifest['predecessor'], best, *nodes.values()] if item}
        for old in self.blobs.glob('*.pt'):
            if str(old.relative_to(self.root)) not in pinned:
                old.unlink()
        fsync_directory(self.blobs)
        return entry


class AttemptLog:
    def __init__(self, run_dir, attempt_id, *, identity, previous=None, recovery=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', attempt_id):
            raise ValueError('Invalid attempt ID')
        self.root = Path(run_dir)
        self.directory = self.root / 'attempts' / attempt_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.attempt_id = attempt_id
        self.lineage = copy.deepcopy((previous or {}).get('lineage', []))
        self.events_lineage = copy.deepcopy((previous or {}).get('events_lineage', []))
        for segment in self.lineage + self.events_lineage:
            path = self.root / segment['path']
            if file_digest(path, segment['end_offset']) != segment['sha256']:
                raise ValueError('Committed attempt log checksum mismatch')
        ends = {segment['path']: segment['end_offset'] for segment in self.lineage + self.events_lineage}
        abandoned = []
        for path in sorted((self.root / 'attempts').glob('*/*.jsonl')):
            name = str(path.relative_to(self.root))
            offset = ends.get(name, 0)
            if path.stat().st_size > offset:
                abandoned.append({'path': name, 'start_offset': offset,
                                  'end_offset': path.stat().st_size, 'sha256': file_digest(path)})
        atomic_json(self.directory / 'identity.json', {'attempt_id': attempt_id, 'identity': identity,
                    'created_unix': time.time(), 'parent_lineage': self.lineage,
                    'abandoned_tails': [tail for tail in abandoned if tail['path'].endswith('/updates.jsonl')],
                    'abandoned_event_tails': [tail for tail in abandoned if tail['path'].endswith('/events.jsonl')],
                    'checkpoint_recovery': recovery})
        self.path = self.directory / 'updates.jsonl'
        self.stream = self.path.open('xb')
        self.events_path = self.directory / 'events.jsonl'
        self.events_stream = self.events_path.open('xb')
        self.attempts = int((previous or {}).get('attempts', 0))
        self.skipped = int((previous or {}).get('skipped', 0))
        self.frames = int((previous or {}).get('frames', 0))

    def append(self, row):
        self.attempts += 1
        self.skipped += int(not row['updated'])
        self.frames += len(row['frames'])
        row = dict(row, attempt_id=self.attempt_id, optimizer_attempt=self.attempts)
        self.stream.write((json.dumps(row, allow_nan=False) + '\n').encode())
        self.stream.flush()

    def event(self, kind, **fields):
        row = dict(schema_version='suction20.event.v1', event_id=uuid.uuid4().hex,
                   attempt_id=self.attempt_id, kind=kind, utc=datetime.now(timezone.utc).isoformat(),
                   unix=time.time(), monotonic=time.monotonic(), **fields)
        self.events_stream.write((json.dumps(row, allow_nan=False) + '\n').encode())
        self.events_stream.flush()
        return row

    def snapshot(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.events_stream.flush()
        os.fsync(self.events_stream.fileno())
        fsync_directory(self.directory)
        event_segment = {'attempt_id': self.attempt_id, 'path': str(self.events_path.relative_to(self.root)),
                         'end_offset': self.events_stream.tell(), 'sha256': file_digest(self.events_path)}
        segment = {'attempt_id': self.attempt_id, 'path': str(self.path.relative_to(self.root)),
                   'end_offset': self.stream.tell(), 'sha256': file_digest(self.path)}
        return {'lineage': self.lineage + [segment], 'events_lineage': self.events_lineage + [event_segment],
                'attempts': self.attempts,
                'skipped': self.skipped, 'frames': self.frames}

    def close(self):
        self.stream.close()
        self.events_stream.close()


class Progress:
    def __init__(self, directory, *, run, experiment, attempt_id, target_step):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        self.state = dict(schema_version=1, run=str(Path(run).resolve()), experiment=experiment,
                          attempt_id=attempt_id, trainer_pid=os.getpid(),
                          cuda_context_ready=False, phase='initializing', global_step=0,
                          target_step=target_step, progress_unix=time.time(), durable_checkpoint=None,
                          validation=None)
        self.update()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def _heartbeat(self):
        while not self._stop.wait(60):
            try:
                self.update()
            except Exception as error:
                self._error = error
                return

    def update(self, *, progressed=False, **values):
        if self._error:
            raise RuntimeError('Status heartbeat publication failed') from self._error
        with self._lock:
            self.state.update(values)
            self.state['heartbeat_unix'] = time.time()
            if progressed:
                self.state['progress_unix'] = time.time()
            atomic_json(self.directory / 'status.json', self.state)

    def pause_requested(self):
        return (self.directory / 'pause.request').exists()

    def validation_grant(self, checkpoint, step):
        request = {'experiment': self.state['experiment'], 'attempt_id': self.state['attempt_id'],
                   'step': step, 'checkpoint_sha256': checkpoint['sha256']}
        request['request_id'] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        atomic_json(self.directory / 'validation.request.json', request)
        self.update(phase='waiting_validation')
        while not self.pause_requested():
            try:
                grant = read_json(self.directory / 'validation.grant.json')
            except FileNotFoundError:
                grant = {}
            if all(grant.get(key) == value for key, value in request.items()) and grant.get('token', grant.get('assignment_token')):
                return grant
            self._stop.wait(1)
            if self._error:
                raise RuntimeError('Status heartbeat publication failed') from self._error
        return None

    def close(self):
        self._stop.set()
        self._thread.join()
