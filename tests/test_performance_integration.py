"""Normal CPU integration: exact continuation, archived migration, sidecar binding."""
import copy
import json
import os
from pathlib import Path

import torch

from scripts import migrate_performance_checkpoint as converter, train_pretrain as entry
from suction20 import pretrain_runtime as runtime, training, run_supervisor as supervisor
from suction20.batching import collate_public
from suction20.performance import attach_performance, load_performance_config, SOURCE_REVISION, SOURCE_WORKTREE_SHA256
from test_pretrain_runtime import harness, state, canonical_rows, same_training
from test_prefetch import ToySource
from test_run_supervisor import config, coordinator


def sidecar(tmp_path):
    path = tmp_path / 'performance.json'
    value = json.loads((__import__('pathlib').Path(__file__).parents[1] / 'configs/performance/cd009-p1.json').read_text())
    value['fps_backend'] = 'numpy'
    value['loader'].update(num_workers=0, pin_memory=False)
    path.write_text(json.dumps(value))
    return path


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.bn = torch.nn.BatchNorm1d(3)

    def set_epoch(self, epoch):
        pass

    def forward(self, inputs, **kwargs):
        xyz = inputs['xyz_cam_m']
        return {'value': self.linear(self.bn(xyz) + torch.rand_like(xyz)),
                'seed_valid': torch.ones(len(xyz), dtype=torch.bool)}


def install_toy(monkeypatch):
    ToySource.data_cfg = {'radius': .02}
    monkeypatch.setattr(entry, 'make_source_dataset', lambda *a, **k: ToySource())
    monkeypatch.setattr(entry, 'make_model', lambda *a: ToyModel())
    monkeypatch.setattr(entry, 'collate_public', collate_public)


def test_prefetched_trainer_preserves_ordinary_sequence_and_resume(tmp_path, harness, monkeypatch):
    config, _, _ = harness
    install_toy(monkeypatch)
    config['data']['smoke_frame_keys'] = [list(key) for key in ToySource.keys]
    config['training'].update(max_optimizer_steps=5, validate_every_optimizer_steps=5)
    original, prefetched = tmp_path / 'original', tmp_path / 'prefetched'
    entry.train(copy.deepcopy(config), original, torch.device('cpu'), pause_at_step=3)
    optimized = attach_performance(config, sidecar(tmp_path))
    entry.train(optimized, prefetched, torch.device('cpu'), pause_at_step=1)
    entry.train(optimized, prefetched, torch.device('cpu'), resume=prefetched / 'last.pt', pause_at_step=3)
    same_training(state(prefetched), state(original))
    assert [r['frame_details'] for r in canonical_rows(prefetched)] == [r['frame_details'] for r in canonical_rows(original)]
    assert all('loader_wait_s' in row and row['performance_identity'] for row in canonical_rows(prefetched))


def test_migration_preserves_state_nodes_and_source_through_normal_resume(tmp_path, harness, monkeypatch):
    config, _, _ = harness
    install_toy(monkeypatch)
    config['data']['smoke_frame_keys'] = [list(key) for key in ToySource.keys]
    label = tmp_path / 'labels'
    label.mkdir()
    runtime.atomic_json(label / 'config_frozen.json', ToySource.data_cfg)
    runtime.atomic_json(label / 'manifest.json', {'passed': True})
    config['data']['label_root'] = str(label)
    monkeypatch.setattr(entry, 'code_revision', lambda: SOURCE_REVISION)
    monkeypatch.setattr(entry, 'worktree_digest', lambda: SOURCE_WORKTREE_SHA256)
    run = tmp_path / 'run'
    entry.train(copy.deepcopy(config), run, torch.device('cpu'), pause_at_step=3)
    before = state(run)
    manifest = runtime.read_json(run / 'checkpoint-manifest.json')
    control = run / 'control'
    status = runtime.read_json(control / 'status.json')
    gone = dict(pid=99999999, start_ticks=1, boot_id='retired-test-process')
    runtime.atomic_json(control / 'worker.json', dict(token=status['attempt_id'], exit_code=75,
        worker_identity=gone, trainer_identity=gone))
    path = sidecar(tmp_path)
    result = converter.migrate(config, run, path, control, manifest['current']['sha256'])
    migrated = state(run)
    same_training(migrated, before)
    assert migrated['runtime_state']['ledger'] == before['runtime_state']['ledger']
    assert migrated['runtime_state']['validations'] == before['runtime_state']['validations']
    after_manifest = runtime.read_json(run / 'checkpoint-manifest.json')
    assert after_manifest['best'] == manifest['best']
    assert after_manifest['nodes'] == manifest['nodes']
    receipt = runtime.verify_performance_migration(run, result['receipt'], migrated['runtime_state']['identity'])
    archive = run / result['receipt']['path']
    assert os.stat(archive.parent / 'source.pt').st_ino == os.stat(run / manifest['current']['path']).st_ino
    monkeypatch.setattr(entry, 'code_revision', training.code_revision)
    monkeypatch.setattr(entry, 'worktree_digest', training.worktree_digest)
    entry.train(attach_performance(config, path), run, torch.device('cpu'), resume=run / 'last.pt')
    final = state(run)
    assert final['global_step'] == 4
    assert final['runtime_state']['performance_migration'] == result['receipt']
    assert runtime.file_digest(archive.parent / 'source.pt') == manifest['current']['sha256']
    assert receipt['global_step'] == 3
    # A subsequent ordinary save may prune source checkpoint names, never its archive.
    runtime.verify_performance_migration(run, result['receipt'], final['runtime_state']['identity'])
    backup = tmp_path / 'backup'
    backup.mkdir()
    backup_control = tmp_path / 'backup-control'
    backup_control.mkdir()
    monkeypatch.setattr(supervisor, 'assignment_paths', lambda *args: (run, backup_control))
    usage = supervisor.shutil.disk_usage
    monkeypatch.setattr(supervisor.shutil, 'disk_usage', lambda path: usage(path)._replace(free=300 * supervisor.GIB))
    destination = Path(supervisor.backup_snapshot({'root': str(tmp_path), 'backup_root': str(backup)},
        {'experiment': config['experiment'], 'token': 'backup-test', 'host': 'L20'},
        {'request_id': 'final', 'final': True}))
    copied_receipt = runtime.read_json(destination / result['receipt']['path'])
    for name, info in copied_receipt['files'].items():
        assert runtime.file_digest(destination / Path(result['receipt']['path']).parent / name) == info['sha256']
    assert canonical_rows(run)[-1]['performance_diagnostics'] == {'active': True, 'until_step_exclusive': 53}


def test_supervisor_binds_sidecar_independently_of_profile(coordinator, tmp_path):
    path = sidecar(tmp_path)
    coordinator.config['performance_configs'] = {'H20': {'E5': {
        'config_path': str(path), 'config_sha256': runtime.file_digest(path)}}}
    assignment = coordinator.assign('E5', 'H20', 0)
    command = supervisor.trainer_command(coordinator.config, assignment)
    assert command[command.index('--performance-config') + 1] == str(path)
    assert assignment['performance_settings'] == json.loads(path.read_text())
    assert assignment['performance_config_sha256'] == runtime.file_digest(path)
    old = dict(assignment)
    for key in ('performance_config_path', 'performance_config_sha256', 'performance_settings'):
        del old[key]
    old['view'] = {'performance_migration_verified': True, 'performance_source_backup_verified': True, 'committed_checkpoint': {
        'analysis': {'identity': {'performance': {'config_sha256': assignment['performance_config_sha256']}}}}}
    supervisor.verify_performance_transition(coordinator.config, old)


def test_loader_worker_resources_measure_owned_child():
    import subprocess
    import sys
    import time
    from suction20.training_telemetry import sample_loader_workers
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    try:
        identities = [supervisor.process_identity(child.pid)]
        samplers = {}
        first = sample_loader_workers(samplers, identities, os.getpid())
        assert first['rss_bytes'] > 0 and first['rss_complete']
        assert not first['cpu_complete']
        assert 'process.cpu_core_equivalents' in first['processes'][0]['missing_reasons']
        time.sleep(.02)
        second = sample_loader_workers(samplers, identities, os.getpid())
        assert second['cpu_complete'] and second['rss_bytes'] > 0
    finally:
        child.terminate()
        child.wait()


def test_scalar_and_batched_sidecars_preserve_profile_and_bind_distinct_hashes(tmp_path):
    default = Path(__file__).parents[1] / 'configs/performance/cd009-p1.json'
    settings = load_performance_config(default)
    assert settings['query_chunk_size'] is None and settings['query_workers'] == 1
    profile = {'runtime': {'raw_config_sha256': 'frozen-profile-sha'},
               'runtime_profile': {'profile_id': 'unchanged'}}
    original = copy.deepcopy(profile)
    scalar = attach_performance(profile, default)
    path = tmp_path / 'batched.json'
    settings['query_chunk_size'] = 64
    path.write_text(json.dumps(settings))
    batched = attach_performance(profile, path)
    assert profile == original
    assert scalar['runtime']['performance']['query_chunk_size'] is None
    assert batched['runtime']['performance']['query_chunk_size'] == 64
    assert scalar['runtime']['performance_config_sha256'] != batched['runtime']['performance_config_sha256']
    for resolved in (scalar, batched):
        assert resolved['runtime']['raw_config_sha256'] == 'frozen-profile-sha'
        assert resolved['runtime_profile'] == original['runtime_profile']
