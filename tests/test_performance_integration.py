"""Normal CPU integration: exact continuation, archived migration, sidecar binding."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from scripts import migrate_performance_checkpoint as converter, train_pretrain as entry
from suction20 import pretrain_runtime as runtime, training, run_supervisor as supervisor
from suction20.batching import collate_public
from suction20.performance import attach_performance, load_performance_config, SOURCE_REVISION, SOURCE_WORKTREE_SHA256
from suction20 import performance
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


def test_v2_schema_is_explicit_and_v1_identity_stays_unchanged(tmp_path, monkeypatch):
    base = Path(__file__).parents[1] / 'configs/performance'
    p1 = load_performance_config(base / 'cd009-p1.json')
    p2 = load_performance_config(base / 'cd009-p2.json')
    assert runtime.file_digest(base / 'cd009-p1.json') == performance.P1_SOURCE_CONFIG_SHA256
    assert 'spatial_index' not in p1
    assert p2['spatial_index'] == {'backend': 'uniform_grid_cpu', 'cell_size_m': .02}
    assert p2['loader'] == p1['loader']
    assert p2['query_chunk_size'] is None and p2['query_workers'] == 1
    for value in (dict(p1, no_instance_voxel_heads=True), dict(p2, backbone_coordinate_fastpath=1),
                  dict(p2, spatial_index={'backend': 'uniform_grid_cpu', 'cell_size_m': .002})):
        with pytest.raises(ValueError):
            performance.validate_performance_config(value)
    from models import geometry_ops, spatial_index
    monkeypatch.setattr(geometry_ops, 'fps_backend_identity', lambda name: {'backend': name})
    monkeypatch.setattr(spatial_index, 'spatial_index_backend_identity', lambda name: {'backend': name, 'binary_sha256': 'grid-build'})
    old = performance.performance_identity(attach_performance({}, base / 'cd009-p1.json'))
    assert old == {'schema_version': performance.SCHEMA_VERSION,
                   'config_sha256': performance.P1_SOURCE_CONFIG_SHA256,
                   'backend_builds': {'fps': {'backend': 'compiled'}}}
    new = performance.performance_identity(attach_performance({}, base / 'cd009-p2.json'))
    assert new['schema_version'] == performance.SCHEMA_VERSION_V2
    assert new['backend_builds']['spatial_index'] == {'backend': 'uniform_grid_cpu', 'binary_sha256': 'grid-build'}


def test_v3_sidecar_binds_native_contact_abi_without_changing_v2(monkeypatch):
    from types import SimpleNamespace
    from models import geometry_ops, spatial_index
    base = Path(__file__).parents[1] / 'configs/performance'
    p2 = load_performance_config(base / 'cd009-p2.json')
    p3 = load_performance_config(base / 'cd009-p3.json')
    assert runtime.file_digest(base / 'cd009-p2.json') == performance.P2_SOURCE_CONFIG_SHA256
    assert p3 == dict(p2, schema_version=performance.SCHEMA_VERSION_V3,
                      contact_query_backend='native_cpu')
    reference = dict(p3, contact_query_backend='numpy')
    assert performance.validate_performance_config(reference) == reference
    monkeypatch.setattr(geometry_ops, 'fps_backend_identity', lambda name: {'backend': name})
    monkeypatch.setattr(spatial_index, 'spatial_index_backend_identity',
                        lambda name: {'backend': name, 'binary_sha256': 'grid-build'})
    monkeypatch.setattr(spatial_index, '_compiled_grid', lambda: SimpleNamespace(CONTACT_ABI_VERSION=1))
    old = performance.performance_identity(attach_performance({}, base / 'cd009-p2.json'))
    new = performance.performance_identity(attach_performance({}, base / 'cd009-p3.json'))
    assert new['schema_version'] == performance.SCHEMA_VERSION_V3
    assert new['backend_builds']['fps'] == old['backend_builds']['fps']
    assert new['backend_builds']['spatial_index'] == dict(
        old['backend_builds']['spatial_index'], contact_abi_version=1)


@pytest.mark.parametrize('p2_start', ['fresh', 'migrated'])
def test_p2_to_p3_preserves_normal_continuation_and_archive(tmp_path, harness, monkeypatch, p2_start):
    from types import SimpleNamespace
    from models import geometry_ops, spatial_index
    from suction20 import prefetch
    cfg, _, _ = harness
    install_toy(monkeypatch)
    # Exercise real config/identity/checkpoint code with a CPU toy network. Native
    # kernel equivalence is covered separately; this test needs no compiled grid.
    monkeypatch.setattr(geometry_ops, 'fps_backend_identity', lambda name: {'backend': name})
    monkeypatch.setattr(spatial_index, 'spatial_index_backend_identity',
                        lambda name: {'backend': name, 'binary_sha256': 'grid-build'})
    monkeypatch.setattr(spatial_index, '_compiled_grid', lambda: SimpleNamespace(CONTACT_ABI_VERSION=1))
    make_loader = prefetch.make_ticket_loader
    def serial_loader(*args, loader_config, **kwargs):
        return make_loader(*args, loader_config=dict(loader_config, num_workers=0, pin_memory=False), **kwargs)
    monkeypatch.setattr(prefetch, 'make_ticket_loader', serial_loader)
    cfg['data']['smoke_frame_keys'] = [list(key) for key in ToySource.keys]
    cfg['training'].update(max_optimizer_steps=5, validate_every_optimizer_steps=5)
    label = tmp_path / 'labels'
    runtime.atomic_json(label / 'config_frozen.json', ToySource.data_cfg)
    runtime.atomic_json(label / 'manifest.json', {'passed': True})
    cfg['data']['label_root'] = str(label)
    run, control = tmp_path / 'run', tmp_path / 'run/control'
    base = Path(__file__).parents[1] / 'configs/performance'
    p1_path, p2_path, p3_path = [base / f'cd009-p{version}.json' for version in (1, 2, 3)]
    p2_config = attach_performance(cfg, p2_path)

    def select_code(revision, worktree):
        for module in (entry, training):
            monkeypatch.setattr(module, 'code_revision', lambda: revision)
            monkeypatch.setattr(module, 'worktree_digest', lambda: worktree)

    def paused_source():
        status = runtime.read_json(control / 'status.json')
        gone = dict(pid=99999999, start_ticks=1, boot_id='retired-test-process')
        runtime.atomic_json(control / 'worker.json', dict(token=status['attempt_id'], exit_code=75,
            worker_identity=gone, trainer_identity=gone))
        return runtime.read_json(run / 'checkpoint-manifest.json')['current']

    select_code(performance.P2_SOURCE_REVISION, performance.P2_SOURCE_WORKTREE_SHA256)
    continuous = tmp_path / 'continuous'
    entry.train(p2_config, continuous, torch.device('cpu'))
    references = []
    if p2_start == 'migrated':
        select_code(SOURCE_REVISION, SOURCE_WORKTREE_SHA256)
        entry.train(copy.deepcopy(cfg), run, torch.device('cpu'), pause_at_step=1)
        source = paused_source()
        select_code(performance.P1_SOURCE_REVISION, performance.P1_SOURCE_WORKTREE_SHA256)
        references.append(converter.migrate(cfg, run, p1_path, control, source['sha256'])['receipt'])
        entry.train(attach_performance(cfg, p1_path), run, torch.device('cpu'), resume=run / 'last.pt', pause_at_step=2)
        source = paused_source()
        select_code(performance.P2_SOURCE_REVISION, performance.P2_SOURCE_WORKTREE_SHA256)
        references.append(converter.migrate(cfg, run, p2_path, control, source['sha256'])['receipt'])
        entry.train(p2_config, run, torch.device('cpu'), resume=run / 'last.pt', pause_at_step=3)
    else:
        entry.train(p2_config, run, torch.device('cpu'), pause_at_step=3)

    source = paused_source()
    before = state(run)
    manifest_before = runtime.read_json(run / 'checkpoint-manifest.json')
    assert performance.approved_migration_source(before['runtime_state']['identity']) == 'p2'
    assert bool(before['runtime_state'].get('performance_migration')) == (p2_start == 'migrated')
    select_code('test-p3-build', 'test-p3-worktree')
    result = converter.migrate(cfg, run, p3_path, control, source['sha256'])
    references.append(result['receipt'])
    migrated = state(run)
    same_training(migrated, before)
    for key in ('ledger', 'validations', 'budget', 'effective_samples_completed'):
        assert migrated['runtime_state'][key] == before['runtime_state'][key]
    manifest_after = runtime.read_json(run / 'checkpoint-manifest.json')
    assert manifest_after['best'] == manifest_before['best']
    assert manifest_after['nodes'] == manifest_before['nodes']
    receipt = runtime.verify_performance_migration(run, result['receipt'], migrated['runtime_state']['identity'])
    assert receipt['source_identity'] == before['runtime_state']['identity']

    entry.train(attach_performance(cfg, p3_path), run, torch.device('cpu'), resume=run / 'last.pt')
    final = state(run)
    same_training(final, state(continuous))
    assert [r['frame_details'] for r in canonical_rows(run)] == [r['frame_details'] for r in canonical_rows(continuous)]
    assert final['runtime_state']['performance_migration'] == result['receipt']
    runtime.verify_performance_migration(run, result['receipt'], final['runtime_state']['identity'])

    # Ordinary backup follows both an existing original/P1 chain and a fresh P2
    # starting point after normal saves have replaced checkpoint predecessors.
    backup = tmp_path / 'backup'
    backup.mkdir()
    monkeypatch.setattr(supervisor, 'assignment_paths', lambda *args: (run, control))
    usage = supervisor.shutil.disk_usage
    monkeypatch.setattr(supervisor.shutil, 'disk_usage', lambda path: usage(path)._replace(free=300 * supervisor.GIB))
    destination = Path(supervisor.backup_snapshot({'root': str(tmp_path), 'backup_root': str(backup)},
        {'experiment': cfg['experiment'], 'token': 'backup-test', 'host': 'L20'},
        {'request_id': 'p3-final', 'final': True}))
    for reference in references:
        archived = runtime.read_json(destination / reference['path'])
        assert runtime.file_digest(destination / reference['path']) == reference['sha256']
        for name, info in archived['files'].items():
            assert runtime.file_digest(destination / Path(reference['path']).parent / name) == info['sha256']


def test_p1_to_p2_resume_retains_recursive_archive_after_predecessor_replacement(tmp_path, harness, monkeypatch):
    cfg, _, _ = harness
    install_toy(monkeypatch)
    # Keep the exact approved P1 sidecar bytes/identity. Worker scheduling has its
    # own spawn regression; this CPU test isolates checkpoint and archive state.
    from suction20 import prefetch
    make_loader = prefetch.make_ticket_loader
    def serial_loader(*args, loader_config, **kwargs):
        return make_loader(*args, loader_config=dict(loader_config, num_workers=0, pin_memory=False), **kwargs)
    monkeypatch.setattr(prefetch, 'make_ticket_loader', serial_loader)
    cfg['data']['smoke_frame_keys'] = [list(key) for key in ToySource.keys]
    cfg['training'].update(max_optimizer_steps=7, validate_every_optimizer_steps=7)
    label = tmp_path / 'labels'
    runtime.atomic_json(label / 'config_frozen.json', ToySource.data_cfg)
    runtime.atomic_json(label / 'manifest.json', {'passed': True})
    cfg['data']['label_root'] = str(label)
    run, control = tmp_path / 'run', tmp_path / 'run/control'

    def select_code(revision, worktree):
        for module in (entry, training):
            monkeypatch.setattr(module, 'code_revision', lambda: revision)
            monkeypatch.setattr(module, 'worktree_digest', lambda: worktree)

    def paused_source():
        status = runtime.read_json(control / 'status.json')
        gone = dict(pid=99999999, start_ticks=1, boot_id='retired-test-process')
        runtime.atomic_json(control / 'worker.json', dict(token=status['attempt_id'], exit_code=75,
            worker_identity=gone, trainer_identity=gone))
        return runtime.read_json(run / 'checkpoint-manifest.json')['current']

    select_code(SOURCE_REVISION, SOURCE_WORKTREE_SHA256)
    entry.train(copy.deepcopy(cfg), run, torch.device('cpu'), pause_at_step=1)
    original = paused_source()
    select_code(performance.P1_SOURCE_REVISION, performance.P1_SOURCE_WORKTREE_SHA256)
    p1_path = Path(__file__).parents[1] / 'configs/performance/cd009-p1.json'
    p1_result = converter.migrate(cfg, run, p1_path, control, original['sha256'])
    p1_config = attach_performance(cfg, p1_path)
    entry.train(p1_config, run, torch.device('cpu'), resume=run / 'last.pt', pause_at_step=3)
    p1_current = paused_source()
    before = state(run)
    assert performance.approved_migration_source(before['runtime_state']['identity']) == 'p1'
    for key in ('code_revision', 'worktree_sha256'):
        invalid = dict(before['runtime_state']['identity'], **{key: 'unapproved'})
        with pytest.raises(ValueError, match='not approved'):
            performance.approved_migration_source(invalid)
    invalid = copy.deepcopy(before['runtime_state']['identity'])
    invalid['performance']['config_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='not approved'):
        performance.approved_migration_source(invalid)

    # Runtime semantics can be tested without requiring a native grid build.
    p2_path = tmp_path / 'p2-reference.json'
    p2 = load_performance_config(p1_path.parent / 'cd009-p2.json')
    p2.update(fps_backend='numpy', spatial_index={'backend': 'kdtree', 'cell_size_m': .02})
    p2['loader'].update(num_workers=0, pin_memory=False)
    runtime.atomic_json(p2_path, p2)
    select_code('approved-test-p2', 'test-p2-worktree')
    p2_result = converter.migrate(cfg, run, p2_path, control, p1_current['sha256'])
    migrated = state(run)
    same_training(migrated, before)
    for key in ('ledger', 'validations', 'budget', 'effective_samples_completed'):
        assert migrated['runtime_state'][key] == before['runtime_state'][key]
    receipt = runtime.verify_performance_migration(run, p2_result['receipt'], migrated['runtime_state']['identity'])
    assert receipt['source_identity'] == before['runtime_state']['identity']
    assert runtime.read_json(run / Path(p2_result['receipt']['path']).parent / 'source-manifest.json')['current']['analysis']['performance_migration'] == p1_result['receipt']

    # The view binds the old assignment hash and the latest paused source backup.
    monkeypatch.setattr(supervisor, 'assignment_paths', lambda *args: (run, control))
    monkeypatch.setattr(supervisor, 'lock_paths', lambda *args: [])
    checked_backups = []
    def checked_backup(config, assignment, checkpoint, done):
        checked_backups.append(checkpoint)
        return True
    monkeypatch.setattr(supervisor, 'final_backup_verified', checked_backup)
    assignment = dict(experiment=cfg['experiment'], host='L20', token=p1_current['attempt_id'],
                      performance_config_sha256=performance.P1_SOURCE_CONFIG_SHA256)
    view = supervisor.inspect_worker({}, assignment)
    assert view['performance_migration_source_config_sha256'] == performance.P1_SOURCE_CONFIG_SHA256
    assert checked_backups[0] == p1_current
    assignment['view'] = view
    coordinator_config = {'performance_configs': {'L20': {cfg['experiment']: {
        'config_path': str(p2_path), 'config_sha256': runtime.file_digest(p2_path)}}}}
    supervisor.verify_performance_transition(coordinator_config, assignment)
    with pytest.raises(supervisor.Refused, match='explicit migration'):
        supervisor.verify_performance_transition(coordinator_config, dict(assignment, performance_config_sha256='other'))

    with pytest.raises(ValueError, match='matching complete runtime'):
        entry.train(p1_config, run, torch.device('cpu'), resume=run / 'last.pt', pause_at_step=5)
    entry.train(attach_performance(cfg, p2_path), run, torch.device('cpu'), resume=run / 'last.pt', pause_at_step=5)
    manifest = runtime.read_json(run / 'checkpoint-manifest.json')
    assert all(manifest[k]['analysis']['performance_migration'] == p2_result['receipt'] for k in ('current', 'predecessor'))
    assert [r['global_step'] for r in canonical_rows(run)] == [1, 2, 3, 4, 5]
    backup = tmp_path / 'backup'
    backup.mkdir()
    usage = supervisor.shutil.disk_usage
    monkeypatch.setattr(supervisor.shutil, 'disk_usage', lambda path: usage(path)._replace(free=300 * supervisor.GIB))
    destination = Path(supervisor.backup_snapshot({'root': str(tmp_path), 'backup_root': str(backup)},
        dict(assignment, token='backup-test'), {'request_id': 'p2-final', 'final': True}))
    for result in (p1_result, p2_result):
        copied = runtime.read_json(destination / result['receipt']['path'])
        assert runtime.file_digest(destination / result['receipt']['path']) == result['receipt']['sha256']
        for name, info in copied['files'].items():
            assert runtime.file_digest(destination / Path(result['receipt']['path']).parent / name) == info['sha256']

    # The latest receipt is insufficient if its original migration is missing.
    prior_receipt = run / p1_result['receipt']['path']
    prior_bytes = prior_receipt.read_bytes()
    prior_receipt.write_bytes(prior_bytes + b' ')
    with pytest.raises(ValueError, match='receipt checksum'):
        runtime.verify_performance_migration(run, p2_result['receipt'], migrated['runtime_state']['identity'])
