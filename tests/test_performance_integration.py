"""Normal CPU integration: exact continuation and immutable sidecar binding."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch

from scripts import train_pretrain as entry
from suction20 import pretrain_runtime as runtime, training, run_supervisor as supervisor
from suction20.batching import collate_public
from suction20.performance import attach_performance, load_performance_config
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


def test_supervisor_binds_sidecar_independently_of_profile(coordinator, tmp_path):
    path = sidecar(tmp_path)
    coordinator.config['performance_configs'] = {'H20': {'E5': {
        'config_path': str(path), 'config_definition': json.loads(path.read_text())}}}
    assignment = coordinator.assign('E5', 'H20', 0)
    command = supervisor.trainer_command(coordinator.config, assignment)
    assert command[command.index('--performance-config') + 1] == str(path)
    assert assignment['performance_settings'] == json.loads(path.read_text())
    old = dict(assignment)
    for key in ('performance_config_path', 'performance_settings'):
        del old[key]
    with pytest.raises(supervisor.Refused, match='sidecar changed'):
        supervisor.verify_assignment_performance(coordinator.config, old)


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


def test_scalar_and_batched_sidecars_preserve_profile_and_bind_actual_settings(tmp_path):
    default = Path(__file__).parents[1] / 'configs/performance/cd009-p1.json'
    settings = load_performance_config(default)
    assert settings['query_chunk_size'] is None and settings['query_workers'] == 1
    profile = {'runtime': {'raw_config_metadata': {'size_bytes': 123}},
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
    assert scalar['runtime']['performance'] != batched['runtime']['performance']
    for resolved in (scalar, batched):
        assert resolved['runtime']['raw_config_metadata'] == {'size_bytes': 123}
        assert resolved['runtime_profile'] == original['runtime_profile']


def test_v2_schema_is_explicit_and_v1_identity_stays_unchanged(tmp_path, monkeypatch):
    base = Path(__file__).parents[1] / 'configs/performance'
    p1 = load_performance_config(base / 'cd009-p1.json')
    p2 = load_performance_config(base / 'cd009-p2.json')
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
    monkeypatch.setattr(spatial_index, 'spatial_index_backend_identity', lambda name: {'backend': name, 'abi_version': 1})
    old = performance.performance_identity(attach_performance({}, base / 'cd009-p1.json'))
    assert old == {'schema_version': performance.SCHEMA_VERSION,
                   'settings': p1,
                   'backend_builds': {'fps': {'backend': 'compiled'}}}
    new = performance.performance_identity(attach_performance({}, base / 'cd009-p2.json'))
    assert new['schema_version'] == performance.SCHEMA_VERSION_V2
    assert new['backend_builds']['spatial_index'] == {'backend': 'uniform_grid_cpu', 'abi_version': 1}


def test_v3_sidecar_binds_native_contact_abi_without_changing_v2(monkeypatch):
    from types import SimpleNamespace
    from models import geometry_ops, spatial_index
    base = Path(__file__).parents[1] / 'configs/performance'
    p2 = load_performance_config(base / 'cd009-p2.json')
    p3 = load_performance_config(base / 'cd009-p3.json')
    assert p3 == dict(p2, schema_version=performance.SCHEMA_VERSION_V3,
                      contact_query_backend='native_cpu')
    reference = dict(p3, contact_query_backend='numpy')
    assert performance.validate_performance_config(reference) == reference
    monkeypatch.setattr(geometry_ops, 'fps_backend_identity', lambda name: {'backend': name})
    monkeypatch.setattr(spatial_index, 'spatial_index_backend_identity',
                        lambda name: {'backend': name, 'abi_version': 1})
    monkeypatch.setattr(spatial_index, '_compiled_grid', lambda: SimpleNamespace(CONTACT_ABI_VERSION=1))
    old = performance.performance_identity(attach_performance({}, base / 'cd009-p2.json'))
    new = performance.performance_identity(attach_performance({}, base / 'cd009-p3.json'))
    assert new['schema_version'] == performance.SCHEMA_VERSION_V3
    assert new['backend_builds']['fps'] == old['backend_builds']['fps']
    assert new['backend_builds']['spatial_index'] == dict(
        old['backend_builds']['spatial_index'], contact_abi_version=1)


def test_v3_optional_execution_fields_preserve_profile_and_payload_identity(tmp_path):
    base = Path(__file__).parents[1] / 'configs'
    p1 = load_performance_config(base / 'performance/cd009-p1.json')
    p2 = load_performance_config(base / 'performance/cd009-p2.json')
    legacy_v3 = dict(p2, schema_version=performance.SCHEMA_VERSION_V3,
                     contact_query_backend='native_cpu')
    profile_path = base / 'runtime_profiles/cd009-v3/L20-E1.json'
    profile = training.load_config(profile_path)
    profile['runtime']['raw_config_metadata'] = runtime.file_metadata(profile_path)
    original = copy.deepcopy(profile)
    definitions = []
    variants = [legacy_v3, dict(legacy_v3, dense_chunk_size_override=None),
                dict(legacy_v3, chunk_input_mode='slice'),
                dict(legacy_v3, dense_chunk_size_override=65536, chunk_input_mode='split')]
    for index, settings in enumerate(variants):
        path = tmp_path / f'performance-{index}.json'
        path.write_text(json.dumps(settings))
        attached = attach_performance(profile, path)
        assert attached['runtime']['performance'] == settings
        assert attached['runtime']['performance_config_metadata'] == runtime.file_metadata(path)
        assert attached['runtime']['raw_config_metadata'] == original['runtime']['raw_config_metadata']
        assert training.semantic_config(attached) == training.semantic_config(original)
        assert attached['model']['dense_chunk_size'] == 16384
        definitions.append(attached['runtime']['performance'])
    assert profile == original
    assert len({json.dumps(value, sort_keys=True) for value in definitions}) == len(variants)
    for invalid in (dict(legacy_v3, dense_chunk_size_override=0),
                    dict(legacy_v3, dense_chunk_size_override=True),
                    dict(legacy_v3, chunk_input_mode='unsupported'),
                    dict(p1, dense_chunk_size_override=None), dict(p2, chunk_input_mode='split')):
        with pytest.raises(ValueError):
            performance.validate_performance_config(invalid)


def test_historical_runtime_compares_available_structure_and_execution_settings():
    current = {'source_label_definition': {'radius_m': .02},
               'semantic_config': {'experiment': 'E5'},
               'worktree_snapshot': {'revision': 'revision'}, 'code_revision': 'revision',
               'budget_contract': 'contract', 'runtime_profile': {'profile_id': 'tiny'},
               'dependencies': {'torch': 'version'}, 'math_settings': {'tf32_matmul': False},
               'performance': {'schema_version': 'version', 'settings': {'query_workers': 1},
                               'backend_builds': {'fps': {'backend': 'numpy', 'abi_version': 1}}}}
    saved = {key: copy.deepcopy(value) for key, value in current.items()
             if key not in ('source_label_definition', 'semantic_config', 'worktree_snapshot')}
    saved['source_label_config_sha256'] = 'retired-value-ignored'
    saved['performance'].pop('settings')
    saved['performance']['config_sha256'] = 'retired-value-ignored'
    config = {'runtime': {'performance': {'query_workers': 1}}}
    performance.validate_runtime_identity(saved, current, config, copy.deepcopy(config))
    changed = copy.deepcopy(saved)
    changed['code_revision'] = 'other-revision'
    with pytest.raises(ValueError, match='identity differs'):
        performance.validate_runtime_identity(changed, current, config, config)
    with pytest.raises(ValueError, match='execution settings'):
        performance.validate_runtime_identity(saved, current, config,
                                              {'runtime': {'performance': {'query_workers': 2}}})
    with pytest.raises(ValueError, match='lacks required'):
        performance.validate_runtime_identity({}, current, config, config)
