from copy import deepcopy
import json
from pathlib import Path

import pytest

from suction20.from_scratch import from_scratch_config
from suction20.performance import attach_performance, validate_performance_config
from suction20.pretrain_budget import resolve_budget


@pytest.mark.parametrize('experiment,voxel_heads', [('E0', True), ('E3', True), ('E5', False)])
def test_fresh_config_preserves_budget_and_source(experiment, voxel_heads, tmp_path):
    root = Path(__file__).resolve().parents[1]
    host = 'H20' if experiment == 'E5' else 'L20'
    source = json.loads((root / f'configs/runtime_profiles/cd009-v3/{host}-{experiment}.json').read_text())
    original = deepcopy(source)
    config, performance = from_scratch_config(source)
    assert source == original
    assert config['loss'] == source['loss']
    assert resolve_budget(config) == resolve_budget(source)
    assert config['runtime_profile']['selection_status'] == 'from_scratch'
    assert 'measured_frames_per_second' not in config['runtime_profile']
    assert config['training']['precision'] == 'fp16'
    assert config['optimizer']['fused'] is True
    assert config['data']['scene_cache_size'] == 90
    assert config['data']['object_cache_size'] == 88
    assert performance['no_instance_voxel_heads'] is voxel_heads
    assert performance['dense_chunk_size_override'] == (131072 if host == 'H20' else 65536)
    assert performance['geometry_threads'] == (8 if host == 'H20' else 4)
    assert config['runtime']['cpu_affinity_count'] == (16 if host == 'H20' else 8)
    assert config['runtime']['cpu_threads'] == 8
    assert config['runtime']['gpu_memory_limit_gib'] == (112 if host == 'H20' else 36)
    assert config['runtime_profile']['hardware_preset'] == host
    assert performance['loader']['num_workers'] == 4
    path = tmp_path / 'performance.json'
    path.write_text(json.dumps(performance))
    attached = attach_performance(config, path)
    assert attached['runtime']['performance'] == performance


def test_fresh_optional_controls_and_legacy_loader_policy():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / 'configs/runtime_profiles/cd009-v3/H20-E5.json').read_text())
    config, performance = from_scratch_config(source, hardware_preset='L20', precision='bf16', workers=6,
        geometry_threads=8, geometry_precision='float64', dense_chunk_size=131072,
        cpu_affinity_count=12, cpu_threads=4, gpu_memory_limit_gib=38,
        compile_dense_heads=True, instance_broadcast_mode='indexed', fused_optimizer=False)
    assert config['training']['precision'] == 'bf16'
    assert not config['optimizer']['fused']
    assert performance['loader']['num_workers'] == 6
    assert performance['geometry_threads'] == 8
    assert performance['compile_dense_heads']
    assert config['experiment'] == 'E5'
    assert config['runtime_profile']['host'] == 'L20'
    assert config['runtime'] == {'cpu_affinity_count': 12, 'cpu_threads': 4, 'gpu_memory_limit_gib': 38}
    legacy = json.loads((root / 'configs/performance/cd009-p3-integrated.json').read_text())
    assert validate_performance_config(legacy) == legacy
    legacy['loader']['num_workers'] = 4
    with pytest.raises(ValueError, match='Legacy'):
        validate_performance_config(legacy)


def test_reference_preset_keeps_common_comparison_and_source_memory_limit():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / 'configs/runtime_profiles/cd009-v3/H20-E5.json').read_text())
    config, performance = from_scratch_config(source, hardware_preset='reference')
    assert performance['dense_chunk_size_override'] == 65536
    assert performance['geometry_threads'] == 4
    assert config['runtime']['cpu_affinity_count'] == 8
    assert config['runtime']['gpu_memory_limit_gib'] == 112
    assert config['runtime_profile']['host'] == 'H20'
