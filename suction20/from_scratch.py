"""Build an independent fresh-start configuration from an existing experiment.

The old profile is a source of the experiment definition and hardware limits,
not a measured-throughput endorsement of the new execution settings.
"""
from __future__ import annotations

from copy import deepcopy

from .performance import SCHEMA_VERSION_V4, validate_performance_config
from .pretrain_budget import resolve_config


def from_scratch_config(base, *, precision='fp16', dense_chunk_size=65536,
                        workers=4, geometry_threads=4, geometry_precision='float32',
                        scene_cache_size=90, object_cache_size=88,
                        voxel_heads=True, instance_broadcast_mode='segment',
                        compile_dense_heads=False, fused_optimizer=True):
    """Return new training/performance dictionaries; never mutate the source."""
    config = deepcopy(base)
    experiment = config.get('experiment', config.get('model', {}).get('experiment'))
    if experiment not in ('E0', 'E1', 'E2', 'E3', 'E4', 'E5'):
        raise ValueError('From-scratch configuration requires an E0–E5 experiment')
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('Precision must be fp32, fp16 or bf16')
    if min(scene_cache_size, object_cache_size) < 1:
        raise ValueError('Metadata cache sizes must be positive')
    performance = validate_performance_config({
        'schema_version': SCHEMA_VERSION_V4,
        'fps_backend': 'compiled',
        'spatial_index': {'backend': 'uniform_grid_cpu', 'cell_size_m': .02},
        'contact_query_backend': 'native_cpu',
        'no_instance_voxel_heads': bool(voxel_heads and experiment in ('E0', 'E1', 'E3')),
        'backbone_coordinate_fastpath': True,
        'dense_chunk_size_override': dense_chunk_size,
        'chunk_input_mode': 'split',
        'geometry_threads': geometry_threads,
        'geometry_precision': geometry_precision,
        'instance_broadcast_mode': instance_broadcast_mode,
        'compile_dense_heads': compile_dense_heads,
        'query_chunk_size': None,
        'query_workers': 1,
        'loader': {'num_workers': workers, 'prefetch_factor': 2,
                   'persistent_workers': True, 'pin_memory': True,
                   'multiprocessing_context': 'spawn'},
        'detailed_cpu_timing': False,
    })
    config['experiment'] = experiment
    config['model']['dense_chunk_size'] = dense_chunk_size
    config['model']['checkpoint_dense'] = False
    config['training']['precision'] = precision
    config['training']['amp'] = precision != 'fp32'
    # AdamW was already the project's optimizer: fused changes execution only.
    config['optimizer']['fused'] = bool(fused_optimizer)
    config['data']['scene_cache_size'] = int(scene_cache_size)
    config['data']['object_cache_size'] = int(object_cache_size)
    old_profile = config.get('runtime_profile', {})
    host = old_profile.get('host', 'custom')
    config['runtime_profile'] = {
        'profile_id': f'cd010-fresh-{host}-{experiment}-dense{dense_chunk_size}-{precision}',
        'training_version': 'v3',
        'selection_status': 'from_scratch',
        'selection_basis': 'Independent fresh-start candidate; GPU throughput is not yet measured.',
        'source_profile_id': old_profile.get('profile_id'),
        'host': host,
        'gpu_model': old_profile.get('gpu_model'),
        'experiment': experiment,
    }
    config['runtime'] = {key: value for key, value in config.get('runtime', {}).items()
                         if key in ('cpu_threads', 'gpu_memory_limit_gib')}
    config['runtime'].setdefault('cpu_threads', 8)
    # Retain the scientific sample budget, optimizer/loss settings and schedule.
    return resolve_config(config), performance
