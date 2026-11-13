"""Build an independent fresh-start configuration from an existing experiment.

The old profile supplies the experiment and source host. Presets incorporate
measured CD010 resource choices; new scheduling combinations remain candidates.
"""
from __future__ import annotations

from copy import deepcopy

from .performance import SCHEMA_VERSION_V4, validate_performance_config
from .pretrain_budget import CONTRACT_VERSION, resolve_config


HARDWARE_PRESETS = {
    'H20': {'dense_chunk_size': 65536, 'workers': 4, 'geometry_threads': 8,
            'cpu_affinity_count': 16, 'cpu_threads': 8, 'gpu_memory_limit_gib': None,
            'gpu_model': 'NVIDIA H20-3e'},
    'L20': {'dense_chunk_size': 65536, 'workers': 4, 'geometry_threads': 4,
            'cpu_affinity_count': 8, 'cpu_threads': 8, 'gpu_memory_limit_gib': None,
            'gpu_model': 'NVIDIA L20'},
    'reference': {'dense_chunk_size': 65536, 'workers': 4, 'geometry_threads': 4,
                  'cpu_affinity_count': 8, 'cpu_threads': 8, 'gpu_memory_limit_gib': None},
}


def from_scratch_config(base, *, hardware_preset=None, precision='fp16', dense_chunk_size=None,
                        workers=None, geometry_threads=None, geometry_precision='float32',
                        cpu_affinity_count=None, cpu_threads=None, gpu_memory_limit_gib=None,
                        no_gpu_memory_limit=False,
                        microbatch_size=None, effective_batch_size=None,
                        geometry_pipeline_depth=1, pose_chunk_size=None,
                        contact_feature_gather_mode='chunk',
                        scene_cache_size=90, object_cache_size=88,
                        voxel_heads=True, instance_broadcast_mode='segment',
                        compile_dense_heads=False, fused_optimizer=True):
    """Return fresh-run dictionaries; optionally change the effective batch.

    An explicit effective-batch change preserves v3 sample milestones and
    derives their update counts. Optimizer hyperparameters, including peak LR,
    are retained; this is a new training recipe, not a checkpoint migration.
    Fresh presets use the full available device memory (JSON null); an explicit
    numeric gpu_memory_limit_gib can still set an allocator cap.
    """
    config = deepcopy(base)
    old_profile = config.get('runtime_profile', {})
    source_host = old_profile.get('host', 'custom')
    preset_name = hardware_preset or (source_host if source_host in ('H20', 'L20') else 'reference')
    if preset_name not in HARDWARE_PRESETS:
        raise ValueError('Hardware preset must be H20, L20 or reference')
    preset = HARDWARE_PRESETS[preset_name]
    dense_chunk_size = preset['dense_chunk_size'] if dense_chunk_size is None else dense_chunk_size
    workers = preset['workers'] if workers is None else workers
    geometry_threads = preset['geometry_threads'] if geometry_threads is None else geometry_threads
    cpu_affinity_count = preset['cpu_affinity_count'] if cpu_affinity_count is None else cpu_affinity_count
    cpu_threads = preset['cpu_threads'] if cpu_threads is None else cpu_threads
    if no_gpu_memory_limit and gpu_memory_limit_gib is not None:
        raise ValueError('Choose no GPU memory limit or a numeric limit, not both')
    if gpu_memory_limit_gib is None:
        gpu_memory_limit_gib = preset['gpu_memory_limit_gib']
    for key, value in (('cpu_affinity_count', cpu_affinity_count), ('cpu_threads', cpu_threads)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if gpu_memory_limit_gib is not None and (
            type(gpu_memory_limit_gib) not in (int, float) or not 0 < gpu_memory_limit_gib < 1000):
        raise ValueError('GPU memory limit must be positive GiB or None for no allocator cap')
    experiment = config.get('experiment', config.get('model', {}).get('experiment'))
    if experiment not in ('E0', 'E1', 'E2', 'E3', 'E4', 'E5'):
        raise ValueError('From-scratch configuration requires an E0–E5 experiment')
    training = config['training']
    effective_batch = training.get('effective_batch_size',
        training.get('microbatch_size', 1) * training.get('gradient_accumulation_steps', 4))
    if effective_batch_size is not None:
        if type(effective_batch_size) is not int or effective_batch_size < 1:
            raise ValueError('effective_batch_size must be a positive integer')
        if effective_batch_size != effective_batch:
            if training.get('budget_contract') != CONTRACT_VERSION:
                raise ValueError('Changing effective batch requires the v3 sample budget contract')
            # These are derived from sample milestones, not independent limits.
            # resolve_config recreates them for the new effective batch.
            for section, key in (('training', 'max_optimizer_steps'),
                                 ('training', 'validate_every_optimizer_steps'),
                                 ('scheduler', 'warmup_optimizer_steps'),
                                 ('scheduler', 'total_optimizer_steps')):
                config.get(section, {}).pop(key, None)
        effective_batch = effective_batch_size
        training['effective_batch_size'] = effective_batch
    if microbatch_size is not None or effective_batch_size is not None:
        micro = training.get('microbatch_size', 1) if microbatch_size is None else microbatch_size
        if type(micro) is not int or micro < 1 or effective_batch % micro:
            raise ValueError('microbatch_size must be a positive divisor of the selected effective batch')
        training['microbatch_size'] = micro
        training['gradient_accumulation_steps'] = effective_batch // micro
    if type(geometry_pipeline_depth) is not int or geometry_pipeline_depth not in (1, 2):
        raise ValueError('geometry_pipeline_depth must be 1 or 2')
    if geometry_pipeline_depth == 2 and (
            experiment not in ('E3', 'E4', 'E5') or training.get('gradient_accumulation_steps', 4) < 2):
        raise ValueError('Depth 2 requires contact grouping and at least two microbatches per update')
    training['geometry_pipeline_depth'] = geometry_pipeline_depth
    if pose_chunk_size is not None:
        if type(pose_chunk_size) is not int or pose_chunk_size < 1:
            raise ValueError('pose_chunk_size must be a positive integer')
        config['model']['pose_chunk_size'] = pose_chunk_size
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('Precision must be fp32, fp16 or bf16')
    if min(scene_cache_size, object_cache_size) < 1:
        raise ValueError('Metadata cache sizes must be positive')
    performance = validate_performance_config({
        'schema_version': SCHEMA_VERSION_V4,
        'fps_backend': 'compiled',
        'spatial_index': {'backend': 'uniform_grid_cpu', 'cell_size_m': .02},
        'contact_query_backend': 'native_cpu',
        'contact_feature_gather_mode': contact_feature_gather_mode,
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
    host = preset_name if preset_name in ('H20', 'L20') else source_host
    config['runtime_profile'] = {
        'profile_id': (f'cd010-fresh-{host}-{experiment}-dense{dense_chunk_size}'
                       f'-cpu{cpu_affinity_count}-geo{geometry_threads}-w{workers}-{precision}'
                       f'-b{training.get("microbatch_size", 1)}-e{effective_batch}'
                       f'-pipe{geometry_pipeline_depth}'
                       f'-pose{config["model"].get("pose_chunk_size", 32)}'
                       f'-gather{contact_feature_gather_mode}'),
        'training_version': 'v3',
        'selection_status': 'from_scratch',
        'selection_basis': (f'{preset_name} resource preset; 64k retained after CD010 measurements. '
                            'This generated scheduling combination requires its own throughput comparison.'),
        'hardware_preset': preset_name,
        'source_profile_id': old_profile.get('profile_id'),
        'host': host,
        'gpu_model': preset.get('gpu_model', old_profile.get('gpu_model')),
        'experiment': experiment,
    }
    config['runtime'] = {'cpu_threads': cpu_threads, 'cpu_affinity_count': cpu_affinity_count,
                         'gpu_memory_limit_gib': gpu_memory_limit_gib}
    # Retain the sample budget/milestones and optimizer/loss hyperparameters.
    return resolve_config(config), performance
