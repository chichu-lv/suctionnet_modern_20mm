"""Strict performance sidecar and shared pretraining identity construction.

Configuration parsing is stdlib-only for supervisor use. Torch and backend imports
are deferred until target-environment identity construction. Hardware profiles are
never rewritten; performance identity is independent of their raw file digest.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 'suction20.performance.v1'
SCHEMA_VERSION_V2 = 'suction20.performance.v2'
SCHEMA_VERSION_V3 = 'suction20.performance.v3'
SCHEMA_VERSION_V4 = 'suction20.performance.v4'


def validate_performance_config(value):
    keys = {'schema_version', 'fps_backend', 'query_chunk_size', 'query_workers', 'loader', 'detailed_cpu_timing'}
    optional_keys = set()
    if not isinstance(value, dict) or value.get('schema_version') not in (SCHEMA_VERSION, SCHEMA_VERSION_V2, SCHEMA_VERSION_V3, SCHEMA_VERSION_V4):
        raise ValueError('Unsupported performance configuration fields/schema')
    fresh = value['schema_version'] == SCHEMA_VERSION_V4
    if value['schema_version'] in (SCHEMA_VERSION_V2, SCHEMA_VERSION_V3, SCHEMA_VERSION_V4):
        keys |= {'spatial_index', 'no_instance_voxel_heads', 'backbone_coordinate_fastpath'}
    if value['schema_version'] in (SCHEMA_VERSION_V3, SCHEMA_VERSION_V4):
        keys.add('contact_query_backend')
        optional_keys = {'dense_chunk_size_override', 'chunk_input_mode'}
    if fresh:
        keys |= {'geometry_threads', 'geometry_precision', 'instance_broadcast_mode', 'compile_dense_heads'}
        optional_keys.add('contact_feature_gather_mode')
    if not keys.issubset(value) or set(value) - keys - optional_keys:
        raise ValueError('Unsupported performance configuration fields/schema')
    if value['schema_version'] in (SCHEMA_VERSION_V2, SCHEMA_VERSION_V3, SCHEMA_VERSION_V4):
        spatial = value['spatial_index']
        if (not isinstance(spatial, dict) or set(spatial) != {'backend', 'cell_size_m'} or
                spatial['backend'] not in ('kdtree', 'uniform_grid_cpu') or
                type(spatial['cell_size_m']) not in (float, int) or spatial['cell_size_m'] != .02):
            raise ValueError('Performance spatial index requires kdtree or uniform_grid_cpu with 20mm cells')
        if any(type(value[key]) is not bool for key in ('no_instance_voxel_heads', 'backbone_coordinate_fastpath')):
            raise ValueError('Performance voxel heads/coordinate fastpath must be boolean')
    if value['fps_backend'] not in ('compiled', 'numpy'):
        raise ValueError('Performance FPS backend must explicitly be compiled or numpy')
    chunk = value['query_chunk_size']
    if chunk is not None and (type(chunk) is not int or not 1 <= chunk <= 64):
        raise ValueError('Performance query chunk must be null (scalar) or an integer in 1..64')
    if type(value['query_workers']) is not int or value['query_workers'] not in (1, 4):
        raise ValueError('Performance query workers must be 1 or 4')
    if value['schema_version'] in (SCHEMA_VERSION_V3, SCHEMA_VERSION_V4):
        contact = value['contact_query_backend']
        if contact not in ('numpy', 'native_cpu'):
            raise ValueError('Performance contact query backend must be numpy or native_cpu')
        if contact == 'native_cpu' and (value['spatial_index']['backend'] != 'uniform_grid_cpu' or
                value['fps_backend'] != 'compiled' or chunk is not None or value['query_workers'] != 1):
            raise ValueError('Native contact query requires uniform_grid_cpu, compiled FPS and scalar query settings')
        dense_chunk = value.get('dense_chunk_size_override')
        if dense_chunk is not None and (type(dense_chunk) is not int or dense_chunk <= 0):
            raise ValueError('Performance dense chunk override must be null or a positive integer')
        if 'chunk_input_mode' in value and value['chunk_input_mode'] not in ('slice', 'split'):
            raise ValueError('Performance chunk input mode must be slice or split')
    if fresh:
        if type(value['geometry_threads']) is not int or value['geometry_threads'] < 1:
            raise ValueError('Geometry threads must be a positive integer')
        if value['geometry_precision'] not in ('float64', 'float32'):
            raise ValueError('Geometry precision must be float64 or float32')
        if value['instance_broadcast_mode'] not in ('indexed', 'segment'):
            raise ValueError('Instance broadcast mode must be indexed or segment')
        if type(value['compile_dense_heads']) is not bool:
            raise ValueError('compile_dense_heads must be boolean')
        if value.get('contact_feature_gather_mode', 'chunk') not in ('chunk', 'zone'):
            raise ValueError('Contact feature gather mode must be chunk or zone')
    if type(value['detailed_cpu_timing']) is not bool:
        raise ValueError('Performance detailed_cpu_timing must be boolean')
    loader = value['loader']
    if not isinstance(loader, dict) or set(loader) != {'num_workers', 'prefetch_factor', 'persistent_workers', 'pin_memory', 'multiprocessing_context'}:
        raise ValueError('Unsupported performance loader fields')
    if type(loader['num_workers']) is not int or loader['num_workers'] < 0:
        raise ValueError('Performance loader worker count must be a nonnegative integer')
    if not fresh and loader['num_workers'] not in (0, 2):
        raise ValueError('Legacy performance loader uses two workers (zero for CPU/reference)')
    if (type(loader['prefetch_factor']) is not int or loader['prefetch_factor'] < 1 or
            (not fresh and loader['prefetch_factor'] != 2) or loader['multiprocessing_context'] != 'spawn'):
        raise ValueError('Performance loader requires a positive prefetch factor and spawn (legacy factor=2)')
    if any(type(loader[key]) is not bool for key in ('persistent_workers', 'pin_memory')):
        raise ValueError('Performance loader pinning/persistence must be boolean')
    return copy.deepcopy(value)


def load_performance_config(path):
    return validate_performance_config(json.loads(Path(path).read_bytes()))


def attach_performance(config, path):
    payload = Path(path).read_bytes()
    result = copy.deepcopy(config)
    runtime = result.setdefault('runtime', {})
    runtime['performance'] = validate_performance_config(json.loads(payload))
    runtime['performance_config_sha256'] = hashlib.sha256(payload).hexdigest()
    return result


def performance_identity(config):
    runtime = config.get('runtime', {})
    if 'performance' not in runtime:
        return None
    performance = validate_performance_config(runtime['performance'])
    digest = runtime.get('performance_config_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Performance sidecar requires its original byte SHA256')
    from models.geometry_ops import fps_backend_identity
    fps_options = {'precision': 'float32'} if performance.get('geometry_precision') == 'float32' else {}
    builds = {'fps': fps_backend_identity(performance['fps_backend'], **fps_options)}
    if performance.get('spatial_index', {}).get('backend') == 'uniform_grid_cpu':
        from models.spatial_index import spatial_index_backend_identity
        builds['spatial_index'] = spatial_index_backend_identity('uniform_grid_cpu')
        if performance.get('geometry_precision') == 'float32':
            from models.spatial_index import _compiled_grid
            if getattr(_compiled_grid(), 'GEOMETRY_FLOAT32_VERSION', None) != 1:
                raise RuntimeError('FP32 geometry requires rebuilding models._spatial_index_cpu')
            builds['spatial_index']['geometry_precision'] = 'float32'
        if performance.get('contact_query_backend') == 'native_cpu':
            from models.spatial_index import _compiled_grid
            abi = getattr(_compiled_grid(), 'CONTACT_ABI_VERSION', None)
            if abi != 1:
                raise RuntimeError('Unsupported native contact query ABI; rebuild models._spatial_index_cpu')
            builds['spatial_index']['contact_abi_version'] = abi
    return {'schema_version': performance['schema_version'], 'config_sha256': digest,
            'backend_builds': builds}


def build_runtime_identity(config, source_data, *, code_revision_value=None, worktree_sha256=None):
    import importlib.metadata
    import torch
    from .pretrain_budget import resolve_budget
    from .pretrain_runtime import file_digest
    from .training import content_digest, semantic_config, code_revision, worktree_digest
    label_root = Path(config['data'].get('label_root', '.'))
    manifest, frozen = label_root / 'manifest.json', label_root / 'config_frozen.json'
    identity = {'source_label_config_sha256': content_digest(source_data),
                'semantic_config_sha256': content_digest(semantic_config(config)),
                'code_revision': code_revision_value if code_revision_value is not None else code_revision(),
                'worktree_sha256': worktree_sha256 if worktree_sha256 is not None else worktree_digest(),
                'manifest_sha256': file_digest(manifest) if manifest.exists() else None,
                'frozen_config_raw_sha256': file_digest(frozen) if frozen.exists() else None,
                'raw_config_sha256': config.get('runtime', {}).get('raw_config_sha256'),
                'budget_contract': resolve_budget(config)['contract_version'],
                'runtime_profile': config.get('runtime_profile'), 'dependencies': {}}
    for package in ('torch', 'numpy', 'scipy', 'spconv-cu120', 'cumm-cu120'):
        try:
            identity['dependencies'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            identity['dependencies'][package] = None
    identity['math_settings'] = {'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    performance = performance_identity(config)
    if performance is not None:
        identity['performance'] = performance
    return identity
