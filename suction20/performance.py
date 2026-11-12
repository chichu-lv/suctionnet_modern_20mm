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
SOURCE_REVISION = '98862f4d6a133c99afa082a0755dbb888b41b38d'
SOURCE_WORKTREE_SHA256 = 'd634627b180cee10be2c42949c8d10cec21ac49424aa734f6b5acf2c5f2c30c8'
P1_SOURCE_REVISION = '996421c664a5a0b84c308aa4ca47c31eaff4035b'
P1_SOURCE_WORKTREE_SHA256 = '23bde51eb6b8f091a15b2aeb336db8a7085b150f304076af44b3a1d83ba7ee20'
P1_SOURCE_CONFIG_SHA256 = '08d2247c5bd433d8fdc16325a9602821a8a5c8c52b0c6cf0148a4991e4f9d188'


def approved_migration_source(identity):
    """Classify only the two explicitly approved source builds, never resume."""
    performance = identity.get('performance')
    if (identity.get('code_revision') == SOURCE_REVISION and
            identity.get('worktree_sha256') == SOURCE_WORKTREE_SHA256 and not performance):
        return 'original'
    if (identity.get('code_revision') == P1_SOURCE_REVISION and
            identity.get('worktree_sha256') == P1_SOURCE_WORKTREE_SHA256 and
            isinstance(performance, dict) and performance.get('schema_version') == SCHEMA_VERSION and
            performance.get('config_sha256') == P1_SOURCE_CONFIG_SHA256):
        return 'p1'
    raise ValueError('Performance migration source is not approved')


def validate_performance_config(value):
    keys = {'schema_version', 'fps_backend', 'query_chunk_size', 'query_workers', 'loader', 'detailed_cpu_timing'}
    if not isinstance(value, dict) or value.get('schema_version') not in (SCHEMA_VERSION, SCHEMA_VERSION_V2):
        raise ValueError('Unsupported performance configuration fields/schema')
    if value['schema_version'] == SCHEMA_VERSION_V2:
        keys |= {'spatial_index', 'no_instance_voxel_heads', 'backbone_coordinate_fastpath'}
    if set(value) != keys:
        raise ValueError('Unsupported performance configuration fields/schema')
    if value['schema_version'] == SCHEMA_VERSION_V2:
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
    if type(value['detailed_cpu_timing']) is not bool:
        raise ValueError('Performance detailed_cpu_timing must be boolean')
    loader = value['loader']
    if not isinstance(loader, dict) or set(loader) != {'num_workers', 'prefetch_factor', 'persistent_workers', 'pin_memory', 'multiprocessing_context'}:
        raise ValueError('Unsupported performance loader fields')
    if type(loader['num_workers']) is not int or loader['num_workers'] not in (0, 2):
        raise ValueError('Performance loader uses two workers (zero only for explicit CPU/reference use)')
    if type(loader['prefetch_factor']) is not int or loader['prefetch_factor'] != 2 or loader['multiprocessing_context'] != 'spawn':
        raise ValueError('Performance loader requires prefetch_factor=2 and spawn')
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
    builds = {'fps': fps_backend_identity(performance['fps_backend'])}
    if performance.get('spatial_index', {}).get('backend') == 'uniform_grid_cpu':
        from models.spatial_index import spatial_index_backend_identity
        builds['spatial_index'] = spatial_index_backend_identity('uniform_grid_cpu')
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
