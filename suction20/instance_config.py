"""Separate instance-model semantics from relocation and query batching.

These helpers compare inspectable settings, never file contents. Runtime
overrides are allowed for loading weights into a new run or inference. Exact
same-run resume retains its separate, strict complete-configuration check.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from pathlib import Path

from .metadata import without_legacy_checksums


# Missing fields in historical saved configurations identify the original chart.
# New run configurations explicitly materialize B_FEATURE_FRAME before saving.
B_LEGACY_FEATURE_FRAME = 'two_fixed_z_charts_rodrigues_v2'
B_FEATURE_FRAME = 'two_fixed_z_charts_rodrigues_z05_v1'


_SEMANTIC_DEFAULTS = {
    'a': {'backbone_points': 2048, 'min_support_points': 100,
          'backbone_coordinate_fastpath': False, 'singleton_bn_policy': 'running_stats'},
    'b': {'phase_seed': 42, 'training_phase': True, 'min_support_points': 100,
          'feature_frame': B_LEGACY_FEATURE_FRAME},
}
_EXECUTION_DEFAULTS = {
    'a': {'backbone_root': None, 'query_chunk_size': 32},
    'b': {'query_chunk': 64},
}


def _options(route, options):
    if route not in _SEMANTIC_DEFAULTS:
        raise ValueError('Instance model route must be a or b')
    if not isinstance(options, dict):
        raise ValueError('Instance model configuration must be a mapping')
    return deepcopy(without_legacy_checksums(options))


def semantic_model_config(route, options):
    """Compare every setting except explicitly route-specific execution fields.

    Omitted fields in saved configurations have their historical values, except
    the corrected B minimum-support gate now defaults to 100. Unknown keys
    remain in the comparison rather than being silently classified as runtime.
    Geometry, singleton BN and B's phase settings are intentionally semantic.
    New B run configs explicitly select the stable chart; missing feature_frame
    identifies legacy weights and must not silently change their sampling grid.
    """
    values = _options(route, options)
    semantic = deepcopy(_SEMANTIC_DEFAULTS[route])
    semantic.update({key: value for key, value in values.items()
                     if key not in _EXECUTION_DEFAULTS[route]})
    return semantic


def model_execution_options(route, options):
    """Report resolved execution settings without changing model semantics."""
    values = _options(route, options)
    result = {key: values.get(key, default)
              for key, default in _EXECUTION_DEFAULTS[route].items()}
    if isinstance(result.get('backbone_root'), Path):
        result['backbone_root'] = str(result['backbone_root'])
    return result


def validate_saved_config_definition(config, identity):
    """Check the checkpoint's own declared configuration before any override."""
    if not isinstance(config, dict) or not isinstance(identity, dict):
        raise ValueError('Checkpoint must include configuration and identity mappings')
    if ('config_definition' in identity and
            without_legacy_checksums(identity['config_definition']) != without_legacy_checksums(config)):
        raise ValueError('Checkpoint config definition mismatch')


def apply_model_runtime_overrides(config, *, backbone_root=None, query_chunk_size=None):
    """Copy configuration and apply only path and route-specific chunk overrides.

    A uses ``query_chunk_size`` and B uses ``query_chunk`` internally. The common
    CLI spelling is ``--query-chunk-size``. Saved checkpoint definitions must be
    validated before calling this helper, and remain unchanged afterwards.
    """
    if not isinstance(config, dict):
        raise ValueError('Instance runtime configuration must be a mapping')
    result = deepcopy(without_legacy_checksums(config))
    route = result.get('route')
    options = _options(route, result.get('model', {}))
    if route == 'b':
        options.setdefault('min_support_points', 100)
        options.setdefault('feature_frame', B_LEGACY_FEATURE_FRAME)
    if backbone_root is not None:
        if route != 'a':
            raise ValueError('backbone_root override is only valid for route A')
        options['backbone_root'] = str(Path(backbone_root).resolve())
    if query_chunk_size is not None:
        if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, Integral) or query_chunk_size < 1:
            raise ValueError('query_chunk_size override must be a positive integer')
        options['query_chunk_size' if route == 'a' else 'query_chunk'] = int(query_chunk_size)
    result['model'] = options
    return result
