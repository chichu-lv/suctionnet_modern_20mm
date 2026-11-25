"""Read-only, versioned source geometry prepared before a training run.

An entry uses one mmap-able NPY byte arena and a small typed JSON layout. It
contains geometry only, never augmented RGB, phase, network tensors or RNG.
Source path/stat and explicit preparation versions are provenance, not content
integrity checks. Missing or stale entries take the normal validated cold path.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .instance_preparation_cache import _freeze


FORMAT = 'instance-static-array-arena-v1'
_STATS = dict(hits=0, missing=0, stale=0, invalid=0)


def static_store_stats():
    return dict(_STATS)


def reset_static_store_stats():
    for name in _STATS:
        _STATS[name] = 0


class DeferredNeighborIndex:
    """Rebuild A's public arbitrary-pose index only if that API is exercised.

    Canonical training already has all query/association results in the static
    packet. Keeping the immutable index inputs retains the fallback contract
    without rebuilding and serializing cKDTrees for every training draw.
    """
    def __init__(self, points, source_ids, components):
        if (points.dtype != np.float64 or source_ids.dtype != np.int64
                or components.dtype != np.int64 or source_ids.ndim != 1
                or points.shape != (len(source_ids), 3)
                or components.shape != source_ids.shape or not np.isfinite(points).all()):
            raise ValueError('Invalid stored component neighbor index')
        self.points, self.source_ids, self.components = points, source_ids, components
        self._index = None

    def _materialize(self):
        if self._index is None:
            from models.instance_route_a_geometry import NeighborIndex
            self._index = NeighborIndex(self.points, self.source_ids, self.components)
        return self._index

    def nearest_anchor(self, points, radius=.005):
        return self._materialize().nearest_anchor(points, radius)

    def associate(self, points, components, radius=.010):
        return self._materialize().associate(points, components, radius)

    def support_ball(self, point, component):
        return self._materialize().support_ball(point, component)


def preparation_version(route):
    if route == 'a':
        from models.instance_route_a import SOURCE_STATIC_GEOMETRY_SPEC
        from models.instance_route_a_geometry import _spatial_index_cpu, _FPS_AXES
        return dict(spec=SOURCE_STATIC_GEOMETRY_SPEC,
                    native_source=getattr(_spatial_index_cpu, 'ROUTE_A_CONTACT_SOURCE_VERSION', None),
                    fps_axes=None if _FPS_AXES is None else list(map(int, _FPS_AXES)))
    if route == 'b':
        from models.instance_route_b_preparation import STATIC_INPUTS_SPEC
        return dict(spec=STATIC_INPUTS_SPEC)
    raise ValueError('Static geometry route must be a or b')


def geometry_location(root, source_path):
    """Preserve the entire resolved source path, without basename collisions."""
    source = Path(source_path).resolve()
    return Path(root) / 'entries' / Path(*source.parts[1:])


def _encode(value, arrays, array_ids):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf' or value.dtype.fields is not None:
            raise ValueError('Static geometry arrays must be plain numeric/bool arrays')
        index = array_ids.get(id(value))
        if index is None:
            index = len(arrays)
            array_ids[id(value)] = index
            arrays.append(np.ascontiguousarray(value).reshape(value.shape))
        return ['array', index]
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays, array_ids)
    if type(value).__name__ == 'NeighborIndex' or isinstance(value, DeferredNeighborIndex):
        if type(value).__module__ not in ('models.instance_route_a_geometry', __name__):
            raise TypeError('Unsupported neighbor index implementation')
        return ['neighbor', *[_encode(getattr(value, name), arrays, array_ids)
                              for name in ('points', 'source_ids', 'components')]]
    if isinstance(value, dict):
        return ['dict', [[_encode(key, arrays, array_ids), _encode(item, arrays, array_ids)]
                         for key, item in value.items()]]
    if isinstance(value, (tuple, list)):
        return ['tuple' if isinstance(value, tuple) else 'list',
                [_encode(item, arrays, array_ids) for item in value]]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite scalar in static geometry layout')
        return ['scalar', value]
    raise TypeError(f'Unsupported static geometry value: {type(value).__name__}')


def _decode(node, arrays, *, depth=0):
    if depth > 32 or not isinstance(node, list) or not node:
        raise ValueError('Malformed static geometry layout')
    kind = node[0]
    if kind == 'array' and len(node) == 2:
        index = node[1]
        if type(index) is not int or not 0 <= index < len(arrays):
            raise ValueError('Invalid static array index')
        return arrays[index]
    if kind == 'scalar' and len(node) == 2:
        value = node[1]
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError('Invalid static scalar')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Invalid static scalar')
        return value
    if kind == 'neighbor' and len(node) == 4:
        return DeferredNeighborIndex(*[_decode(item, arrays, depth=depth + 1) for item in node[1:]])
    if kind in ('dict', 'list', 'tuple') and len(node) == 2 and isinstance(node[1], list):
        if kind == 'dict':
            result = {}
            for pair in node[1]:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('Malformed static dictionary')
                key, value = [_decode(item, arrays, depth=depth + 1) for item in pair]
                if not isinstance(key, (str, int)) or key in result:
                    raise ValueError('Invalid or duplicate static dictionary key')
                result[key] = value
            return result
        result = [_decode(item, arrays, depth=depth + 1) for item in node[1]]
        return tuple(result) if kind == 'tuple' else result
    raise ValueError('Unsupported static geometry layout node')


def _array(value, dtype, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError('Stored static geometry shape/dtype mismatch')


def _association(value, rows, support_count):
    for name in ('indices', 'weights'):
        _array(value[name], 'int64' if name == 'indices' else 'float64', (rows, 3))
    for name in ('valid', 'component_missing'):
        _array(value[name], 'bool', (rows,))
    for name in ('d1', 'd3'):
        _array(value[name], 'float64', (rows,))
    _array(value['counts'], 'int64', (rows,))
    keep = value['indices'] >= 0
    if (not np.isfinite(value['weights']).all() or np.any(value['weights'] < 0)
            or np.any(value['indices'] < -1) or np.any(value['indices'] >= support_count)
            or np.any(value['weights'][~keep] != 0)
            or not np.array_equal(value['counts'], keep.sum(1))
            or not np.array_equal(value['valid'], value['counts'] > 0)):
        raise ValueError('Invalid stored association')


def _validate_static(value, route):
    """Check executable geometry contracts without content fingerprints."""
    if not isinstance(value, dict) or value.get('spec') != preparation_version(route)['spec']:
        raise ValueError('Static preparation version mismatch')
    if route == 'a':
        required = {'spec', 'geometry', 'settings', 'points', 'normals', 'chunks',
                    'rgb_source_order', 'rgb_source_rows', 'duplicate_source_ids', 'source_support_count'}
        if set(value) != required:
            raise ValueError('Invalid A static fields')
        if (not isinstance(value['settings'], tuple) or len(value['settings']) != 3
                or any(type(n) is not int or n < 1 for n in value['settings'])):
            raise ValueError('Invalid stored A geometry settings')
        geometry = value['geometry']
        if 'rgb_np' in geometry or 'source_references' in geometry:
            raise ValueError('Static geometry cannot contain dynamic source fields')
        m, q = len(geometry['points_np']), len(value['points'])
        for name in ('points_np', 'normals_np'):
            _array(geometry[name], 'float64', (m, 3))
            if not np.isfinite(geometry[name]).all():
                raise ValueError('Nonfinite stored support geometry')
        for name in ('source_ids_np', 'components_np'):
            _array(geometry[name], 'int64', (m,))
        for name in ('normal_valid_np', 'rgb_valid_np', 'raw_feature_valid_np'):
            _array(geometry[name], 'bool', (m,))
        for name in ('centroid_np', 'gravity_np'):
            _array(geometry[name], 'float64', (3,))
            if not np.isfinite(geometry[name]).all():
                raise ValueError('Nonfinite stored instance frame')
        if (not np.isclose(np.linalg.norm(geometry['gravity_np']), 1., rtol=0, atol=1e-12)
                or type(geometry['support_sufficient']) is not bool
                or geometry['support_sufficient'] != (m >= value['settings'][1])):
            raise ValueError('Invalid stored instance frame/support')
        for name in ('points', 'normals'):
            _array(value[name], 'float64', (q, 3))
        source_count = value['source_support_count']
        if type(source_count) is not int or source_count < m:
            raise ValueError('Invalid original source support count')
        _array(value['rgb_source_order'], 'int64', (source_count,))
        _array(value['rgb_source_rows'], 'int64', (m,))
        _array(value['duplicate_source_ids'], 'bool', (max(0, source_count - 1),))
        for name in ('rgb_source_order', 'rgb_source_rows'):
            rows = value[name]
            if len(rows) and (rows.min() < 0 or rows.max() >= source_count):
                raise ValueError('Stored RGB source row out of bounds')
        k = len(geometry['backbone_raw_indices'])
        _array(geometry['backbone_raw_indices'], 'int64', (k,))
        _array(geometry['backbone_voxel_keys'], 'int64', (k, 3))
        if k and (geometry['backbone_raw_indices'].min() < 0 or geometry['backbone_raw_indices'].max() >= m):
            raise ValueError('Stored backbone row out of bounds')
        _association(geometry['raw_association'], m, k)
        for name, rows in (('source_index', slice(None)), ('backbone_index', geometry['backbone_raw_indices'])):
            index = geometry[name]
            for field, source in (('points', 'points_np'), ('source_ids', 'source_ids_np'),
                                  ('components', 'components_np')):
                if not np.array_equal(getattr(index, field), geometry[source][rows]):
                    raise ValueError('Stored neighbor index differs from support geometry')
        chunk_size = value['settings'][2]
        if set(value['chunks']) != set(range(0, q, chunk_size)):
            raise ValueError('Incomplete stored query chunks')
        for start, chunk in value['chunks'].items():
            n = min(chunk_size, q - start)
            if len(chunk) != 7 or len(chunk[-1]) != 6:
                raise ValueError('Invalid stored contact tuple')
            for array in chunk[:2]:
                _array(array, 'float64', (n, 3))
                if not np.isfinite(array).all():
                    raise ValueError('Nonfinite stored normalized query')
            _array(chunk[2], 'int64', (n,))
            _array(chunk[3], 'float64', (n,))
            _array(chunk[4], 'bool', (n,))
            if (np.any(chunk[2] < -1) or np.any(chunk[2] >= m)
                    or np.any(chunk[4] & (chunk[2] < 0))):
                raise ValueError('Invalid stored query anchor')
            _association(chunk[5], n, k)
            contact = chunk[-1]
            for array, dtype, shape in zip(contact, ('int64', 'bool', 'float64', 'int64', 'float64', 'int64'),
                    ((n, 128), (n, 128), (n, 128, 8), (n, 3), (n, 8), (n,))):
                _array(array, dtype, shape)
            rows = contact[0][contact[1]]
            if ((len(rows) and (rows.min() < 0 or rows.max() >= m))
                    or not np.isfinite(contact[2]).all() or not np.isfinite(contact[4]).all()
                    or np.any(contact[3] < 0) or np.any(contact[5] < 0)):
                raise ValueError('Invalid stored contact geometry')
    else:
        # Detailed ROI and tensor validation remains in the B input consumer.
        required = {'spec', 'min_support_points', 'image_hw', 'top_left', 'encoded',
                    'context_geometry', 'texture_geometry', 'context_indices', 'texture_indices',
                    'context_valid', 'texture_valid'}
        if set(value) != required or any(name in value['encoded'] for name in ('context', 'texture', 'phase')):
            raise ValueError('Invalid B static fields')
        from models.instance_route_b import TRAINING_INPUTS_SPEC, _validate_training_wire
        ph, pw = value['encoded']['texture_padded_hw']
        _array(value['context_geometry'], 'float32', (1, 10, 128, 128))
        _array(value['texture_geometry'], 'float32', (1, 2, ph, pw))
        _array(value['context_valid'], 'bool', (128, 128))
        _array(value['texture_valid'], 'bool', (ph, pw))
        for array in (value['context_geometry'], value['texture_geometry'], *value['encoded'].values()):
            if isinstance(array, np.ndarray) and array.dtype.kind == 'f' and not np.isfinite(array).all():
                raise ValueError('Nonfinite stored B image geometry')
        zero = np.zeros((), dtype=np.float32)
        encoded = dict(value['encoded'], context=np.broadcast_to(zero, (1, 13, 128, 128)),
                       texture=np.broadcast_to(zero, (1, 5, ph, pw)), sample_uid='static-validation')
        _validate_training_wire(dict(spec=TRAINING_INPUTS_SPEC, min_support_points=value['min_support_points'],
            encoded=encoded, image_hw=value['image_hw'], top_left=value['top_left']))
        for branch in ('context', 'texture'):
            indices, valid = value[branch + '_indices'], value[branch + '_valid']
            _array(indices, 'int64', (*valid.shape, 2))
            _array(valid, 'bool', valid.shape)
            if (np.any(indices < 0) or np.any(indices[..., 0] >= value['image_hw'][0])
                    or np.any(indices[..., 1] >= value['image_hw'][1])):
                raise ValueError('Stored image sampling index out of bounds')


def write_static_packet(directory, value, *, key, route):
    """Write into a private staging directory; the builder publishes it later."""
    _validate_static(value, route)
    arrays, ids = [], {}
    layout = _encode(value, arrays, ids)
    descriptors, size = [], 0
    for array in arrays:
        offset = (size + 63) // 64 * 64
        descriptors.append(dict(dtype=array.dtype.str, shape=list(array.shape),
                                offset=offset, nbytes=array.nbytes))
        size = offset + array.nbytes
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arena = np.lib.format.open_memmap(directory / 'arrays.npy', mode='w+', dtype=np.uint8, shape=(size,))
    for array, descriptor in zip(arrays, descriptors):
        if array.nbytes:
            arena[descriptor['offset']:descriptor['offset'] + array.nbytes] = array.reshape(-1).view(np.uint8)
    arena.flush()
    del arena
    record = dict(format=FORMAT, route=route, version=preparation_version(route), key=key,
                  arrays=descriptors, arena_bytes=size,
                  file_bytes=(directory / 'arrays.npy').stat().st_size, layout=layout)
    (directory / 'packet.json').write_text(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
    return sum((directory / name).stat().st_size for name in ('arrays.npy', 'packet.json'))


def read_static_packet(directory, *, key, route):
    directory = Path(directory)
    before = directory.stat()
    record = json.loads((directory / 'packet.json').read_text())
    if (record['format'] != FORMAT or record['route'] != route
            or record['version'] != preparation_version(route) or _freeze(record['key']) != _freeze(key)):
        return None
    path = directory / 'arrays.npy'
    if path.stat().st_size != record['file_bytes']:
        raise ValueError('Stored geometry arena length changed')
    arena = np.load(path, allow_pickle=False, mmap_mode='c')
    if arena.dtype != np.uint8 or arena.shape != (record['arena_bytes'],):
        raise ValueError('Invalid geometry arena')
    arrays, previous_end = [], 0
    for item in record['arrays']:
        dtype, shape, offset = np.dtype(item['dtype']), item['shape'], item['offset']
        if (dtype.kind not in 'biuf' or dtype.fields is not None or dtype.itemsize not in (1, 2, 4, 8)
                or not isinstance(shape, list) or len(shape) > 8
                or any(type(n) is not int or n < 0 for n in shape)
                or type(offset) is not int or offset < previous_end or offset % 64):
            raise ValueError('Malformed geometry array descriptor')
        size = math.prod(shape) * dtype.itemsize
        if size != item['nbytes'] or offset + size > len(arena):
            raise ValueError('Stored geometry array exceeds arena')
        arrays.append(np.ndarray(tuple(shape), dtype=dtype, buffer=arena, offset=offset))
        previous_end = offset + size
    value = _decode(record['layout'], arrays)
    _validate_static(value, route)
    after = directory.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError('Geometry entry was replaced during reading')
    return value


def read_static_geometry(root, source_path, *, key, route):
    if not root:
        return None
    directory = geometry_location(root, source_path)
    try:
        ready = json.loads((Path(root) / 'cache.json').read_text())
        if (ready['definition']['format'] != FORMAT or ready['definition']['route'] != route
                or ready['definition']['version'] != preparation_version(route)):
            _STATS['stale'] += 1
            return None
        value = read_static_packet(directory, key=key, route=route)
    except FileNotFoundError:
        _STATS['missing'] += 1
        return None
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, EOFError):
        _STATS['invalid'] += 1
        return None
    if value is None:
        _STATS['stale'] += 1
    else:
        _STATS['hits'] += 1
    return value
