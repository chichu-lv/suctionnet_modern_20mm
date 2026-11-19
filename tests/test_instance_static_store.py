"""Persistent immutable geometry keeps draw colors and recovery state live."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_instance_geometry_cache import build_geometry_cache
from suction20.instance_loading import InstancePreparationPool
from suction20.instance_static_store import (DeferredNeighborIndex, FORMAT, _decode, _encode,
    geometry_location, preparation_version, read_static_geometry, read_static_packet,
    reset_static_store_stats, static_store_stats, write_static_packet)
from test_instance_loading import _source
from test_instance_prefetch_resume import datasets
from test_instance_training import assert_nested_equal


ROUTE = 'b' if importlib.util.find_spec('models.instance_route_b') is not None else 'a'


def _config():
    model = dict(min_support_points=100)
    if ROUTE == 'a':
        model.update(backbone_points=32, query_chunk_size=4)
    return dict(route=ROUTE, model=model, rgb_augmentation=True, pretrain=dict(queries=4))


def _static():
    if ROUTE == 'a':
        from models.instance_route_a import prepare_source_static_geometry
        from test_instance_route_a import observation
        obs = observation(side=21)
        return obs, prepare_source_static_geometry(obs, backbone_points=32, query_chunk_size=2)
    from models.instance_route_b_preparation import prepare_source_static_inputs
    from test_instance_route_b_static_roi import observation
    obs = observation('interior')
    return obs, prepare_source_static_inputs(obs)


def _equal_arrays(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
        assert left.dtype == right.dtype
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _equal_arrays(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) == type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _equal_arrays(a, b)
    elif type(left).__name__ in ('NeighborIndex', 'DeferredNeighborIndex'):
        for field in ('points', 'source_ids', 'components'):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    else:
        assert left == right


def test_static_arena_roundtrip_is_exact_cow_and_has_no_eager_trees(tmp_path):
    _, static = _static()
    location = tmp_path / 'entry'
    key = ('example', ROUTE, 128)
    size = write_static_packet(location, static, key=key, route=ROUTE)
    loaded = read_static_packet(location, key=key, route=ROUTE)
    assert size == sum(path.stat().st_size for path in location.iterdir())
    _equal_arrays(static, loaded)
    if ROUTE == 'a':
        index = loaded['geometry']['source_index']
        assert isinstance(index, DeferredNeighborIndex) and index._index is None
        point = static['geometry']['points_np'][:1]
        expected = static['geometry']['source_index'].nearest_anchor(point)
        actual = index.nearest_anchor(point)
        _equal_arrays(expected, actual)
        target = loaded['geometry']['gravity_np']
    else:
        target = loaded['encoded']['gravity']
    target[:] = 0
    # Copy-on-write mapping never modifies the shared on-disk geometry.
    _equal_arrays(static, read_static_packet(location, key=key, route=ROUTE))
    assert read_static_packet(location, key=('different-source',), route=ROUTE) is None


def test_static_arena_preserves_zero_dimensional_arrays_without_pickle():
    arrays, ids = [], {}
    expected = {'scalar_array': np.array(-0., np.float64), 7: (np.array([np.nan]), None)}
    layout = _encode(expected, arrays, ids)
    restored = _decode(json.loads(json.dumps(layout)), arrays)
    _equal_arrays(expected, restored)
    assert restored['scalar_array'].shape == () and np.signbit(restored['scalar_array'])
    with pytest.raises(ValueError, match='numeric'):
        _encode(np.array([object()], dtype=object), [], {})


def test_static_store_rejects_bad_geometry_contracts_at_read_boundary(tmp_path):
    _, static = _static()
    broken = deepcopy(static)
    if ROUTE == 'a':
        first = next(iter(broken['chunks'].values()))
        first[5]['indices'][0, 0] = len(broken['geometry']['backbone_raw_indices'])
    else:
        broken['encoded']['bilinear_valid_map'] = broken['encoded']['bilinear_valid_map'].astype(np.float32)
    with pytest.raises(ValueError):
        write_static_packet(tmp_path / 'broken', broken, key=('x',), route=ROUTE)
    broken = deepcopy(static)
    field = broken['geometry']['points_np'] if ROUTE == 'a' else broken['context_geometry']
    field.flat[0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite|Invalid component'):
        write_static_packet(tmp_path / 'nonfinite', broken, key=('x',), route=ROUTE)


def test_geometry_locations_distinguish_same_basename_at_different_depths(tmp_path):
    a = geometry_location(tmp_path / 'cache', tmp_path / 'a' / 'same' / 'entry.npz')
    b = geometry_location(tmp_path / 'cache', tmp_path / 'b' / 'same' / 'entry.npz')
    assert a != b


def test_replacement_during_packet_read_never_pairs_two_generations(tmp_path, monkeypatch):
    import suction20.instance_static_store as store
    _, static = _static()
    location, replacement = tmp_path / 'current', tmp_path / 'replacement'
    write_static_packet(location, static, key=('same',), route=ROUTE)
    write_static_packet(replacement, static, key=('same',), route=ROUTE)
    original = store.np.load
    def swap(path, **kwargs):
        location.rename(tmp_path / 'old')
        replacement.rename(location)
        return original(path, **kwargs)
    monkeypatch.setattr(store.np, 'load', swap)
    with pytest.raises(ValueError, match='replaced during reading'):
        read_static_packet(location, key=('same',), route=ROUTE)


@pytest.mark.parametrize('workers', [0, 1])
def test_first_draw_uses_disk_geometry_with_new_rgb_and_zero_memory_cache(tmp_path, monkeypatch, workers):
    from suction20.instance_training import _draw_rgb
    source = _source(tmp_path / 'source')
    output = tmp_path / 'geometry'
    cfg = _config()
    result = build_geometry_cache(source.manifest, cfg, output, max_bytes=32 * 1024**2,
                                  max_records=1)
    assert result['built'] == 1 and result['entries_available'] == 1
    assert not result['complete']
    tickets = [(0, _draw_rgb(np.random.default_rng(seed))) for seed in (7, 43)]
    settings = dict(workers=workers, route=ROUTE, geometry_options=cfg['model'],
                    compact_source=True, static_cache_bytes=0)
    with InstancePreparationPool(source, **settings) as pool:
        cold = list(pool.map(tickets))
    monkeypatch.setenv('SUCTION_INSTANCE_GEOMETRY_CACHE_' + ROUTE.upper(), str(output))
    with InstancePreparationPool(source, **settings) as pool:
        cached = list(pool.map(tickets))
        stats = pool.preparation_cache_stats()
    assert stats['geometry_store']['hits'] == 2
    assert stats['entries'] == 0
    for expected, actual in zip(cold, cached):
        _equal_arrays(expected['labels'], actual['labels'])
        key = '_route_a_geometry' if ROUTE == 'a' else '_route_b_training_inputs'
        # Source identity references are intentionally attached per draw.
        left, right = dict(expected['observation'][key]), dict(actual['observation'][key])
        left.pop('source_references', None); right.pop('source_references', None)
        _equal_arrays(left, right)
    cache_key = '_route_a_geometry' if ROUTE == 'a' else '_route_b_training_inputs'
    inner, colors = ('geometry', 'rgb_np') if ROUTE == 'a' else ('encoded', 'context')
    assert not np.array_equal(cached[0]['observation'][cache_key][inner][colors],
                              cached[1]['observation'][cache_key][inner][colors])


def test_stale_and_invalid_entries_fall_back_without_online_writes(tmp_path, monkeypatch):
    source = _source(tmp_path / 'source')
    root = tmp_path / 'geometry'
    cfg = _config()
    build_geometry_cache(source.manifest, cfg, root, max_bytes=32 * 1024**2, max_records=1)
    monkeypatch.setenv('SUCTION_INSTANCE_GEOMETRY_CACHE_' + ROUTE.upper(), str(root))
    record = source.records[0]
    path = (source.root / record['observation_npz']).resolve()
    location = geometry_location(root, path)
    receipt = location / 'packet.json'
    payload = json.loads(receipt.read_text())
    payload['arrays'][0]['offset'] = payload['arena_bytes'] + 64
    receipt.write_text(json.dumps(payload))
    unchanged = receipt.read_text()
    settings = dict(workers=0, route=ROUTE, geometry_options=cfg['model'],
                    compact_source=True, static_cache_bytes=0)
    with InstancePreparationPool(source, **settings) as pool:
        list(pool.map([(0, (np.ones(3), (0, 1, 2), 0, False))]))
        assert pool.preparation_cache_stats()['geometry_store']['invalid'] == 1
    assert receipt.read_text() == unchanged
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    with InstancePreparationPool(source, **settings) as pool:
        list(pool.map([(0, (np.ones(3), (0, 1, 2), 0, False))]))
        assert pool.preparation_cache_stats()['geometry_store']['stale'] == 1


def test_builder_budget_resume_and_source_identity(tmp_path):
    source = _source(tmp_path / 'source')
    config = _config()
    original = source.manifest.read_text()
    limited = build_geometry_cache(source.manifest, config, tmp_path / 'small', max_bytes=1)
    assert limited['budget_exhausted'] and limited['entries_available'] == 0
    root = tmp_path / 'geometry'
    first = build_geometry_cache(source.manifest, config, root, max_bytes=32 * 1024**2, max_records=1)
    with pytest.raises(ValueError, match='exceeds'):
        build_geometry_cache(source.manifest, config, root, max_bytes=1, max_records=1, resume=True)
    resumed = build_geometry_cache(source.manifest, config, root, max_bytes=32 * 1024**2,
                                  workers=2, resume=True)
    assert resumed['reused'] == first['built'] == 1
    assert resumed['complete'] and resumed['entries_available'] == len(source)
    assert source.manifest.read_text() == original
    assert not list(root.glob('geometry-build-*'))
    with pytest.raises(ValueError, match='shrink'):
        build_geometry_cache(source.manifest, config, root, max_bytes=32 * 1024**2,
                             max_records=1, resume=True)


def test_interleaved_serial_pools_do_not_double_count_each_others_reads(tmp_path):
    source = _source(tmp_path / 'source')
    settings = dict(workers=0, route=ROUTE, geometry_options=_config()['model'], compact_source=True)
    with InstancePreparationPool(source, **settings) as first, InstancePreparationPool(source, **settings) as second:
        list(first.map([(0, None)]))
        list(second.map([(1, None)]))
        assert first.preparation_cache_stats()['reader']['npz_reads'] == 2
        assert second.preparation_cache_stats()['reader']['npz_reads'] == 2


@pytest.mark.parametrize('stage', ['pretrain', 'finetune'])
def test_persistent_geometry_preserves_full_checkpoint_resume(tmp_path, datasets, monkeypatch, stage):
    from test_instance_static_preparation import test_compact_static_cache_cold_resume_preserves_training_state
    root = tmp_path / 'geometry'
    build_geometry_cache(datasets[0].manifest, _config(), root, max_bytes=32 * 1024**2)
    monkeypatch.setenv('SUCTION_INSTANCE_GEOMETRY_CACHE_' + ROUTE.upper(), str(root))
    monkeypatch.setenv('SUCTION_INSTANCE_STATIC_CACHE_MB', '0')
    test_compact_static_cache_cold_resume_preserves_training_state(tmp_path, datasets, monkeypatch, stage)
