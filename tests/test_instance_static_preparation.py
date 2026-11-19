"""Static source preparation is disposable, bounded and independent of RGB/RNG."""
import copy
import importlib.util
import os
import pickle

import numpy as np
import pytest
import torch

from suction20.instance_preparation_cache import (
    StaticGeometryCache, source_geometry_key, static_bytes,
)
from suction20.instance_training import source_support_count
from test_instance_prefetch_resume import datasets


def test_static_cache_byte_budget_eviction_and_oversized_entry():
    value = {'array': np.arange(128, dtype=np.float64)}
    size = static_bytes(value)
    cache = StaticGeometryCache(size + 64, max_entries=2)
    first = cache.get_or_prepare(('sample', 1), lambda: value)
    assert cache.get_or_prepare(('sample', 1), lambda: pytest.fail('cache miss')) is first
    cache.get_or_prepare(('sample', 2), lambda: copy.deepcopy(value))
    assert cache.hits == 1 and cache.evictions == 1 and len(cache.entries) == 1
    large = cache.get_or_prepare(('large',), lambda: np.empty(100000))
    assert large.shape == (100000,) and ('large',) not in cache.entries
    assert cache.bytes <= cache.max_bytes
    zero = StaticGeometryCache(0)
    assert zero.get_or_prepare('x', lambda: 3) == 3 and not zero.entries
    owner = np.empty(100000)
    tiny_view = owner[:1]
    assert static_bytes(tiny_view) >= owner.nbytes
    cache.get_or_prepare('view', lambda: tiny_view)
    assert 'view' not in cache.entries


def test_source_geometry_key_tracks_version_query_settings_and_file_metadata(tmp_path):
    path = tmp_path / 'observation.npz'
    path.write_bytes(b'fixture')
    record = dict(observation_npz=path.name, sample_id='sample',
                  observation_identity={'schema': 'v2'}, query_identity={'version': 3})
    spec = dict(root=str(tmp_path), route='a', training_queries=128,
                geometry_options={'backbone_points': 2048, 'query_chunk_size': 32})
    original = source_geometry_key(record, spec)
    assert original == source_geometry_key(copy.deepcopy(record), copy.deepcopy(spec))
    assert original != source_geometry_key(record, dict(spec, route='b'))
    assert original != source_geometry_key(record, dict(spec, training_queries=64))
    assert original != source_geometry_key(record, dict(spec, geometry_options={'backbone_points': 1024}))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    assert original != source_geometry_key(record, spec)


def test_worker_does_not_cache_arrays_under_a_replaced_source_key(tmp_path, monkeypatch):
    from suction20 import instance_loading as loading
    from suction20.instance_data import InstanceDataset
    route = 'b' if importlib.util.find_spec('models.instance_route_b') is not None else 'a'
    module = __import__(f'models.instance_route_{route}', fromlist=['compact_prepared_source_item'])
    path = tmp_path / 'sample.npz'
    path.write_bytes(b'old file')
    record = dict(observation_npz=path.name, sample_id='sample')
    spec = dict(kind='source', root=str(tmp_path), route=route, compact_source=True)
    cache = StaticGeometryCache()
    seen = []
    def read(reader, index, **kwargs):
        if not seen:
            replacement = path.with_suffix('.new')
            replacement.write_bytes(b'new file')
            replacement.replace(path)
        return dict(observation={}, record=record)
    def prepare(item, rgb, spec, *, static_cache=None, static_key=None):
        seen.append((static_cache, static_key))
        return item
    monkeypatch.setattr(InstanceDataset, 'load_uncached', read)
    monkeypatch.setattr(loading, '_WORKER_SPEC', spec)
    monkeypatch.setattr(loading, '_WORKER_STATIC_CACHE', cache)
    monkeypatch.setattr(loading, '_prepare_observation', prepare)
    monkeypatch.setattr(module, 'compact_prepared_source_item', lambda item: item)
    loading._worker_prepare((record, None))
    assert seen[0] == (None, None)
    loading._worker_prepare((record, None))
    assert seen[1][0] is cache and seen[1][1] == source_geometry_key(record, spec)


def test_compact_source_gate_keeps_flattened_count_separate_from_image_count():
    assert source_support_count({'source_support_count': 7, 'support_count': 4225}) == 7
    assert source_support_count({'support_points': np.empty((9, 3))}) == 9
    assert source_support_count({'support_points': np.empty((9, 3)), 'source_support_count': 2}) == 9
    for bad in (-1, 3.5, True):
        with pytest.raises(ValueError, match='source_support_count'):
            source_support_count({'source_support_count': bad})


@pytest.mark.skipif(importlib.util.find_spec('models.instance_route_a') is None, reason='A branch only')
def test_a_explicit_native_benchmark_cannot_silently_use_reference(monkeypatch):
    import models.instance_route_a as route
    marker = object()
    monkeypatch.setattr(route, 'native_contact_geometry', lambda *args: None)
    monkeypatch.setattr(route, 'reference_contact_geometry', lambda *args: marker)
    monkeypatch.setenv('SUCTION_A_CONTACT_BACKEND', 'native')
    with pytest.raises(RuntimeError, match='unavailable'):
        route._contact_geometry(None, None, None, None, None)
    for mode in ('auto', 'reference'):
        monkeypatch.setenv('SUCTION_A_CONTACT_BACKEND', mode)
        assert route._contact_geometry(None, None, None, None, None) is marker


@pytest.mark.skipif(importlib.util.find_spec('models.instance_route_a') is None, reason='A branch only')
def test_a_static_geometry_rebinds_new_rgb_without_geometry_or_weight_reuse(monkeypatch):
    import models.instance_route_a as route
    from test_instance_route_a import InjectedBackbone, observation
    obs = observation(side=21)
    options = dict(backbone_points=32, query_chunk_size=2)
    static = route.prepare_source_static_geometry(obs, **options)
    assert 'rgb_np' not in static['geometry'] and 'source_references' not in static
    snapshot = pickle.dumps(static)
    torch.manual_seed(165)
    model = route.InstanceSuction3D(backbone=InjectedBackbone(), **options).eval()
    another = dict(obs, support_rgb=np.full_like(obs['support_rgb'], .8))
    expected = model(another)
    expected['q_inst'].sum().backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(route, '_instance_geometry', lambda *a: pytest.fail('Rebuilt static geometry'))
    monkeypatch.setattr(route, '_query_geometry', lambda *a: pytest.fail('Rebuilt cached queries'))
    prepared = route.prepare_source_geometry(another, static_geometry=static, **options)
    actual = model(prepared)
    for name, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        else:
            assert actual[name] == value
    actual['q_inst'].sum().backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert pickle.dumps(static) == snapshot
    assert not np.shares_memory(prepared['_route_a_geometry']['geometry']['rgb_np'], obs['support_rgb'])
    invalid = dict(another, support_rgb_valid=np.zeros(len(obs['support_points']), bool))
    with pytest.raises(ValueError, match='RGB validity'):
        route.prepare_source_geometry(invalid, static_geometry=static, **options)
    changed = dict(another, query_points=another['query_points'] + .001)
    with pytest.raises(ValueError, match='canonical queries'):
        route.prepare_source_geometry(changed, static_geometry=static, **options)


@pytest.mark.parametrize('workers', [0, 1])
def test_compact_worker_reuses_static_geometry_with_new_colors_and_cold_restart(tmp_path, workers):
    from test_instance_loading import _source
    from suction20.instance_loading import InstancePreparationPool
    from suction20.instance_selection import SourceEvaluationSelection
    from suction20.instance_training import _draw_rgb
    route = 'b' if importlib.util.find_spec('models.instance_route_b') is not None else 'a'
    dataset = _source(tmp_path)
    dataset = SourceEvaluationSelection(dataset, [1, 0], {'is_subset': False}, [])
    rng = np.random.default_rng(91)
    tickets = [(0, _draw_rgb(rng)) for _ in range(3)]
    options = dict(min_support_points=100)
    if route == 'a':
        options.update(backbone_points=32, query_chunk_size=4)
    settings = dict(workers=workers, route=route, geometry_options=options, compact_source=True)
    with InstancePreparationPool(dataset, **settings) as pool:
        cached = list(pool.map(tickets))
        assert cached[0]['record']['sample_id'] == dataset.records[0]['sample_id']
        stats = pool.preparation_cache_stats()
        assert stats['hits'] == 2 and stats['misses'] == 1
        assert stats['retained_bytes'] <= stats['max_bytes']
    with InstancePreparationPool(dataset, static_cache_bytes=0, **settings) as pool:
        cold = list(pool.map(tickets))
        assert pool.preparation_cache_stats()['hits'] == 0
    key = '_route_a_geometry' if route == 'a' else '_route_b_training_inputs'
    color_key = 'rgb_np' if route == 'a' else 'context'
    subkey = 'geometry' if route == 'a' else 'encoded'
    colors = []
    for actual, expected in zip(cached, cold):
        left, right = actual['observation'][key][subkey], expected['observation'][key][subkey]
        colors.append(left[color_key])
        for name, value in left.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, right[name])
        np.testing.assert_array_equal(actual['labels']['q_inst'], expected['labels']['q_inst'])
    assert any(not np.array_equal(colors[0], value) for value in colors[1:])


@pytest.mark.parametrize('stage', ['pretrain', 'finetune'])
def test_compact_static_cache_cold_resume_preserves_training_state(tmp_path, datasets, monkeypatch, stage):
    import test_instance_prefetch_resume as resume
    original_color = resume.TinyBatchedModel._color

    class CompactColorModel(resume.TinyBatchedModel):
        @staticmethod
        def _color(observation):
            if '_route_a_geometry' in observation:
                return np.asarray(observation['_route_a_geometry']['geometry']['rgb_np'], np.float32).mean(0)
            if '_route_b_training_inputs' in observation:
                context = observation['_route_b_training_inputs']['encoded']['context']
                return np.asarray(context[0, :3], np.float32).mean(axis=(1, 2))
            return original_color(observation)
    monkeypatch.setattr(resume, 'TinyBatchedModel', CompactColorModel)
    monkeypatch.setenv('SUCTION_COMPACT_SOURCE_IPC', '1')
    resume.test_spawn_checkpoint_restores_pending_rgb_sampler_and_optimizer(tmp_path, datasets, stage)
