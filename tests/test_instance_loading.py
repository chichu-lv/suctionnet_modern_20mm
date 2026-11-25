"""Real manifest records cross ordered spawn workers without copying warm caches."""
import itertools
import json
import shutil

import numpy as np
import pytest

from suction20.instance_data import InstanceDataset, InstanceAttemptDataset, build_source_instances
from suction20.instance_loading import InstancePreparationPool
from suction20.instance_selection import SourceEvaluationSelection
from suction20.instance_training import _draw_rgb, _apply_rgb
from test_instance_data import _assets, _camera


def _source(tmp_path):
    root, release, _ = _assets(tmp_path)
    base = root/'scenes/scene_0000/realsense'
    for folder, suffix in [('rgb', 'png'), ('depth', 'png'), ('label', 'png'), ('meta', 'mat')]:
        shutil.copyfile(base/folder/f'0000.{suffix}', base/folder/f'0001.{suffix}')
    manifest, _ = build_source_instances(root, release, tmp_path/'cache', split='train',
                                         scenes=[0], frames=[0, 1], train_queries=4)
    return InstanceDataset(manifest, split='train', training_queries=4)


class _UnpicklableCache:
    def __reduce__(self):
        raise AssertionError('The warm parent dataset cache was pickled')


@pytest.mark.parametrize('workers', [0, 2])
def test_ordered_preparation_matches_reader_and_cpu_color_transform(tmp_path, workers):
    data = _source(tmp_path)
    data._cache['large_parent_only_cache'] = _UnpicklableCache()
    rng = np.random.default_rng(75)
    tickets = [(i, _draw_rgb(rng)) for i in [1, 0, 1]]
    with InstancePreparationPool(data, workers=workers, prefetch=2) as pool:
        output = list(pool.map(tickets))
        assert pool.submitted == pool.completed == len(tickets)
    assert pool.closed
    for actual, (index, rgb) in zip(output, tickets):
        expected = data.load_uncached(index)
        expected['observation'] = _apply_rgb(expected['observation'], rgb)
        assert actual['record'] == expected['record']
        for name in ('rgb', 'support_rgb', 'query_points', 'query_normals', 'query_ids'):
            assert np.array_equal(actual['observation'][name], expected['observation'][name])
        for name in ('q_inst', 'label_valid', 'points', 'normal_out'):
            assert np.array_equal(actual['labels'][name], expected['labels'][name], equal_nan=True)


def test_spawn_uses_selected_view_record_order_and_bounded_infinite_input(tmp_path):
    data = _source(tmp_path)
    view = SourceEvaluationSelection(data, [1, 0], {'is_subset': False}, [])
    produced = []
    def tickets():
        for i in itertools.count():
            index = i % 2
            produced.append(index)
            yield index, None
    with InstancePreparationPool(view, workers=1, prefetch=2) as pool:
        stream = pool.map(tickets())
        first = next(stream)
        assert first['record']['sample_id'] == data.records[1]['sample_id']
        assert len(produced) == 3  # two initially queued, one replacement before yield
        second = next(stream)
        assert second['record']['sample_id'] == data.records[0]['sample_id']
        assert len(produced) == 4
        assert pool.submitted - pool.completed == pool.capacity == 2
        stream.close()
    assert len(produced) == 4 and pool.closed


def test_target_spawn_preserves_recorded_pose_and_split(tmp_path):
    root, _, _ = _assets(tmp_path)
    base = root/'scenes/scene_0000/realsense'
    observation = dict(format='rgbd', rgb_path=str(base/'rgb/0000.png'),
        depth_path=str(base/'depth/0000.png'), instance_mask_path=str(base/'label/0000.png'),
        depth_scale_m=.001, intrinsics=_camera().tolist(), g_cam=[0., 0., -1.])
    row = dict(schema_version='suction20.target.v1', label_kind='attempt_success',
        sample_id='attempt', group_id='group', split='val', observation=observation,
        success_definition='pickup_and_transport', attempt=dict(point_cam_m=[.001, 0., .6],
        normal_out_cam=[0., .6, -.8], instance_id=1, success=1, label_valid=True))
    path = tmp_path/'attempts.jsonl'
    path.write_text(json.dumps(row)+'\n')
    data = InstanceAttemptDataset(path, split='val')
    with InstancePreparationPool(data, workers=1, route='a') as pool:
        actual = list(pool.map([(0, None)]))[0]
    expected = data.load_uncached(0)
    assert actual['record'] == expected['record']
    for key in ('point_cam_m', 'normal_out_cam', 'pose_support_valid', 'label_valid'):
        assert np.array_equal(actual['attempts'][key], expected['attempts'][key])


@pytest.mark.parametrize('route', ['a', 'b'])
def test_route_geometry_and_images_prepared_after_augmentation(tmp_path, route):
    # Branch CI contains its own model; the local union may contain both.
    pytest.importorskip(f'models.instance_route_{route}')
    data = _source(tmp_path)
    options = {'min_support_points': 100}
    if route == 'a':
        options.update(backbone_points=32, query_chunk_size=4)
    rgb = _draw_rgb(np.random.default_rng(8))
    with InstancePreparationPool(data, workers=1, route=route, geometry_options=options) as pool:
        item = next(pool.map([(0, rgb)]))
    prepared_key = '_route_a_geometry' if route == 'a' else '_route_b_inputs'
    assert prepared_key in item['observation']
    expected = _apply_rgb(data.load_uncached(0)['observation'], rgb)
    assert np.array_equal(item['observation']['rgb'], expected['rgb'])
    assert np.array_equal(item['observation']['query_points'], expected['query_points'])


@pytest.mark.parametrize('route', ['a', 'b'])
def test_compact_worker_skips_unused_archive_arrays_before_route_preparation(tmp_path, monkeypatch, route):
    model_module = pytest.importorskip(f'models.instance_route_{route}')
    from suction20 import instance_loading as loading
    from suction20.instance_data import _COMPACT_SOURCE_UNUSED_ARRAYS

    data = _source(tmp_path)
    options = {'min_support_points': 100}
    if route == 'a':
        options.update(backbone_points=32, query_chunk_size=4)
    spec = dict(loading._reader_spec(data), route=route, compact_source=True, geometry_options=options)
    rgb = _draw_rgb(np.random.default_rng(8))
    expected = model_module.compact_prepared_source_item(
        loading._prepare_observation(data.load_uncached(0), rgb, spec))
    path = data.root / data.records[0]['observation_npz']
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original = archive_type.__getitem__

    def read_required(self, key):
        assert key not in _COMPACT_SOURCE_UNUSED_ARRAYS, 'compact worker inflated an unused image'
        return original(self, key)

    monkeypatch.setattr(archive_type, '__getitem__', read_required)
    monkeypatch.setattr(loading, '_WORKER_SPEC', spec)
    actual = loading._worker_prepare((data.records[0], rgb))
    assert actual['record'] == expected['record']

    def arrays(value, prefix=()):
        if isinstance(value, np.ndarray):
            return {prefix: value}
        children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
        return {path: array for key, child in children for path, array in arrays(child, prefix + (key,)).items()}

    actual_arrays, expected_arrays = arrays(actual), arrays(expected)
    assert actual_arrays.keys() == expected_arrays.keys()
    for key in actual_arrays:
        assert actual_arrays[key].dtype == expected_arrays[key].dtype
        np.testing.assert_array_equal(actual_arrays[key], expected_arrays[key])


def test_worker_read_failure_propagates_and_closes_pool(tmp_path):
    data = _source(tmp_path)
    (data.root/data.records[0]['observation_npz']).unlink()
    pool = InstancePreparationPool(data, workers=1)
    with pytest.raises(FileNotFoundError):
        list(pool.map([(0, None)]))
    assert pool.closed
    with pytest.raises(RuntimeError, match='closed'):
        pool.map([(1, None)])


def test_crashed_worker_does_not_leave_an_unbounded_result_wait(tmp_path):
    data = _source(tmp_path)
    pool = InstancePreparationPool(data, workers=1, prefetch=1)
    stream = pool.map(itertools.repeat((0, None)))
    next(stream)
    process = pool._processes[0]
    process.terminate()
    process.join(timeout=5)
    with pytest.raises(RuntimeError, match='worker exited'):
        next(stream)
    assert pool.closed


@pytest.mark.parametrize('workers,prefetch', [(-1, 2), (True, 2), (1.5, 2), (1, 0), (1, True)])
def test_worker_configuration_rejects_invalid_bounds(workers, prefetch):
    with pytest.raises(ValueError):
        InstancePreparationPool([], workers=workers, prefetch=prefetch)
