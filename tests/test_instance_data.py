"""Real-format synthetic assets exercise the actual instance data/teacher path."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat
from scipy.spatial import cKDTree

from suction20.eval_r20 import evaluate_poses, load_scene_eval_context, seal_at_pose
from suction20.full_data import DEFAULT_CONFIG, collision_mask_for_instance, transform_points, voxel_filter
from suction20.instance_data import (InstanceAttemptDataset, InstanceDataset, build_source_instances,
    load_observation, save_observation, source_scene_ids, source_split,
    merge_source_manifests, _save_npz)
from suction20.instance_data import (source_build_output, source_output_lock,
                                    _read_npz, _COMPACT_SOURCE_UNUSED_ARRAYS,
                                    SOURCE_TRAIN_ARRAYS, QUERY_FIELDS, LABEL_ALIASES)
from suction20.instance_geometry import BOUNDARY_ROUNDOFF_M, prepare_instance, stratified_queries_refill
from suction20.instance_teacher import InstanceTeacher, empty_labels, validate_labels


def _camera(size=65):
    return np.array([[600., 0., (size - 1) / 2], [0., 600., (size - 1) / 2], [0., 0., 1.]])


def _observation(size=65):
    return prepare_instance(np.full((size, size, 3), .5, np.float32),
        np.full((size, size), .6), np.ones((size, size), bool), _camera(size), [0., 0., -1.])


def _assets(tmp_path, scene_id=0, protrusion=False):
    root, release = tmp_path / 'dataset', tmp_path / 'labels'
    base = root / 'scenes' / f'scene_{scene_id:04d}' / 'realsense'
    for name in ('rgb', 'depth', 'label', 'meta'):
        (base / name).mkdir(parents=True)
    (root / 'dense_point_clouds').mkdir()
    (release / 'seal_r20/poisson').mkdir(parents=True)
    size = 65
    Image.fromarray(np.full((size, size, 3), 127, np.uint8)).save(base / 'rgb/0000.png')
    Image.fromarray(np.full((size, size), 600, np.uint16)).save(base / 'depth/0000.png')
    Image.fromarray(np.ones((size, size), np.uint8)).save(base / 'label/0000.png')
    pose = np.column_stack((np.eye(3), [0., 0., .6]))[:, :, None]
    savemat(base / 'meta/0000.mat', {'intrinsic_matrix': _camera(), 'factor_depth': 1000.,
                                   'cls_indexes': np.array([[1]]), 'poses': pose})
    np.save(base / 'cam0_wrt_table.npy', np.eye(4))
    np.save(base / 'camera_poses.npy', np.repeat(np.eye(4)[None], 256, axis=0))
    (base.parent / 'object_id_list.txt').write_text('0\n')
    # Full dense disk with a point strictly inside every open Seal sector.
    theta = -np.pi + (np.arange(72) + .5) * (2 * np.pi / 72)
    radii = np.linspace(.001, .05, 50)
    dense = np.array([[r * np.cos(t), r * np.sin(t), 0.] for r in radii for t in theta])
    if protrusion:
        dense = np.concatenate((dense, [[0., 0., -.006], [.025, 0., -.040]]))
    np.savez(root / 'dense_point_clouds/000.npz', points=dense)
    candidates = np.array([[0., 0., 0.], [.002, 0., 0.]])
    np.savez(release / 'seal_r20/poisson/000_labels.npz', points=candidates,
        normals=np.tile([0., 0., -1.], (2, 1)), scores=np.ones(2),
        collision=np.zeros(2, bool), indices=np.arange(2))
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(table_grid_m=.05, table_width_m=.1, table_depth_m=.1)
    (release / 'config_frozen.json').write_text(json.dumps(cfg))
    return root, release, dense


def test_source100_split_and_no_dev():
    assert source_scene_ids('train') == tuple(range(100))
    assert len(source_scene_ids('test')) == 90
    assert source_split(99) == ('train', None)
    assert source_split(100) == ('test', 'seen')
    assert source_split(130) == ('test', 'similar')
    assert source_split(160) == ('test', 'novel')
    with pytest.raises(ValueError):
        source_scene_ids('val')


def test_raw_geometry_normals_queries_and_rgb_mask_roundtrip(tmp_path):
    size = 65
    z = np.full((size, size), .6)
    rgb = np.full((size, size, 3), .7, np.float32)
    mask = np.zeros_like(z, bool)
    mask[5:-5, 5:-5] = True
    z[15, 15] = np.nan
    rgb[20, 20] = np.nan
    z_original = z.copy()
    obs = prepare_instance(rgb, z, mask, _camera(), [0, 0, -1])
    assert np.array_equal(z_original, z, equal_nan=True)
    assert len(obs['query_points']) == 256
    assert np.all(obs['query_normals'][:, 2] == -1)
    ids = obs['query_source_ids']
    assert np.array_equal(obs['query_points'], obs['xyz_map'].reshape(-1, 3)[ids])
    assert len(np.unique(ids)) == 256
    assert not obs['rgb_valid'][20, 20] and np.all(obs['rgb'][20, 20] == 0)
    assert np.all(obs['rgb'][~mask] == 0) and np.all(obs['xyz_map'][~mask] == 0)
    assert np.array_equal(obs['bbox_xyxy'], [5, 5, 60, 60])
    path = tmp_path / 'instance.npz'
    save_observation(path, obs)
    restored = load_observation(path)
    for key in ('query_points', 'query_normals', 'support_points', 'support_rgb_valid', 'mask', 'rgb_valid'):
        assert np.array_equal(restored[key], obs[key])
    obs['query_points'] = obs['query_points'].copy()
    obs['query_points'][0, 0] += 1e-5
    save_observation(path, obs)
    with pytest.raises(ValueError, match='raw source pixels'):
        load_observation(path)


def test_refill_prefix_and_hole_path_rejection():
    y, x = np.mgrid[:128, :128]
    mask = abs(x - y) <= 4
    full, diag = stratified_queries_refill(mask, mask)
    small, _ = stratified_queries_refill(mask, mask, 64)
    assert len(full) == 256 and diag['first_pass_count'] == 46
    assert np.array_equal(small, full[:64])
    rgb = np.ones((65, 65, 3), np.float32)
    depth = np.full((65, 65), .6)
    mask = np.ones_like(depth, bool)
    depth[32, 33] = 0
    obs = prepare_instance(rgb, depth, mask, _camera(), [0, 0, -1])
    assert not obs['normal_valid'][32, 32]  # Endpoints remain; internal hole must reject.
    empty = prepare_instance(rgb, depth, np.zeros_like(mask), _camera(), [0, 0, -1])
    assert empty['query_points'].shape == (0, 3) and not empty['trainable']


@pytest.mark.parametrize('protrusion', [False, True])
def test_target_only_teacher_matches_legacy_components(tmp_path, protrusion):
    root, release, dense = _assets(tmp_path, protrusion=protrusion)
    teacher = InstanceTeacher(root, release)
    p = np.array([[0., 0., .6], [.003, -.002, .6]])
    n = np.tile([0., 0., -1.], (2, 1))
    labels = teacher.evaluate(0, 0, 0, p, n)
    old_context = load_scene_eval_context(root, release, 0, 'realsense')
    old = evaluate_poses(old_context, 0, p, n, np.zeros(2, int))
    assert np.array_equal(labels['seal'], old['S'])
    assert np.array_equal(labels['wrench'], old['W'])
    assert np.array_equal(labels['collision_self'], old['collision'])
    assert np.array_equal(labels['q_inst'], old['Q20'])
    assert labels['collision_self_seal'][0] == protrusion
    cfg = teacher.config
    obstacle = transform_points(voxel_filter(dense, cfg['obstacle_voxel_m']),
        old_context.records[0]['T_world_obj']).astype(np.float32)
    envelope = collision_mask_for_instance(p, n, cKDTree(obstacle), obstacle,
        np.zeros(len(obstacle), int), 0, np.zeros(2, bool), cfg)
    assert np.array_equal(envelope, labels['collision_self_envelope'])
    assert validate_labels(labels, expected_points=p, expected_normals=n)
    changed = p.copy(); changed[0, 0] += 1e-7
    with pytest.raises(ValueError, match='pose differs'):
        validate_labels(labels, expected_points=changed)


def test_teacher_excludes_external_obstacle(tmp_path):
    root, release, _ = _assets(tmp_path)
    teacher = InstanceTeacher(root, release)
    p, n = np.array([[0., 0., .6]]), np.array([[0., 0., -1.]])
    labels = teacher.evaluate(0, 0, 0, p, n)
    context = load_scene_eval_context(root, release, 0, 'realsense')
    context.obstacle_points = np.concatenate((context.obstacle_points, [[0., 0., .55]]))
    context.obstacle_rows = np.concatenate((context.obstacle_rows, [-1]))
    context.obstacle_tree = cKDTree(context.obstacle_points)
    old = evaluate_poses(context, 0, p, n, np.zeros(1, int))
    assert not labels['collision_self'][0] and labels['q_inst'][0] > 0
    assert old['collision'][0] and old['Q20'][0] == 0


def test_unknown_labels_remain_unknown():
    labels = empty_labels(np.array([[0., 0., .6]]), np.array([[0., 0., -1.]]))
    assert not validate_labels(labels)
    assert np.isnan(labels['q_inst'][0]) and not labels['label_valid'][0]
    labels['label_valid'][0] = True
    with pytest.raises(ValueError):
        validate_labels(labels)


def test_real_format_builder_prefix_integrity_and_resume(tmp_path):
    root, release, _ = _assets(tmp_path, scene_id=9)
    manifest, summary = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[9], frames=[0], train_queries=64)
    assert summary['frame_count'] == 1 and not summary['complete_requested_split']
    ds = InstanceDataset(manifest, split='train', training_queries=64)
    item = ds[0]
    assert item['record']['scene_id'] == 9 and item['record']['split'] == 'train'
    assert item['record']['has_supervision'] and len(item['observation']['query_points']) == 64
    assert np.array_equal(item['labels']['Q_inst'], item['labels']['q_inst'])
    with pytest.raises(ValueError, match='exceeds label prefix'):
        InstanceDataset(manifest)[0]
    first = manifest.read_bytes()
    build_source_instances(root, release, tmp_path / 'cache', split='train', scenes=[9], frames=[0], train_queries=64)
    assert manifest.read_bytes() == first
    label_path = manifest.parent / ds.records[0]['label_cache']
    label_path.unlink()
    unlabelled = InstanceDataset(manifest, require_labels=False)[0]
    assert unlabelled['labels'] is None and len(unlabelled['observation']['query_points']) == 256


def test_compact_source_read_skips_unused_arrays_with_identical_retained_values(tmp_path, monkeypatch):
    root, release, _ = _assets(tmp_path)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    dataset = InstanceDataset(manifest, training_queries=8)
    expected = dataset.load_uncached(0)
    path = dataset.root / dataset.records[0]['observation_npz']
    # The public reader, builder and repacker still expose complete observations.
    assert _COMPACT_SOURCE_UNUSED_ARRAYS <= load_observation(path).keys()
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original = archive_type.__getitem__
    requested = []

    def read_required(self, key):
        requested.append(key)
        assert key not in _COMPACT_SOURCE_UNUSED_ARRAYS, 'unused array was materialized'
        return original(self, key)

    monkeypatch.setattr(archive_type, '__getitem__', read_required)
    actual = dataset.load_uncached(0, _compact_source=True)
    assert {'xyz_map', 'normal_map', 'query_points', 'query_normals', 'metadata_json'} <= set(requested)
    assert actual['record'] == expected['record']
    for group in ('observation', 'labels'):
        omitted = _COMPACT_SOURCE_UNUSED_ARRAYS if group == 'observation' else set()
        assert actual[group].keys() == expected[group].keys() - omitted
        for key, value in actual[group].items():
            reference = expected[group][key]
            if isinstance(value, np.ndarray):
                assert value.dtype == reference.dtype
                np.testing.assert_array_equal(value, reference)
            else:
                assert value == reference


def test_compact_source_hint_does_not_narrow_test_observations(tmp_path):
    root, release, _ = _assets(tmp_path, scene_id=100)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='test', scenes=[100], frames=[0], with_labels=False)
    dataset = InstanceDataset(manifest, require_labels=False)
    assert _COMPACT_SOURCE_UNUSED_ARRAYS <= dataset.load_uncached(0, _compact_source=True)['observation'].keys()
    full = dataset.load_uncached(0)['observation']
    for route in ('a', 'b'):
        actual = dataset.load_uncached(0, _compact_source=True, source_route=route)['observation']
        assert actual.keys() == full.keys()


@pytest.mark.parametrize('route', ['a', 'b'])
@pytest.mark.parametrize('storage', ['npz', 'mmap'])
def test_route_source_member_protocol_reads_only_consumers_and_canonical_checks(tmp_path, monkeypatch, route, storage):
    root, release, _ = _assets(tmp_path)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    dataset = InstanceDataset(manifest, training_queries=8)
    expected = dataset.load_uncached(0)
    path = dataset.root / dataset.records[0]['observation_npz']
    with np.load(path, allow_pickle=False) as archive:
        original_arrays = {key: archive[key] for key in archive.files}
        archive_type = type(archive)
    # Unknown, non-consumed arrays do not silently become new worker inputs.
    original_arrays['unconsumed_extension'] = np.zeros((13, 17), np.float64)
    np.savez_compressed(path, **original_arrays)
    mirror_instance = None
    if storage == 'mmap':
        mirror = tmp_path / 'mirror'
        mirror_instance = mirror / path.parent.name / path.stem
        mirror_instance.mkdir(parents=True)
        (mirror / 'cache.json').write_text(json.dumps({
            'format': 'instance-selection-npy-v1', 'source_root': str(dataset.root)}))
        stat = path.stat()
        (mirror_instance / 'members.json').write_text(json.dumps({
            'source': [str(path), stat.st_size, stat.st_mtime_ns], 'members': list(original_arrays)}))
        for key, value in original_arrays.items():
            np.save(mirror_instance / (key + '.npy'), value, allow_pickle=False)
        monkeypatch.setenv('SUCTION_INSTANCE_SELECTION_CACHE', str(mirror))
    original = archive_type.__getitem__
    original_load = np.load
    requested, decoded = [], {}

    def record(key, value):
        requested.append(key)
        assert key in SOURCE_TRAIN_ARRAYS[route] or key == 'metadata_json'
        decoded[key] = value

    def read_required(self, key):
        value = original(self, key)
        if str(self.zip.filename) == str(path):
            assert storage == 'npz', 'valid local mirror unexpectedly fell back to NPZ'
            record(key, value)
        return value

    def map_required(source, *args, **kwargs):
        value = original_load(source, *args, **kwargs)
        if mirror_instance is not None and Path(source).parent == mirror_instance:
            assert kwargs.get('mmap_mode') == 'c'
            record(Path(source).stem, value)
        return value

    monkeypatch.setattr(archive_type, '__getitem__', read_required)
    monkeypatch.setattr(np, 'load', map_required)
    actual = dataset.load_uncached(0, _compact_source=True, source_route=route)
    assert set(requested) == (set(original_arrays) & SOURCE_TRAIN_ARRAYS[route]) | {'metadata_json'}
    assert {'xyz_map', 'normal_map'} <= set(requested)
    assert actual['record'] == expected['record']
    for group in ('observation', 'labels'):
        for key, value in actual[group].items():
            if key == 'source_support_count':
                assert value == len(expected['observation']['support_points'])
                continue
            reference = expected[group][key]
            if isinstance(value, np.ndarray):
                assert value.dtype == reference.dtype
                np.testing.assert_array_equal(value, reference)
            else:
                assert value == reference
    for key in QUERY_FIELDS:
        assert np.shares_memory(actual['observation'][key], decoded[key])
        if storage == 'mmap':
            assert isinstance(actual['observation'][key], np.memmap)
    for alias, source in LABEL_ALIASES.items():
        assert actual['labels'][alias] is actual['labels'][source]
    if route == 'a':
        assert not {'xyz_map', 'normal_map'} & actual['observation'].keys()
        assert 'support_points' in actual['observation']
    else:
        assert not any(key.startswith('support_') for key in actual['observation'])
        assert {'xyz_map', 'normal_map', 'instance_mask', 'normal_valid'} <= actual['observation'].keys()


def test_route_b_source_gate_retains_flattened_support_count(tmp_path):
    root, release, _ = _assets(tmp_path)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    dataset = InstanceDataset(manifest, training_queries=8)
    path = dataset.root / dataset.records[0]['observation_npz']
    observation = load_observation(path)
    observation['support_points'] = observation['support_points'][:7].copy()
    save_observation(path, observation)
    full = dataset.load_uncached(0)['observation']
    actual = dataset.load_uncached(0, _compact_source=True, source_route='b')['observation']
    assert actual['source_support_count'] == len(full['support_points']) == 7
    assert int(np.count_nonzero(actual['mask'])) > 7


@pytest.mark.parametrize('corruption, expected_error', [
    ('query_point', 'raw source pixels'), ('normal_spec', 'version mismatch'),
    ('label_pose', 'pose differs'),
])
def test_compact_source_read_keeps_geometry_schema_and_label_checks(tmp_path, corruption, expected_error):
    root, release, _ = _assets(tmp_path)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    dataset = InstanceDataset(manifest, training_queries=8)
    field = 'label_cache' if corruption == 'label_pose' else 'observation_npz'
    path = dataset.root / dataset.records[0][field]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if corruption == 'normal_spec':
        metadata = json.loads(str(arrays['metadata_json']))
        metadata['normal_spec'] = 'unsupported'
        arrays['metadata_json'] = np.asarray(json.dumps(metadata))
    else:
        arrays['points' if corruption == 'label_pose' else 'query_points'][0, 0] += 1e-5
    np.savez_compressed(path, **arrays)
    for options in ({}, {'_compact_source': True},
                    {'_compact_source': True, 'source_route': 'a'},
                    {'_compact_source': True, 'source_route': 'b'}):
        with pytest.raises(ValueError, match=expected_error):
            dataset.load_uncached(0, **options)


@pytest.mark.parametrize('excluded', [(), ('depth_m',)])
def test_excluded_array_still_collides_with_metadata(tmp_path, excluded):
    path = tmp_path / 'ambiguous.npz'
    np.savez_compressed(path, depth_m=np.zeros((3, 3)),
                        metadata_json=np.asarray(json.dumps({'depth_m': 'ambiguous'})))
    with pytest.raises(ValueError, match='array/metadata key collision'):
        _read_npz(path, exclude_arrays=excluded)
    with pytest.raises(ValueError, match='array/metadata key collision'):
        _read_npz(path, needed_arrays=())


def test_test_insufficient_support_is_retained(tmp_path):
    root, release, _ = _assets(tmp_path, scene_id=100)
    label = np.zeros((65, 65), np.uint8); label[20:23, 20:23] = 1
    Image.fromarray(label).save(root / 'scenes/scene_0100/realsense/label/0000.png')
    manifest, summary = build_source_instances(root, release, tmp_path / 'cache',
        split='test', scenes=[100], frames=[0])
    item = InstanceDataset(manifest, split='test')[0]
    assert len(item['observation']['query_points']) == 0
    assert item['record']['status'] == 'insufficient_support'
    assert summary['visible_instance_count'] == 1 and not item['record']['has_supervision']


def test_dataset_compares_explicit_physics_and_ignores_old_integrity_metadata(tmp_path):
    root, release, _ = _assets(tmp_path)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    original = json.loads(manifest.read_text())
    dataset = InstanceDataset(manifest, training_queries=8)
    assert dataset.identity['teacher_identity']['physics'] == original['label_identity']['physics']
    assert dataset.identity['object_assets'] == {'0': {key: original['label_identity'][key]
        for key in ('dense_path', 'seal_candidates_path')}}
    # Obsolete integrity metadata is deliberately ignored, even when nonsensical.
    old = json.loads(json.dumps(original))
    old['manifest_version'] = 'suction20.instance_manifest.v3'
    old['observation_' + 'sha256'] = 'old-ignored-value'
    old['label_' + 'sha256'] = 'different-old-ignored-value'
    old['label_identity']['physics_' + 'sha256'] = 'not-a-digest'
    manifest.write_text(json.dumps(old) + '\n')
    assert InstanceDataset(manifest, training_queries=8)[0]['record']['has_supervision']
    changed = json.loads(json.dumps(original))
    changed['label_identity']['physics']['wrench_k'] *= 2
    manifest.write_text(json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='teacher contracts'):
        InstanceDataset(manifest, training_queries=8)[0]
    changed['sample_id'] += '_other'
    manifest.write_text(json.dumps(original) + '\n' + json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='mixed declared teacher'):
        InstanceDataset(manifest, training_queries=8)
    unknown = json.loads(json.dumps(original))
    unknown['sample_id'] += '_legacy'
    unknown['label_identity'].pop('physics')
    unknown['label_identity'].pop('teacher_spec')
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in (unknown, original, changed)))
    with pytest.raises(ValueError, match='mixed declared teacher'):
        InstanceDataset(manifest, training_queries=8)
    assert InstanceDataset(manifest, require_labels=False).identity['teacher_identity'] is None
    assert InstanceDataset(manifest, require_labels=False).identity['object_assets'] == {}


def test_archive_load_opens_each_npz_once_without_preflight_scan(tmp_path, monkeypatch):
    import builtins
    root, release, _ = _assets(tmp_path)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    dataset = InstanceDataset(manifest, training_queries=8, cache_size=0)
    opened = []
    original_open, original_path_open = builtins.open, Path.open
    def record_open(file, *args, **kwargs):
        if str(file).endswith('.npz'):
            opened.append(str(Path(file).resolve()))
        return original_open(file, *args, **kwargs)
    def record_path_open(path, *args, **kwargs):
        if str(path).endswith('.npz'):
            opened.append(str(path.resolve()))
        return original_path_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', record_open)
    monkeypatch.setattr(Path, 'open', record_path_open)
    for _ in range(2):
        dataset[0]
    row = dataset.records[0]
    expected = [str((manifest.parent/row[key]).resolve()) for key in ('observation_npz', 'label_cache')]
    assert opened == expected * 2


def test_legacy_label_cache_keeps_unknown_physics_without_rescoring(tmp_path, monkeypatch):
    root, release, _ = _assets(tmp_path)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    row = json.loads(manifest.read_text())
    path = manifest.parent/row['label_cache']
    with np.load(path, allow_pickle=False) as file:
        content = {key: file[key] for key in file.files if key != 'metadata_json'}
        content.update(json.loads(str(file['metadata_json'])))
    content['identity'].pop('physics')
    content['identity'].pop('teacher_spec')
    _save_npz(path, content)
    row['label_identity'] = content['identity']
    row['manifest_version'] = 'suction20.instance_manifest.v3'
    manifest.write_text(json.dumps(row)+'\n')
    assert InstanceDataset(manifest, training_queries=8).identity['teacher_identity']['physics'] is None
    monkeypatch.setattr(InstanceTeacher, 'evaluate', lambda *a, **k: pytest.fail('legacy cache must not require Teacher recomputation'))
    build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    rewritten = json.loads(manifest.read_text())
    assert rewritten['manifest_version'] == 'suction20.instance_manifest.v4'
    assert rewritten['label_identity']['physics'] is None


def test_attempt_pose_and_split_preserved(tmp_path):
    root, _, _ = _assets(tmp_path)
    base = root / 'scenes/scene_0000/realsense'
    observation = dict(format='rgbd', rgb_path=str(base / 'rgb/0000.png'),
        depth_path=str(base / 'depth/0000.png'), instance_mask_path=str(base / 'label/0000.png'),
        depth_scale_m=.001, intrinsics=_camera().tolist(), g_cam=[0., 0., -1.])
    point = [.001, -.002, .6]
    normal = [0., .6, -.8]
    rows = [dict(schema_version='suction20.target.v1', label_kind='attempt_success',
        sample_id='attempt1', group_id='group1', split='val', observation=observation,
        success_definition='pickup_and_transport', attempt=dict(point_cam_m=point,
        normal_out_cam=normal, instance_id=1, success=1, label_valid=True))]
    manifest = tmp_path / 'attempts.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows))
    assert len(InstanceAttemptDataset(manifest, split='train')) == 0
    dataset = InstanceAttemptDataset(manifest, split='val', cache_size=0)
    item = dataset[0]
    assert np.array_equal(item['attempts']['point_cam_m'][0], point)
    assert np.array_equal(item['attempts']['normal_out_cam'][0], normal)
    assert item['attempts']['pose_support_valid'][0]
    assert item['attempts']['pose_source'] == 'planned_pipeline_contact'
    assert not item['attempts']['actual_execution_verified']
    Image.fromarray(np.full((65, 65, 3), 128, np.uint8)).save(base / 'rgb/0000.png')
    replacement = dataset[0]
    assert not np.array_equal(replacement['observation']['rgb'], item['observation']['rgb'])
    assert replacement['record']['sample_id'] == item['record']['sample_id']
    assert dataset.identity['observation_ids'] == [item['record']['sample_id']]


def test_attempt_support_uses_shared_closed_boundary(tmp_path):
    root, _, _ = _assets(tmp_path)
    base = root/'scenes/scene_0000/realsense'
    observation = dict(format='rgbd', rgb_path=str(base/'rgb/0000.png'),
        depth_path=str(base/'depth/0000.png'), instance_mask_path=str(base/'label/0000.png'),
        depth_scale_m=.001, intrinsics=_camera().tolist(), g_cam=[0.,0.,-1.])
    rows = [dict(schema_version='suction20.target.v1', label_kind='attempt_success',
        sample_id=f'a{i}', group_id='g', split='train', observation=observation,
        success_definition='pickup', attempt=dict(point_cam_m=[0.,0.,.6+.005+offset],
            normal_out_cam=[0.,0.,-1.], instance_id=1, success=1, label_valid=True))
        for i, offset in enumerate((.5*BOUNDARY_ROUNDOFF_M, 2*BOUNDARY_ROUNDOFF_M))]
    manifest=tmp_path/'attempts.jsonl';manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    attempts=InstanceAttemptDataset(manifest)[0]['attempts']
    assert attempts['pose_support_valid'].tolist() == [True, False]


@pytest.mark.parametrize('conflict', ['split', 'group', 'rgb_path', 'capture_uid'])
def test_attempt_capture_owner_spans_instances_and_normalized_paths(tmp_path, conflict):
    observation = dict(format='rgbd', rgb_path='rgb.png', depth_path='depth.png',
        instance_mask_path='mask.png', depth_scale_m=.001,
        intrinsics=_camera().tolist(), g_cam=[0., 0., -1.])
    first = dict(schema_version='suction20.target.v1', label_kind='attempt_success',
        sample_id='a', group_id='g1', split='train', observation=observation,
        success_definition='pickup', attempt=dict(point_cam_m=[0., 0., .6],
            normal_out_cam=[0., 0., -1.], instance_id=1, success=1, label_valid=True))
    second = json.loads(json.dumps(first))
    second.update(sample_id='b', group_id='g2', split='val' if conflict != 'group' else 'train')
    second['attempt']['instance_id'] = 2
    for field in ('rgb_path', 'depth_path', 'instance_mask_path'):
        second['observation'][field] = './' + second['observation'][field]
    if conflict == 'rgb_path':
        second['observation']['depth_path'] = 'other_depth.png'
    if conflict == 'capture_uid':
        first['capture_uid'] = second['capture_uid'] = 'capture-17'
        for field in ('rgb_path', 'depth_path', 'instance_mask_path'):
            second['observation'][field] = 'other_' + Path(second['observation'][field]).name
    manifest = tmp_path / 'attempts.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in (first, second)))
    with pytest.raises(ValueError, match='shared target observation/capture crosses split or group'):
        InstanceAttemptDataset(manifest)


def test_attempt_shared_capture_allows_instances_in_same_group_split(tmp_path):
    observation = dict(format='rgbd', rgb_path='rgb.png', depth_path='depth.png',
        instance_mask_path='mask.png', depth_scale_m=.001,
        intrinsics=_camera().tolist(), g_cam=[0., 0., -1.])
    rows = [dict(schema_version='suction20.target.v1', label_kind='attempt_success',
        sample_id=f'a{i}', group_id='g', split='train', capture_uid='capture-17',
        observation=dict(observation), success_definition='pickup',
        attempt=dict(point_cam_m=[0., 0., .6], normal_out_cam=[0., 0., -1.],
            instance_id=i, success=1, label_valid=True)) for i in (1, 2)]
    for field in ('rgb_path', 'depth_path', 'instance_mask_path'):
        rows[1]['observation'][field] = './' + rows[1]['observation'][field]
    # A historical checksum difference must not become an indirect declaration check.
    rows[0]['observation']['legacy'] = {'source_sha256': 'old-one', 'version': 1}
    rows[1]['observation']['legacy'] = {'source_sha256': 'old-two', 'version': 1}
    manifest = tmp_path / 'attempts.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in rows))
    dataset = InstanceAttemptDataset(manifest)
    assert len(dataset) == 2
    assert dataset.observation_owners[('capture_uid', 'capture-17')] == ('train', 'g')
    assert dataset.observation_owners[('depth_path', str((tmp_path / 'depth.png').resolve()))] == ('train', 'g')
    assert len(dataset.identity['observation_provenance']) == 3
    assert dataset.records[0]['observation'] == dataset.records[1]['observation']
    assert 'sha256' not in json.dumps(dataset.records)


def test_teacher_and_new_archives_drop_nested_obsolete_metadata(tmp_path):
    root, release, _ = _assets(tmp_path)
    config_path = release / 'config_frozen.json'
    config = json.loads(config_path.read_text())
    config['source_sha256'] = 'not-verified'
    config['provenance'] = {'version': 'legacy', 'nested': [{'artifact_sha256': 'ignored', 'retained': 7}]}
    config_path.write_text(json.dumps(config))
    teacher = InstanceTeacher(root, release)
    assert teacher.config['provenance'] == {'version': 'legacy', 'nested': [{'retained': 7}]}
    assert 'sha256' not in json.dumps(teacher.config)
    assert 'sha256' not in json.dumps(teacher.identity)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    assert 'sha256' not in manifest.read_text()
    row = json.loads(manifest.read_text())
    label_path = manifest.parent / row['label_cache']
    with np.load(label_path, allow_pickle=False) as archive:
        original = {key: archive[key] for key in archive.files}
    label_metadata = json.loads(str(original['metadata_json']))
    assert 'sha256' not in json.dumps(label_metadata)
    # Simulate an actual old archive, without using the new sanitized writer.
    label_metadata['identity']['physics']['provenance']['old_sha256'] = 'changed-old-value'
    original['metadata_json'] = np.asarray(json.dumps(label_metadata))
    np.savez_compressed(label_path, **original)
    item = InstanceDataset(manifest, training_queries=8)[0]
    assert item['labels']['identity']['physics'] == row['label_identity']['physics']
    assert 'sha256' not in json.dumps(item['labels']['identity'])
    observation = _observation()
    observation['provenance'] = {'nested': [{'old_sha256': 'ignored', 'retained': 8}]}
    destination = tmp_path / 'clean_observation.npz'
    save_observation(destination, observation)
    with np.load(destination, allow_pickle=False) as archive:
        saved_metadata = json.loads(str(archive['metadata_json']))
    assert saved_metadata['provenance'] == {'nested': [{'retained': 8}]}
    assert 'sha256' not in json.dumps(saved_metadata)


def test_merge_scene_shards_relocates_paths_and_rejects_overlap_and_missing(tmp_path):
    first_root, release, _ = _assets(tmp_path/'one', scene_id=0)
    second_root, _, _ = _assets(tmp_path/'two', scene_id=1)
    one, _ = build_source_instances(first_root, release, tmp_path/'shards/one',
        split='train', scenes=[0], frames=[0], train_queries=8)
    two, _ = build_source_instances(second_root, release, tmp_path/'shards/two',
        split='train', scenes=[1], frames=[0], train_queries=8)
    with pytest.raises(ValueError, match='incomplete'):
        merge_source_manifests([one,two], tmp_path/'full', split='train')
    merged, summary = merge_source_manifests([one,two], tmp_path/'merged', split='train', allow_partial=True)
    assert summary['frame_count']==2 and not summary['complete_requested_split']
    dataset = InstanceDataset(merged, training_queries=8)
    assert len(dataset)==2 and all(dataset[i]['record']['has_supervision'] for i in range(2))
    assert all(row['observation_npz'].startswith('../shards/') for row in dataset.records)
    with pytest.raises(ValueError, match='overlapping'):
        merge_source_manifests([one,merged], tmp_path/'overlap', split='train', allow_partial=True)
    with pytest.raises(ValueError, match='crosses requested split'):
        merge_source_manifests([one], tmp_path/'cross', split='test', allow_partial=True)
    broken=json.loads(two.with_suffix('.summary.json').read_text())
    broken['frame_ids']=[0,1];broken['frame_count']=2
    two.with_suffix('.summary.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='missing frames'):
        merge_source_manifests([one,two], tmp_path/'bad', split='train', allow_partial=True)


def test_merge_keeps_explicit_empty_gt_frames(tmp_path):
    root, release, _ = _assets(tmp_path, scene_id=100)
    base=root/'scenes/scene_0100/realsense'
    Image.fromarray(np.zeros((65,65),np.uint8)).save(base/'label/0000.png')
    source,_=build_source_instances(root,release,tmp_path/'shard',split='test',scenes=[100],frames=[0])
    manifest,summary=merge_source_manifests([source],tmp_path/'merged',split='test',allow_partial=True)
    assert manifest.read_text()==''
    assert summary['empty_gt_frames']==[[100,0]] and summary['frame_count']==1


def test_explicit_selection_namespace_is_stable_readable_and_bounded(tmp_path):
    expected = tmp_path/'shards/realsense/train/scenes_0000-0009/frames_0000-0255'
    assert source_build_output(tmp_path, split='train', scenes=range(10)) == expected
    assert source_build_output(tmp_path, split='train', scenes=reversed(range(10))) == expected
    assert source_build_output(tmp_path, split='train') == tmp_path
    assert source_build_output(tmp_path, split='test', camera='kinect') == tmp_path
    assert source_build_output(tmp_path, split='train', frames=[0]) != expected
    scattered = source_build_output(tmp_path, split='all', scenes=range(0, 190, 2), frames=range(0, 256, 2))
    assert all(len(part.encode()) <= 180 for part in scattered.relative_to(tmp_path).parts)
    assert scattered != source_build_output(tmp_path, split='all', scenes=range(1, 190, 2), frames=range(0, 256, 2))
    with pytest.raises(ValueError, match='nonempty'):
        source_build_output(tmp_path, split='train', scenes=[])


def _build_command(root, release, output, scene, *, workers=1):
    script = Path(__file__).resolve().parents[1]/'scripts/build_instance_dataset.py'
    return [sys.executable, str(script), '--dataset-root', str(root), '--label-root', str(release),
            '--output', str(output), '--split', 'train', '--scenes', str(scene), '--frames', '0',
            '--train-queries', '8', '--workers', str(workers)]


@pytest.mark.parametrize('parallel', [False, True])
def test_manual_cli_shards_share_output_without_overwriting_and_merge(tmp_path, parallel):
    root, release, _ = _assets(tmp_path/'first', scene_id=0)
    other, _, _ = _assets(tmp_path/'second', scene_id=1)
    shutil.copytree(other/'scenes/scene_0001', root/'scenes/scene_0001')
    output = tmp_path/'shared-output'
    output.mkdir()
    # A previously published root manifest must not be touched by explicit
    # subsets; neither caller is authorized to replace that publication.
    original = output/'manifest_train.jsonl'
    original.write_text('previous full publication\n')
    commands = [_build_command(root, release, output, scene) for scene in (0, 1)]
    if parallel:
        children = [subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    for command in commands]
        results = []
        try:
            for child in children:
                stdout, stderr = child.communicate(timeout=30)
                assert child.returncode == 0, stderr
                results.append(json.loads(stdout))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill(); child.wait()
    else:
        results = []
        for command in commands:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            assert completed.returncode == 0, completed.stderr
            results.append(json.loads(completed.stdout))
    manifests = [Path(result['manifest']) for result in results]
    assert len(set(manifests)) == 2
    assert original.read_text() == 'previous full publication\n'
    for scene, manifest in enumerate(manifests):
        assert manifest == source_build_output(output, split='train', scenes=[scene], frames=[0])/'manifest_train.jsonl'
        row = json.loads(manifest.read_text())
        assert row['scene_id'] == scene and (manifest.parent/row['observation_npz']).is_file()
    merged, summary = merge_source_manifests(manifests, tmp_path/'merged', split='train', allow_partial=True)
    assert summary['frame_count'] == 2 and summary['visible_instance_count'] == 2
    data = InstanceDataset(merged, training_queries=8)
    assert len(data) == 2 and all(data[i]['record']['has_supervision'] for i in range(2))
    snapshot = merged.read_bytes(), merged.with_suffix('.summary.json').read_bytes()
    with pytest.raises(ValueError, match='reduced/different frame coverage'):
        merge_source_manifests(manifests[:1], merged.parent, split='train', allow_partial=True)
    assert (merged.read_bytes(), merged.with_suffix('.summary.json').read_bytes()) == snapshot


def test_same_shard_writer_refuses_concurrent_cli_and_releases_lock(tmp_path):
    root, release, _ = _assets(tmp_path)
    output = tmp_path/'cache'
    destination = source_build_output(output, split='train', scenes=[0], frames=[0])
    command = _build_command(root, release, output, 0)
    with source_output_lock(destination):
        child = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert child.returncode != 0 and 'already being written' in child.stderr
        assert not (destination/'manifest_train.jsonl').exists()
    child = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert Path(json.loads(child.stdout)['manifest']).is_file()
    lock = destination/'.instance-build.lock'
    first_inode = lock.stat().st_ino
    with source_output_lock(destination):
        pass
    assert lock.stat().st_ino == first_inode


def test_workers_cli_uses_scoped_output_and_preserves_parent_manifest(tmp_path):
    root, release, _ = _assets(tmp_path/'first', scene_id=0)
    other, _, _ = _assets(tmp_path/'second', scene_id=1)
    shutil.copytree(other/'scenes/scene_0001', root/'scenes/scene_0001')
    output = tmp_path/'cache'
    output.mkdir()
    (output/'manifest_train.jsonl').write_text('previous full publication\n')
    child = subprocess.run(_build_command(root, release, output, '0-1', workers=2),
                           capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    report = json.loads(child.stdout)
    manifest = Path(report['manifest'])
    assert manifest == source_build_output(output, split='train', scenes=[0, 1], frames=[0])/'manifest_train.jsonl'
    assert report['frame_count'] == 2 and len(InstanceDataset(manifest, training_queries=8)) == 2
    assert (output/'manifest_train.jsonl').read_text() == 'previous full publication\n'


def test_full_output_cannot_change_camera_before_reading_assets(tmp_path):
    output = tmp_path/'cache'
    output.mkdir()
    (output/'manifest_train.summary.json').write_text(json.dumps({'camera': 'realsense', 'split': 'train'}))
    with pytest.raises(ValueError, match='another camera/split'):
        build_source_instances(tmp_path/'unused', None, output, split='train', camera='kinect', with_labels=False)
