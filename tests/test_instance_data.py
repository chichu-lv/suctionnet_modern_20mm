"""Real-format synthetic assets exercise the actual instance data/teacher path."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat
from scipy.spatial import cKDTree

from suction20.eval_r20 import evaluate_poses, load_scene_eval_context, seal_at_pose
from suction20.full_data import DEFAULT_CONFIG, collision_mask_for_instance, transform_points, voxel_filter
from suction20.instance_data import (InstanceAttemptDataset, InstanceDataset, build_source_instances,
    load_observation, save_observation, source_scene_ids, source_split)
from suction20.instance_geometry import prepare_instance, stratified_queries_refill
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
    sha = save_observation(path, obs)
    restored = load_observation(path, expected_sha256=sha)
    for key in ('query_points', 'query_normals', 'support_points', 'support_rgb_valid', 'mask', 'rgb_valid'):
        assert np.array_equal(restored[key], obs[key])
    with pytest.raises(ValueError, match='hash'):
        load_observation(path, expected_sha256='0' * 64)


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


def test_dataset_binds_frozen_teacher_physics_and_cache_identity(tmp_path):
    root, release, _ = _assets(tmp_path)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    original = json.loads(manifest.read_text())
    dataset = InstanceDataset(manifest, training_queries=8)
    assert dataset.identity['teacher_identity']['physics_sha256'] == original['label_identity']['physics_sha256']
    assert 'dense_sha256' not in dataset.identity['teacher_identity']
    assert dataset.identity['object_assets'] == {'0': {key: original['label_identity'][key]
        for key in ('dense_sha256', 'seal_candidates_sha256')}}
    changed = json.loads(json.dumps(original))
    changed['label_identity']['physics_sha256'] = '0' * 64
    manifest.write_text(json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='cached label identity'):
        InstanceDataset(manifest, training_queries=8)[0]
    changed['sample_id'] += '_other'
    manifest.write_text(json.dumps(original) + '\n' + json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='mixed frozen teacher'):
        InstanceDataset(manifest, training_queries=8)
    changed['label_identity']['physics_sha256'] = original['label_identity']['physics_sha256']
    changed['label_identity']['dense_sha256'] = '0' * 64
    manifest.write_text(json.dumps(original) + '\n' + json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='same object ID has mixed'):
        InstanceDataset(manifest, training_queries=8)
    # Inference must not make teacher assets or even their identity mandatory.
    assert InstanceDataset(manifest, require_labels=False).identity['teacher_identity'] is None
    assert InstanceDataset(manifest, require_labels=False).identity['object_assets'] == {}


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
    with pytest.raises(ValueError, match='changed after dataset identity'):
        dataset[0]
    replacement = InstanceAttemptDataset(manifest, split='val')
    assert replacement.identity['observation_assets_sha256'] != dataset.identity['observation_assets_sha256']
