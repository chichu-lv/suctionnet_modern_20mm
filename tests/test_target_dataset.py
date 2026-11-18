"""Target API fixtures exercise supervision/pose identity without target training."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip('torch')
from suction20.full_data import DEFAULT_CONFIG
from suction20.observation import AugmentConfig, poses_to_original_camera
from suction20.target_dataset import TargetSuction20Dataset, TargetGroupSchedule, collate_target
from suction20.training import configuration_identity


@pytest.fixture
def target_files(tmp_path):
    cfg = dict(DEFAULT_CONFIG, seal_norm_min=.2, seal_norm_max=.6, wrench_threshold_factor=1)
    (tmp_path/'source.json').write_text(json.dumps(cfg))
    x, y = np.meshgrid(np.arange(6)*.004, np.arange(6)*.004)
    xyz = np.column_stack((x.ravel(), y.ravel(), np.full(36, .5))).astype(np.float32)
    ids = np.full(36, 5, np.int64)
    ids[0] = 0
    arrays = dict(xyz=xyz, instance_id=ids, source_indices=np.arange(100, 136),
                  normals_raw=np.tile([0., 0., -1.], (36, 1)), normals_raw_valid=np.ones(36, bool))
    np.savez(tmp_path/'cloud.npz', **arrays)
    obs = dict(format='pointcloud_npz', path='cloud.npz', g_cam=[0., 0., -1.])
    return tmp_path, cfg, arrays, obs


def manifest(root, rows, name='manifest.jsonl'):
    path = root/name
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def row(obs, sample_id='a', group='g1', split='train', mode='dense_proxy', **kwargs):
    return dict(schema_version='suction20.target.v1', sample_id=sample_id,
                group_id=group, split=split, observation=copy.deepcopy(obs), label_kind=mode, **kwargs)


def dense_row(obs, **kwargs):
    return row(obs, labels=dict(path='labels.npz', source_label_config_path='source.json',
                               normal_target_semantics='original_candidate_axis'), **kwargs)


def attempt_row(obs, arrays, *, sample_id='a', group='g1', split='train', index=8, source=True):
    attempt = dict(point_cam_m=arrays['xyz'][index].tolist(), normal_out_cam=[0., 0., 2.],
                   instance_id=5, success=1, label_valid=True)
    if source:
        attempt['source_index'] = int(arrays['source_indices'][index])
    return row(obs, sample_id, group, split, 'attempt_success', attempt=attempt)


def test_sparse_independent_masks_zero_quality_and_directed_axes(target_files):
    root, cfg, arrays, obs = target_files
    np.savez(root/'labels.npz', source_indices=[120, 108, 114, 100],
             gt_valid=[True, False, True, True], gt_B=[0., np.nan, .6, 1.], instance_id_gt=[3, 3, 3, 0],
             gt_seal_raw=[.2, .5, np.inf, .4], seal_valid=[True]*4,
             gt_wrench_eval=[.4, np.nan, .2, .5], wrench_valid=[True, False, True, True],
             collision_flag=[True, False, False, False], collision_valid=[True]*4,
             gt_normal=[[0, 1., 0], [0, 0, 2.], [0, 0, 0], [1., 0, 0]], axis_target_valid=[True]*4)
    ds = TargetSuction20Dataset(manifest(root, [dense_row(obs)]), mode='dense_proxy', split='train', source_label_config=cfg)
    sample = ds[0]
    t, x = sample['targets'], sample['inputs']
    assert t['quality_valid'].sum() == 2 and t['quality_valid'][20] and t['gt_B'][20] == 0
    assert t['gt_B'][14] == pytest.approx(.6) and not t['quality_valid'][8]
    assert t['seal_valid'][8] and t['seal_target'][8] == pytest.approx(.75)
    assert not t['seal_valid'][14]  # Inf raw Seal must not clip into a valid one.
    assert t['collision_valid'][8] and t['collision_free_target'][8] == 1
    assert t['axis_valid'][8] and t['axis_valid'][20] and not t['axis_valid'][14]
    np.testing.assert_array_equal(t['normal_label'][8], [0, 0, 1])  # never camera-flip GT axes
    np.testing.assert_array_equal(x['instance_id_input'], arrays['instance_id'])
    assert t['instance_id_gt'][8] == 3 and x['instance_id_input'][8] == 5
    assert not t['quality_valid'][0] and not t['seal_valid'][0]
    assert not t['quality_valid'][9]  # absent sparse label
    assert 'gt_A' not in t and len(sample['attempts']['success']) == 0
    batch = collate_target([sample])
    assert batch['attempts']['point_cam_m'].shape == (0, 3)
    assert batch['attempts']['feature_point_index'].shape == (0,)
    assert ds.source_label_definition == configuration_identity(cfg)


def test_missing_components_and_axis_masks_are_false(target_files):
    root, cfg, arrays, obs = target_files
    np.savez(root/'labels.npz', gt_B=np.zeros(36), gt_valid=np.ones(36, bool), instance_id_gt=arrays['instance_id'])
    ds = TargetSuction20Dataset(manifest(root, [dense_row(obs)]), mode='dense_proxy', split='train', source_label_config=cfg)
    t = ds[0]['targets']
    assert t['quality_valid'].sum() == 35
    for key in ('seal_valid', 'wrench_valid', 'collision_valid', 'axis_valid'):
        assert not t[key].any()
    assert np.all(t['gt_B'] == 0)


def test_derive_B_only_from_actual_components_and_source_frozen_norm(target_files):
    root, cfg, arrays, obs = target_files
    np.savez(root/'labels.npz', gt_valid=np.ones(36, bool), instance_id_gt=arrays['instance_id'],
             gt_seal_raw=np.full(36, .4), gt_wrench_eval=np.full(36, .3), collision_flag=np.zeros(36, bool))
    path = manifest(root, [dense_row(obs)])
    ds = TargetSuction20Dataset(path, mode='dense_proxy', split='train', source_label_config=cfg)
    np.testing.assert_allclose(ds[0]['targets']['gt_B'][1:], .15)
    np.savez(root/'labels.npz', gt_valid=np.ones(36, bool), instance_id_gt=arrays['instance_id'], gt_seal_raw=np.full(36, .4))
    with pytest.raises(ValueError, match='all actual'):
        ds[0]
    changed = dict(cfg, seal_norm_max=1.)
    (root/'source.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='definition differs'):
        TargetSuction20Dataset(path, mode='dense_proxy', split='train', source_label_config=cfg)


def test_attempt_observation_grouping_packed_anchor_and_actual_axes(target_files):
    root, cfg, arrays, obs = target_files
    np.savez(root/'cloud2.npz', **arrays)
    obs2 = dict(obs, path='cloud2.npz')
    rows = [attempt_row(obs, arrays), attempt_row(obs, arrays, sample_id='b', index=14, source=False),
            attempt_row(obs2, arrays, sample_id='c', group='g2', index=20)]
    rows[1]['attempt']['success'] = 0
    rows[1]['attempt']['point_cam_m'][0] += .001  # exact executed point, not observed anchor
    ds = TargetSuction20Dataset(manifest(root, rows), mode='attempt_success', split='train',
                                source_label_config=cfg, success_definition='pick_and_transfer')
    assert len(ds) == 2 and ds.split_groups == {'train': ['g1', 'g2'], 'val': [], 'test': []}
    samples = [ds[0], ds[1]]
    assert samples[0]['meta']['sample_ids'] == ['a', 'b']
    b = collate_target(samples)
    a = b['attempts']
    assert b['targets'] == {}
    assert a['batch_index'].tolist() == [0, 0, 1]
    assert a['feature_point_index'].tolist() == [8, -1, 36+20]
    assert a['success'].tolist() == [1., 0., 1.]
    np.testing.assert_allclose(a['point_cam_m'][1].numpy(), rows[1]['attempt']['point_cam_m'])
    np.testing.assert_array_equal(a['normal_out_cam'].numpy(), [[0, 0, 1]]*3)
    # A supplied pixel from the wrong input instance must fall back, not leak.
    samples[0]['attempts']['instance_id'][0] = 9
    assert collate_target(samples)['attempts']['feature_point_index'][0] == -1


def test_rigid_attempt_transform_inverse_and_cache_integrity(target_files):
    root, cfg, arrays, obs = target_files
    path = manifest(root, [attempt_row(obs, arrays)])
    aug = AugmentConfig(depth_noise_std_m=0, dropout_max=0)
    ds = TargetSuction20Dataset(path, mode='attempt_success', split='train', source_label_config=cfg,
                                success_definition='pick', augment=aug)
    a, again = ds.get_sample(0, 17), ds.get_sample(0, 17)
    np.testing.assert_array_equal(a['inputs']['xyz_cam_m'], again['inputs']['xyz_cam_m'])
    point, normal = poses_to_original_camera(a['attempts']['point_cam_m'], a['attempts']['normal_out_cam'], a['meta'])
    np.testing.assert_allclose(point[0], arrays['xyz'][8], atol=1e-7)
    np.testing.assert_allclose(normal[0], [0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(a['attempts']['normal_out_cam'] @ a['inputs']['g_cam'], [-1], atol=1e-6)
    original, _ = ds._read_observation(ds.items[0])
    np.testing.assert_array_equal(original['xyz_cam_m'], arrays['xyz'])
    with pytest.raises(ValueError, match='preserve depth'):
        TargetSuction20Dataset(path, mode='attempt_success', split='train', source_label_config=cfg,
                               success_definition='pick', augment=AugmentConfig())
    with pytest.raises(ValueError, match='cannot use random'):
        TargetSuction20Dataset(path, mode='attempt_success', split='val', source_label_config=cfg,
                               success_definition='pick', augment=aug)


@pytest.mark.parametrize('case', ['duplicate_id', 'group_split', 'observation_split', 'success_definition'])
def test_manifest_rejects_leakage_before_split_filter(target_files, case):
    root, cfg, arrays, obs = target_files
    rows = [attempt_row(obs, arrays)]
    other = attempt_row(obs, arrays, sample_id='b')
    if case == 'duplicate_id':
        other['sample_id'] = 'a'
    elif case == 'group_split':
        other['split'] = 'val'
    elif case == 'observation_split':
        other['split'], other['group_id'] = 'val', 'g2'
    else:
        other['success_definition'] = 'different outcome'
    rows.append(other)
    with pytest.raises(ValueError):
        TargetSuction20Dataset(manifest(root, rows), mode='attempt_success', split='train',
                               source_label_config=cfg, success_definition='pick')


def test_rgbd_observation_and_sparse_pixel_label_alignment(target_files):
    root, cfg, _, _ = target_files
    depth = np.full((7, 7), 500, np.uint16)
    depth[0, 0] = 0
    Image.fromarray(depth).save(root/'depth.png')
    Image.fromarray(np.full((7, 7), 7, np.uint8)).save(root/'mask.png')
    obs = dict(format='rgbd', depth_path='depth.png', instance_mask_path='mask.png', depth_scale_m=.001,
               intrinsics=[[100, 0, 3], [0, 100, 3], [0, 0, 1]], g_cam=[0, 0, -1])
    np.savez(root/'labels.npz', source_indices=[24, 8], gt_valid=[True, True], gt_B=[0, .7], instance_id_gt=[5, 5])
    ds = TargetSuction20Dataset(manifest(root, [dense_row(obs)]), mode='dense_proxy', split='train', source_label_config=cfg)
    sample = ds[0]
    assert len(sample['inputs']['xyz_cam_m']) == 48
    assert sample['inputs']['source_indices'][0] == 1
    assert sample['targets']['gt_B'][7] == pytest.approx(.7)
    assert sample['targets']['quality_valid'][23] and sample['targets']['gt_B'][23] == 0
    np.testing.assert_array_equal(sample['inputs']['normal_obs'][23], [0, 0, -1])
    assert sample['meta']['image_shape'] == (7, 7)


def test_observation_only_pca_not_label_axis_and_no_fake_pixel_mapping(target_files):
    root, cfg, arrays, obs = target_files
    stripped = {key: value for key, value in arrays.items() if key in ('xyz', 'instance_id')}
    np.savez(root/'cloud.npz', **stripped)
    record = attempt_row(obs, arrays)
    record['attempt']['source_index'] = 8  # coincides with a row, not a real source pixel
    ds = TargetSuction20Dataset(manifest(root, [record]), mode='attempt_success', split='train',
                                source_label_config=cfg, success_definition='pick')
    sample = ds[0]
    assert sample['meta']['source_index_kind'] == 'row'
    assert sample['inputs']['normal_obs_valid'].all()
    np.testing.assert_array_equal(sample['inputs']['normal_obs'], [[0, 0, -1]]*36)
    assert sample['attempts']['normal_out_cam'][0, 2] == 1  # opposite real execution axis
    assert collate_target([sample])['attempts']['feature_point_index'][0] == -1


def test_uniform_groups_and_exact_resume_without_global_rng(target_files):
    root, cfg, arrays, obs = target_files
    rows = []
    for i in range(4):
        name = f'cloud{i}.npz'
        np.savez(root/name, **arrays)
        rows.append(attempt_row(dict(obs, path=name), arrays, sample_id=str(i), group='large' if i < 3 else 'small'))
    ds = TargetSuction20Dataset(manifest(root, rows), mode='attempt_success', split='train', source_label_config=cfg, success_definition='pick')
    schedule = TargetGroupSchedule(ds, seed=27)
    draws = schedule.next_frames(4000)
    small_fraction = sum(index == 3 for index, _ in draws)/len(draws)
    assert .46 < small_fraction < .54  # uniform records would yield about .25
    state = schedule.state_dict()
    expected = schedule.next_frames(17)
    np.random.seed(1)
    resumed = TargetGroupSchedule(ds, seed=27)
    resumed.load_state_dict(state)
    assert resumed.next_frames(17) == expected
    assert resumed.epoch == schedule.epoch and resumed.position == schedule.position
    with pytest.raises(ValueError, match='differs'):
        TargetGroupSchedule(ds, seed=28).load_state_dict(state)


def test_gt_instance_mapping_must_be_explicit(target_files):
    root, cfg, arrays, obs = target_files
    np.savez(root/'labels.npz', gt_valid=np.ones(36, bool), gt_B=np.zeros(36))
    record = dense_row(obs)
    ds = TargetSuction20Dataset(manifest(root, [record]), mode='dense_proxy', split='train', source_label_config=cfg)
    with pytest.raises(ValueError, match='mapping'):
        ds[0]
    record['labels']['instance_mapping'] = 'input_instance_ids'
    ds = TargetSuction20Dataset(manifest(root, [record]), mode='dense_proxy', split='train', source_label_config=cfg)
    sample = ds[0]
    np.testing.assert_array_equal(sample['targets']['instance_id_gt'], arrays['instance_id'])
    assert sample['meta']['instance_gt_source'] == 'explicit_shared_input_mapping'


def test_unlabeled_observation_helper_has_no_supervision_or_label_dependencies(target_files):
    from suction20.target_dataset import load_target_observation
    root, _, arrays, obs = target_files
    sample = load_target_observation(obs, root)
    assert sample['targets'] == {} and not len(sample['attempts']['success'])
    assert sample['meta']['label_kind'] == 'unlabeled'
    np.testing.assert_array_equal(sample['inputs']['xyz_cam_m'], arrays['xyz'])
    packed = collate_target([sample])
    assert packed['targets'] == {} and packed['attempts']['point_cam_m'].shape == (0, 3)
    Image.fromarray(np.full((5, 5), 500, np.uint16)).save(root/'unlabeled_depth.png')
    Image.fromarray(np.ones((5, 5), np.uint8)).save(root/'unlabeled_mask.png')
    rgbd = dict(format='rgbd', depth_path='unlabeled_depth.png', instance_mask_path='unlabeled_mask.png',
                depth_scale_m=.001, intrinsics=[[100, 0, 2], [0, 100, 2], [0, 0, 1]], g_cam=[0, 0, -1])
    loaded = load_target_observation(rgbd, root)
    assert len(loaded['inputs']['xyz_cam_m']) == 25
    assert loaded['inputs']['normal_obs_valid'].sum() == 9
    assert loaded['targets'] == {}


def test_historical_opaque_sampler_requires_fresh_finetuning(target_files):
    root, cfg, arrays, obs = target_files
    ds = TargetSuction20Dataset(manifest(root, [attempt_row(obs, arrays)]), mode='attempt_success',
                                split='train', source_label_config=cfg, success_definition='pick')
    sampler = TargetGroupSchedule(ds)
    historical = sampler.state_dict()
    historical['identity'].pop('manifest_definition')
    historical['identity']['manifest_digest'] = 'retired-opaque-id'
    with pytest.raises(ValueError, match='opaque observation IDs.*fresh finetuning'):
        sampler.load_state_dict(historical)
