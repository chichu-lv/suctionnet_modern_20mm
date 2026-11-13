"""Small on-disk contract fixtures; these are not real-data acceptance evidence."""
import json
import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat
from suction20.datasets import PublicSuction20Dataset, public_frame_keys, resolve_source_format, split_scene_ids
from suction20.full_data import read_frame
from suction20.observation import AugmentConfig, apply_rigid, depth_geometry, poses_to_original_camera


@pytest.fixture
def source_fixture(tmp_path):
    raw, labels = tmp_path/'raw', tmp_path/'labels'
    base = raw/'scenes/scene_0000/realsense'
    for folder in ('depth', 'label', 'meta'):
        (base/folder).mkdir(parents=True)
    labels.mkdir()
    cfg = dict(camera='realsense', radius_m=.02, depth_min_m=.2, depth_max_m=1.2,
               seal_norm_min=0, seal_norm_max=.73, wrench_threshold_factor=1)
    (labels/'config_frozen.json').write_text(json.dumps(cfg))
    depth = np.full((7, 7), 500, np.uint16)
    classes = np.full((7, 7), 4, np.uint8)
    classes[:, 4:] = 8
    classes[0] = 0
    Image.fromarray(depth).save(base/'depth/0000.png')
    Image.fromarray(classes).save(base/'label/0000.png')
    K = np.array([[100., 0, 3], [0, 100, 3], [0, 0, 1]])
    pose = np.repeat(np.eye(4)[:3, :, None], 2, axis=2)
    pose[:, :3, 1] = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    savemat(base/'meta/0000.mat', dict(intrinsic_matrix=K, factor_depth=1000, cls_indexes=np.array([4, 8]), poses=pose))
    np.save(base/'cam0_wrt_table.npy', np.eye(4))
    np.save(base/'camera_poses.npy', np.repeat(np.eye(4)[None], 256, axis=0))
    xyz, normals, valid, _, pixels, _ = read_frame(raw, 0, 0, cfg)
    ids = np.where(classes.reshape(-1) == 4, 1, np.where(classes.reshape(-1) == 8, 2, 0))
    known = ids > 0
    known[24] = False
    candidates = labels/'seal_r20/poisson'
    candidates.mkdir(parents=True)
    for object_id in (3, 7):
        np.savez(candidates/f'{object_id:03d}_labels.npz', points=np.array([[0., 0, 0], [0, 0, .01]]), normals=np.array([[0., 0, -1], [0., 1., 0]]), scores=[.3, .5], collision=[False, False], indices=[48, 987])
    frame = dict(xyz=xyz, instance_id=ids, class_id=classes.reshape(-1), source_indices=pixels,
                 candidate_row=np.where(known, 1, -1), candidate_id=np.where(known, 987, -1),
                 gt_valid=known, gt_seal_norm=np.zeros(49), gt_wrench_eval=np.full(49, .75),
                 gt_A=np.zeros(49), gt_B=np.zeros(49), collision_flag=np.ones(49, bool),
                 gt_normal=np.tile([1., 0, 0], (49, 1)), normal_valid=np.zeros(49, bool),
                 match_distance=np.where(known, .001, np.inf), config_version='legacy')
    path = labels/'scenes/scene_0000/realsense/graspness_r20_v2/0000.npz'
    path.parent.mkdir(parents=True)
    np.savez(path, **frame)
    return raw, labels, path, frame, normals, valid


def dataset(fixture, **kwargs):
    return PublicSuction20Dataset(fixture[0], fixture[1], [(0, 0)], **kwargs)


def test_scene_records_reuse_loaded_calibration(source_fixture, monkeypatch):
    original_load = np.load
    names = []

    def counted(path, *args, **kwargs):
        names.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', counted)
    ds = dataset(source_fixture, scene_cache_size=90, object_cache_size=88)
    ds.get_sample(0, 7)
    ds.get_sample(0, 8)
    assert sum(name.endswith('cam0_wrt_table.npy') for name in names) == 1
    assert sum(name.endswith('camera_poses.npy') for name in names) == 1


def test_legacy_axis_identity_zero_unknown_and_frozen_config(source_fixture):
    ds = dataset(source_fixture)
    sample = ds[0]
    inp, target = sample['inputs'], sample['targets']
    assert ds.data_cfg['wrench_threshold_factor'] == 1
    assert ds.data_cfg['seal_norm_max'] == .73
    assert sample['meta']['source_format'] == 'legacy_dc4ad6b'
    np.testing.assert_allclose(target['normal_label'][8], [0, 1, 0])
    np.testing.assert_allclose(target['normal_label'][12], [-1, 0, 0])
    assert target['axis_valid'][8]  # known zero, collision, old normal_valid=False
    assert not target['quality_valid'][24] and not target['axis_valid'][24]
    assert inp['candidate_eligible'][24]  # unknown remains an inference proposal
    np.testing.assert_array_equal(target['gt_B'], source_fixture[3]['gt_B'])
    np.testing.assert_array_equal(inp['normal_obs_valid'], source_fixture[5])
    np.testing.assert_allclose(inp['normal_obs'][source_fixture[5]], source_fixture[4][source_fixture[5]])
    assert not np.any(inp['normal_obs'][~source_fixture[5]])
    assert not set(inp) & {'gt_B', 'normal_label', 'candidate_row', 'gt_valid'}


def test_schema_precedence_new_axes_and_legacy_index_validation(source_fixture):
    _, _, path, frame, normals, valid = source_fixture
    frame.update(schema_version='r20-v2.1', normals_raw=normals, normals_raw_valid=valid)
    np.savez(path, **frame)
    sample = dataset(source_fixture)[0]
    np.testing.assert_allclose(sample['targets']['normal_label'][8], [1, 0, 0])
    assert sample['meta']['source_format'] == 'r20_v2_1'
    del frame['schema_version']
    frame['candidate_id'][8] = 1
    np.savez(path, **frame)
    with pytest.raises(ValueError, match='candidate_id'):
        dataset(source_fixture)[0]
    assert resolve_source_format({'adapter_schema': 'r20-v2.1'}) == 'legacy_dc4ad6b'
    with pytest.raises(ValueError, match='schema'):
        resolve_source_format({'schema_version': 'unrecognized'})


def test_input_mask_separation_and_source_pixel_alignment(source_fixture):
    _, _, path, frame, _, _ = source_fixture
    permutation = np.arange(48, -1, -1)
    for key, value in list(frame.items()):
        if isinstance(value, np.ndarray) and value.ndim and len(value) == 49:
            frame[key] = value[permutation]
    np.savez(path, **frame)
    mask = np.zeros((7, 7), np.int64)
    mask[:, :4] = 27
    sample = dataset(source_fixture, instance_source=lambda key: mask)[0]
    np.testing.assert_array_equal(sample['inputs']['instance_id_input'], mask.ravel()[permutation])
    np.testing.assert_array_equal(sample['targets']['instance_id_gt'], frame['instance_id'])
    np.testing.assert_array_equal(sample['inputs']['normal_obs_valid'], source_fixture[5][permutation])
    assert set(sample['inputs']['instance_id_input']) == {0, 27}


def test_depth_dropout_recomputes_normals_and_selects_every_trace(source_fixture):
    cfg = AugmentConfig(yaw_max_deg=0, translation_max_m=0, depth_noise_std_m=.0005, depth_noise_clip_m=.0015, dropout_max=.7)
    ds = dataset(source_fixture, augment=cfg)
    sample, again = ds.get_sample(0, 7), ds.get_sample(0, 7)
    pixels = sample['inputs']['source_indices']
    assert 0 < len(pixels) < 49
    np.testing.assert_array_equal(sample['inputs']['xyz_cam_m'], again['inputs']['xyz_cam_m'])
    for name in ('candidate_row', 'candidate_id', 'normal_valid'):
        np.testing.assert_array_equal(sample['meta'][name], source_fixture[3][name][pixels])
    np.testing.assert_array_equal(sample['targets']['instance_id_gt'], source_fixture[3]['instance_id'][pixels])
    kept = np.zeros(49, bool)
    kept[pixels] = True
    for row, pixel in enumerate(pixels):
        y, x = divmod(int(pixel), 7)
        if x in (0, 6) or y in (0, 6) or not all(kept[j] for j in (pixel-1, pixel+1, pixel-7, pixel+7)):
            assert not sample['inputs']['normal_obs_valid'][row]
    assert np.any(abs(sample['inputs']['xyz_cam_m'][:, 2]-.5) > 1e-6)


def test_rigid_rotation_and_inverse_preserve_directed_physics(source_fixture):
    sample = dataset(source_fixture)[0]
    p, n = sample['inputs']['xyz_cam_m'].copy(), sample['targets']['normal_label'].copy()
    dot = n @ sample['inputs']['g_cam']
    T = np.array([[0., 0, 1, .01], [0, 1, 0, -.02], [-1, 0, 0, .03], [0, 0, 0, 1]])
    apply_rigid(sample, T)
    np.testing.assert_allclose(sample['targets']['normal_label'] @ sample['inputs']['g_cam'], dot)
    p_back, n_back = poses_to_original_camera(sample['inputs']['xyz_cam_m'], sample['targets']['normal_label'], sample['meta'])
    np.testing.assert_allclose(p_back, p, atol=1e-7)
    np.testing.assert_allclose(n_back, n, atol=1e-7)
    np.testing.assert_array_equal(sample['targets']['gt_B'], source_fixture[3]['gt_B'])


def test_organized_depth_jump_and_normal_direction():
    z = np.full((7, 7), .5)
    z[:, 4:] += .02
    p, n, valid, keep = depth_geometry(z, np.diag([100., 100., 1.]))
    assert not valid[:, 3:5].any()
    assert valid[2, 2] and n[2, 2, 2] == -1
    assert np.all(np.sum(n*p, axis=-1)[valid] < 0)


def test_scene_split_view_sampling_and_epoch_determinism():
    train, val = set(split_scene_ids('train')), set(split_scene_ids('val'))
    assert len(train) == 90 and len(val) == 10 and not train & val
    keys = public_frame_keys('train', seed=8, epoch=2)
    assert len(keys) == len(set(keys)) == 5760
    assert keys == public_frame_keys('train', seed=8, epoch=2)
    assert keys != public_frame_keys('train', seed=8, epoch=3)
    assert all(sum(k[0] == scene for k in keys) == 64 for scene in train)
    assert {key[0] for key in public_frame_keys('val')} == val
    assert {key[2] for key in public_frame_keys('val')} == set(range(0, 256, 8))


def test_observation_only_inference_requires_no_label_root(source_fixture):
    from suction20.datasets import PublicObservationDataset
    # The supplied segmentation has arbitrary input IDs unrelated to GT IDs.
    mask = np.full((7, 7), 57, np.int64)
    ds = PublicObservationDataset(source_fixture[0], [(0, 0)],
                                  data_cfg=dict(depth_min_m=.2, depth_max_m=1.2),
                                  instance_source=lambda key: mask)
    sample = ds[0]
    assert sample['targets'] == {}
    assert np.all(sample['inputs']['instance_id_input'] == 57)
    np.testing.assert_array_equal(sample['inputs']['xyz_cam_m'], source_fixture[3]['xyz'])
    np.testing.assert_array_equal(sample['inputs']['normal_obs_valid'], source_fixture[5])


def test_new_schema_without_normal_cache_derives_from_current_depth(source_fixture):
    _, _, path, frame, normals, valid = source_fixture
    frame['schema_version'] = 'r20-v2.1'
    np.savez(path, **frame)
    sample = dataset(source_fixture)[0]
    np.testing.assert_array_equal(sample['inputs']['normal_obs_valid'], valid)
    np.testing.assert_allclose(sample['inputs']['normal_obs'][valid], normals[valid])
    np.testing.assert_allclose(sample['targets']['normal_label'][8], [1, 0, 0])


def test_test_split_api_uses_all_views_without_reading_official_test():
    keys = public_frame_keys('test_seen')
    assert len(keys) == 30*256
    assert {key[2] for key in keys} == set(range(256))


def test_selective_payload_matches_full_read_augmented_sample(source_fixture, monkeypatch):
    from suction20 import datasets as module
    original = module.read_view_payload

    def old_payload(path, source_format, compatibility, corrupt_depth):
        with np.load(path, allow_pickle=False) as archive:
            frame = {key: archive[key] for key in archive.files}
        effective = resolve_source_format(frame, source_format, compatibility)
        derive = corrupt_depth or not {'normals_raw', 'normals_raw_valid'} <= set(frame)
        return frame, effective, derive

    augment = AugmentConfig(depth_noise_std_m=.0005, dropout_max=.1)
    monkeypatch.setattr(module, 'read_view_payload', old_payload)
    expected = dataset(source_fixture, augment=augment).get_sample(0, 17)
    monkeypatch.setattr(module, 'read_view_payload', original)
    actual_dataset = dataset(source_fixture, augment=augment)
    actual = actual_dataset.get_sample(0, 17)
    for section in ('inputs', 'targets', 'meta'):
        assert actual[section].keys() == expected[section].keys()
        for key, value in actual[section].items():
            np.testing.assert_array_equal(value, expected[section][key])
    frame, effective, derive = original(source_fixture[2], 'legacy_dc4ad6b', 'auto', True)
    assert effective == 'legacy_dc4ad6b' and derive
    assert 'gt_normal' not in frame and 'config_version' not in frame
    assert {'candidate_id', 'candidate_row', 'normal_valid', 'match_distance'} <= frame.keys()
    assert all(set(value) == {'normals', 'indices'} for value in actual_dataset._objects.values())
    assert all(value['normals'].dtype == np.float64 for value in actual_dataset._objects.values())
    assert actual['production_timing']['prepare_total_s'] >= 0
    assert actual['cache_info']['object_array_bytes'] > 0
