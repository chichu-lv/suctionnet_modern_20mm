import copy
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from suction20.datasets import collate_public, derive_masks


def sample():
    inputs = dict(xyz_cam_m=np.array([[-.001, .002, .50], [-.0005, .002, .50], [.005, .002, .50]], np.float32),
                  normal_obs=np.tile(np.array([0, 0, -1], np.float32), (3, 1)), normal_obs_valid=np.array([True, True, False]),
                  instance_id_input=np.array([1, 1, 0], np.int64), source_indices=np.array([42, 17, 91], np.int64), g_cam=np.array([0, 0, -1], np.float32))
    targets = dict(gt_B=np.array([0., .8, .9], np.float32), gt_valid=np.array([True, True, False]),
                   normal_label=inputs['normal_obs'].copy(), instance_id_gt=np.array([3, 3, 0], np.int64))
    derive_masks(inputs, targets)
    return dict(inputs=inputs, targets=targets, meta={'sample_uid': (0, 'realsense', 0)})


def test_inverse_map_coordinate_order_instance_batch_and_target_identity():
    a, b = sample(), sample()
    b['targets']['gt_B'][0] = .4
    batch = collate_public([a, b])
    x, y = batch['inputs'], batch['targets']
    assert x['voxel_coords'].dtype == torch.int32 and x['inverse_map'].dtype == torch.int64
    assert x['voxel_features'].shape == (4, 8)
    assert x['voxel_coords'].is_contiguous()
    assert x['voxel_coords'].stride() == (4, 1)
    assert x['voxel_features'].is_contiguous()
    assert x['point_offsets'].tolist() == [0, 3, 6]
    assert x['batch_index'].tolist() == [0, 0, 0, 1, 1, 1]
    assert x['inverse_map'].tolist() == [0, 0, 1, 2, 2, 3]
    assert x['voxel_coords'].tolist() == [[0, 0, 0, 0], [0, 0, 0, 3], [1, 0, 0, 0], [1, 0, 0, 3]]
    assert y['instance_id_gt'].tolist() == [3, 3, 0, 3, 3, 0]
    assert x['instance_id_input'].tolist() == [1, 1, 0, 1, 1, 0]
    np.testing.assert_allclose(y['gt_B'].numpy(), [0, .8, .9, .4, .8, .9])
    assert torch.equal(x['voxel_features'][x['inverse_map']][0], x['voxel_features'][x['inverse_map']][1])
    assert x['source_indices'].tolist() == [42, 17, 91, 42, 17, 91]
    restored = x['voxel_coords'][x['inverse_map']][:, [3, 2, 1]] + x['voxel_origins'][x['batch_index']]
    np.testing.assert_array_equal(restored.numpy(), np.floor(x['xyz_cam_m'].numpy()/.002).astype(np.int64))


def test_targets_and_trace_cannot_leak_into_features():
    a = sample()
    b = copy.deepcopy(a)
    b['targets']['gt_B'][:] = [1, .2, -10]
    b['targets']['normal_label'][:] *= -1
    b['targets']['gt_valid'][:] = False
    b['inputs']['source_indices'][:] += 3000
    x, z = collate_public([a])['inputs'], collate_public([b])['inputs']
    for key in ('voxel_features', 'voxel_coords', 'candidate_eligible'):
        assert torch.equal(x[key], z[key])
    assert not set(x) & set(b['targets'])


def test_empty_frame_keeps_batch_offsets_and_gravity():
    a, b = sample(), sample()
    for group in ('inputs', 'targets'):
        for key in list(b[group]):
            if key != 'g_cam':
                b[group][key] = b[group][key][:0]
    x = collate_public([a, b])['inputs']
    assert x['point_offsets'].tolist() == [0, 3, 3]
    assert x['g_cam'].shape == (2, 3) and x['voxel_origins'].shape == (2, 3)
