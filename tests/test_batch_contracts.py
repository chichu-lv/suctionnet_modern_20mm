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


@pytest.mark.parametrize('batch_size', [1, 2])
def test_integer_keys_match_axis_unique_packed_values(monkeypatch, batch_size):
    from suction20 import batching
    rng = np.random.default_rng(18)
    samples = []
    for _ in range(batch_size):
        item = sample()
        for section in ('inputs', 'targets'):
            for key, value in item[section].items():
                if key != 'g_cam':
                    item[section][key] = np.resize(value, (300,) + value.shape[1:])
        item['inputs']['xyz_cam_m'] = (rng.integers(-8, 9, size=(300, 3)) * .001).astype(np.float32)
        samples.append(item)
    actual = collate_public(samples)

    def axis_unique(grid):
        origin = grid.min(axis=0)
        unique, inverse = np.unique(grid - origin, axis=0, return_inverse=True)
        return origin, unique, inverse

    monkeypatch.setattr(batching, 'unique_voxels_xyz', axis_unique)
    expected = collate_public(samples)
    for section in ('inputs', 'targets'):
        for key, value in actual[section].items():
            assert value.dtype == expected[section][key].dtype
            assert torch.equal(value, expected[section][key]), key
    if batch_size == 1:
        assert np.shares_memory(actual['inputs']['xyz_cam_m'].numpy(), samples[0]['inputs']['xyz_cam_m'])
        assert np.shares_memory(actual['targets']['gt_B'].numpy(), samples[0]['targets']['gt_B'])


def test_cpu_contexts_share_final_storage_and_separate_supervision():
    from suction20.cpu_geometry import CpuGeometry, SamplerTargetsCPU
    batch = collate_public([sample(), sample()])
    geometry = CpuGeometry.from_inputs(batch['inputs'])
    labels = SamplerTargetsCPU.from_targets(batch['targets'])
    for key in ('xyz_cam_m', 'normal_obs', 'normal_obs_valid', 'instance_id_input', 'batch_index', 'point_offsets'):
        assert np.shares_memory(getattr(geometry, key), batch['inputs'][key].numpy())
    assert not hasattr(geometry, 'quality_valid') and not hasattr(geometry, 'gt_A')
    assert np.shares_memory(labels.quality_valid, batch['targets']['quality_valid'].numpy())
    assert labels.gt_A is None
    assert geometry.xyz64 is geometry.xyz64
    np.testing.assert_array_equal(geometry.xyz64, batch['inputs']['xyz_cam_m'].numpy().astype(np.float64))
    assert [rows.tolist() for rows in geometry.frame_rows] == [[0, 1, 2], [3, 4, 5]]
