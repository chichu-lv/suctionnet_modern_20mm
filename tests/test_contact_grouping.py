import copy

import numpy as np
import pytest
import torch

from models.contact_grouping import ContactGrouping, SpatialIndex, ZONE_NAMES


FROZEN = {'radius_m': .020, 'self_height_threshold_m': .004,
          'envelope_segments_m': [[-.002, .025, .022], [.025, .070, .028], [.070, .150, .028]]}


def scene(xyz, instances=None, batch=None):
    xyz = torch.as_tensor(np.asarray(xyz), dtype=torch.float32)
    count = len(xyz)
    return {'xyz_cam_m': xyz,
            'instance_id_input': torch.ones(count, dtype=torch.long) if instances is None else torch.tensor(instances),
            'batch_index': torch.zeros(count, dtype=torch.long) if batch is None else torch.tensor(batch),
            'normal_obs': torch.tensor([[0., 0., 1.]]).repeat(count, 1),
            'normal_obs_valid': torch.ones(count, dtype=torch.bool),
            'g_cam': torch.tensor([[0., 0., -1.]]).repeat(max(batch or [0]) + 1, 1),
            'F_local': torch.randn(count, 256, requires_grad=True)}


def run_query(group, inputs, *, point=(0., 0., 0.), normal=(0., 0., 1.), random_roll=False):
    return group.query(SpatialIndex.from_inputs(inputs), torch.tensor([point]), torch.tensor([normal]),
                       torch.ones(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                       torch.ones(1, dtype=torch.bool), random_roll=random_roll)


def test_complete_zones_before_limits_and_prelimit_lip_sector_counts():
    # A global nearest128 crop would contain only core and miss lip/approach.
    core = np.column_stack((np.linspace(-.005, .005, 200), np.zeros((200, 2))))
    angle = np.repeat((np.arange(8) + .5) * 2 * np.pi / 8, 10)
    lip = np.column_stack((.020 * np.cos(angle), .020 * np.sin(angle), np.zeros(80)))
    xyz = np.concatenate((core, lip, [[0., 0., .12]]))
    # The other frame occupies exactly the same physical coordinates.
    inputs = scene(np.concatenate((xyz, xyz)), [1] * 280 + [0] + [1] * 280 + [0],
                   [0] * 281 + [1] * 281)
    group = ContactGrouping(FROZEN)
    q = run_query(group, inputs)
    assert q['core']['counts'].tolist() == [200]
    assert q['core']['mask'].sum() == 32
    assert q['lip']['counts'].tolist() == [80]
    assert q['lip']['mask'].sum() == 64
    assert q['lip_sector_counts'].tolist() == [[10] * 8]
    assert q['approach']['counts'].tolist() == [1]
    assert q['approach']['indices'][q['approach']['mask']].tolist() == [280]
    for name in ZONE_NAMES:
        chosen = q[name]['indices'][q[name]['mask']]
        assert len(chosen) == len(np.unique(chosen))
        assert np.all(chosen < 281)


def test_frozen_envelope_self_clearance_and_segment_boundaries():
    # Exact binary fractions make equality tests independent of decimal rounding.
    frozen = {'radius_m': .020, 'self_height_threshold_m': .00390625,
              'envelope_segments_m': [[-.001953125, .03125, .015625],
                                      [.03125, .0625, .03125], [.0625, .125, .03125]]}
    points = [[0, 0, 0], [0, 0, .00390625], [0, 0, .0078125], [0, 0, .03125],
              [0, 0, .0625], [0, 0, .125], [.02, 0, .02], [0, 0, -.001953125],
              [0, 0, .126]]
    group = ContactGrouping(frozen)
    q = run_query(group, scene(points, [1, 1, 1, 0, 0, 0, 0, 0, 0]))
    assert q['cup']['indices'][q['cup']['mask']].tolist() == [2, 3, 7]
    assert q['mount']['indices'][q['mount']['mask']].tolist() == [4]
    assert q['approach']['indices'][q['approach']['mask']].tolist() == [5]


def test_actual_center_axis_and_random_roll_control_queries():
    group = ContactGrouping(FROZEN)
    inputs = scene([[.019, 0, 0], [0, 0, .12]], [1, 0])
    origin = run_query(group, inputs)
    shifted = run_query(group, inputs, point=(.002, 0., 0.))
    rotated = run_query(group, inputs, normal=(1., 0., 0.))
    assert origin['lip']['counts'][0] == 1 and origin['core']['counts'][0] == 0
    assert shifted['lip']['counts'][0] == 0 and shifted['core']['counts'][0] == 1
    assert origin['approach']['counts'][0] == 1 and rotated['approach']['counts'][0] == 0
    torch.manual_seed(7)
    roll_one = run_query(group, inputs, random_roll=True)
    torch.manual_seed(7)
    roll_two = run_query(group, inputs, random_roll=True)
    assert np.array_equal(roll_one['lip_sector_counts'], roll_two['lip_sector_counts'])
    assert not np.array_equal(roll_one['lip_sector_counts'], origin['lip_sector_counts'])
    assert np.array_equal(run_query(group, inputs)['lip_sector_counts'], origin['lip_sector_counts'])


def test_empty_masks_and_shared_projection_retain_feature_gradients():
    group = ContactGrouping(FROZEN, chunk_size=1)
    inputs = scene([[0, 0, 0], [.02, 0, 0], [0, 0, .12]], [1, 1, 0])
    points = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    normals = torch.tensor([[0., 0., 1.]]).repeat(2, 1).requires_grad_()
    result = group(inputs, points, normals, torch.ones(2, dtype=torch.long),
                   torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool))
    assert result['zone_counts'].tolist() == [[1, 1, 0, 0, 1], [0, 0, 0, 0, 0]]
    for name in ZONE_NAMES:
        assert result[name][1].eq(0).all()
    sum(result[name].sum() for name in ZONE_NAMES).backward()
    assert normals.grad is None
    assert inputs['F_local'].grad.abs().sum() > 0
    assert group.local_projection[0].weight.grad.abs().sum() > 0
    assert group.point_mlp[0].weight.grad.abs().sum() > 0
    assert 'spatial_index' in inputs and 'grouping_timings' in inputs
    with pytest.raises(ValueError, match='frozen'):
        ContactGrouping({})


def test_per_zone_gather_preserves_contact_outputs_and_gradients_with_shared_rows():
    torch.manual_seed(27)
    reference = ContactGrouping(FROZEN, chunk_size=2, checkpointed=False, chunk_input_mode='split')
    optimized = copy.deepcopy(reference)
    optimized.feature_gather_mode = 'zone'
    inputs = scene([[0., 0., 0.], [.007, 0., 0.], [.020, 0., 0.],
                    [0., 0., .010], [0., 0., .050], [0., 0., .120]])
    # Nearby poses reuse source rows across chunk boundaries; the last pose
    # supplies ordinary empty-zone padding and all five regions are exercised.
    points = torch.tensor([[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.],
                           [-.001, 0., 0.], [1., 1., 1.]])
    normals = torch.tensor([[0., 0., 1.]]).repeat(5, 1)
    weights = {name: torch.randn(5, 64) for name in ZONE_NAMES}
    compared = []
    for group in (reference, optimized):
        encoded = {**inputs, 'F_local': inputs['F_local'].detach().clone().requires_grad_()}
        result = group(encoded, points, normals, torch.ones(5, dtype=torch.long),
                       torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.bool))
        assert result['zone_counts'][:4].gt(0).all()
        assert all(result[name][-1].eq(0).all() for name in ZONE_NAMES)
        sum((result[name] * weights[name]).sum() for name in ZONE_NAMES).backward()
        assert encoded['F_local'].grad.abs().sum() > 0
        compared.append([*(result[name].detach() for name in ZONE_NAMES),
                         encoded['F_local'].grad, *(parameter.grad for parameter in group.parameters())])
    for actual, expected in zip(compared[1], compared[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
