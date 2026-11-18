"""Route-A geometry and differentiable model checks on explicit CPU fixtures.

Injected backbones test the complete route-A head/contact graph. They do not
claim to validate CUDA SpConv kernels, real SuctionNet teacher labels or speed.
"""
import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from models.instance_route_a import InstanceSuction3D, SingletonRunningBatchNorm1d
from models.instance_route_a_geometry import (
    NeighborIndex, contact_coordinates, deterministic_fps, lip_summary,
    voxel_representatives,
)


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class InjectedBackbone(nn.Module):
    def __init__(self, with_bn=False):
        super().__init__()
        self.projection = nn.Linear(12, 256)
        self.norm = nn.BatchNorm1d(256) if with_bn else nn.Identity()

    def forward(self, inputs):
        assert inputs['voxel_features'].shape[1] == 12
        assert inputs['voxel_coords'].dtype == torch.int32
        assert inputs['voxel_coords'].is_contiguous()
        return self.norm(self.projection(inputs['voxel_features']))


def observation(side=41):
    yy, xx = np.mgrid[:side, :side]
    xyz = np.stack(((xx - side // 2) * .001,
                    (yy - side // 2) * .001, np.full_like(xx, .6, dtype=float)), axis=-1)
    points = xyz.reshape(-1, 3)
    return {'rgb': np.full((side, side, 3), .5, np.float32),
            'support_points': points, 'support_rgb': np.full_like(points, .5),
            'support_normals': np.tile([0., 0., -1.], (len(points), 1)),
            'support_normal_valid': np.ones(len(points), bool),
            'support_rgb_valid': np.ones(len(points), bool),
            'support_components': np.ones(len(points), np.int64),
            'support_source_ids': np.arange(len(points)),
            'support_uv': np.column_stack((xx.ravel(), yy.ravel())),
            'centroid': points.mean(0), 'gravity': np.array([0., 1., 0.]),
            'query_points': np.array([[0., 0., .6], [.003, .004, .6], [-.004, -.002, .6]]),
            'query_normals': np.tile([0., 0., -1.], (3, 1)),
            'query_ids': np.array([5, 7, 9])}


def test_voxel_representatives_and_fps_ties_and_prefix():
    points = np.array([[.001, .001, .001], [.001, .001, .001], [.003, .001, .001]])
    ids = np.array([8, 2, 7])
    rows, keys = voxel_representatives(points, ids, np.zeros(3))
    assert ids[rows].tolist() == [2, 7]
    assert keys.tolist() == [[0, 0, 0], [1, 0, 0]]
    points = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 2, 0]], dtype=float)
    ids = np.array([9, 4, 1, 8])
    full = deterministic_fps(points, ids, 99)
    assert ids[full].tolist() == [1, 8, 4, 9]
    np.testing.assert_array_equal(deterministic_fps(points, ids, 2), full[:2])
    perm = np.array([2, 0, 3, 1])
    np.testing.assert_array_equal(ids[perm][deterministic_fps(points[perm], ids[perm], 4)], ids[full])


def test_interpolation_uses_one_two_three_neighbors_and_component_boundary():
    points = np.array([[0, 0, 0], [.005, 0, 0], [.009, 0, 0], [.019, 0, 0]])
    index = NeighborIndex(points, np.arange(4), np.array([1, 1, 1, 2]))
    query = np.array([[0, 0, 0], [.015, 0, 0], [.019, 0, 0], [.030, 0, 0], [0, 0, 0]])
    actual = index.associate(query, np.array([1, 1, 1, 1, 99]))
    assert actual['counts'].tolist() == [3, 2, 1, 0, 0]
    np.testing.assert_array_equal(actual['weights'][0], [1, 0, 0])
    np.testing.assert_allclose(actual['weights'][1, :2], [1 / .006, 1 / .010] / np.sum([1 / .006, 1 / .010]))
    assert actual['indices'][2, 0] == 2  # nearby other-component point is excluded.
    assert actual['d3'][1] == pytest.approx(.015)  # not the truncated interpolation radius.
    assert np.isnan(actual['d3'][-1])
    assert actual['component_missing'].tolist() == [False, False, False, False, True]


def test_equal_distance_neighbors_resolve_cutoff_ties_by_source_id():
    points = np.array([[.005, 0, 0], [-.005, 0, 0], [0, .005, 0], [0, -.005, 0]])
    ids = np.array([40, 30, 20, 10])
    index = NeighborIndex(points, ids, np.ones(4, int))
    actual = index.associate(np.zeros((1, 3)), np.ones(1, int))
    assert ids[actual['indices'][0]].tolist() == [10, 20, 30]
    np.testing.assert_allclose(actual['weights'][0], [1 / 3] * 3)
    anchor, _ = index.nearest_anchor(np.zeros((1, 3)))
    assert ids[anchor[0]] == 10


def test_region_boundaries_are_half_open_with_closed_outer_lip():
    points = np.array([[r, 0, 0] for r in [0, .010, .018, .022, .02200001]])
    _, _, _, masks = contact_coordinates(points, np.zeros(3), np.array([0., 0., 1.]))
    assert [np.flatnonzero(m).tolist() for m in masks] == [[0], [1], [2, 3]]


def test_harmonics_retain_low_order_structure_and_ignore_basis_phase():
    theta = 2 * np.pi * (np.arange(72) + .5) / 72
    tangent = .020 * np.column_stack((np.cos(theta), np.sin(theta), np.zeros(72)))
    patterns = [np.where(np.cos(theta) > 0, .003, -.003),
                np.where(np.cos(2 * theta) > 0, .003, -.003),
                np.where(np.arange(72) % 2, .003, -.003)]
    summaries = [lip_summary(h, tangent, np.array([0., 0., 1.]))[0] for h in patterns]
    assert summaries[0][4] > summaries[1][4] + .04
    assert summaries[1][5] > summaries[0][5] + .04
    assert max(summaries[2][4:7]) < 1e-12
    np.testing.assert_allclose([s[3] for s in summaries], [.075] * 3)
    angle = .713
    frame = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    rotated, _ = lip_summary(patterns[0], tangent, np.array([0., 0., 1.]), basis=frame)
    np.testing.assert_allclose(rotated, summaries[0], atol=1e-13)
    constant, support = lip_summary(np.full(6, .003), tangent[:6], np.array([0., 0., 1.]))
    assert support == 6 and constant[7] == 0
    np.testing.assert_allclose(constant[3:7], 0., atol=1e-15)


def test_model_shapes_dimensional_contract_and_backward():
    torch.manual_seed(1)
    model = InstanceSuction3D(backbone=InjectedBackbone())
    obs = observation()
    encoded = model.encode_instance(obs)
    assert encoded['sparse_inputs']['voxel_features'].shape[1] == 12
    np.testing.assert_allclose(encoded['sparse_inputs']['voxel_features'][:, 9:].detach().numpy(),
                               np.tile([0, 1, 0], (len(encoded['backbone_source_ids']), 1)))
    result = model(obs)
    assert result['query_ids'].tolist() == [5, 7, 9]
    assert result['valid'].all() and result['lip_summary'].shape == (3, 8)
    assert model.query_fusion[0].in_features == 531
    assert model.seal_head[0].in_features == 200
    assert model.wrench_head[0].in_features == 329
    assert model.task_head[0].in_features == 260
    for name in ('q_inst', 'seal', 'wrench'):
        assert result[name].shape == (3,)
        assert ((result[name] >= 0) & (result[name] <= 1)).all()
    torch.testing.assert_close(result['ell'], torch.full((3,), math.log(.1)))
    torch.testing.assert_close(result['log_variance'], math.log(2) + 2 * result['ell'])
    loss = (result['q_inst'] - .8).square().mean() + (result['seal'] - .7).square().mean()
    loss = loss + (result['wrench'] - .6).square().mean() + result['ell'].mean()
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(result['task_logit'], torch.ones(3))
    loss.backward()
    assert model.backbone.projection.weight.grad.abs().sum() > 0
    assert model.query_fusion[0].weight.grad.abs().sum() > 0
    assert model.local_point_mlp[0].weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    groups = model.parameter_groups()
    group_ids = [id(p) for values in groups.values() for p in values]
    assert len(group_ids) == len(set(group_ids)) == len(list(model.parameters()))


def test_exact_pose_axis_rebuilds_contacts_and_query_chunks_are_equivalent():
    model = InstanceSuction3D(backbone=InjectedBackbone(), query_chunk_size=32).eval()
    obs = observation()
    encoded = model.encode_instance(obs)
    first = model.score_at_pose(encoded, obs['query_points'], obs['query_normals'])
    model.query_chunk_size = 1
    chunked = model.score_at_pose(encoded, obs['query_points'], obs['query_normals'])
    for key in ('q_inst', 'seal', 'wrench', 'ell', 'task_logit', 'lip_summary'):
        torch.testing.assert_close(first[key], chunked[key], atol=2e-7, rtol=2e-6)
    p = np.array([.0004, -.0002, .6])
    n = np.array([.5, 0., -math.sqrt(.75)])
    oblique = model.score_at_pose(encoded, p, n)
    np.testing.assert_array_equal(oblique['points'].detach().numpy()[0], p)
    np.testing.assert_allclose(oblique['normal_out'].detach().numpy()[0], n, atol=1e-15)
    assert oblique['lip_summary'][0, 3] > 0
    assert first['lip_summary'][0, 3] == 0
    with pytest.raises(ValueError, match='all zero'):
        model.score_at_pose(encoded, p, n, np.array([1]))


def test_missing_backbone_features_do_not_invalidate_raw_geometry():
    model = InstanceSuction3D(backbone=InjectedBackbone(), backbone_points=1)
    obs = observation()
    result = model(obs)
    assert result['valid'].all()
    assert not result['query_feature_valid'].any()
    assert (result['region_counts'] > 0).all()
    assert torch.isfinite(result['q_inst']).all()


def test_raw_lip_summary_is_computed_before_64_point_region_fps():
    theta = 2 * np.pi * (np.arange(144) + .5) / 144
    height = .002 * np.cos(3 * theta)
    ring = np.column_stack((.020 * np.cos(theta), .020 * np.sin(theta), .6 - height))
    points = np.vstack(([0., 0., .6], ring))
    obs = observation()
    obs.update(support_points=points, support_rgb=np.full_like(points, .5),
               support_normals=np.tile([0., 0., -1.], (len(points), 1)),
               support_normal_valid=np.ones(len(points), bool), support_rgb_valid=np.ones(len(points), bool),
               support_components=np.ones(len(points), int), support_source_ids=np.arange(len(points)),
               centroid=points.mean(0), query_points=np.array([[0., 0., .6]]),
               query_normals=np.array([[0., 0., -1.]]), query_ids=np.array([0]))
    result = InstanceSuction3D(backbone=InjectedBackbone())(obs)
    assert result['region_counts'][0, 2] == 144
    assert result['lip_summary'][0, 0] == pytest.approx(np.log1p(144), abs=1e-6)
    assert result['independent_angular_support'][0] == 144
    assert result['lip_summary'][0, 6] == pytest.approx(.001 / .04, abs=1e-6)


def test_invalid_rgb_normals_and_invalid_queries_cannot_poison_gradients():
    obs = observation()
    obs['support_rgb'][0] = np.nan
    obs['support_rgb_valid'][0] = False
    obs['support_normals'][1] = np.nan
    obs['support_normal_valid'][1] = False
    model = InstanceSuction3D(backbone=InjectedBackbone())
    encoded = model.encode_instance(obs)
    assert torch.isfinite(encoded['sparse_inputs']['voxel_features']).all()
    points = np.array([[0., 0., .6], [np.nan, 0, .6], [1e50, 0, .6], [1, 1, 1]])
    result = model.score_at_pose(encoded, points, np.tile([0., 0., -1.], (4, 1)))
    assert result['valid'].tolist() == [True, False, False, False]
    for key in ('q_inst', 'seal', 'wrench', 'ell', 'task_logit'):
        assert torch.isfinite(result[key]).all() and (result[key][1:] == 0).all()
    result['q_inst'].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_empty_queries_and_small_support_return_explicit_invalid():
    obs = observation(side=7)
    model = InstanceSuction3D(backbone=InjectedBackbone())
    result = model(obs)
    assert not result['valid'].any()
    assert set(result['reason']) == {'insufficient_support'}
    empty = model.score_at_pose(model.encode_instance(obs), np.empty((0, 3)), np.empty((0, 3)))
    assert empty['q_inst'].shape == (0,) and empty['reason'] == []
    for key in ('support_points', 'support_rgb', 'support_normals'):
        obs[key] = np.empty((0, 3))
    for key in ('support_normal_valid', 'support_rgb_valid'):
        obs[key] = np.empty(0, bool)
    for key in ('support_components', 'support_source_ids'):
        obs[key] = np.empty(0, int)
    obs['centroid'] = np.zeros(3)
    result = model(obs)
    assert not result['valid'].any() and torch.isfinite(result['q_inst']).all()


def test_singleton_bn_policy_preserves_normal_path_and_handles_singleton():
    torch.manual_seed(2)
    reference = nn.BatchNorm1d(4).double()
    safe = SingletonRunningBatchNorm1d(4).double()
    safe.load_state_dict(reference.state_dict())
    batch = torch.randn(8, 4, dtype=torch.double)
    torch.testing.assert_close(safe(batch), reference(batch), atol=0, rtol=0)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(safe.state_dict()[name], value, atol=0, rtol=0)
    before = copy.deepcopy(safe.state_dict())
    single = torch.randn(1, 4, dtype=torch.double, requires_grad=True)
    response = safe(single)
    response.sum().backward()
    assert torch.isfinite(response).all() and torch.isfinite(single.grad).all()
    assert safe.singleton_fallback_count == 1
    for name in ('running_mean', 'running_var', 'num_batches_tracked'):
        torch.testing.assert_close(safe.state_dict()[name], before[name], atol=0, rtol=0)
    model = InstanceSuction3D(backbone=InjectedBackbone(with_bn=True), backbone_points=1)
    assert isinstance(model.backbone.norm, SingletonRunningBatchNorm1d)
    result = model(observation())
    result['q_inst'].sum().backward()
    assert model.backbone.norm.singleton_fallback_count == 1
    assert model.backbone.norm.num_batches_tracked == 0
    assert model.architecture_metadata()['singleton_bn_policy'] == 'running_stats'


def test_real_wrapper_supports_12_channels_without_changing_legacy_default(monkeypatch):
    import models.backbones.spconv_resunet14 as wrapper

    class ConstructorProbe(nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.layer = nn.Linear(inputs, outputs)

    monkeypatch.setattr(wrapper, '_load_external', lambda root: ConstructorProbe)
    legacy = wrapper.SpconvResUNet14('verified-placeholder')
    current = InstanceSuction3D(backbone_root='verified-placeholder')
    assert legacy.network.layer.in_features == 8
    assert current.backbone.network.layer.in_features == 12
    assert not current.backbone_injected
    with pytest.raises(ValueError, match='requires backbone_root'):
        InstanceSuction3D()

