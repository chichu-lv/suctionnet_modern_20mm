"""CPU contract fixtures deliberately inject a tiny NON-PRODUCTION backbone."""
import math
import pytest
import torch
from torch import nn

from models import Suction20Net
from models.backbones.spconv_resunet14 import coordinate_rows
from models.heads import AxisHead, tangent_basis


class TinyTestBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 256)

    def forward(self, inputs):
        return self.linear(inputs['voxel_features'])


def model_inputs():
    count = 18
    return {'xyz_cam_m': torch.arange(count * 3).reshape(count, 3).float() * .001,
            'normal_obs': torch.tensor([[0., 0., -1.]]).repeat(count, 1),
            'normal_obs_valid': torch.tensor([True] * 16 + [False] * 2),
            'instance_id_input': torch.tensor([1] * 4 + [2] * 4 + [1] * 8 + [0] * 2),
            'batch_index': torch.tensor([0] * 8 + [1] * 10),
            'g_cam': torch.tensor([[0., 0., -1.], [0., 1., 0.]]),
            'voxel_features': torch.randn(count, 8), 'inverse_map': torch.arange(count),
            'voxel_coords': torch.stack((torch.arange(count) // 9, torch.arange(count),
                                         torch.zeros(count), torch.zeros(count)), -1).int()}


def test_coordinate_remap_and_missing_detection():
    coords = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3], [1, 1, 2, 3]])
    assert coordinate_rows(coords, coords[[2, 0, 1]]).tolist() == [1, 2, 0]
    bad = coords.clone()
    bad[-1, -1] = 9
    with pytest.raises(ValueError, match='missing'):
        coordinate_rows(coords, bad)


def test_axis_identity_bound_invalid_and_directed():
    head = AxisHead()
    feature = torch.randn(4, 256)
    normals = torch.tensor([[0., 0., -1.], [1., 0., 0.], [0., 0., 0.], [0., 0., 1.]])
    valid = torch.tensor([True, True, False, True])
    output = head(feature, normals, valid)
    torch.testing.assert_close(output, normals)
    with torch.no_grad():
        head.net[-1].bias.fill_(10)
    output = head(feature, normals, valid)
    assert torch.all((output[valid] * normals[valid]).sum(-1) >= math.cos(math.pi / 6) - 1e-6)
    assert output[2].eq(0).all()
    u, _ = tangent_basis(torch.tensor([[0., 0., 1.]]))
    torch.testing.assert_close(u, torch.tensor([[0., -1., 0.]]))


def test_e1_contract_gradient_isolation_and_parameter_partition():
    torch.manual_seed(42)
    model = Suction20Net({'seed_count': 12, 'dense_chunk_size': 5}, {}, TinyTestBackbone())
    inputs = model_inputs()
    output = model(inputs)
    assert output['q'].shape == (2, 12)
    assert output['seed_valid'].sum(1).tolist() == [8, 8]
    assert output['q'][~output['seed_valid']].eq(0).all()
    output['q'].sum().backward()
    assert all(p.grad is None for p in model.axis_head.parameters())
    assert model.backbone.linear.weight.grad.abs().sum() > 0
    all_parameters = [p for group in model.parameter_groups().values() for p in group]
    assert len({id(p) for p in all_parameters}) == len(list(model.parameters())) == len(all_parameters)
    model.zero_grad(set_to_none=True)
    output = model(inputs)
    output['normal_out_all'][:, 0].sum().backward()
    assert model.axis_head.net[-1].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        one, two = model(inputs), model(inputs)
    torch.testing.assert_close(one['q'], two['q'])
    assert torch.equal(one['seed_point_index'], two['seed_point_index'])


def test_explicit_pose_batch_anchor_and_axis_direction():
    model = Suction20Net({'seed_count': 4}, {}, TinyTestBackbone()).eval()
    encoded = model.encode_scene(model_inputs())
    points = encoded['xyz_cam_m'][[0, 8, 0]].clone()
    points[-1] += 1
    normals = torch.tensor([[0., 0., -2.], [0., 0., 2.], [0., 0., -1.]])
    instance = torch.ones(3, dtype=torch.long)
    with pytest.raises(ValueError, match='pose_batch_index'):
        model.score_at_pose(encoded, points, normals, instance)
    result = model.score_at_pose(encoded, points, normals, instance,
                                 pose_batch_index=torch.tensor([0, 1, 0]))
    assert result['pose_feature_valid'].tolist() == [True, True, False]
    assert result['feature_point_index'].tolist() == [0, 8, -1]
    assert result['q'][-1] == 0


def test_finetune_feature_groups_and_canonical_success_head():
    model = Suction20Net({'use_task_success_head': True}, {}, TinyTestBackbone())
    groups = {key: {id(p) for p in params} for key, params in model.parameter_groups().items()}
    for module in (model.fusion, model.pose_fusion):
        assert {id(p) for p in module.parameters()} <= groups['decoder']
    for module in (model.q0_head, model.axis_head, model.q_head, model.task_success_head):
        assert {id(p) for p in module.parameters()} <= groups['heads']
    assert not groups['decoder'] & groups['heads']
    assert model.task_success_head[-1].bias.eq(0).all()
    assert 'task_success_logit' in model(model_inputs())
    disabled = Suction20Net({'use_task_success_head': False, 'task_success_head': True}, {}, TinyTestBackbone())
    assert disabled.task_success_head is None


def test_empty_packed_observations_and_unassociated_executed_pose():
    class MustNotRunEmptyBackbone(TinyTestBackbone):
        def forward(self, inputs):
            raise AssertionError('An empty active set must not reach sparse kernels')

    model = Suction20Net({'seed_count': 12, 'use_task_success_head': True}, {}, MustNotRunEmptyBackbone())
    inputs = {key: value if key == 'g_cam' else value[:0] for key, value in model_inputs().items()}
    output = model(inputs)
    assert output['q0_all'].shape == (0,)
    assert output['normal_out_all'].shape == (0, 3)
    assert output['seed_valid'].shape == (2, 12)
    assert not output['seed_valid'].any()
    for key in ('q', 'point_cam_m', 'normal_out_cam', 'pose_features', 'task_success_logit'):
        assert output[key].eq(0).all()
    (output['q0_all'].sum() + output['normal_out_all'].sum() + output['q'].sum()).backward()
    encoded = model.encode_scene(inputs)
    result = model.score_at_pose(encoded, torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]),
                                 torch.ones(1, dtype=torch.long), feature_point_index=torch.zeros(1, dtype=torch.long),
                                 pose_batch_index=torch.zeros(1, dtype=torch.long))
    assert not result['pose_feature_valid'].any()
    assert result['q'].eq(0).all() and result['h'].eq(0).all()
    assert result['feature_point_index'].tolist() == [-1]
    assert coordinate_rows(inputs['voxel_coords'], inputs['voxel_coords']).numel() == 0


def test_sparse_wrapper_materializes_row_major_coordinates(monkeypatch):
    """Exercise the kernel boundary with the column-major collator layout."""
    import sys
    import types
    from models.backbones.spconv_resunet14 import SpconvResUNet14

    class CheckedSparseTensor:
        def __init__(self, features, indices, shape, batch_size):
            assert indices.dtype == torch.int32
            assert indices.is_contiguous() and features.is_contiguous()
            self.features, self.indices = features, indices

    class FakeSparseKernel(nn.Module):
        def forward(self, sparse):
            # Reorder output deliberately, so this also exercises row remapping.
            sparse.features = torch.nn.functional.pad(sparse.features.flip(0), (0, 504))
            sparse.indices = sparse.indices.flip(0)
            return sparse

    package = types.ModuleType('spconv')
    submodule = types.ModuleType('spconv.pytorch')
    submodule.SparseConvTensor = CheckedSparseTensor
    package.pytorch = submodule
    monkeypatch.setitem(sys.modules, 'spconv', package)
    monkeypatch.setitem(sys.modules, 'spconv.pytorch', submodule)
    wrapper = SpconvResUNet14.__new__(SpconvResUNet14)
    nn.Module.__init__(wrapper)
    wrapper.network = FakeSparseKernel()
    wrapper.projection = nn.Identity()
    inputs = model_inputs()
    coords = inputs['voxel_coords'].T.contiguous().T
    assert not coords.is_contiguous()
    inputs['voxel_coords'] = coords
    expected_coords = coords.clone()
    result = wrapper(inputs)
    torch.testing.assert_close(result[:, :8], inputs['voxel_features'])
    torch.testing.assert_close(inputs['voxel_coords'], expected_coords)
