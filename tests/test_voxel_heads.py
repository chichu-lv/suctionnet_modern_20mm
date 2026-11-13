"""CPU equivalence of voxel heads with unequal raw-point labels and occupancy."""
import pytest
import torch

from models import Suction20Net
from suction20.losses import source_losses
from test_contact_grouping import FROZEN
from test_model import TinyTestBackbone, model_inputs


def repeated_voxel_inputs():
    inputs = model_inputs()
    # Two frames, unequal occupancy, and different instances in the same voxel.
    inputs['inverse_map'] = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2,
                                          3, 3, 3, 4, 4, 5, 5, 5, 5, 5])
    inputs['voxel_features'] = inputs['voxel_features'][:6]
    inputs['voxel_coords'] = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0],
                                           [1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0]],
                                          dtype=torch.int32)
    inputs['normal_obs'][1] = torch.tensor([1., 0., 0.])
    return inputs


@pytest.mark.parametrize('experiment', ['E0', 'E1', 'E3'])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_voxel_heads_preserve_raw_losses_gradients_rng_and_initialization(experiment, checkpointed):
    config = {'experiment': experiment, 'seed_count': 6, 'dense_chunk_size': 5,
              'pose_chunk_size': 2, 'checkpoint_dense': checkpointed}
    models = []
    for enabled in (False, True):
        torch.manual_seed(42)
        models.append(Suction20Net(config, FROZEN, TinyTestBackbone(),
                                  performance={'no_instance_voxel_heads': enabled}).train())
    baseline, candidate = models
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, candidate.state_dict()[name], rtol=0, atol=0)
    for model in models:
        names = {id(param): name for name, param in model.named_parameters()}
        groups = {group: [names[id(p)] for p in params]
                  for group, params in model.parameter_groups().items()}
        if model is baseline:
            expected_groups = groups
        else:
            assert groups == expected_groups
    inputs = repeated_voxel_inputs()
    targets = {'gt_A': torch.linspace(0, 1, 18), 'gt_B': torch.linspace(1, 0, 18),
               'quality_valid': inputs['normal_obs_valid'].clone(),
               'instance_id_gt': inputs['instance_id_input'],
               'axis_valid': inputs['normal_obs_valid'],
               'normal_label': torch.roll(inputs['normal_obs'], 1, dims=-1)}
    targets['quality_valid'][3] = False
    rng = torch.get_rng_state()
    results = []
    for model in models:
        torch.set_rng_state(rng)
        local_inputs = dict(inputs, voxel_features=inputs['voxel_features'].clone().requires_grad_())
        encoded = model.encode_scene(local_inputs)
        if model is candidate:
            assert 'F_fused' not in encoded
            assert encoded['F_fused_voxel'].shape == (6, 256)
        else:
            assert encoded['F_fused'].shape == (18, 256)
        assert encoded['F_local'].shape == (18, 256)
        q0, axis = model.predict_dense(encoded)
        assert q0.shape == (18,)
        assert not torch.equal(axis[0], axis[1]), 'Axis must retain each raw observed normal'
        gathered = model.gather_fused_features(encoded, torch.tensor([0, 1, 4, 8, 17]))
        output = model(local_inputs, sampler_mode='train', targets=targets)
        losses = source_losses(output, targets, local_inputs, quality_target=model.quality_target)
        losses['total'].backward()
        results.append((output, losses, gathered, local_inputs['voxel_features'].grad.clone(),
                        {name: None if p.grad is None else p.grad.clone()
                         for name, p in model.named_parameters()}, torch.get_rng_state()))
    first, second = results
    assert torch.equal(first[-1], second[-1])
    for key in first[0]:
        torch.testing.assert_close(first[0][key], second[0][key], rtol=2e-5, atol=2e-6)
    for key in first[1]:
        torch.testing.assert_close(first[1][key], second[1][key], rtol=2e-5, atol=2e-6)
    for index in (2, 3):
        torch.testing.assert_close(first[index], second[index], rtol=2e-5, atol=2e-6)
    for name, grad in first[4].items():
        if grad is None:
            assert second[4][name] is None
        else:
            assert torch.isfinite(grad).all() and torch.isfinite(second[4][name]).all()
            torch.testing.assert_close(grad, second[4][name], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('experiment', ['E2', 'E4', 'E5'])
def test_instance_fusion_keeps_raw_point_features(experiment):
    model = Suction20Net({'experiment': experiment}, FROZEN, TinyTestBackbone(),
                         performance={'no_instance_voxel_heads': True})
    encoded = model.encode_scene(repeated_voxel_inputs())
    assert 'F_fused_voxel' not in encoded
    assert encoded['F_fused'].shape == (18, 256)


def test_voxel_heads_empty_scene_preserves_pose_and_backward_contract():
    model = Suction20Net({'seed_count': 4}, FROZEN, TinyTestBackbone(),
                         performance={'no_instance_voxel_heads': True})
    inputs = {k: value if k == 'g_cam' else value[:0]
              for k, value in repeated_voxel_inputs().items()}
    output = model(inputs)
    assert output['q0_all'].shape == (0,)
    assert not output['seed_valid'].any()
    (output['q0_all'].sum() + output['normal_out_all'].sum() + output['q'].sum()).backward()
    encoded = model.encode_scene(inputs)
    pose = model.score_at_pose(encoded, torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]),
                               torch.ones(1, dtype=torch.long),
                               pose_batch_index=torch.zeros(1, dtype=torch.long))
    assert not pose['pose_feature_valid'].any()
    assert pose['q'].eq(0).all()
