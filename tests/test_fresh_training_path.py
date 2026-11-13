"""One ordinary optimizer update through the combined fresh-training paths."""
import math

import pytest
import torch

from models import Suction20Net
from suction20.cpu_geometry import CpuGeometry, SamplerTargetsCPU
from suction20.losses import source_losses
from suction20.precision import resolve_precision
from suction20.training import make_optimizer, optimizer_update
from test_contact_grouping import FROZEN
from test_model import TinyTestBackbone, model_inputs


@pytest.mark.parametrize('experiment', ['E3', 'E5'])
def test_fresh_bf16_native_geometry_training_update(experiment):
    pytest.importorskip('models._fps_cpu')
    pytest.importorskip('models._spatial_index_cpu')
    torch.manual_seed(42)
    inputs = model_inputs()
    # Repeated voxels exercise point-space supervision after E3 head sharing.
    inputs['inverse_map'] = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2,
                                          3, 3, 3, 4, 4, 5, 5, 5, 5, 5])
    inputs['voxel_features'] = inputs['voxel_features'][:6]
    inputs['voxel_coords'] = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0],
                                           [1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0]],
                                          dtype=torch.int32)
    inputs['point_offsets'] = torch.tensor([0, 8, 18])
    valid = inputs['normal_obs_valid'] & (inputs['instance_id_input'] > 0)
    targets = {
        'gt_A': torch.linspace(0., 1., 18), 'gt_B': torch.linspace(1., 0., 18),
        'quality_valid': valid, 'instance_id_gt': inputs['instance_id_input'],
        'axis_valid': valid, 'normal_label': torch.tensor([[-.1, .15, -.98]]).repeat(18, 1),
        'seal_target': torch.linspace(.2, .8, 18), 'seal_valid': valid,
        'wrench_target': torch.linspace(.8, .2, 18), 'wrench_valid': valid,
        'collision_free_target': (torch.arange(18) % 2).float(), 'collision_valid': valid,
    }
    model = Suction20Net(
        {'experiment': experiment, 'seed_count': 6, 'dense_chunk_size': 5,
         'pose_chunk_size': 2, 'checkpoint_dense': True},
        FROZEN, TinyTestBackbone(), performance={
            'fps_backend': 'compiled', 'geometry_precision': 'float32', 'geometry_threads': 4,
            'spatial_index': {'backend': 'uniform_grid_cpu', 'cell_size_m': .02},
            'contact_query_backend': 'native_cpu', 'chunk_input_mode': 'split',
            'instance_broadcast_mode': 'segment', 'no_instance_voxel_heads': experiment == 'E3',
        }).train()
    precision = resolve_precision({'training': {'precision': 'bf16'}}, 'cpu')
    scaler = precision.make_scaler()
    optimizer = make_optimizer(model, {'lr_peak': .001}, 'cpu')
    before = model.backbone.linear.weight.detach().clone()
    with precision.autocast():
        prediction = model(inputs, sampler_mode='train',
                           cpu_geometry=CpuGeometry.from_inputs(inputs),
                           sampler_targets_cpu=SamplerTargetsCPU.from_targets(targets))
    losses = source_losses(prediction, targets, inputs, quality_target=model.quality_target)
    assert prediction['q'].shape == (2, 6) and prediction['seed_valid'].all()
    assert prediction['q0_all'].shape == (18,) and prediction['q0_all'].dtype == torch.bfloat16
    assert 'native_query_wall_s' in model.last_grouping_timings
    assert losses['total'].dtype == torch.float32 and torch.isfinite(losses['total'])
    scaler.scale(losses['total']).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.backbone.linear.weight.grad.abs().sum() > 0
    assert model.contact_grouping.point_mlp[0].weight.grad.abs().sum() > 0
    if experiment == 'E5':
        assert model.instance_module.pool_projection[0].weight.grad.abs().sum() > 0
        assert {'seal', 'wrench', 'collision_free'} <= losses.keys()
    updated, norm = optimizer_update(optimizer, scaler, model.parameters())
    assert updated and math.isfinite(norm) and norm > 0
    assert not torch.equal(before, model.backbone.linear.weight)
    assert optimizer.state[model.backbone.linear.weight]['step'].item() == 1
