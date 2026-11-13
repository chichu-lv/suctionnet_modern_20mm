"""Detached CPU jobs and two live training graphs for the geometry pipeline."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from models import Suction20Net
from suction20.cpu_geometry import CpuGeometry, SamplerTargetsCPU
from suction20.losses import source_losses
from test_contact_grouping import FROZEN
from test_model import TinyTestBackbone, model_inputs


def make_model():
    pytest.importorskip('models._fps_cpu')
    pytest.importorskip('models._spatial_index_cpu')
    return Suction20Net(
        {'experiment': 'E5', 'seed_count': 6, 'dense_chunk_size': 8,
         'pose_chunk_size': 3, 'checkpoint_dense': False}, FROZEN, TinyTestBackbone(),
        performance={'fps_backend': 'compiled', 'geometry_precision': 'float32',
                     'geometry_threads': 2, 'contact_query_backend': 'native_cpu',
                     'spatial_index': {'backend': 'uniform_grid_cpu', 'cell_size_m': .02},
                     'chunk_input_mode': 'split', 'instance_broadcast_mode': 'segment'})


def batch():
    inputs = model_inputs()
    inputs['point_offsets'] = torch.tensor([0, 8, 18])
    valid = inputs['normal_obs_valid'] & (inputs['instance_id_input'] > 0)
    targets = {'gt_A': torch.linspace(0., 1., 18), 'gt_B': torch.linspace(1., 0., 18),
               'quality_valid': valid, 'instance_id_gt': inputs['instance_id_input'],
               'axis_valid': valid, 'normal_label': torch.tensor([[-.1, .15, -.98]]).repeat(18, 1),
               'seal_target': torch.linspace(.2, .8, 18), 'seal_valid': valid,
               'wrench_target': torch.linspace(.8, .2, 18), 'wrench_valid': valid,
               'collision_free_target': (torch.arange(18) % 2).float(), 'collision_valid': valid}
    kwargs = {'cpu_geometry': CpuGeometry.from_inputs(inputs),
              'sampler_targets_cpu': SamplerTargetsCPU.from_targets(targets)}
    return inputs, targets, kwargs


def test_staged_query_preserves_prediction_and_has_no_worker_rng_or_shared_stats():
    torch.manual_seed(42)
    model = make_model().eval()
    inputs, _, kwargs = batch()
    with torch.no_grad():
        expected = model(inputs, **kwargs)
        prepared = model.prepare_forward(inputs, **kwargs)
        old_stats = dict(model.contact_grouping.last_query_timings)
        old_rng = torch.get_rng_state()
        with ThreadPoolExecutor(max_workers=1) as executor:
            geometry = executor.submit(prepared.geometry_job).result()
        assert torch.equal(old_rng, torch.get_rng_state())
        assert model.contact_grouping.last_query_timings == old_stats
        actual = model.finish_forward(prepared, geometry_result=geometry)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
    assert geometry.timings['cpu_geometry_job_s'] >= geometry.timings['query_cpu_seconds']
    assert geometry.timings['cpu_geometry_finished_monotonic'] >= geometry.timings['cpu_geometry_started_monotonic']
    assert 'native_query_wall_s' in model.last_grouping_timings


def test_two_inflight_graphs_finish_and_update_with_finite_gradients():
    torch.manual_seed(42)
    model = make_model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.backbone.linear.weight.detach().clone()
    jobs = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(2):
            inputs, targets, kwargs = batch()
            prepared = model.prepare_forward(inputs, sampler_mode='train', targets=targets, **kwargs)
            jobs.append((prepared, executor.submit(prepared.geometry_job), inputs, targets))
        for prepared, future, inputs, targets in jobs:
            prediction = model.finish_forward(prepared, geometry_result=future.result())
            losses = source_losses(prediction, targets, inputs, quality_target=model.quality_target)
            assert torch.isfinite(losses['total'])
            (losses['total'] / 2).backward()
    for module in (model.backbone.linear, model.contact_grouping.point_mlp[0],
                   model.instance_module.pool_projection[0], model.q_head[-1]):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.backbone.linear.weight)
