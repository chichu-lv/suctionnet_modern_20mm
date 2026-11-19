"""Packed route-A execution contracts on explicit CPU backbone fixtures.

These tests check row ownership, gradients and precision boundaries. They do
not measure real SpConv/H20 throughput or establish new-run convergence.
"""
import copy
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from models.instance_route_a import InstanceSuction3D, prepare_source_geometry_cpu
from test_instance_route_a import InjectedBackbone, observation


class CaptureBackbone(InjectedBackbone):
    def __init__(self, *, with_bn=False):
        super().__init__(with_bn=with_bn)
        self.calls = []

    def forward(self, inputs):
        self.calls.append({name: value.detach().clone() if torch.is_tensor(value) else value
                           for name, value in inputs.items()})
        self.calls[-1]['autocast_enabled'] = torch.is_autocast_enabled(inputs['voxel_features'].device.type)
        return super().forward(inputs)


def samples():
    first, second = observation(side=15), observation(side=17)
    first['support_rgb'][:] = .2
    second['support_rgb'][:] = .8
    translation = np.array([.13, -.04, .2])
    second['support_points'] = second['support_points'] + translation
    second['query_points'] = second['query_points'] + translation
    second['centroid'] = second['support_points'].mean(0)
    # Source IDs intentionally overlap: they are scoped to each observation.
    return [first, second]


def assert_scores(actual, expected, *, atol=2e-6, rtol=2e-5):
    assert set(actual) == set(expected)
    for name, value in actual.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected[name], atol=atol, rtol=rtol, equal_nan=True)
        else:
            assert value == expected[name]


def test_packed_sparse_microbatch_uses_one_backbone_and_keeps_object_rows_separate():
    torch.manual_seed(8)
    backbone = CaptureBackbone()
    model = InstanceSuction3D(backbone=backbone, backbone_points=64, query_chunk_size=128).eval()
    observations = samples()
    expected = [model(item) for item in observations]
    backbone.calls.clear()
    actual = model.forward_instances(observations)
    assert len(backbone.calls) == 1
    packed = backbone.calls[0]
    assert packed['g_cam'].shape == (2, 3)
    assert packed['voxel_coords'][:, 0].tolist() == [0] * 64 + [1] * 64
    assert packed['voxel_features'].shape == (128, 12)
    assert packed['voxel_features'].dtype == torch.float32
    assert not packed['autocast_enabled']
    assert not torch.equal(actual[0]['q_inst'], actual[1]['q_inst'])
    for result, reference in zip(actual, expected):
        assert_scores(result, reference)


def test_identical_voxels_in_two_objects_are_distinct_sparse_coordinates():
    model = InstanceSuction3D(backbone=CaptureBackbone(), backbone_points=32).eval()
    obs = observation(side=15)
    encoded = model.encode_instances([obs, copy.deepcopy(obs)])
    coords = model.backbone.calls[0]['voxel_coords']
    assert len(torch.unique(coords, dim=0)) == 64
    torch.testing.assert_close(coords[:32, 1:], coords[32:, 1:])
    assert encoded[0]['source_index'] is not encoded[1]['source_index']
    assert encoded[0]['backbone_features'].shape == encoded[1]['backbone_features'].shape == (32, 256)
    torch.testing.assert_close(encoded[0]['context'], encoded[1]['context'])


def test_training_sparse_bn_sees_combined_voxels_and_updates_once():
    torch.manual_seed(9)
    model = InstanceSuction3D(backbone=CaptureBackbone(with_bn=True), backbone_points=1).train()
    observations = samples()
    output = model.forward_instances(observations)
    norm = model.backbone.norm
    assert len(model.backbone.calls) == 1
    assert norm.num_batches_tracked == 1
    assert norm.singleton_fallback_count == 0
    with torch.no_grad():
        pre_norm = model.backbone.projection(model.backbone.calls[0]['voxel_features'])
    torch.testing.assert_close(norm.running_mean, pre_norm.mean(0) * norm.momentum)
    torch.testing.assert_close(norm.running_var, torch.ones_like(norm.running_var) * (1 - norm.momentum)
                               + pre_norm.var(0, unbiased=True) * norm.momentum)
    sum(item['q_inst'].mean() for item in output).backward()
    assert torch.isfinite(model.backbone.projection.weight.grad).all()


def test_packed_dense_backward_matches_instance_mean_without_bn():
    torch.manual_seed(10)
    one = InstanceSuction3D(backbone=InjectedBackbone(), backbone_points=32, query_chunk_size=128)
    batch = copy.deepcopy(one)
    observations = samples()
    for obs in observations:
        result = one(obs)
        ((result['q_inst'] - .8).square().mean() / 2).backward()
    results = batch.forward_instances(observations)
    torch.stack([(result['q_inst'] - .8).square().mean() for result in results]).mean().backward()
    for (name, a), (other, b) in zip(one.named_parameters(), batch.named_parameters()):
        assert name == other
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=2e-4)


def test_invalid_instances_and_empty_queries_keep_batch_and_output_positions():
    model = InstanceSuction3D(backbone=CaptureBackbone(), backbone_points=32).eval()
    small = observation(side=7)
    empty = observation(side=5)
    empty.update(query_points=np.empty((0, 3)), query_normals=np.empty((0, 3)), query_ids=np.empty(0, int))
    actual = model.forward_instances([small, observation(side=15), empty])
    assert len(model.backbone.calls) == 1
    assert model.backbone.calls[0]['g_cam'].shape == (3, 3)
    assert model.backbone.calls[0]['voxel_coords'][:, 0].unique().tolist() == [1]
    assert actual[0]['valid'].tolist() == [False] * 3
    assert set(actual[0]['reason']) == {'insufficient_support'}
    assert actual[1]['valid'].all()
    assert actual[2]['q_inst'].shape == (0,) and actual[2]['reason'] == []
    assert model.forward_instances([]) == []


def test_total_dense_query_cap_splits_prepared_blocks_without_changing_ownership():
    model = InstanceSuction3D(backbone=CaptureBackbone(), backbone_points=32, query_chunk_size=4).eval()
    observations = samples()
    observations[1].update(query_points=np.tile(observations[1]['query_points'], (2, 1)),
                           query_normals=np.tile(observations[1]['query_normals'], (2, 1)),
                           query_ids=np.arange(6) + 20)
    prepared = [model.prepare_source_geometry(obs) for obs in observations]
    reference = model.forward_instances(prepared)
    seen = []
    hook = model.local_point_mlp.register_forward_pre_hook(lambda module, args: seen.append(len(args[0])))
    model.configure_execution(dense_query_chunk=3)
    actual = model.forward_instances(prepared)
    hook.remove()
    assert sum(seen) == 9 and max(seen) <= 3
    for result, expected in zip(actual, reference):
        assert_scores(result, expected)
    assert actual[1]['query_ids'].tolist() == list(range(20, 26))


def test_worker_output_survives_pickle_and_cannot_be_reused_with_replaced_support(monkeypatch):
    import models.instance_route_a as route_a
    model = InstanceSuction3D(backbone=InjectedBackbone(), backbone_points=32, query_chunk_size=2).eval()
    observations = samples()
    expected = model.forward_instances(observations)
    prepared = [pickle.loads(pickle.dumps(prepare_source_geometry_cpu(
        item, backbone_points=32, query_chunk_size=2))) for item in observations]
    monkeypatch.setattr(route_a, '_query_geometry', lambda *a, **k: pytest.fail('Rebuilt worker query geometry'))
    actual = model.forward_instances(prepared)
    for result, reference in zip(actual, expected):
        assert_scores(result, reference)
    wrong_source = dict(prepared[0], support_rgb=prepared[0]['support_rgb'].copy())
    with pytest.raises(ValueError, match='different source observation'):
        model.encode_instances([wrong_source])
    model.backbone_points = 33
    with pytest.raises(ValueError, match='different model settings'):
        model.encode_instances([prepared[0]])


def test_bf16_dense_keeps_geometry_double_sparse_float_and_evaluation_float():
    model = InstanceSuction3D(backbone=CaptureBackbone(), backbone_points=32, query_chunk_size=128).train()
    model.configure_execution(precision='bf16_dense', dense_query_chunk=32)
    observed = []
    hook = model.local_point_mlp[0].register_forward_hook(lambda m, a, out: observed.append(out.dtype))
    results = model.forward_instances(samples())
    assert observed == [torch.bfloat16]
    assert model.backbone.calls[0]['voxel_features'].dtype == torch.float32
    assert not model.backbone.calls[0]['autocast_enabled']
    assert all(item['q_inst'].dtype == torch.float32 and item['points'].dtype == torch.float64 for item in results)
    loss = torch.stack([(item['q_inst'] - .8).square().mean() + item['ell'].mean() for item in results]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    observed.clear()
    model.eval()
    bf16_config_eval = model.forward_instances(samples())
    assert observed == [torch.float32]
    model.configure_execution(precision='fp32', dense_query_chunk=32)
    fp32_eval = model.forward_instances(samples())
    for first, second in zip(bf16_config_eval, fp32_eval):
        assert_scores(first, second, atol=0, rtol=0)
    hook.remove()


@pytest.mark.parametrize('options', [{'precision': 'fp16'}, {'compile_mode': 'unknown'},
                                   {'dense_query_chunk': 0}, {'dense_query_chunk': True},
                                   {'dense_query_chunk': 1.5}])
def test_execution_options_are_explicit_and_validated(options):
    model = InstanceSuction3D(backbone=InjectedBackbone())
    with pytest.raises(ValueError):
        model.configure_execution(**options)


def test_optional_dense_compile_does_not_register_new_weight_names(monkeypatch):
    calls = []
    def compile_probe(module, **options):
        calls.append((module, options))
        return lambda value: module(value)
    monkeypatch.setattr(torch, 'compile', compile_probe)
    model = InstanceSuction3D(backbone=InjectedBackbone())
    keys = tuple(model.state_dict())
    parameters = [id(p) for p in model.parameters()]
    model.configure_execution(compile_mode='reduce-overhead')
    assert len(calls) == 10
    assert all(row[1] == {'mode': 'reduce-overhead', 'dynamic': True} for row in calls)
    assert tuple(model.state_dict()) == keys
    assert [id(p) for p in model.parameters()] == parameters
    assert not any(module is model.backbone for module, _ in calls)
    model.parameter_groups()
    model.configure_execution()
    assert model._compiled_dense == {}
