"""Bounded checks for the paired diagnostic's import and chain-rule mechanics."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('paired_pipeline_diagnostic',
    Path(__file__).resolve().parents[1] / 'scripts/benchmark_paired_pipeline.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def test_shared_backbone_dependency_allows_suffix_chunk_change(tmp_path):
    roots = [paired.Checkout(tmp_path / label) for label in ('baseline', 'candidate')]
    for index, checkout in enumerate(roots):
        (checkout.root / 'models').mkdir(parents=True)
        (checkout.root / 'models/heads.py').write_text(
            'def mlp(widths):\n    return widths\n\n'
            f'def chunk_apply(fn, values):\n    return fn(values) + {index}\n')
    result = paired.shared_backbone_dependencies(roots)
    assert len(result['models/heads.py::mlp_ast']) == 64
    (roots[1].root / 'models/heads.py').write_text('def mlp(widths):\n    return list(widths)\n')
    with pytest.raises(ValueError, match='Shared-backbone dependency changed'):
        paired.shared_backbone_dependencies(roots)


def test_checkout_context_keeps_lazy_imports_separate_and_restores_caller(tmp_path):
    contexts, modules = [], []
    original_models = sys.modules.get('models')
    for label in ('old', 'new'):
        root = tmp_path / label
        (root / 'models').mkdir(parents=True)
        (root / 'models/__init__.py').write_text('')
        (root / 'models/value.py').write_text(f'VALUE = {label!r}\n')
        (root / 'models/consumer.py').write_text('def read():\n    from models.value import VALUE\n    return VALUE\n')
        context = paired.Checkout(root)
        with context.active():
            modules.append(context.module('models.consumer'))
        contexts.append(context)
    for _ in range(2):
        for label, context, module in zip(('old', 'new'), contexts, modules):
            with context.active():
                assert module.read() == label
            assert sys.modules.get('models') is original_models


class ToyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6))
        self.head = nn.Linear(6, 2)

    def forward(self, inputs, *, sampler_mode, targets):
        return {'prediction': self.head(self.backbone(inputs))}


def toy_loss(prediction, targets, inputs):
    return {'total': (prediction['prediction'] - targets).square().mean()}


def test_sequential_suffix_cotangents_reproduce_full_backprop_through_one_graph():
    torch.manual_seed(42)
    original = ToyPipeline().train()
    optimized, direct = copy.deepcopy(original), copy.deepcopy(original)
    inputs, targets = torch.randn(8, 4), torch.randn(8, 2)
    scale, accumulation = 16., 4
    direct_loss = toy_loss(direct(inputs, sampler_mode='train', targets=targets), targets, inputs)['total']
    names, parameters = zip(*direct.named_parameters())
    direct_gradients = dict(zip(names, torch.autograd.grad(direct_loss * scale / accumulation, parameters)))
    features = original.backbone(inputs)
    results = [paired.run_suffix(model, features, inputs, targets, toy_loss,
        amp=False, gradient_scale=scale, accumulation=accumulation) for model in (original, optimized)]
    comparison = paired.Comparisons(amp=False)
    comparison.check(results[0], results[1], 'paired')
    assert not comparison.failures
    assert paired.gradient_health([results[0]['cotangent']])['nonzero']
    backbone_parameters = list(original.backbone.named_parameters())
    for index, result in enumerate(results):
        vjp = paired.backbone_vjp(features, backbone_parameters, result['cotangent'], retain_graph=index == 0)
        combined = {**result['suffix_gradients'], **{'backbone.' + key: value for key, value in vjp.items()}}
        for name, gradient in direct_gradients.items():
            torch.testing.assert_close(combined[name], gradient, atol=1e-6, rtol=1e-4)
    assert int(original.backbone[1].num_batches_tracked) == 1
    assert int(optimized.backbone[1].num_batches_tracked) == 0
    assert all(parameter.grad is None for parameter in original.parameters())


class ToyQuery:
    def __init__(self):
        self.index = object()
        self.calls = 0
        self.last_query_timings = {}
        self.last_timings = {}

    def query(self, index, points, normals, instances, batch, valid, *, random_roll=False):
        assert index is self.index
        self.calls += 1
        self.last_query_timings['calls'] = self.calls
        self.last_timings['query_calls'] = self.calls
        roll = torch.rand(len(points)) if random_roll else torch.zeros(len(points))
        geometry = np.zeros((len(points), 1, 7), dtype=np.float32)
        geometry[:, 0, :3] = points.numpy()
        geometry[:, 0, 3:6] = normals.numpy()
        geometry[:, 0, 6] = roll.numpy()
        return {'core': {'indices': np.arange(len(points), dtype=np.int64)[:, None],
                         'mask': valid.numpy()[:, None].copy(),
                         'geometry': geometry, 'counts': valid.numpy().astype(np.int64)},
                'lip_sector_counts': np.zeros((len(points), 8), dtype=np.int64)}


def captured_query_pair(*, change_axis):
    points = torch.tensor([[.01, .02, .40], [.03, .04, .42]])
    normals = torch.tensor([[.1, .2, .9746794], [.2, -.1, .9746794]])
    changed = normals.clone()
    if change_axis:
        changed[0, 0] = torch.nextafter(changed[0, 0], torch.tensor(float('inf')))
    instances, batch, valid = torch.tensor([1, 2]), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool)
    captures = []
    torch.manual_seed(42)
    initial = torch.get_rng_state().clone()
    for axes in (normals, changed):
        model, records = SimpleNamespace(contact_grouping=ToyQuery()), []
        paired.install_query_capture(model, records)
        torch.set_rng_state(initial)
        model.contact_grouping.query(model.contact_grouping.index, points, axes, instances, batch, valid, random_roll=True)
        captures.append(records)
    return captures


def test_query_replay_tolerates_axis_rounding_but_keeps_actual_selection_exact():
    baseline, candidate = captured_query_pair(change_axis=True)
    assert paired.fingerprint(baseline[0]['value']) != paired.fingerprint(candidate[0]['value'])
    actual = paired.fingerprint(candidate[0]['value'])
    grouping = candidate[0]['grouping']
    timings = copy.deepcopy((grouping.last_query_timings, grouping.last_timings))
    torch.rand(3)
    caller_rng = torch.get_rng_state().clone()
    comparisons = paired.Comparisons(amp=True)
    checks = paired.compare_query_captures(baseline, candidate, comparisons, 'query')
    assert not comparisons.failures
    assert checks == [dict(fixed_input_mode='replay_baseline_query', actual_input_equal=False,
        actual_rng_before_equal=True, fixed_input_output_exact=True, fixed_input_rng_after_exact=True)]
    assert grouping.calls == 2 and baseline[0]['grouping'].calls == 1
    assert torch.equal(torch.get_rng_state(), caller_rng)
    assert (grouping.last_query_timings, grouping.last_timings) == timings
    assert paired.fingerprint(candidate[0]['value']) == actual

    # Even a passing fixed-input replay cannot hide a changed actual selection.
    candidate[0]['value']['core']['indices'][0, 0] += 1
    comparisons = paired.Comparisons(amp=True)
    checks = paired.compare_query_captures(baseline, candidate, comparisons, 'query')
    assert checks[0]['fixed_input_output_exact']
    assert comparisons.failures == ['query.0.actual.discrete.core.indices.sha256']
    assert torch.equal(torch.get_rng_state(), caller_rng)
    assert (grouping.last_query_timings, grouping.last_timings) == timings


def test_identical_query_inputs_skip_replay_and_keep_descriptor_check_exact():
    baseline, candidate = captured_query_pair(change_axis=False)
    caller_rng = torch.get_rng_state().clone()
    comparisons = paired.Comparisons(amp=True)
    checks = paired.compare_query_captures(baseline, candidate, comparisons, 'query')
    assert not comparisons.failures
    assert checks[0]['fixed_input_mode'] == 'actual_same_input'
    assert candidate[0]['grouping'].calls == 1
    geometry = candidate[0]['value']['core']['geometry']
    geometry[0, 0, 3] = np.nextafter(geometry[0, 0, 3], np.float32(np.inf))
    comparisons = paired.Comparisons(amp=True)
    checks = paired.compare_query_captures(baseline, candidate, comparisons, 'query')
    assert not checks[0]['fixed_input_output_exact']
    assert comparisons.failures == ['query.0.fixed_input.core.geometry.sha256']
    assert candidate[0]['grouping'].calls == 1
    assert torch.equal(torch.get_rng_state(), caller_rng)
