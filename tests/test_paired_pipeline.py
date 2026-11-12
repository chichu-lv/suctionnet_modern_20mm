"""Bounded checks for the paired diagnostic's import and chain-rule mechanics."""
import copy
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

spec = importlib.util.spec_from_file_location('paired_pipeline_diagnostic',
    Path(__file__).resolve().parents[1] / 'scripts/benchmark_paired_pipeline.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


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
