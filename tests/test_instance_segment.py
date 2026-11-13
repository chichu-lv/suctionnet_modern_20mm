"""Normal-path mathematics and integration for from-scratch instance reduction."""
import copy

import pytest
import torch
from torch import nn

from models.heads import compile_dense_modules, mlp
from models.instance_fusion import instance_layout, segment_broadcast
from models.suction20net import Suction20Net
from test_contact_grouping import FROZEN
from test_model import TinyTestBackbone
from test_voxel_heads import repeated_voxel_inputs


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_segment_broadcast_uses_complete_fp32_gradient_sums(dtype):
    # Unequal segments, background, an absent instance and multiple tiles.
    slots = torch.tensor([3, 1, 1, 3, 0, 3, 1, 1, 3, 0, 1])
    torch.manual_seed(9)
    features = torch.randn(5, 7, dtype=dtype, requires_grad=True)
    weights = torch.randn(len(slots), 7, dtype=dtype)
    layout = instance_layout(slots, len(features), tile_size=3)
    output = segment_broadcast(features, slots, layout)
    torch.testing.assert_close(output, features.detach()[slots], rtol=0, atol=0)
    (output * weights).sum().backward()
    expected = torch.stack([weights[slots == row].float().sum(0)
                            for row in range(len(features))]).to(dtype)
    torch.testing.assert_close(features.grad, expected, rtol=1e-6, atol=1e-6)
    assert layout.tiles_per_instance.tolist() == [1, 2, 0, 2, 0]


@pytest.mark.parametrize('checkpointed', [False, True])
def test_instance_segment_matches_pointwise_model_gradients(checkpointed):
    config = {'experiment': 'E2', 'dense_chunk_size': 5,
              'checkpoint_dense': checkpointed}
    torch.manual_seed(42)
    reference = Suction20Net(config, FROZEN, TinyTestBackbone(),
                            performance={'chunk_input_mode': 'split'})
    optimized = Suction20Net(config, FROZEN, TinyTestBackbone(), performance={
        'chunk_input_mode': 'split', 'instance_broadcast_mode': 'segment'})
    optimized.load_state_dict(reference.state_dict())
    inputs = repeated_voxel_inputs()
    # Different raw positions and instances share voxels; only the instance128
    # broadcast may be shared, not the full feature/relative-coordinate MLP.
    inputs['xyz_cam_m'][1] += torch.tensor([.002, -.003, .001])
    weights = torch.linspace(-.7, .8, len(inputs['xyz_cam_m']))
    results = []
    for model in (reference, optimized):
        data = dict(inputs, voxel_features=inputs['voxel_features'].clone().requires_grad_())
        encoded = model.encode_scene(data)
        q0, axis = model.predict_dense(encoded)
        objective = (q0 * weights).sum() + encoded['F_fused'].square().mean()
        objective.backward()
        results.append((encoded['F_fused'].detach(), q0.detach(), axis.detach(),
                        data['voxel_features'].grad,
                        {name: parameter.grad for name, parameter in model.named_parameters()}))
    for first, second in zip(results[0][:-1], results[1][:-1]):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
    for name, expected in results[0][-1].items():
        actual = results[1][-1][name]
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    assert optimized.instance_module.group_points


def test_dense_compile_keeps_state_dict_and_backward(monkeypatch):
    # Exercise Dynamo/AOTAutograd without requiring a CUDA compiler in CPU CI.
    # Production leaves the default Inductor backend selected by Module.compile.
    original_compile = torch.compile
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs:
                        original_compile(fn, backend='aot_eager', dynamic=True))
    torch.manual_seed(17)
    expected = nn.ModuleDict({'dense': mlp((8, 16, 4)), 'outside': nn.Linear(4, 1)})
    actual = copy.deepcopy(expected)
    keys = tuple(actual.state_dict())
    compile_dense_modules(actual, ['dense'])
    assert tuple(actual.state_dict()) == keys
    assert actual['outside']._compiled_call_impl is None
    inputs = torch.randn(11, 8)
    reference = expected['outside'](expected['dense'](inputs))
    output = actual['outside'](actual['dense'](inputs))
    torch.testing.assert_close(output, reference)
    reference.square().mean().backward()
    output.square().mean().backward()
    for parameter, other in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(parameter.grad, other.grad)
    restored = copy.deepcopy(expected)
    restored.load_state_dict(actual.state_dict(), strict=True)
