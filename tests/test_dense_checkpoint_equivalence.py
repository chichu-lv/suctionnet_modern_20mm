"""CPU equivalence of dense activation recomputation; no sparse CUDA claim."""
import copy

import pytest
import torch

from models import Suction20Net
from test_contact_grouping import FROZEN
from test_model import TinyTestBackbone, model_inputs


@pytest.mark.parametrize('experiment', ['E3', 'E5'])
def test_dense_checkpoint_toggle_preserves_training_outputs_gradients_and_rng(experiment):
    torch.manual_seed(42)
    checked = Suction20Net({'experiment': experiment, 'seed_count': 6,
                           'dense_chunk_size': 5, 'pose_chunk_size': 2,
                           'checkpoint_dense': True}, FROZEN, TinyTestBackbone()).train()
    direct = Suction20Net(dict(checked.model_config, checkpoint_dense=False),
                         FROZEN, TinyTestBackbone()).train()
    direct.load_state_dict(copy.deepcopy(checked.state_dict()))
    inputs = model_inputs()
    targets = {'quality_valid': inputs['normal_obs_valid'],
               'gt_A': torch.linspace(0, 1, len(inputs['xyz_cam_m']))}
    rng = torch.get_rng_state()
    results = []
    for model in (checked, direct):
        torch.set_rng_state(rng)
        output = model(inputs, sampler_mode='train', targets=targets)
        keys = ['q0_all', 'q', 'normal_out_all']
        if experiment == 'E5':
            keys += ['seal_aux', 'wrench_aux', 'collision_free_logit']
        sum(output[key].square().sum() for key in keys).backward()
        results.append((output, {name: None if p.grad is None else p.grad.clone()
                                 for name, p in model.named_parameters()}, torch.get_rng_state()))
    first, second = results
    assert not torch.equal(rng, first[2]), 'Exercise stochastic training sampling/group rolls'
    assert torch.equal(first[2], second[2])
    for key in first[0]:
        torch.testing.assert_close(first[0][key], second[0][key], rtol=0, atol=0)
    assert first[1].keys() == second[1].keys()
    for name, gradient in first[1].items():
        if gradient is None:
            assert second[1][name] is None
        else:
            torch.testing.assert_close(gradient, second[1][name], rtol=1e-6, atol=1e-7)
