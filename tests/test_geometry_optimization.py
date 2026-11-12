"""Small normal-path comparisons for exact FPS, queries and CPU model contexts."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from models import Suction20Net
from models.candidate_sampler import fps_indices
from models.contact_grouping import ContactGrouping, SpatialIndex, ZONE_NAMES
from models.geometry_ops import fps_backend_identity, make_fps_backend
from suction20.cpu_geometry import CpuGeometry, SamplerTargetsCPU
from test_contact_grouping import FROZEN, scene
from test_model import TinyTestBackbone, model_inputs


@pytest.fixture
def compiled_fps():
    pytest.importorskip('models._fps_cpu', reason='optional C extension not built')
    return make_fps_backend('compiled')


def test_compiled_fps_matches_reference_order_and_records_build(compiled_fps):
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(300, 3)).astype(np.float32)
    rows = rng.permutation(250)
    for count in (32, len(rows)):
        expected = fps_indices(xyz, rows, count, first=rows[9])
        np.testing.assert_array_equal(compiled_fps(xyz, rows, count, first=rows[9]), expected)
    identity = fps_backend_identity('compiled')
    assert identity['abi_version'] == 1
    assert identity['compile_flags'] == ['-O3', '-fno-fast-math', '-ffp-contract=off']
    assert identity['source_sha256'] == hashlib.sha256(Path('models/csrc/fps_cpu.c').read_bytes()).hexdigest()
    assert len(identity['binary_sha256']) == 64 and identity['compiler']


def test_bounded_four_ball_queries_match_reference_and_rng(compiled_fps):
    rng = np.random.default_rng(7)
    contact = np.column_stack((rng.uniform(-.022, .022, (800, 2)), rng.uniform(-.004, .004, 800)))
    obstacles = rng.uniform([-.025, -.025, .01], [.025, .025, .14], (240, 3))
    xyz = np.concatenate((contact, obstacles))
    inputs = scene(np.concatenate((xyz, xyz)), ([1]*800 + [2]*240)*2,
                   [0]*len(xyz) + [1]*len(xyz))
    inputs['point_offsets'] = torch.tensor([0, len(xyz), 2*len(xyz)])
    old_index = SpatialIndex.from_inputs(inputs)
    index = SpatialIndex.from_cpu_geometry(CpuGeometry.from_inputs(inputs))
    np.testing.assert_array_equal(index.xyz, old_index.xyz)
    group = ContactGrouping(FROZEN, fps_fn=compiled_fps, query_chunk_size=2, detailed_cpu_timing=True)
    args = (torch.tensor([[0., 0., 0.], [.002, 0., 0.], [0., .002, 0.]]*2),
            torch.tensor([[0., 0., 1.]]*6), torch.ones(6, dtype=torch.long),
            torch.tensor([0, 1, 0, 1, 0, 1]), torch.ones(6, dtype=torch.bool))
    torch.manual_seed(9)
    before = torch.get_rng_state()
    expected = group.query_reference(old_index, *args, random_roll=True)
    after = torch.get_rng_state()
    for workers in (1, 4):
        group.query_workers = workers
        torch.set_rng_state(before)
        actual = group.query(index, *args, random_roll=True)
        assert torch.equal(torch.get_rng_state(), after)
        for name in ZONE_NAMES:
            for field in expected[name]:
                np.testing.assert_array_equal(actual[name][field], expected[name][field])
        np.testing.assert_array_equal(actual['lip_sector_counts'], expected['lip_sector_counts'])
        assert group.last_query_timings['ball_calls'] == 24
        assert group.last_query_timings['ball_api_calls'] == 4  # Two chunks per frame.
        assert group.last_query_timings['fps_calls'] > 0
        assert all(group.last_query_timings[key] >= 0 for key in ('ball_query_s', 'region_filter_s', 'fps_s', 'descriptor_pack_s'))


def test_cpu_context_compiled_model_preserves_outputs_gradients_and_rng(compiled_fps):
    torch.manual_seed(42)
    cfg = {'experiment': 'E5', 'seed_count': 6, 'dense_chunk_size': 5, 'pose_chunk_size': 2,
           'checkpoint_dense': False}
    reference = Suction20Net(cfg, FROZEN, TinyTestBackbone()).train()
    optimized = Suction20Net(cfg, FROZEN, TinyTestBackbone(),
                            performance={'fps_backend': 'compiled', 'query_chunk_size': 2,
                                         'query_workers': 1, 'detailed_cpu_timing': True}).train()
    optimized.load_state_dict(copy.deepcopy(reference.state_dict()))
    # Reference retains pre-optimization scalar query and duplicate local geometry.
    reference.contact_grouping.query = reference.contact_grouping.query_reference
    inputs = model_inputs()
    inputs['point_offsets'] = torch.tensor([0, 8, 18])
    targets = {'quality_valid': inputs['normal_obs_valid'], 'gt_A': torch.linspace(0, 1, 18)}
    geometry = CpuGeometry.from_inputs(inputs)
    sampler_targets = SamplerTargetsCPU.from_targets(targets)
    rng = torch.get_rng_state()
    outputs, gradients, rngs = [], [], []
    for model in (reference, optimized):
        torch.set_rng_state(rng)
        context = {} if model is reference else dict(cpu_geometry=geometry, sampler_targets_cpu=sampler_targets)
        # CPU targets remain outside encode_scene and are sufficient for sampling.
        output = model(inputs, sampler_mode='train', targets=targets if model is reference else None, **context)
        sum(output[key].square().sum() for key in ('q0_all', 'normal_out_all', 'q', 'seal_aux', 'wrench_aux', 'collision_free_logit')).backward()
        outputs.append(output)
        gradients.append({n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()})
        rngs.append(torch.get_rng_state())
    assert torch.equal(rngs[0], rngs[1])
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    for name, grad in gradients[0].items():
        if grad is None:
            assert gradients[1][name] is None
        else:
            torch.testing.assert_close(grad, gradients[1][name], rtol=1e-6, atol=1e-7)
    assert optimized.last_grouping_timings['q0_d2h_host_s'] >= 0
    assert optimized.last_grouping_timings['sampler_cpu_s'] >= 0
