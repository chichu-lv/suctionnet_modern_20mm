"""Small normal-path comparisons for exact FPS, queries and CPU model contexts."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from models import Suction20Net
from models.candidate_sampler import fps_indices
from models.contact_grouping import ContactGrouping, SpatialIndex, ZONE_NAMES
from models.geometry_ops import fps_backend_identity, make_fps_backend
from suction20.cpu_geometry import CpuGeometry, SamplerTargetsCPU
from suction20.performance import load_performance_config
from test_contact_grouping import FROZEN, scene
from test_model import TinyTestBackbone, model_inputs


@pytest.fixture
def compiled_fps():
    pytest.importorskip('models._fps_cpu', reason='optional C extension not built')
    return make_fps_backend('compiled')


@pytest.fixture
def native_contact():
    module = pytest.importorskip('models._spatial_index_cpu', reason='optional native grid/contact extension not built')
    assert module.CONTACT_ABI_VERSION == 1
    assert callable(module.contact_query)
    return module


def two_frame_contact_scene():
    """A dense surface and normal obstacles, repeated in two packed frames."""
    rng = np.random.default_rng(7)
    contact = np.column_stack((rng.uniform(-.022, .022, (800, 2)), rng.uniform(-.004, .004, 800)))
    obstacles = rng.uniform([-.025, -.025, .01], [.025, .025, .14], (240, 3))
    xyz = np.concatenate((contact, obstacles))
    inputs = scene(np.concatenate((xyz, xyz)), ([1]*800 + [2]*240)*2,
                   [0]*len(xyz) + [1]*len(xyz))
    inputs['point_offsets'] = torch.tensor([0, len(xyz), 2*len(xyz)])
    observed = rng.normal([0., 0., 1.], [.08, .08, .02], (2*len(xyz), 3))
    observed /= np.linalg.norm(observed, axis=1, keepdims=True)
    inputs['normal_obs'] = torch.tensor(observed, dtype=torch.float32)
    inputs['normal_obs_valid'][::11] = False
    inputs['normal_obs'][~inputs['normal_obs_valid']] = 0
    args = (torch.tensor([[0., 0., 0.], [.002, 0., 0.], [0., .002, 0.]]*2),
            torch.tensor([[.08, -.04, 1.], [0., 0., 1.], [-.06, .05, 1.],
                          [.03, .07, 1.], [0., 0., 1.], [.08, -.04, 1.]]),
            torch.ones(6, dtype=torch.long), torch.tensor([0, 1, 0, 1, 0, 1]),
            torch.tensor([True, True, True, True, True, False]))
    return inputs, args


def assert_query_equal(expected, actual):
    for name in ZONE_NAMES:
        for field in ('indices', 'mask', 'geometry', 'counts'):
            np.testing.assert_array_equal(actual[name][field], expected[name][field],
                                          err_msg=f'{name}.{field}')
            if field == 'geometry':
                # Numerical equality hides signed-zero drift caught by the real
                # paired query comparisons. Descriptor identity is byte-exact.
                assert actual[name][field].tobytes() == expected[name][field].tobytes(), name
    np.testing.assert_array_equal(actual['lip_sector_counts'], expected['lip_sector_counts'])


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
    assert identity['source_version'] == 'fps_cpu_v1'
    assert identity['binary_filename'].startswith('_fps_cpu.') and identity['compiler']


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


def test_native_contact_matches_p2_queries_and_rng(compiled_fps, native_contact):
    inputs, args = two_frame_contact_scene()
    geometry = CpuGeometry.from_inputs(inputs)
    index = SpatialIndex.from_cpu_geometry(geometry, backend='uniform_grid_cpu', cell_size_m=.02)
    common = dict(fps_fn=compiled_fps, query_chunk_size=None, query_workers=1,
                  detailed_cpu_timing=True,
                  spatial_index={'backend': 'uniform_grid_cpu', 'cell_size_m': .02})
    p2 = ContactGrouping(FROZEN, **common)
    native = ContactGrouping(FROZEN, contact_query_backend='native_cpu', **common)
    torch.manual_seed(9)
    before = torch.get_rng_state()
    expected = p2.query(index, *args, random_roll=True)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    actual = native.query(index, *args, random_roll=True)
    assert torch.equal(torch.get_rng_state(), after)
    assert_query_equal(expected, actual)
    assert 'native_query_wall_s' in native.last_query_timings
    assert native.last_query_timings['fps_calls'] > 0
    # The sixth pose is padding, but still participates in the one roll draw.
    assert not actual['lip_sector_counts'][-1].any()
    for name in ZONE_NAMES:
        assert not actual[name]['mask'][-1].any()
        assert actual[name]['counts'][-1] == 0
    # Exercise both sides of observation-normal validity in packed descriptors.
    selected_validity = np.concatenate([actual[name]['geometry'][actual[name]['mask']][:, 6]
                                        for name in ZONE_NAMES])
    assert np.any(selected_validity == 0) and np.any(selected_validity == 1)


@pytest.mark.parametrize('random_roll', [False, True])
def test_native_contact_anchor_signed_zero_matches_reference_bytes(compiled_fps, native_contact, random_roll):
    inputs = scene([[.125, -.25, .5]], [1])
    # A valid unit observed normal with signed-zero components also exercises
    # the normal-dot descriptor for an orthogonal candidate axis.
    inputs['normal_obs'] = torch.tensor([[-1., -0., -0.]])
    index = SpatialIndex.from_inputs(inputs, backend='uniform_grid_cpu')
    axes = [[x, y, z] for x in (-.1, .1) for y in (-.2, .2) for z in (-1., 1.)]
    axes += [[0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]
    count = len(axes)
    args = (inputs['xyz_cam_m'].repeat(count, 1), torch.tensor(axes),
            torch.ones(count, dtype=torch.long), torch.zeros(count, dtype=torch.long),
            torch.ones(count, dtype=torch.bool))
    reference = ContactGrouping(FROZEN, fps_fn=compiled_fps)
    native = ContactGrouping(FROZEN, fps_fn=compiled_fps, contact_query_backend='native_cpu')
    torch.manual_seed(19)
    before = torch.get_rng_state()
    expected = reference.query(index, *args, random_roll=random_roll)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    actual = native.query(index, *args, random_roll=random_roll)
    assert torch.equal(torch.get_rng_state(), after)
    assert_query_equal(expected, actual)
    assert np.all(expected['core']['geometry'][:, 0, :3].view(np.uint32) == 0)
    assert np.all(actual['core']['counts'] == 1)


def test_cpu_context_compiled_model_preserves_outputs_gradients_and_rng(compiled_fps, native_contact):
    torch.manual_seed(42)
    cfg = {'experiment': 'E5', 'seed_count': 6, 'dense_chunk_size': 5, 'pose_chunk_size': 2,
           'checkpoint_dense': False}
    reference = Suction20Net(cfg, FROZEN, TinyTestBackbone()).train()
    optimized = Suction20Net(cfg, FROZEN, TinyTestBackbone(),
                            performance={'fps_backend': 'compiled', 'query_chunk_size': 2,
                                         'query_workers': 1, 'detailed_cpu_timing': True}).train()
    optimized.load_state_dict(copy.deepcopy(reference.state_dict()))
    native = Suction20Net(cfg, FROZEN, TinyTestBackbone(),
                         performance={'fps_backend': 'compiled', 'query_chunk_size': None,
                                      'query_workers': 1, 'detailed_cpu_timing': True,
                                      'contact_query_backend': 'native_cpu',
                                      'spatial_index': {'backend': 'uniform_grid_cpu', 'cell_size_m': .02}}).train()
    native.load_state_dict(copy.deepcopy(reference.state_dict()))
    settings = load_performance_config(Path(__file__).parents[1] / 'configs/performance/cd009-p3-integrated.json')
    settings['dense_chunk_size_override'] = 9  # Small ordinary scene still spans multiple chunks.
    integrated = Suction20Net(cfg, FROZEN, TinyTestBackbone(), performance=settings).train()
    integrated.load_state_dict(copy.deepcopy(reference.state_dict()))
    assert cfg['dense_chunk_size'] == 5 and integrated.model_config == cfg
    assert integrated.execution_settings == {
        'profile_dense_chunk_size': 5, 'effective_dense_chunk_size': 9,
        'dense_chunk_override_source': 'performance_sidecar', 'chunk_input_mode': 'split'}
    assert integrated.contact_grouping.chunk_size == 2
    assert integrated.contact_grouping.chunk_input_mode == 'split'
    # Reference retains pre-optimization scalar query and duplicate local geometry.
    reference.contact_grouping.query = reference.contact_grouping.query_reference
    inputs = model_inputs()
    inputs['point_offsets'] = torch.tensor([0, 8, 18])
    targets = {'quality_valid': inputs['normal_obs_valid'], 'gt_A': torch.linspace(0, 1, 18)}
    geometry = CpuGeometry.from_inputs(inputs)
    sampler_targets = SamplerTargetsCPU.from_targets(targets)
    rng = torch.get_rng_state()
    outputs, gradients, rngs = [], [], []
    for model in (reference, optimized, native, integrated):
        torch.set_rng_state(rng)
        context = {} if model is reference else dict(cpu_geometry=geometry, sampler_targets_cpu=sampler_targets)
        # CPU targets remain outside encode_scene and are sufficient for sampling.
        output = model(inputs, sampler_mode='train', targets=targets if model is reference else None, **context)
        sum(output[key].square().sum() for key in ('q0_all', 'normal_out_all', 'q', 'seal_aux', 'wrench_aux', 'collision_free_logit')).backward()
        outputs.append(output)
        gradients.append({n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()})
        rngs.append(torch.get_rng_state())
    for i, model in enumerate((optimized, native, integrated), start=1):
        assert torch.equal(rngs[0], rngs[i])
        assert reference.state_dict().keys() == model.state_dict().keys()
        for key in outputs[0]:
            # Changing GEMM chunk shapes can change floating-point reduction
            # order; existing geometry-only comparisons remain exact.
            tolerance = dict(rtol=1e-4, atol=1e-6) if model is integrated else dict(rtol=0, atol=0)
            torch.testing.assert_close(outputs[0][key], outputs[i][key], **tolerance)
        for name, grad in gradients[0].items():
            if grad is None:
                assert gradients[i][name] is None
            else:
                tolerance = dict(rtol=1e-4, atol=1e-6) if model is integrated else dict(rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(grad, gradients[i][name], **tolerance)
    assert optimized.last_grouping_timings['q0_d2h_host_s'] >= 0
    assert optimized.last_grouping_timings['sampler_cpu_s'] >= 0
    assert 'native_query_wall_s' in native.last_grouping_timings
