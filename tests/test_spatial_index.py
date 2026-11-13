"""Exact grid membership, ownership and unchanged downstream geometry contracts."""
import gc
import hashlib
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from models.contact_grouping import ContactGrouping, SpatialIndex, ZONE_NAMES
from models.geometry_ops import make_fps_backend
from models.spatial_index import CpuGridIndex, spatial_index_backend_identity
from suction20.cpu_geometry import CpuGeometry
from test_contact_grouping import FROZEN, scene


@pytest.fixture(autouse=True)
def native_grid():
    return pytest.importorskip('models._spatial_index_cpu', reason='optional grid extension not built')


def geometry(xyz, offsets):
    return SimpleNamespace(xyz64=np.ascontiguousarray(xyz, dtype=np.float64),
                           instance_id_input=np.ones(len(xyz), dtype=np.int64),
                           normal_obs=np.tile([0., 0., 1.], (len(xyz), 1)),
                           normal_obs_valid=np.ones(len(xyz), dtype=bool),
                           point_offsets=np.asarray(offsets, dtype=np.int64))


def test_grid_exact_fp64_boundaries_negative_cells_global_rows_and_empty_frames():
    rng = np.random.default_rng(42)
    radius = .03
    expanded = radius+1e-8
    boundary = np.array([[expanded, 0, 0], [np.nextafter(expanded, np.inf), 0, 0],
                         [-expanded, 0, 0], [-np.nextafter(expanded, np.inf), 0, 0],
                         [-.02, -.02, -.02], [0., 0., 0.]])
    xyz = np.concatenate((rng.uniform(-.1, .1, (2000, 3)), boundary))
    ctx = geometry(np.concatenate((xyz, xyz)), [0, len(xyz), len(xyz), 2*len(xyz)])
    index = CpuGridIndex.from_cpu_geometry(ctx)
    assert index.xyz64 is ctx.xyz64
    assert [index.frame_has_points(i) for i in range(3)] == [True, False, True]
    for frame in range(3):
        start, stop = ctx.point_offsets[frame:frame+2]
        for center in (np.zeros(3), np.array([-.02, -.02, -.02]), np.ones(3)):
            delta = ctx.xyz64[start:stop]-center
            distance = (delta[:, 0]**2+delta[:, 1]**2)+delta[:, 2]**2
            expected = np.flatnonzero(distance <= expanded*expanded)+start
            actual = index.ball(frame, center, radius)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.int64 and actual.flags.c_contiguous
    assert 'tree_build_s' not in index.setup_timings
    assert index.setup_timings['index_build_s'] >= 0
    assert index.setup_timings['spatial_index_backend'] == 'uniform_grid_cpu'


def test_native_pins_coordinates_and_query_buffers_outlive_index(native_grid):
    xyz = np.array([[0., 0., 0.], [.02, 0., 0.], [-.02, 0., 0.]], dtype=np.float64)
    ref = weakref.ref(xyz)
    handle = native_grid.create(xyz, 0, len(xyz), .02)
    del xyz
    gc.collect()
    assert ref() is not None  # Handle independently pins its Py_buffer exporter.
    first = np.frombuffer(native_grid.ball(handle, 0., 0., 0., .03), dtype=np.int64)
    second = np.frombuffer(native_grid.ball(handle, 0., 0., 0., 0.), dtype=np.int64)
    np.testing.assert_array_equal(first, [0, 1, 2])
    np.testing.assert_array_equal(second, [0])
    assert not np.shares_memory(first, second)
    del handle
    gc.collect()
    assert ref() is None
    np.testing.assert_array_equal(first, [0, 1, 2])
    np.testing.assert_array_equal(second, [0])


def test_grid_factory_and_all_contact_regions_match_kdtree_with_roll_rng():
    pytest.importorskip('models._fps_cpu')
    rng = np.random.default_rng(7)
    contact = np.column_stack((rng.uniform(-.022, .022, (800, 2)), rng.uniform(-.004, .004, 800)))
    obstacles = rng.uniform([-.025, -.025, .01], [.025, .025, .14], (240, 3))
    xyz = np.concatenate((contact, obstacles))
    inputs = scene(np.concatenate((xyz, xyz)), ([1]*800+[2]*240)*2,
                   [0]*len(xyz)+[1]*len(xyz))
    inputs['point_offsets'] = torch.tensor([0, len(xyz), 2*len(xyz)])
    kd = SpatialIndex.from_inputs(inputs)
    indices = [SpatialIndex.from_inputs(inputs, backend='uniform_grid_cpu'),
               SpatialIndex.from_cpu_geometry(CpuGeometry.from_inputs(inputs), backend='uniform_grid_cpu')]
    args = (torch.tensor([[0., 0., 0.], [.002, 0., 0.], [0., .002, 0.]]*2),
            torch.tensor([[0., 0., 1.]]*6), torch.ones(6, dtype=torch.long),
            torch.tensor([0, 1, 0, 1, 0, 1]), torch.ones(6, dtype=torch.bool))
    group = ContactGrouping(FROZEN, fps_fn=make_fps_backend('compiled'), detailed_cpu_timing=True)
    before = torch.get_rng_state()
    expected = group.query(kd, *args, random_roll=True)
    after = torch.get_rng_state()
    for index in indices:
        torch.set_rng_state(before)
        actual = group.query(index, *args, random_roll=True)
        assert torch.equal(torch.get_rng_state(), after)
        for name in ZONE_NAMES:
            for field in expected[name]:
                np.testing.assert_array_equal(actual[name][field], expected[name][field])
        np.testing.assert_array_equal(actual['lip_sector_counts'], expected['lip_sector_counts'])
        assert group.last_query_timings['ball_api_calls'] == 24
        assert group.last_query_timings['fps_calls'] > 0
        # Four retained balls remain independent, including compatibility batching.
        balls = index.balls(0, np.zeros((4, 3)), np.array([.01, .02, .03, .04]))
        for rows, radius in zip(balls, [.01, .02, .03, .04]):
            np.testing.assert_array_equal(rows, kd.ball(0, np.zeros(3), radius))


def test_grid_build_identity_records_source_binary_compiler_and_exact_flags():
    identity = spatial_index_backend_identity()
    assert identity['backend'] == 'uniform_grid_cpu' and identity['abi_version'] == 1
    assert identity['compile_flags'] == ['-O3', '-fno-fast-math', '-ffp-contract=off', '-pthread']
    assert identity['link_flags'] == ['-pthread']
    assert identity['source_sha256'] == hashlib.sha256(Path('models/csrc/spatial_index_cpu.cpp').read_bytes()).hexdigest()
    assert len(identity['binary_sha256']) == 64 and identity['compiler']


def test_native_rejects_invalid_buffers_and_nonfinite_geometry(native_grid):
    with pytest.raises(ValueError, match='float64'):
        native_grid.create(np.zeros((2, 3), dtype=np.float32), 0, 2, .02)
    with pytest.raises(ValueError, match='finite'):
        native_grid.create(np.full((2, 3), np.nan), 0, 2, .02)
    index = CpuGridIndex.from_cpu_geometry(geometry(np.zeros((2, 3)), [0, 2]))
    with pytest.raises(ValueError, match='finite'):
        index.ball(0, [np.nan, 0, 0], .02)


def test_contact_forward_uses_configured_grid_and_reuses_scene_index():
    inputs = scene([[0, 0, 0], [.02, 0, 0], [0, 0, .12]], [1, 1, 0])
    group = ContactGrouping(FROZEN, checkpointed=False,
                            spatial_index={'backend': 'uniform_grid_cpu', 'cell_size_m': .02})
    args = (torch.zeros((1, 3)), torch.tensor([[0., 0., 1.]]),
            torch.ones(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.bool))
    output = group(inputs, *args)
    index = inputs['spatial_index']
    assert isinstance(index, CpuGridIndex)
    assert 'tree_build_s' not in group.last_timings
    assert group.last_timings['spatial_index_backend'] == 'uniform_grid_cpu'
    assert output['zone_counts'].tolist() == [[1, 1, 0, 0, 1]]
    sum(output[name].sum() for name in ZONE_NAMES).backward()
    assert inputs['F_local'].grad.abs().sum() > 0
    group(inputs, *args)
    assert inputs['spatial_index'] is index
    assert group.last_timings['index_build_s'] == 0
