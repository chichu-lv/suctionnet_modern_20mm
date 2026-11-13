"""Normal-path coverage for explicit fresh-training FP32 geometry and threads."""
import os

import numpy as np
import pytest
import torch

from models.contact_grouping import ContactGrouping, ZONE_NAMES
from models.geometry_ops import fps_backend_identity, make_fps_backend
from models.spatial_index import SpatialIndex
from suction20.cpu_geometry import CpuGeometry
from test_contact_grouping import FROZEN, scene


def test_compiled_fp32_fps_uses_float32_distances():
    native = pytest.importorskip('models._fps_cpu')
    assert native.FPS_FLOAT32_VERSION == 1
    xyz = np.random.default_rng(17).normal(size=(180, 3)).astype(np.float32)
    rows = np.arange(3, 173, dtype=np.int64)
    compiled = make_fps_backend('compiled', precision='float32')
    reference = make_fps_backend('numpy', precision='float32')
    actual = compiled(xyz, rows, 32, first=19)
    np.testing.assert_array_equal(actual, reference(xyz, rows, 32, first=19))
    assert len(np.unique(actual)) == 32 and actual[0] == 19
    assert fps_backend_identity('compiled', precision='float32')['precision'] == 'float32'


def test_fp32_grid_keeps_complete_raw_neighborhoods_without_xyz64_copy():
    pytest.importorskip('models._spatial_index_cpu')
    xyz = np.random.default_rng(18).uniform(-.025, .025, (1800, 3)).astype(np.float32)
    inputs = scene(np.concatenate((xyz, xyz)), batch=[0]*len(xyz) + [1]*len(xyz))
    inputs['point_offsets'] = torch.tensor([0, len(xyz), 2*len(xyz)])
    geometry = CpuGeometry.from_inputs(inputs)
    index = SpatialIndex.from_cpu_geometry(geometry, backend='uniform_grid_cpu', precision='float32')
    assert index.xyz is geometry.xyz_cam_m and index.xyz.dtype == np.float32
    assert 'xyz64' not in vars(geometry)
    for center, radius in (([0., 0., 0.], .05), ([-.02, 0., .02], .018), ([.02, -.02, 0.], .025)):
        delta = xyz - np.asarray(center, dtype=np.float32)
        distance = (delta[:, 0]**2 + delta[:, 1]**2) + delta[:, 2]**2
        limit = np.float32(radius + 1e-8)
        expected = np.flatnonzero(distance <= limit*limit)
        for frame in (0, 1):
            np.testing.assert_array_equal(index.ball(frame, center, radius), expected + frame*len(xyz))
    assert len(index.ball(0, [0., 0., 0.], .05)) == len(xyz)


def test_fp32_native_contact_preserves_zones_and_parallel_results():
    pytest.importorskip('models._spatial_index_cpu')
    # Ordinary dense contact surface, with points 2 micrometers either side
    # of radial and segment boundaries. Count all points before FPS limits.
    core = np.column_stack((np.linspace(-.005, .005, 200), np.zeros((200, 2))))
    angle = np.repeat((np.arange(8) + .5)*2*np.pi/8, 10)
    lip = np.column_stack((.020*np.cos(angle), .020*np.sin(angle), np.zeros(80)))
    radial = [[.018-2e-6, 0., 0.], [.018+2e-6, 0., 0.],
              [.022-2e-6, 0., 0.], [.022+2e-6, 0., 0.]]
    obstacles = [[0., 0., .012], [0., 0., .025-2e-6], [0., 0., .025+2e-6],
                 [0., 0., .04], [0., 0., .10]]
    xyz = np.concatenate((core, lip, radial, obstacles))
    instance = [1]*284 + [2]*5
    inputs = scene(np.concatenate((xyz, xyz)), instance*2,
                   [0]*len(xyz) + [1]*len(xyz))
    index = SpatialIndex.from_inputs(inputs, backend='uniform_grid_cpu', precision='float32')
    frames = torch.tensor([0, 1]*8)
    args = (torch.zeros(16, 3), torch.tensor([[0., 0., 1.]]).repeat(16, 1),
            torch.ones(16, dtype=torch.long), frames, torch.ones(16, dtype=torch.bool))
    expected_counts = dict(core=201, lip=82, cup=2, mount=2, approach=1)
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    baseline = None
    for workers in (1, 4, 8):
        group = ContactGrouping(FROZEN, contact_query_backend='native_cpu',
                                spatial_index={'backend': 'uniform_grid_cpu'},
                                geometry_precision='float32', geometry_threads=workers)
        actual = group.query(index, *args, random_roll=False)
        assert group.last_query_timings['native_workers'] == min(workers, available, 16)
        for name in ZONE_NAMES:
            value = actual[name]
            np.testing.assert_array_equal(value['counts'], np.full(16, expected_counts[name]))
            assert np.isfinite(value['geometry']).all()
            for row, frame in enumerate(frames.tolist()):
                chosen = value['indices'][row, value['mask'][row]]
                assert len(np.unique(chosen)) == len(chosen)
                assert np.all((chosen >= frame*len(xyz)) & (chosen < (frame+1)*len(xyz)))
            if baseline is not None:
                for field in ('indices', 'mask', 'counts', 'geometry'):
                    np.testing.assert_array_equal(value[field], baseline[name][field])
        if baseline is not None:
            np.testing.assert_array_equal(actual['lip_sector_counts'], baseline['lip_sector_counts'])
        baseline = actual
