"""Exact source-order scan, contiguous FPS and optional native stage profile."""
import numpy as np
import pytest

from models import instance_route_a_geometry as geometry
from test_instance_route_a_native import compare_contacts, encoded_support


@pytest.fixture(autouse=True)
def require_current_native():
    native = pytest.importorskip('models._spatial_index_cpu')
    if getattr(native, 'ROUTE_A_CONTACT_PROFILE_VERSION', None) != 1:
        pytest.skip('build the current Route-A contact extension')


@pytest.mark.parametrize('count', [32768, 32769])
def test_ordered_compact_scan_and_grid_cutover_preserve_reference(count):
    rng = np.random.default_rng(19)
    points = rng.uniform(-.06, .06, (count, 3))
    encoded = encoded_support(points, source_ids=np.arange(count, dtype=np.int64),
                              components=rng.integers(0, 2, count))
    centers = points[[100, 1000, 8000]]
    normals = np.array([[0., 0., 1.], [.36, .48, .8], [0., -.6, .8]])
    compare_contacts(encoded, centers, normals, encoded['components_np'][[100, 1000, 8000]],
                     np.ones(3, bool), workers=2)


def test_optional_stage_profile_preserves_outputs_and_reports_cpu_work():
    rng = np.random.default_rng(4)
    points = rng.uniform(-.04, .04, (4096, 3))
    encoded = encoded_support(points, source_ids=np.arange(len(points), dtype=np.int64))
    centers = np.zeros((4, 3))
    normals = np.tile([0., 0., 1.], (4, 1))
    args = (encoded, centers, normals, np.zeros(4, np.int64), np.ones(4, bool))
    plain = geometry.native_contact_geometry(*args)
    profile = {}
    detailed = geometry.native_contact_geometry(*args, workers=2, profile=profile)
    for left, right in zip(plain, detailed):
        np.testing.assert_array_equal(left, right)
    assert set(profile) == {'query_ms', 'regions_ms', 'lip_ms', 'fps_and_pack_ms'}
    assert all(np.isfinite(value) and value > 0 for value in profile.values())


def test_contiguous_fps_keeps_first_source_id_across_argmax_lanes():
    x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))) * .001
    points = np.repeat(points, 2, axis=0)
    encoded = encoded_support(points, source_ids=np.arange(len(points), dtype=np.int64))
    compare_contacts(encoded, np.zeros((1, 3)), np.array([[0., 0., 1.]]),
                     np.zeros(1, np.int64), np.ones(1, bool))


def test_neighbor_index_reuses_tree_only_when_row_identity_matches():
    points = np.array([[0., 0., 0.], [.001, 0., 0.], [-.001, 0., 0.]])
    ordered = geometry.NeighborIndex(points, np.arange(3), np.zeros(3, np.int64))
    assert ordered.all_tree is ordered.groups[0][1]
    shuffled = geometry.NeighborIndex(points, np.array([20, 30, 10]), np.zeros(3, np.int64))
    assert shuffled.all_tree is not shuffled.groups[0][1]
    for index in (ordered, shuffled):
        anchors, distances = index.nearest_anchor(points)
        np.testing.assert_array_equal(anchors, np.arange(3))
        np.testing.assert_array_equal(distances, np.zeros(3))
