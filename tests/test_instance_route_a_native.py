"""A's native grid/contact engine versus its FP64 three-region reference."""
import pickle

import numpy as np
import pytest

from models import instance_route_a_geometry as geometry


@pytest.fixture(autouse=True)
def require_native():
    native = pytest.importorskip('models._spatial_index_cpu')
    if getattr(native, 'ROUTE_A_CONTACT_ABI_VERSION', None) != 1:
        pytest.skip('build the optional Route-A contact extension')
    assert geometry._FPS_AXES is not None


def encoded_support(points, *, seed=42, source_ids=None, components=None):
    points = np.asarray(points, np.float64)
    rng = np.random.default_rng(seed)
    count = len(points)
    if source_ids is None:
        source_ids = rng.permutation(count).astype(np.int64) * 11 + 7
    if components is None:
        components = np.zeros(count, np.int64)
    normals = rng.normal(size=points.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return dict(points_np=points, source_ids_np=np.asarray(source_ids, np.int64),
                components_np=np.asarray(components, np.int64), normals_np=normals,
                normal_valid_np=rng.random(count) > .15, rgb_valid_np=rng.random(count) > .2,
                raw_feature_valid_np=rng.random(count) > .25,
                gravity_np=np.array([.0, .6, -.8]),
                source_index=geometry.NeighborIndex(points, source_ids, components))


def compare_contacts(encoded, points, normals, components, valid, *, workers=1):
    reference = geometry.reference_contact_geometry(encoded, points, normals, components, valid)
    actual = geometry.native_contact_geometry(encoded, points, normals, components, valid, workers=workers)
    assert actual is not None
    for idx, (left, right) in enumerate(zip(actual, reference)):
        assert left.dtype == right.dtype and left.shape == right.shape
        if idx in (0, 1, 3, 5):
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, rtol=2e-13, atol=2e-13)
    return actual


@pytest.mark.parametrize('count', [0, 1, 31, 257, 3000])
def test_native_contacts_match_current_a_random_and_partial_valid(count):
    rng = np.random.default_rng(100 + count)
    points = rng.uniform(-.035, .035, (count, 3))
    encoded = encoded_support(points, components=rng.integers(0, 3, count))
    centers = rng.uniform(-.01, .01, (17, 3))
    normals = rng.normal(size=centers.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    components = rng.integers(-1, 4, len(centers))
    valid = rng.random(len(centers)) > .2
    centers[~valid], normals[~valid] = np.nan, np.nan
    compare_contacts(encoded, centers, normals, components, valid)
    compare_contacts(encoded, centers[:0], normals[:0], components[:0], valid[:0])


def test_native_contact_threads_are_deterministic_and_handles_do_not_escape():
    rng = np.random.default_rng(908)
    encoded = encoded_support(rng.uniform(-.04, .04, (8000, 3)))
    points = rng.uniform(-.02, .02, (33, 3))
    normals = rng.normal(size=points.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    components, valid = np.zeros(len(points), np.int64), np.ones(len(points), bool)
    before = set(encoded)
    first = compare_contacts(encoded, points, normals, components, valid)
    threaded = compare_contacts(encoded, points, normals, components, valid, workers=3)
    for actual, expected in zip(threaded, first):
        np.testing.assert_array_equal(actual, expected)
    assert set(encoded) == before
    restored = pickle.loads(pickle.dumps(encoded))
    assert isinstance(restored['source_index'], geometry.NeighborIndex)
    for name in encoded:
        if isinstance(encoded[name], np.ndarray):
            np.testing.assert_array_equal(restored[name], encoded[name])


def test_native_contacts_keep_fps_when_region_smaller_than_budget_and_xyz_tied():
    # The old five-zone engine skips FPS when M <= K. A must still return the
    # current FPS order, and never duplicate a selected source row.
    points = np.array([[.001, 0., 0.], [.002, 0., 0.], [-.003, 0., 0.],
                       [.001, 0., 0.], [0., .004, 0.], [0., -.004, 0.]])
    ids = np.array([90, 10, 50, 30, 20, 40])
    encoded = encoded_support(points, source_ids=ids)
    actual = compare_contacts(encoded, np.zeros((1, 3)), np.array([[0., 0., 1.]]),
                              np.zeros(1, np.int64), np.ones(1, bool))
    selected = actual[0][0, actual[1][0]]
    assert len(np.unique(selected)) == len(points)
    assert ids[selected[0]] == ids.min()
    assert not np.array_equal(ids[selected], np.sort(ids))


def test_native_region_boundaries_and_roundoff_guard_use_original_reference(monkeypatch):
    tolerance = geometry.BOUNDARY_ROUNDOFF_M
    radii = [0., .010, .018, .022]
    offsets = [0., -.5*tolerance, .5*tolerance, -tolerance, tolerance,
               -1.5*tolerance, 1.5*tolerance]
    points = [[max(0., radius+offset), 0., height]
              for radius in radii for offset in offsets
              for height in (0., .040, -.040, .040+tolerance, -.040-tolerance,
                             .040+2*tolerance, -.040-2*tolerance)]
    encoded = encoded_support(points)
    original = geometry.reference_contact_geometry
    calls = []

    def record_reference(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(geometry, 'reference_contact_geometry', record_reference)
    actual = geometry.native_contact_geometry(encoded, np.zeros((1, 3)), np.array([[0., 0., 1.]]),
                                              np.zeros(1, np.int64), np.ones(1, bool))
    assert calls == [1]
    expected = original(encoded, np.zeros((1, 3)), np.array([[0., 0., 1.]]),
                        np.zeros(1, np.int64), np.ones(1, bool))
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_native_angular_support_threshold_and_raw_lip_before_fps(monkeypatch):
    # Seven angular sites sit around the validity threshold; hundreds of raw
    # lip points ensure summaries cannot be incorrectly computed after FPS.
    angles = np.r_[0., 1e-6, 2e-6 + 1e-14, np.linspace(.4, 5.8, 300)]
    height = .003*np.sin(angles) + .001*np.cos(2*angles)
    points = np.column_stack((.020*np.cos(angles), .020*np.sin(angles), height))
    encoded = encoded_support(points)
    calls = []
    original = geometry.reference_contact_geometry

    def record_reference(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(geometry, 'reference_contact_geometry', record_reference)
    actual = geometry.native_contact_geometry(encoded, np.zeros((1, 3)), np.array([[0., 0., 1.]]),
                                              np.zeros(1, np.int64), np.ones(1, bool))
    assert calls == [1]
    expected = original(encoded, np.zeros((1, 3)), np.array([[0., 0., 1.]]),
                        np.zeros(1, np.int64), np.ones(1, bool))
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    assert actual[3][0, 2] == len(points)
    assert actual[1][0, 64:].sum() == 64
    assert actual[4][0, 0] == np.log1p(len(points))


def test_native_large_support_and_negative_camera_grid_cells():
    rng = np.random.default_rng(529)
    points = rng.uniform(-.04, .04, (32769, 3)) + [-.6, -.2, .4]
    components = rng.integers(0, 2, len(points))
    encoded = encoded_support(points, components=components)
    centers = np.array([[-.6, -.2, .4], [-.62, -.21, .42]])
    normals = np.array([[0., 0., 1.], [.36, .48, .8]])
    compare_contacts(encoded, centers, normals, np.array([0, 1]), np.ones(2, bool))


@pytest.mark.parametrize('missing', ['extension', 'abi', 'fps_arithmetic'])
def test_native_extension_unavailable_has_explicit_reference_fallback(monkeypatch, missing):
    if missing == 'extension':
        monkeypatch.setattr(geometry, '_spatial_index_cpu', None)
    elif missing == 'abi':
        monkeypatch.setattr(geometry, '_spatial_index_cpu', object())
    else:
        monkeypatch.setattr(geometry, '_FPS_AXES', None)
    # Fallback decision is independent of observation allocation/validation.
    assert geometry.native_contact_geometry({}, None, None, None, None) is None


def test_native_rejects_misaligned_inputs_and_nonunit_selected_normals():
    encoded = encoded_support(np.zeros((1, 3)))
    with pytest.raises(ValueError, match='aligned'):
        geometry.native_contact_geometry(encoded, np.zeros((1, 3)), np.zeros((2, 3)),
                                          np.zeros(1, int), np.ones(1, bool))
    with pytest.raises(ValueError, match='normalized'):
        geometry.native_contact_geometry(encoded, np.zeros((1, 3)), np.zeros((1, 3)),
                                          np.zeros(1, int), np.ones(1, bool))
    for workers in (0, -1, True, 1.5, 257):
        with pytest.raises(ValueError, match='workers'):
            geometry.native_contact_geometry(encoded, None, None, None, None, workers=workers)


def test_native_contact_integration_preserves_predictions_and_parameter_gradients(monkeypatch):
    import copy
    import torch
    from models import instance_route_a as route_a
    from test_instance_route_a import InjectedBackbone, observation

    torch.manual_seed(72)
    obs = observation(side=41)
    xyz = obs['support_points'].copy()
    xy = xyz[:, :2]
    xyz[:, 2] += .003*np.sin(55*xy[:, 0]) + .002*np.cos(39*xy[:, 1])
    normals = np.column_stack((-.165*np.cos(55*xy[:, 0]), .078*np.sin(39*xy[:, 1]),
                               np.ones(len(xyz))))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    obs.update(support_points=xyz, support_normals=normals, centroid=xyz.mean(0),
               query_points=xyz[[512, 840, 1100]], query_normals=normals[[512, 840, 1100]])
    rng = np.random.default_rng(75)
    obs['support_rgb'] = rng.random(xyz.shape)
    model = route_a.InstanceSuction3D(backbone=InjectedBackbone(), backbone_points=128)
    baseline = copy.deepcopy(model)
    original = route_a.native_contact_geometry
    calls = []

    def count_native(*args, **kwargs):
        value = original(*args, **kwargs)
        assert value is not None
        calls.append(len(args[1]))
        return value

    monkeypatch.setattr(route_a, 'native_contact_geometry', count_native)
    actual = model(obs)
    assert calls
    sum(value.float().square().sum() for value in actual.values()
        if isinstance(value, torch.Tensor) and value.requires_grad).backward()
    monkeypatch.setattr(route_a, 'native_contact_geometry', lambda *args, **kwargs: None)
    expected = baseline(obs)
    sum(value.float().square().sum() for value in expected.values()
        if isinstance(value, torch.Tensor) and value.requires_grad).backward()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-6, atol=1e-7, equal_nan=True)
    for (name, actual_parameter), (other_name, expected_parameter) in zip(
            model.named_parameters(), baseline.named_parameters()):
        assert name == other_name
        if actual_parameter.grad is None:
            assert expected_parameter.grad is None
        else:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=2e-6, atol=1e-7)
