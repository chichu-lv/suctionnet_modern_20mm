"""Exact reference equivalence for compact geometry and exposure-only inputs."""
from copy import deepcopy
import pickle

import numpy as np
import pytest
import torch

import models.instance_route_b as route_b
from models.instance_route_b_preparation import prepare_source_static_inputs
from test_instance_route_b import plane_observation
from test_instance_route_b_batch import assert_predictions_close


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(kind):
    obs = plane_observation(45, 59)
    if kind == "interior":
        obs["mask"][:] = False
        obs["mask"][7:37, 9:51] = True
    elif kind == "edge":
        obs["mask"][:] = False
        obs["mask"][:40, :54] = True
    elif kind == "empty":
        obs["mask"][:] = False
    obs["instance_mask"] = obs["mask"].copy()
    # Retain a wider instance bbox than valid-depth geometry and a depth hole.
    obs["mask"][14:17, 16:19] = False
    obs["xyz_map"][14:17, 16:19] = np.nan
    obs["normal_map"][20, 20] = np.inf
    obs["normal_valid"][21, 22] = False
    obs["rgb_valid"] = np.ones(obs["mask"].shape, dtype=bool)
    obs["rgb"][20, 20] = np.nan  # Explicit valid + NaN follows reference zero-before-normalize behavior.
    obs["rgb_valid"][18, 19] = False
    obs["support_points"] = obs["xyz_map"][obs["mask"]].copy()
    return obs


def compact(obs, static=None):
    prepared = route_b.prepare_source_inputs_cpu(obs, sample_image_channels=True, static_inputs=static)
    return route_b.compact_prepared_source_item({"observation": prepared, "record": {}, "labels": {}})["observation"]


def test_static_cache_owns_arrays_without_color_pose_or_observation_references():
    obs = observation("interior")
    static = prepare_source_static_inputs(obs)
    def arrays(value):
        if isinstance(value, np.ndarray):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                assert key not in {"rgb", "rgb_valid", "observation", "sources", "phase",
                                   "query_points", "query_normals", "query_features", "texture_feature"}
                yield from arrays(item)
    source_arrays = [value for value in obs.values() if isinstance(value, np.ndarray)]
    for array in arrays(static):
        assert all(not np.shares_memory(array, source) for source in source_arrays)


@pytest.mark.parametrize("kind", ["empty", "full", "interior", "edge"])
def test_static_geometry_and_sampled_channels_match_full_camera_reference(kind):
    obs = observation(kind)
    reference = route_b.prepare_shared_images(obs)
    static = prepare_source_static_inputs({k: v for k, v in obs.items()
                                          if k not in ("rgb", "rgb_valid", "query_points", "query_normals")})
    current = route_b._restore_training_encoded(compact(obs, static)["_route_b_training_inputs"], "cpu")
    height, width = reference["mask"].shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    for key, expected in reference.items():
        if key == "observation":
            continue
        actual = route_b._geometry_at_pixels(current, key, y, x) if key in route_b._GEOMETRY_MAPS else current[key]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=key)
        else:
            assert actual == expected, key
    # Spatial storage is still the ROI after the consumer receives the ticket.
    assert current["mask"].shape == static["encoded"]["mask"].shape
    if kind in ("interior", "empty"):
        assert current["mask"].numel() < height * width


@pytest.mark.parametrize("kind", ["empty", "full", "interior", "edge"])
def test_roi_gather_preserves_original_camera_boundaries_and_source_ids(kind):
    obs = observation(kind)
    reference = route_b.prepare_shared_images(obs)
    current = route_b._restore_training_encoded(compact(obs)["_route_b_training_inputs"], "cpu")
    uv = torch.tensor(((-2., -3.), (-.5000001, 5.), (-.5, 5.), (58.49, 44.49),
                       (58.5, 44.5), (9., 7.), (50., 36.), (25., 25.),
                       (1e100, -1e100), (float("nan"), 2.), (1., float("inf"))), dtype=torch.float64)
    for key in route_b._GEOMETRY_MAPS:
        actual = route_b._nearest_geometry(current, key, uv)
        expected = route_b._nearest_map(reference[key], uv)
        for item, ref in zip(actual, expected):
            torch.testing.assert_close(item, ref, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["full", "interior", "edge"])
def test_cached_static_inputs_match_outputs_gradients_and_phase_across_exposures(kind, monkeypatch):
    torch.manual_seed(74)
    obs = observation(kind)
    static = prepare_source_static_inputs(obs)
    saved = deepcopy(static)
    model = route_b.InstanceSuction2DFastV3(query_chunk=2).train()
    # A repeated exposure must neither rebuild geometry nor retain the first RGB.
    monkeypatch.setattr(route_b, "prepare_source_static_inputs",
                        lambda *a, **k: pytest.fail("Rebuilt supplied static geometry"))
    phases = []
    for update, multiplier in ((0, .25), (19, .75)):
        exposure = dict(obs, rgb=obs["rgb"] * np.float32(multiplier))
        model.set_training_context(update)
        small = pickle.loads(pickle.dumps(compact(exposure, static)))
        phases.append(model._canonical_phase(small))
        reference = route_b.prepare_source_inputs_cpu(exposure)
        def score(value):
            model.zero_grad(set_to_none=True)
            output = model(value)
            sum(output[key].square().mean() for key in ("q_inst", "seal", "wrench", "ell", "task_logit")).backward()
            return output, {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        expected, old_grad = score(reference)
        actual, new_grad = score(small)
        assert_predictions_close([actual], [expected], rtol=0, atol=0)
        assert old_grad.keys() == new_grad.keys()
        for name in old_grad:
            torch.testing.assert_close(new_grad[name], old_grad[name], rtol=0, atol=0, msg=name)
    assert not torch.equal(phases[0], phases[1])
    for key, value in static.items():
        if isinstance(value, dict):
            for field, array in value.items():
                np.testing.assert_equal(array, saved[key][field])
        else:
            np.testing.assert_equal(value, saved[key])


def test_arbitrary_poses_keep_full_support_fallback_and_four_corner_depth_rules():
    torch.manual_seed(41)
    obs = observation("edge")
    # A depth discontinuity must disallow bilinear texture for the same four
    # ORIGINAL camera pixels even though geometry storage has an ROI origin.
    obs["xyz_map"][20:, 22:, 2] += .02
    model = route_b.InstanceSuction2DFastV3().eval()
    small = compact(obs)
    p = np.concatenate((obs["query_points"], [[-.04, -.02, .6005], [.3, .3, .6], [0., 0., -.1]]))
    n = np.tile([0., 0., -1.], (len(p), 1))
    with torch.no_grad():
        full = model.encode_instance(obs)
        roi = model.encode_instance(small)
        expected = model.score_at_pose(full, p, n, phase=.37)
        actual = model.score_at_pose(roi, p, n, phase=.37)
    assert_predictions_close([actual], [expected], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["spec", "extra", "origin", "geometry_shape", "dtype", "support_count", "texture"])
def test_training_wire_rejects_wrong_layout_before_device_upload(mutation):
    wire = compact(observation("interior"))["_route_b_training_inputs"]
    if mutation == "spec":
        wire["spec"] = "old full-camera wire"
    elif mutation == "extra":
        wire["encoded"]["raw_rgb"] = np.zeros((45, 59, 3), np.float32)
    elif mutation == "origin":
        wire["top_left"] = (999, 0)
    elif mutation == "geometry_shape":
        wire["encoded"]["normal_map"] = wire["encoded"]["normal_map"][:-1]
    elif mutation == "dtype":
        wire["encoded"]["xyz_map"] = wire["encoded"]["xyz_map"].astype(np.float32)
    elif mutation == "support_count":
        wire["encoded"]["support_count"] += 1
    else:
        wire["encoded"]["texture"] = wire["encoded"]["texture"][..., :-1]
    with pytest.raises(ValueError, match="[Cc]ompact route-B"):
        route_b._restore_training_encoded(wire, "cpu")


def test_wire_accepts_cpu_tensors_and_transfers_only_support_count_outside_encoded():
    obs = observation("interior")
    # Keep the reader's original support-count semantics, even if a public
    # caller supplied a flattened array with a different valid-pixel count.
    obs["support_points"] = np.zeros((17, 3), np.float64)
    small = compact(obs)
    assert "support_points" not in small and small["source_support_count"] == 17
    wire = small["_route_b_training_inputs"]
    reference = route_b._restore_training_encoded(wire, "cpu")
    wire["encoded"] = {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value
                       for key, value in wire["encoded"].items()}
    current = route_b._restore_training_encoded(wire, "cpu")
    for key, value in reference.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(current[key], value, rtol=0, atol=0)
    count_only = dict(obs, source_support_count=29)
    count_only.pop("support_points")
    assert compact(count_only)["source_support_count"] == 29
