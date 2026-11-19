"""Static four-pixel rules preserve live projected sampling and feature gradients."""
from copy import deepcopy

import numpy as np
import pytest
import torch

import models.instance_route_b as route_b
from models.instance_route_b_preparation import bilinear_validity_map
from test_instance_route_b_static_roi import compact, observation


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def slot_inputs(kind, depth_step):
    obs = observation(kind)
    # Deliberately create adjacent quads immediately on either side of the
    # FP64 tolerance, as well as holes and camera/compact-rectangle edges.
    obs["xyz_map"][21:, :, 2] += depth_step
    encoded = route_b._restore_training_encoded(compact(obs)["_route_b_training_inputs"], "cpu")
    rng = torch.Generator().manual_seed(281)
    texture = torch.randn((1, 16, *encoded["texture_padded_hw"]), generator=rng)
    y, x = torch.meshgrid(torch.arange(-1, 47, dtype=torch.float64),
                          torch.arange(-1, 61, dtype=torch.float64), indexing="ij")
    uv = torch.stack((x + .25, y + .75), -1).reshape(1, -1, 2)
    # Include nearest-floor ties and finite-but-unrepresentable off-image grids.
    uv = torch.cat((uv, uv.new_tensor([[[-.5, 6.], [58.5, 44.5],
                                       [1e100, -1e100], [float("nan"), 2.]]])), 1)
    k = encoded["intrinsics"]
    points = torch.cat(((uv - k[:2, 2]) / k.diag()[:2] * .6,
                        torch.full((*uv.shape[:-1], 1), .6, dtype=torch.float64)), -1)
    return encoded, texture, points


@pytest.mark.parametrize("kind", ["empty", "full", "interior", "edge"])
@pytest.mark.parametrize("depth_step", [.010 - 1e-12, .010 + 1e-12, .010 + 2e-12, .02])
def test_static_quad_map_matches_live_four_corner_sampling_and_texture_gradient(kind, depth_step):
    encoded, texture, points = slot_inputs(kind, depth_step)
    model = route_b.InstanceSuction2DFastV3().eval()
    results, gradients = [], []
    for cached in (False, True):
        item = dict(encoded)
        if not cached:
            item.pop("bilinear_valid_map")
        feature = texture.clone().requires_grad_()
        item["texture_feature"] = feature
        result = model._read_slots(item, points)
        result["texture"].square().sum().backward()
        results.append(result)
        gradients.append(feature.grad)
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0, equal_nan=True, msg=key)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


def test_quad_map_is_only_geometry_and_does_not_mutate_inputs():
    obs = observation("interior")
    first = compact(obs)
    changed = dict(obs, rgb=np.zeros_like(obs["rgb"]),
                   query_points=obs["query_points"][::-1].copy(),
                   query_normals=-obs["query_normals"])
    second = compact(changed)
    a = first["_route_b_training_inputs"]["encoded"]
    b = second["_route_b_training_inputs"]["encoded"]
    np.testing.assert_array_equal(a["bilinear_valid_map"], b["bilinear_valid_map"])
    xyz, mask = torch.as_tensor(a["xyz_map"]), torch.as_tensor(a["mask"])
    saved_xyz, saved_mask = xyz.clone(), mask.clone()
    result = bilinear_validity_map(xyz, mask)
    assert result.dtype == torch.bool and result.shape == mask.shape
    assert not result[-1].any() and not result[:, -1].any()
    torch.testing.assert_close(xyz, saved_xyz, rtol=0, atol=0)
    torch.testing.assert_close(mask, saved_mask, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["old_spec", "map_missing", "map_dtype", "map_shape"])
def test_quad_wire_rejects_mismatched_static_layout_before_upload(mutation):
    wire = deepcopy(compact(observation("interior"))["_route_b_training_inputs"])
    if mutation == "old_spec":
        wire["spec"] = "route_b_training_roi_fp32_geometry64_v2"
    elif mutation == "map_missing":
        wire["encoded"].pop("bilinear_valid_map")
    elif mutation == "map_dtype":
        wire["encoded"]["bilinear_valid_map"] = wire["encoded"]["bilinear_valid_map"].astype(np.float32)
    else:
        wire["encoded"]["bilinear_valid_map"] = wire["encoded"]["bilinear_valid_map"][:-1]
    with pytest.raises(ValueError, match="[Cc]ompact route-B"):
        route_b._restore_training_encoded(wire, "cpu")


def test_reused_ring_basis_preserves_gates_for_both_charts():
    normals = torch.tensor([[0., 0., -1.], [.6, 0., .8], [1., 0., 0.]], dtype=torch.float64)
    points = torch.tensor([[0., 0., .6]] * 3, dtype=torch.float64)
    phase = torch.tensor([0., .371, 5.9], dtype=torch.float64)
    for frame in (route_b.B_FEATURE_FRAME, route_b.B_LEGACY_FEATURE_FRAME):
        directions = route_b.ring_directions(normals, phase, feature_frame=frame)
        observed = points[:, None, None, :] + points.new_tensor(route_b.RADII_M)[None, :, None, None] * directions[0][:, None]
        valid = torch.ones(observed.shape[:-1], dtype=torch.bool)
        expected = route_b.projected_ring_gate(observed, valid, points, normals, phase, feature_frame=frame)
        actual = route_b.projected_ring_gate(observed, valid, points, normals, phase,
                                             feature_frame=frame, _directions=directions)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
