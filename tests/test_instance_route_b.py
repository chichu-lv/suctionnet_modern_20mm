"""Executable route-B geometry/network contracts; no robot-performance claims."""
import importlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from models.instance_route_b import (
    ANGLES, RADII_M, InstanceSuction2DFastV3, MaskedRingEncoder,
    fixed_z_frames, prepare_shared_images, projected_ring_gate,
    ring_directions, ring_support_summary, sample_feature_at_indices,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


def plane_observation(height=64, width=80):
    y, x = np.mgrid[:height, :width]
    k = np.array(((600., 0., (width - 1) / 2), (0., 610., (height - 1) / 2), (0., 0., 1.)))
    z = np.full((height, width), .6, dtype=np.float64)
    xyz = np.stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z), -1)
    normal = np.zeros_like(xyz)
    normal[..., 2] = -1
    mask = np.ones((height, width), dtype=bool)
    rgb = np.stack((x / width, y / height, np.full_like(z, .4)), -1).astype(np.float32)
    uv = np.array(((width // 2, height // 2), (width // 2 - 8, height // 2 + 3),
                   (width // 2 + 9, height // 2 - 4)))
    return dict(rgb=rgb, xyz_map=xyz, mask=mask, instance_mask=mask.copy(),
                normal_map=normal, normal_valid=mask.copy(), gravity=np.array((0., 1., 0.)),
                intrinsics=k, query_points=xyz[uv[:, 1], uv[:, 0]],
                query_normals=normal[uv[:, 1], uv[:, 0]], sample_uid="test/instance")


def test_context_native_roi_preserve_geometry_and_invalid_support():
    obs = plane_observation(61, 79)
    obs["mask"] = obs["mask"].copy()
    obs["mask"][10:13, 20:25] = False
    obs["xyz_map"][10:13, 20:25] = np.nan
    before = obs["xyz_map"].copy()
    result = prepare_shared_images(obs)
    assert result["context"].shape == (1, 13, 128, 128)
    assert result["texture"].shape == (1, 5, 96, 96)
    assert result["texture_roi_hw"] == (67, 85)
    assert result["bbox_xyxy"] == (0, 0, 79, 61)
    torch.testing.assert_close(result["context_affine_inverse"] @ result["context_affine"], torch.eye(3, dtype=torch.float64))
    assert torch.isfinite(result["context"]).all() and torch.isfinite(result["texture"]).all()
    assert torch.all(result["texture"][..., 67:, :] == 0)
    assert torch.all(result["texture"][..., :, 85:] == 0)
    np.testing.assert_equal(obs["xyz_map"], before)


def test_context_uses_full_mask_not_valid_depth_bbox():
    obs = plane_observation(48, 300)
    obs["mask"][:, :100] = False
    obs["mask"][:, 200:] = False
    result = prepare_shared_images(obs)
    assert result["bbox_xyxy"] == (0, 0, 300, 48)
    assert result["texture_roi_hw"] == (54, 306)
    assert result["texture_padded_hw"] == (64, 320)
    assert result["context_affine"][0, 0] < 1


def test_query_feature_sampling_has_correct_stride_and_position_gradient():
    model = InstanceSuction2DFastV3()
    l1 = torch.zeros(1, 32, 32, 32)
    l1[0, 0, 7, 4] = 1.
    l1.requires_grad_()
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    l2 = (100 * yy + xx).float()[None, None].expand(1, 32, -1, -1).clone().requires_grad_()
    encoded = {"intrinsics": torch.eye(3, dtype=torch.float64), "context_affine": torch.eye(3, dtype=torch.float64),
               "query_features": (l1, l2)}
    points = torch.tensor(((16., 28., 1.), (20., 28., 1.)), dtype=torch.float64)
    fq = model._query_semantics(encoded, points)
    torch.testing.assert_close(fq[:, 0], torch.tensor((1., 0.)))
    torch.testing.assert_close(fq[:, 32], torch.tensor((352., 352.5)))
    fq[0, 0].backward()
    assert l1.grad[0, 0, 7, 4] == 1
    assert torch.count_nonzero(l1.grad) == 1


def test_gate_matches_frozen_numpy_reference_and_preserves_parallax_height(monkeypatch):
    refs = Path(__file__).resolve().parents[1] / "docs/instance_routes/reference"
    monkeypatch.syspath_prepend(str(refs))
    reference = importlib.import_module("contact_boundary_contract")
    p = torch.tensor(((0., 0., .6),), dtype=torch.float64)
    n = torch.tensor(((1., 0., -1.),), dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)
    phase = torch.tensor((-math.pi / ANGLES,), dtype=torch.float64)
    radial, _, _ = ring_directions(n, phase)
    ideal = p[:, None, None, :] + p.new_tensor(RADII_M)[None, :, None, None] * radial[:, None]
    scale = 1 + (-.006) / (n * p).sum(-1)
    observed = ideal * scale[:, None, None, None]
    valid = torch.ones((1, 9, ANGLES), dtype=torch.bool)
    gate = projected_ring_gate(observed, valid, p, n, phase)
    expected = reference.projected_ring_gate(observed[0].numpy(), valid[0].numpy(), p[0].numpy(), n[0].numpy(), phase=float(phase[0]))
    for key in ("contact_valid", "in_region", "radial_ok", "tangential_ok", "positive_radial"):
        np.testing.assert_array_equal(gate[key][0].numpy(), expected[key])
    np.testing.assert_allclose(gate["rho_m"][0].numpy(), expected["rho_m"], atol=1e-14, rtol=0)
    assert not gate["contact_valid"][0, 6, 0]
    assert not gate["contact_valid"][0, 6, 36]
    np.testing.assert_allclose(gate["rho_m"][0, 6, [0, 36]].numpy(), [.024254558441, .012254558441], atol=1e-12)
    support, raw = ring_support_summary(gate["contact_valid"], valid, gate["height_m"])
    expected_summary = reference.ring_support_summary(expected["contact_valid"], valid[0].numpy(), expected["height_m"])
    np.testing.assert_allclose(torch.cat((support, raw), -1)[0].numpy(), expected_summary["summary"], atol=1e-12)
    assert raw[0, 11] == pytest.approx(.15)


def test_centre_phase_independence_and_positive_z_chart():
    p = torch.tensor(((0., 0., .6),), dtype=torch.float64)
    n = torch.tensor(((0., 0., 1.),), dtype=torch.float64)
    e1, e2, chart = fixed_z_frames(n)
    assert chart.item()
    torch.testing.assert_close(torch.cross(e1, e2, dim=-1), n)
    observed = p[:, None, None, :].expand(1, 9, ANGLES, 3).clone()
    observed[:, 0, :, 0] += .002
    valid = torch.zeros(1, 9, ANGLES, dtype=torch.bool)
    valid[:, 0] = True
    first = projected_ring_gate(observed, valid, p, n, p.new_tensor((0.,)))
    second = projected_ring_gate(observed, valid, p, n, p.new_tensor((1.137,)))
    assert first["contact_valid"][:, 0].all()
    assert torch.equal(first["contact_valid"][:, 0], second["contact_valid"][:, 0])


def test_support_summary_preserves_cyclic_gap_and_raw_rejected_extrema():
    contact = torch.zeros(3, 9, ANGLES, dtype=torch.bool)
    contact[0, 1:, 18:54] = True
    contact[1, 1:, ::2] = True
    observed = contact.clone()
    height = torch.zeros(3, 9, ANGLES, dtype=torch.float64)
    observed[2, 7, 0] = observed[2, 7, 1] = True
    height[2, 7, 0], height[2, 7, 1] = .060, -.070
    support, raw = ring_support_summary(contact, observed, height)
    torch.testing.assert_close(support[0, 2:4], torch.tensor((.5, .5), dtype=torch.float64))
    torch.testing.assert_close(support[1, 2:4], torch.tensor((.5, 1 / 72), dtype=torch.float64))
    torch.testing.assert_close(support[2, :2], torch.tensor((0., 1.), dtype=torch.float64))
    torch.testing.assert_close(raw[2, -4:], torch.tensor((2 / 216, 1., 1.5, 1.75), dtype=torch.float64))


def test_ring_encoder_masking_ln_and_cyclic_shift():
    torch.manual_seed(4)
    encoder = MaskedRingEncoder()
    for norm in encoder.norms:
        nn_bias = norm.bias
        with torch.no_grad():
            nn_bias.fill_(.7)
    tokens = torch.randn(2, 75, ANGLES)
    mask = torch.ones(2, ANGLES, dtype=torch.bool)
    mask[0, 4:30] = False
    mask[1] = False
    tokens = torch.where(mask[:, None], tokens, torch.full_like(tokens, torch.nan))
    original = encoder(tokens, mask)
    shifted = encoder(torch.roll(tokens, 13, -1), torch.roll(mask, 13, -1))
    assert torch.isfinite(original).all()
    assert torch.count_nonzero(original[1]) == 0
    torch.testing.assert_close(original, shifted, atol=1e-6, rtol=1e-6)


def test_bilinear_texture_requires_all_four_original_depth_pixels():
    model = InstanceSuction2DFastV3()
    yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    z = torch.full((4, 4), .6, dtype=torch.float64)
    xyz = torch.stack((xx * z, yy * z, z), -1)
    mask = torch.ones(4, 4, dtype=torch.bool)
    normal = torch.zeros_like(xyz)
    normal[..., 2] = -1
    texture = (xx + 10 * yy).float()[None, None].expand(1, 16, -1, -1).clone()
    encoded = dict(xyz_map=xyz, mask=mask, normal_map=normal, normal_valid=mask.clone(),
                   texture_feature=texture, texture_origin_uv=torch.zeros(2, dtype=torch.float64),
                   intrinsics=torch.eye(3, dtype=torch.float64))
    query = torch.tensor([[[.75, .75, .6]]], dtype=torch.float64)
    first = model._read_slots(encoded, query)
    assert first["bilinear_valid"].item()
    assert first["texture"][0, 0, 0] == pytest.approx(13.75)
    encoded["xyz_map"][2, 2, 2] = .62
    second = model._read_slots(encoded, query)
    assert not second["bilinear_valid"].item()
    assert second["texture"][0, 0, 0] == 11
    encoded["mask"][1, 1] = False
    third = model._read_slots(encoded, query)
    assert not third["observed"].item()
    assert torch.count_nonzero(third["texture"]) == 0


def test_real_network_backward_chunk_independence_and_exact_historical_axis():
    torch.manual_seed(10)
    observation = plane_observation()
    model = InstanceSuction2DFastV3(training_phase=False).eval()
    encoded = model.encode_instance(observation)
    normals = observation["query_normals"].copy()
    normals[1] = np.array((.18, .04, -1.)) / np.linalg.norm((.18, .04, -1.))
    complete = model.score_at_pose(encoded, observation["query_points"], normals, query_chunk=64)
    chunked = model.score_at_pose(encoded, observation["query_points"], normals, query_chunk=1)
    for key in ("q_inst", "seal", "wrench", "ell", "task_logit"):
        torch.testing.assert_close(complete[key], chunked[key], atol=2e-6, rtol=2e-6)
    assert complete["valid"].all()
    assert complete["diagnostics"]["fusion"].shape == (3, 330)
    torch.testing.assert_close(complete["normals_out_cam"], torch.tensor(normals, dtype=torch.float64))
    aligned = model.score_at_pose(encoded, observation["query_points"], observation["query_normals"])
    assert not torch.equal(complete["diagnostics"]["contact_valid"][1], aligned["diagnostics"]["contact_valid"][1])
    loss = sum(complete[key].sum() for key in ("q_inst", "seal", "wrench", "ell", "task_logit"))
    loss.backward()
    for module in (model.encoder.stem[0], model.texture_stem[0], model.midlevel_query_projections[0],
                   model.ring_encoders[2].convs[0], model.query_fusion[0]):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    groups = model.parameter_groups()
    ids = [id(p) for params in groups.values() for p in params]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    torch.testing.assert_close(complete["ell"], torch.full((3,), math.log(.1)))
    torch.testing.assert_close(complete["log_variance"], math.log(2.) + 2 * complete["ell"])


def test_training_phase_is_deterministic_order_independent_and_step_bound():
    obs = plane_observation()
    model = InstanceSuction2DFastV3().train()
    encoded = model.encode_instance(obs)
    model.set_training_context(7, seed=123)
    first = model.score_at_pose(encoded, obs["query_points"], obs["query_normals"], query_chunk=1)
    again = model.score_at_pose(encoded, obs["query_points"], obs["query_normals"], query_chunk=3)
    reverse = model.score_at_pose(encoded, obs["query_points"][::-1], obs["query_normals"][::-1])
    torch.testing.assert_close(first["q_inst"], again["q_inst"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(first["diagnostics"]["phase"], reverse["diagnostics"]["phase"].flip(0))
    model.set_training_context(8, seed=123)
    changed = model.score_at_pose(encoded, obs["query_points"], obs["query_normals"])
    assert not torch.equal(first["diagnostics"]["phase"], changed["diagnostics"]["phase"])
    model.eval()
    assert torch.count_nonzero(model.score_at_pose(encoded, obs["query_points"], obs["query_normals"])["diagnostics"]["phase"]) == 0


def test_empty_and_unsupported_queries_are_finite_and_explicitly_invalid():
    obs = plane_observation()
    model = InstanceSuction2DFastV3().eval()
    encoded = model.encode_instance(obs)
    empty = model.score_at_pose(encoded, np.empty((0, 3)), np.empty((0, 3)))
    assert empty["q_inst"].shape == empty["valid"].shape == (0,)
    p = np.array(((np.nan, 0., .6), (0., 0., -.6), (.9, .9, .6), (1e200, 1e200, .6), (0., 0., .6)))
    n = np.array(((0., 0., -1.),) * 4 + ((0., 0., 0.),))
    invalid = model.score_at_pose(encoded, p, n)
    assert not invalid["valid"].any()
    for key in ("q_inst", "seal", "wrench", "ell", "task_logit", "log_variance"):
        assert torch.isfinite(invalid[key]).all()
    assert torch.count_nonzero(invalid["q_inst"]) == 0
    invalid["q_inst"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    obs["mask"][:] = False
    no_instance = model(obs)
    assert not no_instance["valid"].any() and torch.count_nonzero(no_instance["q_inst"]) == 0
    with pytest.raises(ValueError, match="one instance"):
        model.score_at_pose(encoded, obs["query_points"], obs["query_normals"], query_batch_index=[0, 1, 0])


def test_subpixel_sample_zero_padding_without_border_fabrication():
    feature = torch.ones(1, 1, 4, 4, requires_grad=True)
    positions = torch.tensor(((1., 2.), (-4., 0.), (8., 8.), (float('nan'), 0.),
                              (1e200, 1e200)), dtype=torch.float64)
    sampled = sample_feature_at_indices(feature, positions)
    torch.testing.assert_close(sampled[:, 0], torch.tensor((1., 0., 0., 0., 0.)))
    sampled.sum().backward()
    assert torch.isfinite(feature.grad).all()
    assert feature.grad.sum() == 1
