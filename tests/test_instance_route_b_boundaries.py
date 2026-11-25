"""Route-B support, chart-policy and center-slot boundary regressions.

These are bounded synthetic geometry checks using the production B network;
they make no claims about dataset accuracy, latency or chart-seam continuity.
"""
import json
import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from models.instance_route_b import (
    ANGLES, RADII_M, InstanceSuction2DFastV3, fixed_z_frames,
    prepare_shared_images, projected_ring_gate, ring_directions,
)
from suction20.instance_config import B_FEATURE_FRAME, B_LEGACY_FEATURE_FRAME
from suction20.instance_geometry import NORMAL_SPEC, PRECISION_SPEC, SAMPLER_SPEC
from suction20.instance_inference import infer_instance, load_instance_checkpoint
from suction20.instance_training import attempt_loss, evaluate_instance_attempts
from test_instance_inference import checkpoint_state
from test_instance_route_b import plane_observation


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng():
        torch.manual_seed(19)
        result = InstanceSuction2DFastV3(training_phase=False).eval()
    return result


def sparse_plane(count):
    """Keep a central supported query while varying observed pixel count."""
    observation = plane_observation(32, 32)
    yy, xx = np.mgrid[:32, :32]
    source_ids = np.arange(32 * 32).reshape(32, 32)
    order = np.lexsort((source_ids.ravel(), ((xx - 16) ** 2 + (yy - 16) ** 2).ravel()))
    mask = np.zeros((32, 32), dtype=bool)
    mask.ravel()[order[:count]] = True
    observation.update(
        mask=mask, instance_mask=mask.copy(), normal_valid=mask.copy(),
        support_points=observation["xyz_map"][mask].copy(),
        query_points=observation["xyz_map"][16, 16][None].copy(),
        query_normals=observation["normal_map"][16, 16][None].copy(),
        query_valid=np.ones(1, bool), query_ids=np.array([0], dtype=np.int64),
        query_source_ids=np.array([16 * 32 + 16], dtype=np.int64),
        normal_spec=NORMAL_SPEC, precision_spec=PRECISION_SPEC,
        sampler_spec=SAMPLER_SPEC, sample_uid=f"boundary/points-{count}",
    )
    return observation


def score_observation(model, observation):
    encoded = model.encode_instance(observation)
    prediction = model.score_at_pose(encoded, observation["query_points"], observation["query_normals"])
    return encoded, prediction


@pytest.mark.parametrize("count, sufficient", [(99, False), (100, True), (101, True)])
@torch.no_grad()
def test_default_support_threshold_at_boundary(model, count, sufficient):
    encoded, prediction = score_observation(model, sparse_plane(count))
    assert model.min_support_points == 100
    assert encoded["support_count"] == count
    assert encoded["support_sufficient"] is sufficient
    assert encoded["instance_valid"] is sufficient
    assert prediction["valid"].tolist() == [sufficient]
    assert prediction["diagnostics"]["support_count"].tolist() == [count]
    assert prediction["diagnostics"]["support_sufficient"].tolist() == [sufficient]
    assert prediction["diagnostics"]["query_input_valid"].tolist() == [True]


@torch.no_grad()
def test_custom_support_threshold_reaches_model_scoring():
    custom = InstanceSuction2DFastV3(min_support_points=101, training_phase=False).eval()
    assert custom.architecture_metadata()["minimum_support_points"] == 101
    for count, sufficient in ((100, False), (101, True)):
        encoded, prediction = score_observation(custom, sparse_plane(count))
        assert encoded["instance_valid"] is sufficient
        assert prediction["valid"].tolist() == [sufficient]


def test_support_count_uses_mask_finite_xyz_and_positive_depth_not_normals():
    observation = sparse_plane(107)
    rows = np.argwhere(observation["mask"])
    # Five exclusions are independent; the other 102 pixels have usable depth.
    for (row, col), coordinate, value in zip(rows[:4], (0, 1, 2, 2), (np.nan, np.inf, 0., -.6)):
        observation["xyz_map"][row, col, coordinate] = value
    observation["mask"][tuple(rows[4])] = False
    observation["normal_valid"][:] = False
    observation["normal_map"][:] = np.nan
    observation["support_points"] = np.repeat(observation["query_points"], 1000, axis=0)
    prepared = prepare_shared_images(observation, min_support_points=102)
    assert prepared["support_count"] == 102
    assert prepared["instance_valid"] is True
    assert prepared["support_points"].shape == (102, 3)
    assert not prepared["normal_valid"].any()
    assert torch.isfinite(prepared["xyz_map"]).all()
    assert torch.isfinite(prepared["context"]).all()
    assert torch.isfinite(prepared["texture"]).all()
    assert prepare_shared_images(observation, min_support_points=103)["instance_valid"] is False


@pytest.mark.parametrize("count, flattened_count, sufficient", [(10, 1000, False), (100, 0, True)])
@torch.no_grad()
def test_model_gate_ignores_flattened_support_override(model, count, flattened_count, sufficient):
    observation = sparse_plane(count)
    observation["support_points"] = np.repeat(observation["query_points"], flattened_count, axis=0)
    encoded, prediction = score_observation(model, observation)
    assert encoded["support_count"] == count
    assert prediction["valid"].tolist() == [sufficient]


@torch.no_grad()
def test_tiny_instance_is_rejected_by_score_attempt_loss_evaluation_and_inference(model):
    observation = sparse_plane(10)
    _, prediction = score_observation(model, observation)
    assert prediction["valid"].tolist() == [False]
    item = {
        "observation": observation,
        "record": {"observation_uid": "tiny-instance"},
        "attempts": {
            "point_cam_m": observation["query_points"],
            "normal_out_cam": observation["query_normals"],
            "success": np.array([1]), "label_valid": np.array([True]),
            "pose_support_valid": np.array([True]), "pose_valid": np.array([True]),
        },
    }
    # An optimistic outer action flag must not override the model's support gate.
    assert attempt_loss(model, item) is None
    metrics, rows = evaluate_instance_attempts(model, [item])
    assert metrics["total_attempts"] == metrics["unsupported_attempts"] == 1
    assert metrics["supported_attempts"] == 0
    assert rows[0]["supported"] is False
    assert rows[0]["probability"] is None
    result = infer_instance(model, observation, stage="finetune")
    assert result["status"] == "no_pose"
    assert result["reason"] == "insufficient_support"
    assert result["poses"] == []
    assert result["counts"]["geometry_valid_queries"] == 0


@torch.no_grad()
def test_metadata_is_json_serializable_and_matches_real_layers_and_features(model):
    metadata = json.loads(json.dumps(model.architecture_metadata(), allow_nan=False))
    encoded, prediction = score_observation(model, sparse_plane(100))
    diagnostics = prediction["diagnostics"]
    assert metadata["input_channels"] == model.encoder.stem[0].in_channels == encoded["context"].shape[1]
    texture_convs = [module for module in model.texture_stem if isinstance(module, nn.Conv2d)]
    assert metadata["texture_input_channels"] == texture_convs[0].in_channels == encoded["texture"].shape[1]
    assert metadata["texture_feature_channels"] == texture_convs[-1].out_channels == encoded["texture_feature"].shape[1]
    assert metadata["global_feature_channels"] == model.global_projection[0].out_features == encoded["global_feature"].shape[-1]
    assert metadata["query_feature_channels"] == sum(layer.out_channels for layer in model.midlevel_query_projections) == diagnostics["fq"].shape[-1]
    assert metadata["ring_feature_channels"] == sum(2 * branch.convs[-1].out_channels for branch in model.ring_encoders)
    assert all(branch.convs[0].in_channels == metadata["ring_encoder_input_channels"] for branch in model.ring_encoders)
    assert metadata["ring_encoder_input_channels"] == 3 * metadata["ring_token_channels"]
    assert metadata["support_summary_channels"] == diagnostics["support_summary"].shape[-1] == 18
    assert metadata["raw_summary_channels"] == diagnostics["raw_summary"].shape[-1] == 12
    dimensions = ("global_feature_channels", "query_feature_channels", "ring_feature_channels",
                  "support_summary_channels", "raw_summary_channels", "query_descriptor_channels")
    assert sum(metadata[key] for key in dimensions) == metadata["fusion_channels"]
    assert metadata["fusion_channels"] == model.query_fusion[0].in_features == diagnostics["fusion"].shape[-1] == 330
    assert metadata["feature_frame"] == B_FEATURE_FRAME
    assert metadata["feature_frame_seam_continuous"] is False
    assert metadata["minimum_support_points"] == model.min_support_points == 100
    assert metadata["angles"] == ANGLES == 72
    assert metadata["radii_m"] == list(RADII_M)
    assert metadata["unique_projected_slots"] == 577
    assert metadata["broadcast_projected_slots"] == 648


def normal_at_z(z, azimuth_degrees=13.7):
    azimuth = math.radians(azimuth_degrees)
    radius = math.sqrt(max(0., 1. - z * z))
    return (radius * math.cos(azimuth), radius * math.sin(azimuth), z)


def test_current_chart_is_right_handed_at_poles_one_degree_and_switch_boundary():
    z_values = (-1., -.9, 0., .5 - 1e-10, .5, .5 + 1e-10, math.cos(math.radians(1.)), 1.)
    normals = torch.tensor([normal_at_z(z) for z in z_values], dtype=torch.float64)
    e1, e2, positive = fixed_z_frames(normals)
    basis = torch.stack((e1, e2, normals), -1)
    assert torch.isfinite(basis).all()
    identity = torch.eye(3, dtype=torch.float64).expand(len(normals), -1, -1)
    torch.testing.assert_close(basis.transpose(-1, -2) @ basis, identity, atol=2e-14, rtol=0)
    torch.testing.assert_close(torch.cross(e1, e2, dim=-1), normals, atol=2e-14, rtol=0)
    torch.testing.assert_close(torch.linalg.det(basis), torch.ones(len(normals), dtype=torch.float64), atol=2e-14, rtol=0)
    assert torch.equal(positive, normals[:, 2] > .5)
    denominator = 1 + torch.where(positive, normals[:, 2], -normals[:, 2])
    assert float(denominator.min()) == .5
    # Main chart is stable at -Z. Its singularity is +Z, where the other chart applies.
    assert denominator[0] == denominator[-1] == 2
    assert positive[-2:].all()


def test_current_and_legacy_frames_are_identical_below_switch_including_ring_gates():
    normals = torch.tensor([normal_at_z(z) for z in (-1., -.4, 0., .5)], dtype=torch.float64)
    phase = torch.tensor((0., .1, 1.2, 4.5), dtype=torch.float64)
    current = ring_directions(normals, phase, feature_frame=B_FEATURE_FRAME)
    legacy = ring_directions(normals, phase, feature_frame=B_LEGACY_FEATURE_FRAME)
    for a, b in zip(current, legacy):
        assert torch.equal(a, b)
    points = torch.tensor([[0., 0., .6]], dtype=torch.float64).expand(len(normals), -1)
    radii = torch.tensor(RADII_M, dtype=torch.float64)
    observed = points[:, None, None] + radii[None, :, None, None] * current[0][:, None]
    valid = torch.ones(observed.shape[:-1], dtype=torch.bool)
    current_gate = projected_ring_gate(observed, valid, points, normals, phase, feature_frame=B_FEATURE_FRAME)
    legacy_gate = projected_ring_gate(observed, valid, points, normals, phase, feature_frame=B_LEGACY_FEATURE_FRAME)
    assert current_gate.keys() == legacy_gate.keys()
    for key in current_gate:
        assert torch.equal(current_gate[key], legacy_gate[key]), key


def test_changed_chart_semantics_are_explicit_and_legacy_one_degree_stays_finite():
    assert B_FEATURE_FRAME == "two_fixed_z_charts_rodrigues_z05_v1"
    assert B_LEGACY_FEATURE_FRAME == "two_fixed_z_charts_rodrigues_v2"
    normals = torch.tensor([normal_at_z(.8), normal_at_z(math.cos(math.radians(1.)))], dtype=torch.float64)
    current = fixed_z_frames(normals, feature_frame=B_FEATURE_FRAME)
    legacy = fixed_z_frames(normals, feature_frame=B_LEGACY_FEATURE_FRAME)
    assert current[2].tolist() == [True, True]
    assert legacy[2].tolist() == [False, False]
    assert not torch.allclose(current[0], legacy[0])
    assert torch.isfinite(legacy[0]).all() and torch.isfinite(legacy[1]).all()
    legacy_model = InstanceSuction2DFastV3(feature_frame=B_LEGACY_FEATURE_FRAME)
    metadata = legacy_model.architecture_metadata()
    assert metadata["feature_frame"] == B_LEGACY_FEATURE_FRAME
    assert metadata["feature_frame_switch"] == "1 - nz <= 1e-4"
    assert metadata["feature_frame_seam_continuous"] is False


@pytest.mark.parametrize("saved_frame", [None, B_FEATURE_FRAME])
@torch.no_grad()
def test_real_checkpoint_loading_preserves_legacy_or_explicit_chart(model, tmp_path, saved_frame):
    state = checkpoint_state(model, route="b")
    state["config"]["model"] = {"training_phase": False}
    if saved_frame is not None:
        state["config"]["model"].update(feature_frame=saved_frame, min_support_points=100)
        state["identity"]["model_architecture"] = model.architecture_metadata()
    state["identity"]["config_definition"] = deepcopy(state["config"])
    path = tmp_path / "best.pt"
    torch.save(state, path)
    loaded, metadata = load_instance_checkpoint(path, query_chunk_size=7)
    expected = B_LEGACY_FEATURE_FRAME if saved_frame is None else saved_frame
    assert loaded.feature_frame == expected
    assert loaded.min_support_points == 100
    assert not loaded.training
    assert metadata["architecture"]["feature_frame"] == expected
    assert metadata["runtime_model_config"]["feature_frame"] == expected
    assert metadata["runtime_model_config"]["min_support_points"] == 100
    assert metadata["runtime_model_config"]["query_chunk"] == 7
    assert metadata["model_compatibility"]["semantic_model"]["feature_frame"] == expected
    assert ("feature_frame" in metadata["checkpoint_identity"]["config_definition"]["model"]) is (saved_frame is not None)
    torch.testing.assert_close(loaded.encoder.stem[0].weight, model.encoder.stem[0].weight, rtol=0, atol=0)
    observation = sparse_plane(100)
    observation["query_normals"] = np.asarray([normal_at_z(.8)])
    _, prediction = score_observation(loaded, observation)
    assert prediction["valid"].tolist() == [True]
    assert prediction["diagnostics"]["chart_positive_z"].tolist() == [expected == B_FEATURE_FRAME]


def test_switch_seam_is_not_claimed_to_be_an_integer_circular_shift(model):
    normals = torch.tensor([normal_at_z(.5 - 1e-10), normal_at_z(.5 + 1e-10)], dtype=torch.float64)
    e1, e2, positive = fixed_z_frames(normals)
    assert positive.tolist() == [False, True]
    roll = math.degrees(math.atan2(float(e1[0] @ e2[1]), float(e1[0] @ e1[1])))
    # This azimuth gives a 152.6-degree transition: 2.4 degrees off a 5-degree slot shift.
    assert abs(math.remainder(roll, 360. / ANGLES)) == pytest.approx(2.4, abs=1e-7)
    radial, _, _ = ring_directions(normals, torch.zeros(2, dtype=torch.float64))
    distances = [(.020 * (radial[0] - radial[1].roll(shift, 0))).norm(dim=-1).mean()
                 for shift in range(ANGLES)]
    assert .0008 < float(torch.stack(distances).min()) < .0009
    assert model.architecture_metadata()["feature_frame_seam_continuous"] is False


@torch.no_grad()
def test_center_only_broadcast_keeps_frozen_summary_weights_and_one_source_pixel(model):
    observation = sparse_plane(1)
    encoded = model.encode_instance(observation)
    points = torch.as_tensor(observation["query_points"], dtype=torch.float64)
    normals = torch.as_tensor(observation["query_normals"], dtype=torch.float64)
    # Inspect contact statistics directly: the model must still reject this tiny instance.
    _, support, raw, diagnostics = model.contact_features(encoded, points, normals, points.new_zeros(1))
    assert encoded["instance_valid"] is False
    expected_support = torch.tensor([[1., 0.] + [0., 1.] * 8], dtype=torch.float64)
    expected_raw = torch.tensor([[1 / 3, 0., 0., 0.] + [0.] * 8], dtype=torch.float64)
    torch.testing.assert_close(support, expected_support, rtol=0, atol=0)
    torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
    assert support.shape == (1, 18) and raw.shape == (1, 12)
    assert diagnostics["observed_valid"][0, 0].sum() == ANGLES
    assert diagnostics["contact_valid"][0, 0].all()
    assert diagnostics["unique_source_pixels_per_radius"].tolist() == [[1] + [0] * 8]
    assert diagnostics["source_ids"][0, 0].unique().tolist() == observation["query_source_ids"].tolist()
    center_metadata = model.architecture_metadata()["center_summary_semantics"]
    assert center_metadata["center_core_raw_coverage_weight"] == 1 / 3


@pytest.mark.parametrize("threshold", [0, -1, True, 100.])
def test_support_threshold_rejects_invalid_configuration(threshold):
    with pytest.raises(ValueError, match="positive integer"):
        InstanceSuction2DFastV3(min_support_points=threshold)
    with pytest.raises(ValueError, match="positive integer"):
        prepare_shared_images(sparse_plane(100), min_support_points=threshold)
