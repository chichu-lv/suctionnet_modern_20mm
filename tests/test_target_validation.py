"""Fixed-set target validation tests; no GPU or real target-data claims."""
import json

import numpy as np
import pytest
import torch
from torch import nn

from suction20.target_dataset import empty_attempts
from suction20.target_validation import evaluate_target


class Dataset(list):
    augment = None

    def __init__(self, samples, mode):
        super().__init__(samples)
        self.mode = mode


class PoseModel(nn.Module):
    """Score encodes the executed x coordinate, making pose replacement visible."""
    def __init__(self, logit_scale=2.):
        super().__init__()
        self.calls, self.encoded_count, self.dense_count = [], 0, 0
        self.logit_scale = logit_scale

    def forward(self, *args, **kwargs):
        raise AssertionError("fixed validation must never sample proposals")

    def encode_scene(self, inputs):
        self.encoded_count += 1
        return inputs

    def predict_dense(self, encoded):
        self.dense_count += 1
        count = len(encoded["xyz_cam_m"])
        return torch.rand(count), torch.tensor([1., 0., 0.]).expand(count, 3)

    def score_at_pose(self, encoded, points, normals, instance_id, feature_point_index, pose_batch_index):
        self.calls.append({"point": points.clone(), "normal": normals.clone(), "instance": instance_id.clone(),
                           "anchor": feature_point_index.clone(), "batch": pose_batch_index.clone()})
        valid = instance_id == 1
        return {"q": points[:, 0], "task_success_logit": points[:, 0] * self.logit_scale - 1,
                "pose_feature_valid": valid}


def observation(count, group, name, mode):
    return {"inputs": {"xyz_cam_m": np.column_stack([np.zeros(count), np.arange(count) * .001,
                                                        np.full(count, .5)]).astype(np.float32),
                       "normal_obs": np.tile([0., 0., -1.], (count, 1)).astype(np.float32),
                       "normal_obs_valid": np.ones(count, bool), "instance_id_input": np.ones(count, np.int64),
                       "source_indices": np.arange(count, dtype=np.int64) * 2, "g_cam": np.array([0., 0., -1.], np.float32)},
            "targets": {}, "attempts": empty_attempts(),
            "meta": {"group_id": group, "observation_id": name, "sample_ids": [name], "label_kind": mode,
                     "source_index_kind": "pixel", "scene_id": group}}


def dense_sample(count, group, name, value):
    sample = observation(count, group, name, "dense_proxy")
    sample["inputs"]["xyz_cam_m"][:, 0] = value
    sample["targets"] = {"quality_valid": np.ones(count, bool), "gt_B": np.zeros(count, np.float32),
                         "instance_id_gt": np.ones(count, np.int64), "axis_valid": np.ones(count, bool),
                         "normal_label": np.tile([0., 1., 0.], (count, 1)).astype(np.float32)}
    return sample


def test_dense_all_fixed_queries_chunk512_gt_instance_frame_group_means():
    sample = dense_sample(1030, "A", "a0", .1)
    sample["inputs"]["xyz_cam_m"][1000:, 0] = .9
    sample["inputs"]["normal_obs_valid"][-5:] = False
    sample["targets"]["gt_B"][-5:] = np.nan
    sample["targets"]["instance_id_gt"][1000:] = 2  # Input instances remain all 1.
    sample["targets"]["axis_valid"][500:] = False
    samples = [sample, dense_sample(2, "A", "a1", .3), dense_sample(1, "B", "b0", .9)]
    model = PoseModel()
    result = evaluate_target(model, Dataset(samples, "dense_proxy"), {}, "cpu")
    assert result["fixed_query_count"] == 1028
    assert result["evaluated_query_count"] == 1028
    assert result["labeled_axis_query_count"] == 503
    assert result["predicted_axis_query_count"] == 525
    assert result["conditional_axis_diagnostic"]
    assert result["masked_q_mae"] == pytest.approx(.65)
    assert result["frames"][0]["masked_q_mae"] == pytest.approx(.5)
    assert [len(call["point"]) for call in model.calls] == [512, 512, 1, 2, 1]
    assert model.encoded_count == 3 and model.dense_count == 1
    np.testing.assert_array_equal(model.calls[0]["normal"][:500], np.tile([0, 1, 0], (500, 1)))
    np.testing.assert_array_equal(model.calls[0]["normal"][500:], np.tile([1, 0, 0], (12, 1)))
    assert result["selection_metric"] == "masked_q_mae"
    assert result["selection_direction"] == "min"
    assert "AP_R20" not in result
    json.dumps(result, allow_nan=False)
    # q0 changes on every call, but every fixed physical query is unchanged.
    old_calls = model.calls.copy()
    model.calls = []
    again = evaluate_target(model, Dataset(samples, "dense_proxy"), {}, "cpu")
    assert again["masked_q_mae"] == result["masked_q_mae"]
    for first, second in zip(old_calls, model.calls):
        torch.testing.assert_close(first["point"], second["point"])


def test_dense_unmapped_and_empty_are_not_fake_negative_labels():
    sample = dense_sample(3, "A", "a0", .2)
    sample["inputs"]["instance_id_input"][1] = 0
    sample["targets"]["quality_valid"][2] = False
    result = evaluate_target(PoseModel(), Dataset([sample], "dense_proxy"), {}, "cpu")
    assert result["fixed_query_count"] == 2
    assert result["evaluated_query_count"] == 1
    assert result["skipped_query_count"] == 1
    assert result["masked_q_mae"] == pytest.approx(.2)
    result = evaluate_target(PoseModel(), Dataset([dense_sample(0, "A", "empty", 0)], "dense_proxy"), {}, "cpu")
    assert result["masked_q_mae"] is None
    assert result["selection_value"] is None
    assert result["evaluated_query_count"] == 0


def attempt_sample():
    sample = observation(3, "session_a", "frame0", "attempt_success")
    sample["attempts"] = {
        "point_cam_m": np.array([[.9, 0, .5], [.9, 0, .5], [.1, 0, .5], [.2, 0, .5], [np.nan, 0, .5], [.5, 0, .5]], np.float32),
        "normal_out_cam": np.tile([0., 1., 0.], (6, 1)).astype(np.float32),
        "instance_id": np.array([1, 1, 1, 2, 1, 1], np.int64),
        "source_index": np.array([0, 2, 4, -1, -1, -1], np.int64),
        "success": np.array([1, 0, 0, 1, 1, -1], np.float32),
        "label_valid": np.array([True, True, True, True, True, False]),
    }
    return sample


def test_attempt_fixed_executed_poses_baseline_vs_task_and_skip_counts():
    sample = attempt_sample()
    model = PoseModel()
    dataset = Dataset([sample], "attempt_success")
    baseline = evaluate_target(model, dataset, {}, "cpu", source_baseline=True)
    task = evaluate_target(model, dataset, {}, "cpu")
    for result in (baseline, task):
        assert result["attempt_AUROC"] == .75
        assert result["attempt_PR_AUC"] == .5
        assert result["n_success"] == 1 and result["n_failure"] == 2
        assert result["attempt_count"] == 6 and result["evaluated_attempt_count"] == 3
        assert result["skipped_attempt_count"] == 3
        assert result["unlabeled_attempt_count"] == result["invalid_pose_attempt_count"] == result["unmapped_attempt_count"] == 1
        assert result["group_count"] == result["scene_count"] == 1
        assert result["selection_metric"] == "attempt_AUROC"
        json.dumps(result, allow_nan=False)
    assert "attempt_BCE" not in baseline and "attempt_Brier" not in baseline
    assert task["attempt_BCE"] > 0 and task["attempt_Brier"] > 0
    assert model.dense_count == 0
    for call in model.calls:
        np.testing.assert_array_equal(call["point"], sample["attempts"]["point_cam_m"][:4])
        np.testing.assert_array_equal(call["normal"], sample["attempts"]["normal_out_cam"][:4])
        np.testing.assert_array_equal(call["anchor"], [0, 1, 2, -1])
        assert not call["batch"].any()


def test_single_class_task_uses_finite_stable_bce_fallback():
    sample = attempt_sample()
    sample["attempts"] = {k: v[:1] for k, v in sample["attempts"].items()}
    sample["attempts"]["success"][:] = 0
    model = PoseModel(logit_scale=1000)
    result = evaluate_target(model, Dataset([sample], "attempt_success"), {}, "cpu")
    assert result["attempt_AUROC"] is None
    assert result["attempt_BCE"] == pytest.approx(899)
    assert result["selection_value"] == result["attempt_BCE"]
    assert result["selection_metric"] == "attempt_BCE" and result["selection_direction"] == "min"
    baseline = evaluate_target(model, Dataset([sample], "attempt_success"), {}, "cpu", source_baseline=True)
    assert baseline["selection_value"] is None
    assert "attempt_BCE" not in baseline


@pytest.mark.parametrize("mode", ["dense_proxy", "attempt_success"])
def test_empty_target_set_is_json_safe_and_undefined(mode):
    result = evaluate_target(PoseModel(), Dataset([], mode), {}, "cpu")
    assert result["selection_value"] is None
    assert result["group_count"] == result["observation_count"] == 0
    json.dumps(result, allow_nan=False)


def test_validation_refuses_augmented_dataset():
    data = Dataset([], "dense_proxy")
    data.augment = object()
    with pytest.raises(ValueError, match="without augmentation"):
        evaluate_target(PoseModel(), data, {}, "cpu")
