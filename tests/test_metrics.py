import numpy as np
import pytest

from suction20.metrics import (
    aggregate_scene_metrics, ap_r20, attempt_metrics, evaluate_view_metrics,
    masked_quality_metrics, normal_angle_metrics, pose_nms, proposal_coverage,
)


def test_nms_joint_directed_rule_and_deterministic_ties():
    points = np.array([[0., 0, 0], [.001, 0, 0], [.002, 0, 0], [.02, 0, 0], [0, 0, 0]])
    normals = np.array([[0., 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, 1]])
    result = pose_nms(points, normals, np.ones(5), [1, 1, 1, 1, 2],
                      proposal_scores=[.2, .3, .2, .2, .2], point_indices=[0, 1, 3, 2, 4])
    np.testing.assert_array_equal(result["scene_indices"], [1, 3, 2, 4])
    # Close opposite normals remain, and two input instances do not suppress.


def test_nms_caps_and_top1_before_scene_cap():
    count = 61
    points = np.column_stack([np.arange(count) * .02, np.zeros((count, 2))])
    result = pose_nms(points, np.tile([0., 0, 1], (count, 1)),
                      np.linspace(1, 0, count), np.arange(count) // 10 + 1)
    assert len(result["per_instance_indices"]) == 61
    assert len(result["scene_indices"]) == 50
    owners = np.arange(count) // 10 + 1
    metrics = evaluate_view_metrics(np.ones(count), np.zeros(count, bool), owners,
                                    np.arange(1, 9), **result)
    assert metrics["instance_Top1_Q20_at_0_4"] == 7 / 8
    assert metrics["no_candidate_rate"] == 1 / 8
    assert metrics["AP_R20"] == 1


def test_nms_ten_per_input_instance_and_empty():
    points = np.column_stack([np.arange(15) * .02, np.zeros((15, 2))])
    result = pose_nms(points, np.tile([0, 0, 1], (15, 1)), np.ones(15), np.ones(15))
    assert len(result["per_instance_indices"]) == 10
    empty = pose_nms(np.empty((0, 3)), np.empty((0, 3)), [], [])
    assert empty["scene_indices"].shape == (0,)


def test_ap_fixed_denominators_and_gt_ownership():
    assert ap_r20([]) == 0
    assert ap_r20([1]) == pytest.approx(np.sum(1 / np.arange(1, 51)) / 50)
    # Split input masks share GT owner, and a high-score background is failure.
    result = evaluate_view_metrics([1, .5, 1], [False, True, False], [0, 2, 2],
                                    [1, 2], [0, 1, 2], [0, 1, 2])
    assert result["instance_Top1_Q20_at_0_4"] == .5
    assert result["collision_rate"] == 1
    assert result["no_candidate_rate"] == .5
    assert result["AP_R20"] == ap_r20([0, .5, 1])


def test_mean_views_then_scenes_not_global_view_average():
    result = aggregate_scene_metrics([{"scene_id": 0, "AP_R20": 1},
                                      {"scene_id": 0, "AP_R20": 1},
                                      {"scene_id": 1, "AP_R20": 0}])
    assert result["AP_R20"] == .5
    assert result["view_count"] == 3


def test_masked_diagnostics_and_known_zero():
    result = masked_quality_metrics([.5, 1, 1, np.nan], [0, 1, 1, np.nan],
                                     [True, True, True, False], [1, 2, 2, 3])
    assert result["masked_q_mae"] == .25
    assert result["quality_valid_count"] == 3
    assert result["quality_coverage"] == .75
    result = proposal_coverage([0, 2, -1], [1, 1, 2, 3], [True, True, False, True],
                               [.5, 0, 1, .8], [True, True, False])
    assert result["proposal_coverage"] == .5
    result = normal_angle_metrics([[0, 0, -1], [np.nan] * 3], [[0, 0, 1], [np.nan] * 3], [True, False])
    assert result["normal_median_deg"] == 180


def test_attempt_ties_distinct_score_ap_and_single_class():
    # Top tie contains one positive and one negative; grouped AP is .5.
    result = attempt_metrics([.9, .9, .1], [1, 0, 0])
    assert result["attempt_PR_AUC"] == .5
    assert result["attempt_AUROC"] == .75
    assert "attempt_Brier" not in result
    one_class = attempt_metrics([.1, .2], [1, 1], calibrated=True)
    assert one_class["attempt_AUROC"] is None
    assert one_class["attempt_PR_AUC"] == 1
    assert one_class["attempt_Brier"] == pytest.approx((.81 + .64) / 2)
