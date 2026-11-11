"""Fixed-query target validation; no proposal sampling or arbitrary-pose claims.

Dense validation averages points within GT instances, instances within frames,
frames within groups, then groups equally. Attempt ranking and BCE/Brier use
pooled fixed executed-pose records; task_loss separately averages frames/groups.
Counts describe every skipped record explicitly.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import attempt_metrics
from .training import move_to_device


QUERY_CHUNK = 512


def _mean_groups(records, key):
    groups = defaultdict(list)
    for row in records:
        if row.get(key) is not None:
            groups[row["group_id"]].append(row[key])
    means = [float(np.mean(values)) for values in groups.values()]
    return (float(np.mean(means)) if means else None), len(means)


def _numpy(value):
    return value.detach().cpu().numpy()


def _dense_frame(model, encoded, inputs, targets):
    query = torch.where(targets["quality_valid"].bool() & inputs["normal_obs_valid"].bool())[0]
    axis_valid = targets["axis_valid"].bool()
    predicted_axis = None
    if len(query) and (~axis_valid[query]).any():
        _, predicted_axis = model.predict_dense(encoded)
    errors, identities = [], []
    labeled_axes, predicted_axes = 0, 0
    for rows in query.split(QUERY_CHUNK):
        if not len(rows):
            continue
        have_axis = axis_valid[rows]
        normals = targets["normal_label"][rows].clone()
        if predicted_axis is not None:
            normals[~have_axis] = predicted_axis[rows[~have_axis]]
        out = model.score_at_pose(
            encoded, inputs["xyz_cam_m"][rows], normals, inputs["instance_id_input"][rows],
            feature_point_index=rows, pose_batch_index=inputs["batch_index"][rows])
        valid = out["pose_feature_valid"].bool()
        labels = targets["gt_B"][rows][valid]
        estimate = out["q"][valid]
        if not torch.isfinite(labels).all() or not torch.isfinite(estimate).all():
            raise ValueError("Nonfinite scores on valid dense validation queries")
        owner = targets["instance_id_gt"][rows][valid]
        if (owner <= 0).any():
            raise ValueError("Dense quality_valid queries must belong to GT foreground instances")
        errors.append(_numpy((estimate - labels).abs()))
        identities.append(_numpy(owner))
        labeled_axes += int((valid & have_axis).sum())
        predicted_axes += int((valid & ~have_axis).sum())
    error = np.concatenate(errors) if errors else np.empty(0)
    identity = np.concatenate(identities) if identities else np.empty(0, np.int64)
    instances = [float(error[identity == i].mean()) for i in np.unique(identity)]
    return {"masked_q_mae": float(np.mean(instances)) if instances else None,
            "fixed_query_count": len(query), "evaluated_query_count": len(error),
            "skipped_query_count": len(query) - len(error),
            "labeled_axis_query_count": labeled_axes, "predicted_axis_query_count": predicted_axes,
            "evaluated_instance_count": len(instances)}


def _attempt_frame(model, encoded, attempts, source_baseline):
    points, normals = attempts["point_cam_m"], attempts["normal_out_cam"]
    labeled = attempts["label_valid"].bool()
    finite_pose = (torch.isfinite(points).all(-1) & torch.isfinite(normals).all(-1) &
                   (normals.norm(dim=-1) > 1e-6) & (attempts["instance_id"] > 0))
    if not torch.isin(attempts["success"][labeled], attempts["success"].new_tensor([0, 1])).all():
        raise ValueError("Valid attempt outcomes must be binary")
    query = torch.where(labeled & finite_pose)[0]
    scores, outcomes, losses = [], [], []
    for rows in query.split(QUERY_CHUNK):
        if not len(rows):
            continue
        out = model.score_at_pose(
            encoded, points[rows], normals[rows], attempts["instance_id"][rows],
            feature_point_index=attempts["feature_point_index"][rows],
            pose_batch_index=attempts["batch_index"][rows])
        valid = out["pose_feature_valid"].bool()
        labels = attempts["success"][rows][valid].float()
        if source_baseline:
            score = out["q"][valid]
        else:
            if "task_success_logit" not in out:
                raise ValueError("Attempt target validation requires a task-success head")
            logits = out["task_success_logit"][valid]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite valid task-success logits")
            score = logits.sigmoid()
            losses.append(_numpy(F.binary_cross_entropy_with_logits(logits, labels, reduction="none")))
        if not torch.isfinite(score).all():
            raise ValueError("Nonfinite valid executed-pose scores")
        scores.append(_numpy(score))
        outcomes.append(_numpy(labels))
    score = np.concatenate(scores) if scores else np.empty(0)
    outcome = np.concatenate(outcomes) if outcomes else np.empty(0)
    loss = np.concatenate(losses) if losses else np.empty(0)
    return {"attempt_count": len(points), "labeled_attempt_count": int(labeled.sum()),
            "evaluated_attempt_count": len(score), "skipped_attempt_count": len(points) - len(score),
            "unlabeled_attempt_count": int((~labeled).sum()),
            "invalid_pose_attempt_count": int((labeled & ~finite_pose).sum()),
            "unmapped_attempt_count": len(query) - len(score),
            "task_loss": float(loss.mean()) if len(loss) else None}, score, outcome, loss


@torch.no_grad()
def evaluate_target(model, dataset, config, device, *, source_baseline=False):
    """Evaluate every fixed target query/executed pose and return JSON metrics.

    ``source_baseline=True`` reads physical q at executed poses; it never treats
    that proxy as a calibrated success probability. No real target evaluator is
    assumed for dense labels, so this function never emits AP-R20.
    """
    from .target_dataset import collate_target

    mode = getattr(dataset, "mode", config.get("mode", config.get("target_mode")))
    if mode not in {"dense_proxy", "attempt_success"}:
        raise ValueError("Target validation requires dense_proxy or attempt_success mode")
    if getattr(dataset, "augment", None) is not None:
        raise ValueError("Target validation requires a fixed dataset without augmentation")
    records, all_scores, all_outcomes, all_losses = [], [], [], []
    groups, scenes = set(), set()
    unknown_scene = False
    model.eval()
    for sample in dataset:
        meta = sample["meta"]
        group = str(meta["group_id"])
        groups.add(group)
        if meta.get("scene_id") is not None:
            scenes.add(str(meta["scene_id"]))
        else:
            unknown_scene = True
        packed = collate_target([sample], config.get("model", {}).get("voxel_size_m", .002))
        packed = move_to_device(packed, device)
        inputs = packed["inputs"]
        encoded = model.encode_scene(inputs)
        if mode == "dense_proxy":
            row = _dense_frame(model, encoded, inputs, packed["targets"])
        else:
            row, scores, outcomes, losses = _attempt_frame(model, encoded, packed["attempts"], source_baseline)
            all_scores.append(scores)
            all_outcomes.append(outcomes)
            all_losses.append(losses)
        row.update(group_id=group,
                   observation_id=str(meta.get("observation_id", meta.get("sample_id", len(records)))),
                   sample_ids=list(meta.get("sample_ids", [])))
        records.append(row)
    result = {"mode": mode, "source_baseline": bool(source_baseline), "frame_count": len(records),
              "observation_count": len(records), "group_count": len(groups),
              "scene_count": None if unknown_scene else len(scenes), "frames": records}
    if mode == "dense_proxy":
        metric, valid_groups = _mean_groups(records, "masked_q_mae")
        for key in ("fixed_query_count", "evaluated_query_count", "skipped_query_count",
                    "labeled_axis_query_count", "predicted_axis_query_count", "evaluated_instance_count"):
            result[key] = sum(row[key] for row in records)
        result.update(masked_q_mae=metric, valid_group_count=valid_groups,
                      valid_frame_count=sum(row["masked_q_mae"] is not None for row in records),
                      conditional_axis_diagnostic=result["predicted_axis_query_count"] > 0,
                      evaluation_scope="fixed labeled points/poses; no arbitrary-pose R20 evaluation",
                      selection_metric="masked_q_mae", selection_value=metric, selection_direction="min")
    else:
        scores = np.concatenate(all_scores) if all_scores else np.empty(0)
        outcomes = np.concatenate(all_outcomes) if all_outcomes else np.empty(0)
        losses = np.concatenate(all_losses) if all_losses else np.empty(0)
        result.update(attempt_metrics(scores, outcomes, calibrated=not source_baseline))
        result["valid_frame_count"] = sum(row["evaluated_attempt_count"] > 0 for row in records)
        result["valid_group_count"] = len({row["group_id"] for row in records if row["evaluated_attempt_count"]})
        if not source_baseline:
            # Stable logit BCE also handles confidently incorrect saturated
            # sigmoid outputs without clipping the true task loss to ~36.
            result["attempt_BCE"] = float(losses.mean()) if len(losses) else None
            result["task_loss"], _ = _mean_groups(records, "task_loss")
        for key in ("attempt_count", "labeled_attempt_count", "evaluated_attempt_count", "skipped_attempt_count",
                    "unlabeled_attempt_count", "invalid_pose_attempt_count", "unmapped_attempt_count"):
            result[key] = sum(row[key] for row in records)
        metric, direction = "attempt_AUROC", "max"
        if result[metric] is None and not source_baseline:
            metric, direction = "attempt_BCE", "min"
        result.update(selection_metric=metric, selection_value=result[metric], selection_direction=direction,
                      evaluation_scope="fixed executed poses; ranking association, not new-policy success rate")
    return result
