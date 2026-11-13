"""Deterministic prediction and actual-pose R20 evaluation, with GT isolated."""
from __future__ import annotations

import copy
import numpy as np
import torch

from .datasets import collate_public
from .runner import make_source_dataset
from .training import EXPERIMENTS, content_digest, move_to_device
from .precision import resolve_precision


def checkpoint_inference_config(config, state):
    """Bind model semantics to saved weights while allowing runtime relocation.

    E0/E1 and disabled feature branches can have identical tensor shapes, so
    strict state loading alone cannot detect a conflicting inference config.
    """
    if state.get("format_version") != "suction20.checkpoint.v1":
        raise ValueError("Unsupported checkpoint format")
    if content_digest(state["source_label_config"]) != state["source_label_config_sha256"]:
        raise ValueError("Checkpoint source label configuration digest mismatch")
    saved = copy.deepcopy(state["model_config"])
    experiment = state.get("experiment") or saved.get("experiment", "E1")
    requested_experiment = config.get("experiment") or config["model"].get("experiment", experiment)
    if experiment != requested_experiment:
        raise ValueError("Inference experiment differs from checkpoint")
    target, instance, grouping, auxiliary = EXPERIMENTS[experiment]
    expected = dict(quality_target=target, use_instance_fusion=instance,
                    use_contact_grouping=grouping, use_aux_heads=auxiliary)
    for key, value in expected.items():
        if saved.get(key, value) != value:
            raise ValueError(f"Checkpoint experiment conflicts with model.{key}")
        saved.setdefault(key, value)
    if state.get("quality_target", target) != target:
        raise ValueError("Checkpoint quality target conflicts with experiment")
    runtime = {"backbone_root", "dense_chunk_size", "checkpoint_dense"}
    requested = dict(config["model"])
    # Accept the documented legacy aliases but compare their effective values.
    for mapping in (saved, requested):
        if "task_success_head" in mapping:
            mapping.setdefault("use_task_success_head", mapping.pop("task_success_head"))
        if "num_seeds" in mapping:
            mapping.setdefault("seed_count", mapping.pop("num_seeds"))
    defaults = {"use_task_success_head": False, "seed_count": 512, "voxel_size_m": .002}
    for key, value in requested.items():
        if key not in runtime and key != "experiment" and value != saved.get(key, defaults.get(key)):
            raise ValueError(f"Inference model.{key} differs from checkpoint")
    resolved = copy.deepcopy(config)
    resolved["experiment"] = experiment
    resolved["model"] = {**saved, **{k: v for k, v in requested.items() if k in runtime}}
    saved_precision = state.get("training_config", {}).get("training", {}).get("precision")
    if saved_precision is not None:
        resolved.setdefault("training", {}).setdefault("precision", saved_precision)
    return resolved


def use_task_success_ranking(state, config):
    """A present but untrained task head must not replace physical q ranking."""
    trained = (state.get("stage") == "target_finetune" and
               state.get("target_mode") == "attempt_success" and state.get("global_step", 0) > 0)
    score = config.get("ranking", {}).get("final_score", "task_success_score" if trained else "q")
    if score not in {"q", "q_B", "task_success_score"}:
        raise ValueError(f"Unsupported inference ranking score: {score}")
    if score == "task_success_score" and not trained:
        raise ValueError("Task-success ranking requires a trained attempt_success checkpoint")
    return score == "task_success_score"


@torch.no_grad()
def predict_sample(model, sample, config, device, *, include_dense=False):
    """Predict using observations only, returning physical poses in original camera.

    Optional dense arrays remain in the input coordinate frame for validation
    against synchronized targets; targets themselves are never accessed here.
    """
    observation = {"inputs": sample["inputs"], "targets": {}, "meta": sample.get("meta", {})}
    packed = collate_public([observation], config["model"].get("voxel_size_m", .002))
    inputs = move_to_device(packed["inputs"], device)
    model.eval()
    with resolve_precision(config, device, inference=True).autocast():
        output = model(inputs, sampler_mode="inference")
    valid = output["seed_valid"][0].bool()
    index = output["seed_point_index"][0][valid]
    def array(value):
        return value.detach().float().cpu().numpy()
    result = {
        "point_index": index.cpu().numpy(),
        "point_cam_m": array(output["point_cam_m"][0][valid]),
        "normal_out_cam": array(output["normal_out_cam"][0][valid]),
        "q": array(output["q"][0][valid]), "q0": array(output["q0_all"][index]),
        "instance_id": inputs["instance_id_input"][index].cpu().numpy(),
        "source_indices": inputs["source_indices"][index].cpu().numpy(),
    }
    if "task_success_logit" in output:
        result["task_success_score"] = array(output["task_success_logit"][0][valid].sigmoid())
    from .observation import poses_to_original_camera
    result["point_cam_m"], result["normal_out_cam"] = poses_to_original_camera(
        result["point_cam_m"], result["normal_out_cam"], observation["meta"])
    if include_dense:
        result["normal_out_all"] = array(output["normal_out_all"])
        result["q0_all"] = array(output["q0_all"])
    return result


def nms_prediction(prediction, config, *, task_success=False):
    from .metrics import pose_nms
    rank = config.get("ranking", {})
    if rank.get("uncertainty_lambda", 0) != 0:
        raise ValueError("Residual uncertainty ranking is not implemented")
    score = prediction["task_success_score"] if task_success else prediction["q"]
    return pose_nms(prediction["point_cam_m"], prediction["normal_out_cam"], score,
                    prediction["instance_id"], proposal_scores=prediction["q0"],
                    point_indices=prediction["point_index"],
                    distance=rank.get("nms_distance_m", .010), angle_deg=rank.get("nms_angle_deg", 15.),
                    max_per_instance=rank.get("max_per_instance", 10), max_per_scene=rank.get("max_per_scene", 50))


def evaluate_model(model, config, device, *, split="val", frame_keys=None, progress=None):
    from .eval_r20 import evaluate_poses, load_scene_eval_context
    from .metrics import (aggregate_scene_metrics, evaluate_view_metrics,
                          masked_quality_metrics, normal_angle_metrics, proposal_coverage)
    for key, expected in {"nms_distance_m": .010, "nms_angle_deg": 15.,
                          "max_per_instance": 10, "max_per_scene": 50}.items():
        if config.get("ranking", {}).get(key, expected) != expected:
            raise ValueError(f"AP-R20 protocol requires ranking.{key}={expected}")
    dataset = make_source_dataset(config, split=split, frame_keys=frame_keys, augment=False)
    records = []
    if progress is not None:
        progress(0, len(dataset), None)
    context, previous_scene = None, None
    for sample in dataset:
        scene, camera, frame = sample["meta"]["sample_uid"]
        prediction = predict_sample(model, sample, config, device, include_dense=True)
        nms = nms_prediction(prediction, config)
        selected = nms["per_instance_indices"]
        if (scene, camera) != previous_scene:
            context = load_scene_eval_context(config["data"]["source_root"], config["data"]["label_root"],
                                              scene, camera, dataset.data_cfg)
            previous_scene = (scene, camera)
        gt = sample["targets"]["instance_id_gt"]
        ownership = gt[prediction["point_index"]]
        physical = evaluate_poses(context, frame, prediction["point_cam_m"][selected],
                                  prediction["normal_out_cam"][selected], ownership[selected] - 1)
        quality = np.zeros(len(ownership), np.float32)
        collision = np.zeros(len(ownership), bool)
        quality[selected], collision[selected] = physical["Q20"], physical["collision"]
        record = evaluate_view_metrics(quality, collision, ownership, np.unique(gt[gt > 0]),
                                        selected, nms["scene_indices"])
        targets = sample["targets"]
        m = targets["quality_valid"][prediction["point_index"]]
        target_name = getattr(model, "quality_target", config["model"].get("quality_target", "gt_B"))
        labels = sample["targets"][target_name][prediction["point_index"]]
        error = masked_quality_metrics(prediction["q"], labels, m, ownership)
        angles = normal_angle_metrics(prediction["normal_out_all"], targets["normal_label"], targets["axis_valid"])
        dense_error = masked_quality_metrics(prediction["q0_all"], targets[target_name], targets["quality_valid"], gt)
        record.update(scene_id=int(scene), frame_id=int(frame),
                      normal_angle_median_deg=angles["normal_median_deg"],
                      normal_angle_p90_deg=angles["normal_p90_deg"], normal_count=angles["normal_count"],
                      masked_seed_q_mae=error["masked_q_mae"],
                      masked_seed_q_smooth_l1=error["masked_q_smooth_l1"],
                      labeled_seed_count=error["quality_valid_count"],
                      labeled_seed_fraction=error["quality_coverage"],
                      masked_dense_q0_mae=dense_error["masked_q_mae"],
                      labeled_point_count=dense_error["quality_valid_count"],
                      labeled_point_fraction=dense_error["quality_coverage"])
        coverage = proposal_coverage(prediction["point_index"], gt, targets["gt_valid"], targets["gt_B"])
        record["proposal_coverage_at_512"] = coverage.pop("proposal_coverage")
        record.update(coverage)
        records.append(record)
        if progress is not None:
            progress(len(records), len(dataset), sample["meta"]["sample_uid"])
    result = aggregate_scene_metrics(records)
    for key in ("AP_R20", "instance_Top1_Q20_at_0_4", "collision_rate", "no_candidate_rate"):
        result.setdefault(key, None)
    result["frames"] = records
    result["split"] = split
    result["evaluation_scope"] = "actual predicted poses; geometric proxy, not physical success"
    return result
