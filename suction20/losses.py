"""Masked source/target objectives with instance → frame → batch reduction."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_WEIGHTS = {"q": 1., "q0": 1., "normal": 2., "seal": .5,
                   "wrench": .25, "collision_free": .1}


def hierarchical_mean(values: Tensor, valid: Tensor, batch_index: Tensor,
                      instance_id: Tensor) -> Tensor:
    """Exclude empty instances/frames, never average all batch instances together."""
    valid = valid.bool() & (instance_id > 0)
    if not valid.any():
        return values[valid].sum()
    frame_means = []
    for frame in torch.unique(batch_index[valid], sorted=True):
        frame_mask = valid & (batch_index == frame)
        instances = torch.unique(instance_id[frame_mask], sorted=True)
        frame_means.append(torch.stack([
            values[frame_mask & (instance_id == inst)].mean() for inst in instances
        ]).mean())
    return torch.stack(frame_means).mean()


def _regression(pred: Tensor, target: Tensor, valid: Tensor,
                batch_index: Tensor, instance_id: Tensor, beta: float,
                *, binary: bool = False) -> Tensor:
    # Evaluate only labeled entries: an unknown NaN must not poison gradients.
    valid = valid.bool() & (instance_id > 0)
    selected = pred.float()[valid]
    if not selected.numel():
        return selected.sum()
    if binary:
        value = F.binary_cross_entropy_with_logits(selected, target.float()[valid], reduction="none")
    else:
        value = F.smooth_l1_loss(selected, target.float()[valid], beta=beta, reduction="none")
    return hierarchical_mean(value, torch.ones_like(value, dtype=torch.bool),
                             batch_index[valid], instance_id[valid])


def source_losses(pred: dict, targets: dict, inputs: dict, *,
                  quality_target: str = "gt_B", beta: float = .05,
                  weights: dict | None = None) -> dict[str, Tensor]:
    """Also handles dense target data with separate component availability masks."""
    if quality_target not in ("gt_A", "gt_B"):
        raise ValueError("quality_target must be gt_A or gt_B")
    weight = DEFAULT_WEIGHTS | (weights or {})
    batch, instance = inputs["batch_index"], targets["instance_id_gt"]
    quality_valid = targets["quality_valid"].bool()
    seed = pred["seed_point_index"].reshape(-1)
    seed_valid = pred["seed_valid"].reshape(-1).bool()
    seed = seed[seed_valid]
    out = {"q0": _regression(pred["q0_all"], targets[quality_target], quality_valid,
                             batch, instance, beta)}
    valid = quality_valid[seed]
    out["q"] = _regression(pred["q"].reshape(-1)[seed_valid], targets[quality_target][seed], valid,
                           batch[seed], instance[seed], beta)
    axis_valid = targets.get("axis_valid", torch.zeros_like(quality_valid)).bool() & (instance > 0)
    axis = pred["normal_out_all"].float()[axis_valid]
    if axis.numel():
        label = F.normalize(targets["normal_label"].float()[axis_valid], dim=-1)
        errors = 1 - (axis * label).sum(-1).clamp(-1, 1)
        out["normal"] = hierarchical_mean(errors, torch.ones_like(errors, dtype=torch.bool),
                                            batch[axis_valid], instance[axis_valid])
    else:
        out["normal"] = axis.sum()
    for name, prediction, target, mask in [
        ("seal", "seal_aux", "seal_target", "seal_valid"),
        ("wrench", "wrench_aux", "wrench_target", "wrench_valid"),
        ("collision_free", "collision_free_logit", "collision_free_target", "collision_valid"),
    ]:
        if prediction not in pred:
            continue
        availability = targets.get(mask, quality_valid).bool()
        if target not in targets:
            out[name] = pred[prediction].reshape(-1)[:0].sum()
            continue
        availability = availability & torch.isfinite(targets[target])
        out[name] = _regression(pred[prediction].reshape(-1)[seed_valid], targets[target][seed],
                                availability[seed], batch[seed], instance[seed],
                                beta, binary=name == "collision_free")
    out["total"] = sum(out[name] * weight[name] for name in out)
    return out


def attempt_loss(logits: Tensor, attempts: dict, pose_feature_valid: Tensor) -> Tensor:
    """Mean valid attempts per frame, then mean supervised frames."""
    valid = attempts["label_valid"].bool() & pose_feature_valid.bool()
    for key in ("point_cam_m", "normal_out_cam"):
        valid &= torch.isfinite(attempts[key]).all(-1)
    valid &= attempts["normal_out_cam"].norm(dim=-1) > 1e-6
    selected = logits.float()[valid]
    if not selected.numel():
        return selected.sum()
    loss = F.binary_cross_entropy_with_logits(selected, attempts["success"].float()[valid], reduction="none")
    frames = attempts["batch_index"][valid]
    return torch.stack([loss[frames == b].mean() for b in torch.unique(frames)]).mean()
