"""Target training phases and isolated executed-action supervision."""
from __future__ import annotations

import copy
import torch
from torch import nn

from .losses import attempt_loss, source_losses
from .training import decay_groups


def set_bn_eval_and_freeze_affine(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(False)


def apply_phase(model, optimizer, step):
    """Successful steps 0..199 train heads; step 200 unlocks decoder."""
    groups = model.parameter_groups()
    for name, params in groups.items():
        for parameter in params:
            parameter.requires_grad_(name == "heads" or (name == "decoder" and step >= 200))
    set_bn_eval_and_freeze_affine(model)
    for group in optimizer.param_groups:
        group["lr"] = ((3e-4 if step < 200 else 1e-4) if group["role"] == "heads"
                       else (0. if step < 200 else 1e-5))
    return "heads_only" if step < 200 else "decoder_and_heads"


def make_finetune_optimizer(model, config):
    semantic = model.parameter_groups()
    flattened = [p for ps in semantic.values() for p in ps]
    if len({id(p) for p in flattened}) != len(flattened) or {id(p) for p in flattened} != {id(p) for p in model.parameters()}:
        raise ValueError("Parameter groups must be disjoint and exhaustive")
    # Register both groups once; preserve heads' Adam moments at step 200.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    set_bn_eval_and_freeze_affine(model)
    groups = []
    for role in ("heads", "decoder"):
        for group in decay_groups(model, semantic[role], weight_decay=config.get("weight_decay", 1e-4)):
            group["role"] = role
            groups.append(group)
    optimizer = torch.optim.AdamW(groups, lr=3e-4, betas=tuple(config.get("betas", (.9, .999))),
                                  eps=config.get("eps", 1e-8))
    apply_phase(model, optimizer, 0)
    return optimizer


def target_loss(model, packed, mode, loss_config=None):
    inputs, targets = packed["inputs"], packed["targets"]
    if mode == "dense_proxy":
        prediction = model(inputs, sampler_mode="dense_proxy", targets=targets)
        cfg = loss_config or {}
        return source_losses(prediction, targets, inputs, quality_target="gt_B",
                             beta=cfg.get("smooth_l1_beta", .05), weights=cfg.get("weights")), {}
    if mode != "attempt_success":
        raise ValueError(f"Unknown target mode: {mode}")
    attempts = packed["attempts"]
    encoded = model.encode_scene(inputs)
    pose = model.score_at_pose(encoded, attempts["point_cam_m"], attempts["normal_out_cam"],
                               attempts["instance_id"], pose_batch_index=attempts["batch_index"],
                               feature_point_index=attempts["feature_point_index"])
    loss = attempt_loss(pose["task_success_logit"], attempts, pose["pose_feature_valid"])
    return {"total": loss, "task_success": loss}, {
        "attempt_count": len(attempts["success"]),
        "unmapped_attempt_count": int((~pose["pose_feature_valid"]).sum())}


def resolve_finetune_config(config, checkpoint):
    if checkpoint.get("quality_target") != "gt_B":
        raise ValueError("Target adaptation requires a gt_B physical checkpoint")
    if config.get("mode") not in {"dense_proxy", "attempt_success"}:
        raise ValueError("Target mode must be dense_proxy or attempt_success")
    if not config.get("source_replay", {}).get("enabled", True):
        raise ValueError("The implemented protocol requires 1:1 source replay")
    training = config["training"]
    expected = {"target_frames_per_optimizer_step": 2, "effective_batch_size": 4,
                "world_size": 1, "microbatch_size": 1,
                "accumulated_microbatches_per_optimizer_step": 4,
                "encoder_frozen": True, "bn_mode": "eval", "bn_affine_trainable": False}
    for key, value in expected.items():
        if training.get(key, value) != value:
            raise ValueError(f"Unsupported target training.{key}")
    if config["source_replay"].get("source_frames_per_optimizer_step", 2) != 2:
        raise ValueError("Source replay requires two frames per update")
    loss = config.get("loss", {})
    if loss.get("target_weight", 1.) != 1. or loss.get("source_weight", .5) != .5 or loss.get("additional_divide_by_accumulation_steps", False):
        raise ValueError("Target/source loss weights must be 1.0/0.5 without extra division")
    augmentation = config.get("augmentation", {})
    if augmentation.get("scale", False) or augmentation.get("depth_noise_std_m", 0) != 0 or augmentation.get("point_dropout_fraction", [0., 0.]) != [0., 0.]:
        raise ValueError("Target protocol permits only metric rigid augmentation")
    if not 0 <= training["max_optimizer_steps"] <= 1000:
        raise ValueError("Target protocol is bounded to at most 1000 updates")
    phases = [dict(name="heads_only", start_optimizer_step=0, end_optimizer_step_exclusive=200,
                   train_groups=["heads"], lr={"heads": 3e-4, "decoder": 0.}),
              dict(name="decoder_and_heads", start_optimizer_step=200, end_optimizer_step_exclusive=1000,
                   train_groups=["decoder", "heads"], lr={"heads": 1e-4, "decoder": 1e-5})]
    if training.get("phases", phases) != phases:
        raise ValueError("Unsupported target phase schedule")
    result = copy.deepcopy(config)
    result["experiment"] = checkpoint.get("experiment", "E5")
    result["model"] = dict(checkpoint["model_config"])
    result["model"]["use_task_success_head"] = config["mode"] == "attempt_success"
    result["model"].pop("task_success_head", None)
    if result.get("runtime", {}).get("backbone_root"):
        result["model"]["backbone_root"] = result["runtime"]["backbone_root"]
    result["data"] = dict(result["source_replay"])
    source_training = checkpoint["training_config"]
    result["source_loss"] = copy.deepcopy(source_training.get("source_loss", source_training.get("loss", {})))
    result["sampling"] = copy.deepcopy(source_training.get("sampling", {}))
    result.setdefault("augmentation", {}).setdefault("depth_noise_std_m", 0.)
    result["augmentation"].setdefault("point_dropout_fraction", [0., 0.])
    return result


def target_selection(metrics, mode):
    """Undefined AUROC remains None; BCE is only the documented fallback."""
    if mode == "dense_proxy":
        value = metrics.get("masked_q_mae")
        return None if value is None else ("masked_q_mae", -value)
    auc = metrics.get("attempt_AUROC")
    if auc is not None:
        return "attempt_AUROC", auc
    bce = metrics.get("attempt_BCE")
    return None if bce is None else ("attempt_BCE", -bce)
