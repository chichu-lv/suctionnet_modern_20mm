"""Reproducible schedules, optimizer boundaries and checkpoint transport."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

EXPERIMENTS = {
    "E0": ("gt_A", False, False, False), "E1": ("gt_B", False, False, False),
    "E2": ("gt_B", True, False, False), "E3": ("gt_B", False, True, False),
    "E4": ("gt_B", True, True, False), "E5": ("gt_B", True, True, True),
}


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as stream:
        # PyYAML's implicit scalar rules treat JSON numbers such as 1e-08
        # as strings. Preserve JSON's numeric types for selected profiles.
        config = json.load(stream) if path.suffix.lower() == ".json" else yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    config = copy.deepcopy(config)
    experiment = config.get("experiment")
    if experiment is not None:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment {experiment}")
        target, instance, grouping, auxiliary = EXPERIMENTS[experiment]
        expected = dict(quality_target=target, use_instance_fusion=instance,
                        use_contact_grouping=grouping, use_aux_heads=auxiliary)
        model = config.setdefault("model", {})
        for name, value in expected.items():
            if name in model and model[name] != value:
                raise ValueError(f"{experiment}: conflicting model.{name}")
            model[name] = value
    return config


def content_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def semantic_config(config: dict) -> dict:
    """Permit relocating files and devices, but preserve all training semantics."""
    result = copy.deepcopy(config)
    result.pop("runtime", None)
    result.pop("init_checkpoint", None)
    for section in ("data", "source_replay"):
        for key in ("source_root", "label_root", "frozen_label_config", "adapter_cache"):
            result.get(section, {}).pop(key, None)
    result.get("target_data", {}).pop("manifest", None)
    result.get("model", {}).pop("backbone_root", None)
    return result


def validate_resume_config(previous: dict, current: dict) -> None:
    if semantic_config(previous) != semantic_config(current):
        raise ValueError("Resume configuration changed training semantics")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class FrameSchedule:
    """Separate deterministic view RNG, with position at next consumed frame.

    Workers receive explicit augmentation seeds; prefetch cannot consume the
    model RNG or alter the resumed sequence.
    """
    def __init__(self, scene_ids, *, seed=20261110, views_per_scene=64, frame_keys=None):
        self.scenes = list(scene_ids)
        self.seed, self.views = int(seed), int(views_per_scene)
        self.fixed_keys = [tuple(k) for k in frame_keys] if frame_keys is not None else None
        self.epoch, self.position = 0, 0
        self.keys = self._make_epoch()
        if not self.keys:
            raise ValueError("Training frame schedule is empty")

    def _make_epoch(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        keys = list(self.fixed_keys) if self.fixed_keys is not None else [
            (int(s), int(f)) for s in self.scenes
            for f in rng.choice(256, self.views, replace=False)
        ]
        rng.shuffle(keys)
        return keys

    def next(self):
        if self.position == len(self.keys):
            self.epoch += 1
            self.position = 0
            self.keys = self._make_epoch()
        key = self.keys[self.position]
        aug_seed = int(np.random.SeedSequence([self.seed, self.epoch, self.position, 1]).generate_state(1)[0])
        self.position += 1
        return key, aug_seed

    def state_dict(self):
        return {"scene_ids": self.scenes, "seed": self.seed, "views_per_scene": self.views,
                "frame_keys": self.fixed_keys, "epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        for name, current in (("scene_ids", self.scenes), ("seed", self.seed), ("views_per_scene", self.views)):
            if state[name] != current:
                raise ValueError(f"Resume schedule {name} differs")
        fixed = [tuple(k) for k in state["frame_keys"]] if state["frame_keys"] is not None else None
        if fixed != self.fixed_keys:
            raise ValueError("Resume frame list differs")
        self.epoch, self.position = state["epoch"], state["position"]
        self.keys = self._make_epoch()
        if not 0 <= self.position <= len(self.keys):
            raise ValueError("Invalid resumed schedule position")


def learning_rate(step: int, *, peak=1e-3, start=1e-5, final=1e-5,
                  warmup=2880, total=57600) -> float:
    if step < warmup:
        return start + (peak - start) * step / max(1, warmup - 1)
    progress = min(1., (step - warmup) / max(1, total - warmup - 1))
    return final + .5 * (peak - final) * (1 + math.cos(math.pi * progress))


def decay_groups(model: nn.Module, parameters=None, *, weight_decay=1e-4, lr=None):
    selected = {id(p) for p in (parameters if parameters is not None else model.parameters()) if p.requires_grad}
    norm_ids = {id(p) for m in model.modules() if isinstance(m, (nn.modules.batchnorm._BatchNorm,
                nn.LayerNorm, nn.GroupNorm, nn.modules.instancenorm._InstanceNorm)) for p in m.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if id(p) in selected:
            (no_decay if name.endswith("bias") or id(p) in norm_ids else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.}]
    if lr is not None:
        for group in groups:
            group["lr"] = lr
    return groups


def make_optimizer(model, config, device):
    """Keep AdamW and its parameter groups; fused execution is an opt-in."""
    options = dict(lr=config.get("lr_peak", .001),
                   betas=tuple(config.get("betas", (.9, .999))),
                   eps=config.get("eps", 1e-8))
    if "fused" in config:
        if type(config["fused"]) is not bool:
            raise ValueError("optimizer.fused must be a boolean")
        options["fused"] = config["fused"] and torch.device(device).type == "cuda"
    return torch.optim.AdamW(decay_groups(model, weight_decay=config.get("weight_decay", 1e-4)),
                             **options)


def optimizer_update(optimizer, scaler, parameters, max_norm=10.) -> tuple[bool, float]:
    if not scaler.is_enabled():
        norm = torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)
        value = float(norm)
        updated = math.isfinite(value)
        if updated:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return updated, value
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= old_scale
    optimizer.zero_grad(set_to_none=True)
    return updated, float(norm)


def move_to_device(value, device, *, non_blocking=False):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {k: move_to_device(v, device, non_blocking=non_blocking) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(v, device, non_blocking=non_blocking) for v in value)
    return value


def select_training_gpu_inputs(inputs):
    keys = {"voxel_features", "voxel_coords", "inverse_map", "xyz_cam_m", "normal_obs",
            "normal_obs_valid", "instance_id_input", "batch_index", "g_cam"}
    return {key: value for key, value in inputs.items() if key in keys}


def code_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def worktree_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard"], text=True).splitlines()
    digest = hashlib.sha256()
    for name in sorted(set(paths)):
        path = root / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, scaler, schedule, *, global_step, config,
                    source_label_config, best_metric=None, **extra):
    state = {"format_version": "suction20.checkpoint.v1", "stage": "public_pretrain",
             "model_state_dict": model.state_dict(), "model_config": config["model"],
             "quality_target": config["model"].get("quality_target", "gt_B"),
             "source_label_config": source_label_config,
             "source_label_config_sha256": content_digest(source_label_config),
             "experiment": config.get("experiment"), "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": {"global_step": global_step, "config": config.get("scheduler", {})},
             "scaler_state_dict": scaler.state_dict(), "global_step": global_step,
             "epoch": schedule.epoch, "next_microbatch_in_epoch": schedule.position,
             "sampler_state": schedule.state_dict(), "rng_state": capture_rng(),
             "split_definition": config["data"].get("split_definition", {
                 "train": [s for s in range(100) if s % 10 != 9],
                 "val": list(range(9, 100, 10))}),
             "best_metric": best_metric, "training_config": config,
             "code_revision": extra["code_revision"] if "code_revision" in extra else code_revision(),
             "code_worktree_sha256": extra["code_worktree_sha256"] if "code_worktree_sha256" in extra else worktree_digest()}
    state.update(extra)
    write_checkpoint_state(path, state)


def write_checkpoint_state(path, state):
    """Publish an already assembled state without touching its tensor/RNG values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .pretrain_runtime import fsync_directory
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, model, *, optimizer=None, scaler=None, schedule=None,
                    resume=False, allow_new_task_head=False, source_label_config=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != "suction20.checkpoint.v1":
        raise ValueError("Unsupported checkpoint format")
    if source_label_config is not None and content_digest(source_label_config) != state["source_label_config_sha256"]:
        raise ValueError("Source label definition differs from checkpoint")
    if allow_new_task_head and not resume:
        result = model.load_state_dict(state["model_state_dict"], strict=False)
        if result.unexpected_keys or any(not k.startswith("task_success_head.") for k in result.missing_keys):
            raise ValueError(f"Incompatible model state: {result}")
    else:
        model.load_state_dict(state["model_state_dict"], strict=True)
    if resume:
        if optimizer is None or scaler is None or schedule is None:
            raise ValueError("Resume requires optimizer, scaler and schedule")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        schedule.load_state_dict(state["sampler_state"])
        restore_rng(state["rng_state"])
    return state


def source_supervision_counts(prediction, packed):
    """Per-term denominators for logging, using CPU targets and tiny seed copies.

    Small seed tensors are copied to CPU; this may synchronize pending CUDA work.
    It does not consume RNG or change loss reduction.
    These counts describe existing loss reduction; they do not renormalize it.
    """
    inputs, targets = packed['inputs'], packed['targets']
    batch = inputs['batch_index'].numpy()
    instance = targets['instance_id_gt'].numpy()
    quality = targets['quality_valid'].numpy().astype(bool)
    valid_seed = prediction['seed_valid'].detach().cpu().numpy().astype(bool)
    selected = prediction['seed_point_index'].detach().cpu().numpy().reshape(-1)[valid_seed.reshape(-1)]
    def count(mask, indices=None):
        owned = mask & (instance > 0)
        return int(np.unique(batch[owned] if indices is None else batch[indices][owned[indices]]).size)
    counts = {'q0': count(quality), 'q': count(quality, selected),
              'normal': count(targets['axis_valid'].numpy().astype(bool)) if 'axis_valid' in targets else 0}
    for name, output, target, mask in [('seal', 'seal_aux', 'seal_target', 'seal_valid'),
                                      ('wrench', 'wrench_aux', 'wrench_target', 'wrench_valid'),
                                      ('collision_free', 'collision_free_logit', 'collision_free_target', 'collision_valid')]:
        if output in prediction:
            available = targets[mask].numpy().astype(bool) if mask in targets else quality.copy()
            available &= np.isfinite(targets[target].numpy()) if target in targets else False
            counts[name] = count(available, selected)
    return counts
