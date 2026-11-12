"""Shared source construction and model invocation for command-line programs."""
from __future__ import annotations

from pathlib import Path

import torch

from .datasets import PublicSuction20Dataset, public_frame_keys, split_scene_ids
from .observation import AugmentConfig


def augmentation_from_config(config):
    if config.get("enabled", True) is False:
        return None
    if config.get("scale", False):
        raise ValueError("Metric R20 geometry does not support scale augmentation")
    yaw = config.get("yaw_about_gravity_deg", [-180, 180])
    translation = config.get("translation_in_table_plane_m", [-.02, .02])
    dropout = config.get("point_dropout_fraction", [0, .05])
    if yaw[0] != -yaw[1] or translation[0] != -translation[1] or dropout[0] != 0:
        raise ValueError("Expected symmetric rotation/translation and zero-based dropout intervals")
    return AugmentConfig(yaw_max_deg=float(yaw[1]), translation_max_m=float(translation[1]),
                         depth_noise_std_m=float(config.get("depth_noise_std_m", .0005)),
                         depth_noise_clip_m=float(config.get("depth_noise_clip_m", .0015)),
                         dropout_max=float(dropout[1]))


def make_source_dataset(config, *, split="train", frame_keys=None, augment=False):
    data = config["data"]
    camera = data.get("camera", "realsense")
    if frame_keys is None:
        if split == "train":
            frame_keys = data.get("smoke_frame_keys") or [
                (s, f) for s in split_scene_ids("train") for f in range(256)]
        elif split == "smoke_val":
            frame_keys = data["smoke_val_frame_keys"]
        else:
            frame_keys = public_frame_keys(split, camera=camera)
    return PublicSuction20Dataset(data["source_root"], data["label_root"], frame_keys,
                                  camera=camera, augment=augmentation_from_config(config.get("augmentation", {})) if augment else None,
                                  instance_source=data.get("instance_mask_root") or "gt",
                                  source_format=data.get("source_format", "legacy_dc4ad6b"),
                                  compatibility=data.get("compatibility", "auto"),
                                  adapter_cache=data.get("adapter_cache"), seed=config.get("seed", 42))


def make_model(config, frozen_config, device):
    from models.suction20net import Suction20Net
    sampling = config.get("sampling", {})
    supported = {"instance_balanced": True, "high_A_threshold": .2,
                 "warmup_epochs": 2,
                 "warmup_fractions": {"proposal": 0., "high_A": .5, "low_A": .5},
                 "main_fractions": {"proposal": .5, "high_A": .25, "low_A": .25},
                 "grouping_axis": "detached_predicted_normal"}
    for key, value in sampling.items():
        if key not in supported or value != supported[key]:
            raise ValueError(f"Unsupported sampling policy: {key}={value}")
    model_config = dict(config["model"])
    if config.get("experiment"):
        model_config["experiment"] = config["experiment"]
    return Suction20Net(model_config, frozen_config).to(device)


def override_paths(config, args):
    for name in ("source_root", "label_root"):
        value = getattr(args, name, None)
        if value:
            config["data"][name] = str(Path(value).resolve())
    if getattr(args, "backbone_root", None):
        config["model"]["backbone_root"] = str(Path(args.backbone_root).resolve())
    return config


def add_runtime_arguments(parser):
    parser.add_argument("--source-root")
    parser.add_argument("--label-root")
    parser.add_argument("--backbone-root")
    parser.add_argument("--device", default="cuda")


def require_device(device):
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; a real sparse GPU backbone is required for model smoke")
    if result.type == "cuda" and result.index is None:
        # Some CUDA APIs (notably the allocator limit) require an explicit
        # device index even though Tensor.to accepts the CLI's bare "cuda".
        result = torch.device("cuda", torch.cuda.current_device())
    return result
