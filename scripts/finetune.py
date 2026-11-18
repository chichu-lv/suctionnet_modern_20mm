"""Adapt a selected gt_B model using dense proxy labels or executed attempts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from suction20.datasets import collate_public, frame_key, split_scene_ids
from suction20.finetuning import (apply_phase, make_finetune_optimizer, resolve_finetune_config,
                                 set_bn_eval_and_freeze_affine, target_loss, target_selection)
from suction20.losses import source_losses
from suction20.runner import (add_runtime_arguments, augmentation_from_config, make_model,
                              make_source_dataset, require_device)
from suction20.training import (FrameSchedule, configuration_identity, load_checkpoint, load_config,
                               move_to_device, optimizer_update, save_checkpoint, seed_everything,
                               validate_resume_config)


def finetune(config, run_dir, device, *, init_checkpoint=None, resume=None, stop_at_step=None):
    from suction20.target_dataset import TargetSuction20Dataset, TargetGroupSchedule, collate_target
    from suction20.target_validation import evaluate_target
    if bool(init_checkpoint) == bool(resume):
        raise ValueError("Provide exactly one initialization or resume checkpoint")
    run_dir = Path(run_dir)
    if (run_dir / "last.pt").exists() and not resume:
        raise ValueError("Run already has a checkpoint; resume it or choose a new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(resume or init_checkpoint, map_location="cpu", weights_only=False)
    source_cfg = configuration_identity(state["source_label_config"])
    if not isinstance(source_cfg, dict):
        raise ValueError("Checkpoint source label definition must be a mapping")
    config = resolve_finetune_config(config, state)
    seed_everything(config.get("seed", 42))
    runtime = config.get("runtime", {})
    torch.set_num_threads(runtime.get("cpu_threads", 8))
    memory_limit = runtime.get("gpu_memory_limit_gib", 64)
    if device.type == "cuda" and memory_limit is not None:
        torch.cuda.set_per_process_memory_fraction(min(1., memory_limit * 1024**3 /
                                                       torch.cuda.get_device_properties(device).total_memory), device)
    mode = config["mode"]
    target_data = config["target_data"]
    target_kwargs = dict(mode=mode, source_label_config=source_cfg,
                         success_definition=target_data.get("success_definition"), seed=config.get("seed", 42),
                         normal_estimation=target_data.get("normal_estimation"))
    target = TargetSuction20Dataset(target_data["manifest"], split="train",
                                    augment=augmentation_from_config(config["augmentation"]), **target_kwargs)
    validation = TargetSuction20Dataset(target_data["manifest"], split="val", augment=None, **target_kwargs)
    if not len(target) or not len(validation):
        raise ValueError("Target adaptation requires independent nonempty train and validation groups")
    source = make_source_dataset(config, augment=True)
    if configuration_identity(source.data_cfg) != configuration_identity(source_cfg):
        raise ValueError("Source replay label definition differs from checkpoint")
    source_index = {key: index for index, key in enumerate(source.keys)}
    source_schedule = FrameSchedule(split_scene_ids("train"), seed=config.get("sampling_seed", 20261110))
    target_schedule = TargetGroupSchedule(target, seed=config.get("seed", 42))
    model = make_model(config, source_cfg, device)
    optimizer = make_finetune_optimizer(model, config.get("optimizer", {}))
    amp = bool(config["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loaded = load_checkpoint(resume or init_checkpoint, model, optimizer=optimizer, scaler=scaler,
                              schedule=source_schedule, resume=bool(resume),
                              allow_new_task_head=mode == "attempt_success", source_label_config=source_cfg)
    step, best = 0, None
    source_ref = str(init_checkpoint)
    if resume:
        if loaded.get("stage") != "target_finetune" or loaded.get("target_mode") != mode:
            raise ValueError("Resume requires a target checkpoint of the same mode")
        validate_resume_config(loaded["training_config"], config)
        if "target_manifest_definition" not in loaded:
            raise ValueError("Legacy target sampler uses opaque observation IDs; exact resume is unavailable. "
                             "Load the weights as initialization for a fresh finetuning run.")
        if loaded["target_manifest_definition"] != target.manifest_definition:
            raise ValueError("Target manifest changed on resume")
        target_schedule.load_state_dict(loaded["target_sampler_state"])
        step, best = loaded["global_step"], loaded.get("best_target_metric")
        source_ref = loaded["source_checkpoint_ref"]
    config_path = run_dir / "resolved_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "source_config_frozen.json").write_text(json.dumps(source_cfg, indent=2) + "\n")
    cap = config["training"]["max_optimizer_steps"]
    if stop_at_step is not None:
        cap = min(cap, stop_at_step)
    def save(name):
        save_checkpoint(run_dir / name, model, optimizer, scaler, source_schedule,
                        global_step=step, config=config, source_label_config=source_cfg, best_metric=best,
                        stage="target_finetune", target_mode=mode, source_checkpoint_ref=source_ref,
                        target_manifest_definition=target.manifest_definition, target_split_groups=target.split_groups,
                        success_definition=target.success_definition, target_sampler_state=target_schedule.state_dict(),
                        task_success_head_present=mode == "attempt_success",
                        parameter_phase="heads_only" if step < 200 else "decoder_and_heads",
                        bn_mode="eval_frozen_affine", best_target_metric=best, uncertainty_lambda=0.)
    def validate(baseline=False):
        nonlocal best
        metrics = evaluate_target(model, validation, config, device, source_baseline=baseline)
        with (run_dir / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps({**metrics, "global_step": step, "source_baseline": baseline}, allow_nan=False) + "\n")
        selection = target_selection(metrics, mode)
        if not baseline and selection is not None and (best is None or selection[1] > best["value"]):
            best = {"metric": selection[0], "value": selection[1], "global_step": step}
            save("best.pt")
    if not resume:
        validate(baseline=True)
        save("last.pt")
    skips = 0
    while step < cap:
        model.train()
        phase = apply_phase(model, optimizer, step)
        set_bn_eval_and_freeze_affine(model)
        optimizer.zero_grad(set_to_none=True)
        totals, observations = {"target": 0., "source": 0.}, []
        unmapped = 0
        for _ in range(2):
            index, seed = target_schedule.next()
            sample = target.get_sample(index, seed)
            packed = move_to_device(collate_target([sample], config["model"].get("voxel_size_m", .002)), device)
            with torch.autocast(device_type=device.type, enabled=amp):
                losses, diagnostics = target_loss(model, packed, mode, config.get("loss"))
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite target loss")
            scaler.scale(.5 * losses["total"]).backward()
            totals["target"] += .5 * float(losses["total"].detach())
            unmapped += diagnostics.get("unmapped_attempt_count", 0)
            observations.append(str(sample["meta"].get("observation_id", index)))
            del sample, packed, losses
        for _ in range(2):
            key, seed = source_schedule.next()
            sample = source.get_sample(source_index[frame_key(key, config["data"].get("camera", "realsense"))], seed)
            packed = move_to_device(collate_public([sample], config["model"].get("voxel_size_m", .002)), device)
            model.set_epoch(source_schedule.epoch)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(packed["inputs"], sampler_mode="train", targets=packed["targets"])
            cfg = config["source_loss"]
            losses = source_losses(prediction, packed["targets"], packed["inputs"], quality_target="gt_B",
                                    beta=cfg.get("smooth_l1_beta", .05), weights=cfg.get("weights"))
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite source replay loss")
            scaler.scale(.25 * losses["total"]).backward()
            totals["source"] += .25 * float(losses["total"].detach())
            del sample, packed, losses, prediction
        updated, norm = optimizer_update(optimizer, scaler, [p for p in model.parameters() if p.requires_grad],
                                         config.get("optimizer", {}).get("grad_clip_norm", 10.))
        if updated:
            step += 1
            skips = 0
        else:
            skips += 1
        row = {"global_step": step, "updated": updated, "parameter_phase": phase, "loss": totals,
               "grad_norm": norm if torch.isfinite(torch.tensor(norm)) else None,
               "target_observations": observations, "unmapped_attempt_count": unmapped}
        with (run_dir / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if skips >= 10:
            raise FloatingPointError("Ten consecutive skipped optimizer updates")
        if updated:
            if step == 200 or step % config.get("validation", {}).get("every_optimizer_steps", 100) == 0 or step == cap:
                validate()
            save("last.pt")
            if step == 200:
                save("step_200.pt")
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-manifest")
    parser.add_argument("--stop-at-step", type=int)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--init-checkpoint")
    initial.add_argument("--resume")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.target_manifest:
        config["target_data"]["manifest"] = str(Path(args.target_manifest).resolve())
    for key in ("source_root", "label_root"):
        value = getattr(args, key)
        if value:
            config["source_replay"][key] = str(Path(value).resolve())
    # Backbone path is transport metadata, applied to the inherited architecture.
    if args.backbone_root:
        config.setdefault("runtime", {})["backbone_root"] = str(Path(args.backbone_root).resolve())
    finetune(config, args.run_dir, require_device(args.device), init_checkpoint=args.init_checkpoint,
             resume=args.resume, stop_at_step=args.stop_at_step)


if __name__ == "__main__":
    main()
