"""Train one public R20 experiment; smoke configs bound successful updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from suction20.datasets import collate_public, frame_key, split_scene_ids
from suction20.losses import source_losses
from suction20.runner import add_runtime_arguments, make_model, make_source_dataset, override_paths, require_device
from suction20.training import (FrameSchedule, content_digest, decay_groups, learning_rate,
                               load_checkpoint, load_config, move_to_device, optimizer_update,
                               save_checkpoint, seed_everything, validate_resume_config, worktree_digest)


def train(config, run_dir, device, *, resume=None, init_checkpoint=None, stop_at_step=None):
    run_dir = Path(run_dir)
    if (run_dir / "last.pt").exists() and not resume:
        raise ValueError("Run already has a checkpoint; resume it or choose a new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(config.get("runtime", {}).get("cpu_threads", 8))
    if device.type == "cuda":
        memory_limit = config.get("runtime", {}).get("gpu_memory_limit_gib", 64) * 1024**3
        fraction = min(1., memory_limit / torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    train_cfg = config["training"]
    if train_cfg.get("world_size", 1) != 1:
        raise ValueError("This trainer supports one GPU; no silent distributed batch change")
    microbatch = train_cfg.get("microbatch_size", 1)
    accumulation = train_cfg.get("gradient_accumulation_steps", 4)
    if microbatch * accumulation != train_cfg.get("effective_batch_size", 4):
        raise ValueError("Effective batch differs from microbatch × accumulation")
    dataset = make_source_dataset(config, augment=True)
    key_to_index = {key: i for i, key in enumerate(dataset.keys)}
    schedule = FrameSchedule(split_scene_ids("train"), seed=config.get("sampling_seed", 20261110),
                             views_per_scene=config["data"].get("views_per_train_scene_per_epoch", 64),
                             frame_keys=config["data"].get("smoke_frame_keys"))
    model = make_model(config, dataset.data_cfg, device)
    opt_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(decay_groups(model, weight_decay=opt_cfg.get("weight_decay", 1e-4)),
                                  lr=opt_cfg.get("lr_peak", .001), betas=tuple(opt_cfg.get("betas", (.9, .999))),
                                  eps=opt_cfg.get("eps", 1e-8))
    amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best = 0, None
    if resume or init_checkpoint:
        state = load_checkpoint(resume or init_checkpoint, model, optimizer=optimizer, scaler=scaler,
                                schedule=schedule, resume=bool(resume), source_label_config=dataset.data_cfg)
        if state["quality_target"] != config["model"]["quality_target"]:
            raise ValueError("Cannot relabel checkpoint quality target")
        if resume:
            validate_resume_config(state["training_config"], config)
            step, best = state["global_step"], state.get("best_metric")
    cap = int(train_cfg["max_optimizer_steps"])
    if stop_at_step is not None:
        cap = min(cap, stop_at_step)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "source_config_frozen.json").write_text(json.dumps(dataset.data_cfg, indent=2) + "\n")
    identity = {"source_label_config_sha256": content_digest(dataset.data_cfg),
                "worktree_sha256": worktree_digest(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (run_dir / "runtime.json").write_text(json.dumps(identity, indent=2) + "\n")
    scheduler = config.get("scheduler", {})
    parameters = [p for p in model.parameters() if p.requires_grad]
    probe = parameters[0]
    skips = 0
    optimizer.zero_grad(set_to_none=True)
    while step < cap:
        started = time.perf_counter()
        model.train()
        lr = learning_rate(step, peak=opt_cfg.get("lr_peak", .001),
                           start=scheduler.get("warmup_start_lr", 1e-5),
                           final=scheduler.get("final_lr", 1e-5),
                           warmup=scheduler.get("warmup_optimizer_steps", 2880),
                           total=scheduler.get("total_optimizer_steps", 57600))
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals, consumed, frame_stats = {}, [], []
        before = probe.detach().clone()
        for _ in range(accumulation):
            samples = []
            for _ in range(microbatch):
                key, aug_seed = schedule.next()
                normalized_key = frame_key(key, config["data"].get("camera", "realsense"))
                index = key_to_index[normalized_key]
                samples.append(dataset.get_sample(index, aug_seed))
                consumed.append(list(normalized_key))
            model.set_epoch(schedule.epoch)
            packed = collate_public(samples, config["model"].get("voxel_size_m", .002))
            frame_stats.append({"point_count": len(packed["inputs"]["xyz_cam_m"]),
                                "voxel_count": len(packed["inputs"]["voxel_coords"])})
            inputs = move_to_device(packed["inputs"], device)
            targets = move_to_device(packed["targets"], device)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(inputs, sampler_mode="train", targets=targets)
            frame_stats[-1]["valid_seed_count"] = int(prediction["seed_valid"].sum())
            if hasattr(model, "last_grouping_timings"):
                frame_stats[-1]["grouping_timings"] = dict(model.last_grouping_timings)
            loss = source_losses(prediction, targets, inputs,
                                 quality_target=config["model"]["quality_target"],
                                 beta=config.get("loss", {}).get("smooth_l1_beta", .05),
                                 weights=config.get("loss", {}).get("weights"))
            if not torch.isfinite(loss["total"]):
                raise FloatingPointError("Non-finite training loss; no optimizer update performed")
            scaler.scale(loss["total"] / accumulation).backward()
            for name, value in loss.items():
                totals[name] = totals.get(name, 0.) + float(value.detach()) / accumulation
            del samples, packed, prediction, loss, inputs, targets
        updated, grad_norm = optimizer_update(optimizer, scaler, parameters, opt_cfg.get("grad_clip_norm", 10.))
        if updated:
            step += 1
            skips = 0
        else:
            skips += 1
        row = dict(global_step=step, updated=updated, loss=totals,
                   grad_norm=grad_norm if np.isfinite(grad_norm) else None, gradients_finite=bool(np.isfinite(grad_norm)), lr=lr,
                   parameter_probe_delta=float((probe.detach() - before).norm()), frames=consumed,
                   frame_stats=frame_stats, amp_scale=scaler.get_scale(),
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   elapsed_seconds=time.perf_counter() - started,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        with (run_dir / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if skips >= 10:
            raise FloatingPointError("Ten consecutive skipped optimizer updates")
        if updated:
            save_checkpoint(run_dir / "last.pt", model, optimizer, scaler, schedule, global_step=step,
                            config=config, source_label_config=dataset.data_cfg, best_metric=best)
        validate_every = train_cfg.get("validate_every_optimizer_steps", 5760)
        if updated and (step % validate_every == 0 or step == cap):
            from suction20.validation import evaluate_model
            metrics = evaluate_model(model, config, device,
                                     split="smoke_val" if config["data"].get("smoke_val_frame_keys") else "val")
            with (run_dir / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(global_step=step, **metrics), allow_nan=False) + "\n")
            candidate = (metrics["instance_Top1_Q20_at_0_4"], metrics["AP_R20"])
            if best is None or candidate > tuple(best["values"]):
                best = {"values": candidate, "global_step": step}
                save_checkpoint(run_dir / "best.pt", model, optimizer, scaler, schedule, global_step=step,
                                config=config, source_label_config=dataset.data_cfg, best_metric=best)
            save_checkpoint(run_dir / "last.pt", model, optimizer, scaler, schedule, global_step=step,
                            config=config, source_label_config=dataset.data_cfg, best_metric=best)
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stop-at-step", type=int)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--resume")
    initial.add_argument("--init-checkpoint")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    config = override_paths(load_config(args.config), args)
    train(config, args.run_dir, require_device(args.device), resume=args.resume,
          init_checkpoint=args.init_checkpoint, stop_at_step=args.stop_at_step)


if __name__ == "__main__":
    main()
