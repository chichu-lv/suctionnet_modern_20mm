"""Train one public R20 experiment; smoke configs bound successful updates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
import uuid
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from suction20.performance import attach_performance, build_runtime_identity
from suction20.precision import resolve_precision, UpdateLossAccumulator
from suction20.geometry_pipeline import GeometryPipeline, GeometryCompletion
from suction20.pretrain_budget import CONTRACT_VERSION, resolve_budget, resolve_config, sampling_epoch, validate_resume_progress
from suction20.datasets import collate_public, frame_key, split_scene_ids
from suction20.losses import source_losses
from suction20.runner import add_runtime_arguments, make_model, make_source_dataset, override_paths, require_device
from suction20.training import (FrameSchedule, content_digest, make_optimizer, learning_rate,
                               load_checkpoint, load_config, move_to_device, select_training_gpu_inputs, optimizer_update,
                               save_checkpoint, seed_everything, validate_resume_config, worktree_digest,
                               capture_rng, restore_rng, semantic_config, code_revision, source_supervision_counts)
from suction20.pretrain_runtime import (AttemptLog, CheckpointStore, Progress, RunLock,
                                       atomic_json, file_digest, read_json, normalize_gpu_uuid)


def train(config, run_dir, device, *, resume=None, init_checkpoint=None, stop_at_step=None,
          pause_at_step=None, control_dir=None, validation_gate=False, attempt_id=None,
          profile_update=None):
    """Return successful step count; CLI maps durable paused status to exit 75."""
    config = resolve_config(config)
    budget = resolve_budget(config)
    run_dir = Path(run_dir).absolute()
    lock = RunLock(run_dir)
    progress = None
    try:
        progress = Progress(control_dir or run_dir / "control", run=run_dir,
                            experiment=config.get("experiment"), attempt_id=attempt_id or uuid.uuid4().hex,
                            target_step=int(config["training"]["max_optimizer_steps"]))
        return _train(config, run_dir, device, progress, resume=resume, init_checkpoint=init_checkpoint,
                      stop_at_step=stop_at_step, pause_at_step=pause_at_step, validation_gate=validation_gate,
                      profile_update=profile_update)
    except Exception as error:
        if progress is not None:
            progress.update(phase="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if progress is not None:
            progress.close()
        lock.close()


def _train(config, run_dir, device, progress, *, resume, init_checkpoint, stop_at_step,
           pause_at_step, validation_gate, profile_update=None):
    budget = resolve_budget(config)
    effective_batch = budget["effective_batch_size"]
    profile = config.get("runtime_profile")
    performance = config.get("runtime", {}).get("performance")
    fresh_execution = bool(performance and performance.get("schema_version") == "suction20.performance.v4")
    if budget["contract_version"] == CONTRACT_VERSION:
        allowed_status = ("selected", "from_scratch") if fresh_execution else ("selected",)
        if not isinstance(profile, dict) or profile.get("training_version") != "v3" or profile.get("selection_status") not in allowed_status or not profile.get("profile_id") or profile["profile_id"] == "pending":
            raise ValueError("Formal v3 requires a selected hardware runtime_profile")
    progress.update(budget=budget, contract_version=budget["contract_version"], runtime_profile=profile,
                    effective_samples_completed=0, target_effective_samples=budget["target_samples"],
                    completed_validation_nodes=[], last_completed_validation=None)
    store = CheckpointStore(run_dir)
    if (store.manifest or (run_dir / "last.pt").exists()) and not resume:
        raise ValueError("Run already has a checkpoint; resume it or choose a new run directory")
    if resume and init_checkpoint:
        raise ValueError("Resume and initialization are mutually exclusive")
    if fresh_execution and init_checkpoint:
        raise ValueError("From-scratch performance requires new initialization, not init_checkpoint")
    if pause_at_step is not None and pause_at_step < 1:
        raise ValueError("pause_at_step must be positive")
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
    pipeline_depth = train_cfg.get("geometry_pipeline_depth", 1)
    if type(pipeline_depth) is not int or pipeline_depth not in (1, 2):
        raise ValueError("training.geometry_pipeline_depth must be 1 or 2")
    if pipeline_depth == 2 and (not fresh_execution or accumulation < 2 or
                               config.get("experiment") not in ("E3", "E4", "E5")):
        raise ValueError("Geometry pipeline depth 2 requires fresh v4 contact training and accumulation >= 2")
    if profile_update is not None and (type(profile_update) is not int or profile_update < 1):
        raise ValueError("profile_update must be a positive successful-update boundary")
    dataset = make_source_dataset(config, augment=True)
    key_to_index = {key: i for i, key in enumerate(dataset.keys)}
    schedule = FrameSchedule(split_scene_ids("train"), seed=config.get("sampling_seed", 20261110),
                             views_per_scene=config["data"].get("views_per_train_scene_per_epoch", 64),
                             frame_keys=config["data"].get("smoke_frame_keys"))
    model = make_model(config, dataset.data_cfg, device)
    opt_cfg = config.get("optimizer", {})
    optimizer = make_optimizer(model, opt_cfg, device)
    precision = resolve_precision(config, device)
    scaler = precision.make_scaler()
    label_root = Path(config["data"].get("label_root", "."))
    frozen_path = label_root / "config_frozen.json"
    if frozen_path.exists():
        config["data"]["frozen_label_config"] = str(frozen_path.resolve())
    manifest_path = label_root / "manifest.json"
    formal = budget["formal"]
    if formal and (not manifest_path.is_file() or not frozen_path.is_file()):
        raise ValueError("Formal pretraining requires the frozen release manifest and label configuration")
    if formal and stop_at_step is not None:
        raise ValueError("Use pause_at_step for formal training; bounded stops add off-protocol validation")
    if formal and init_checkpoint:
        raise ValueError("Formal experiments require independent initialization")
    identity = build_runtime_identity(config, dataset.data_cfg, code_revision_value=code_revision(),
                                      worktree_sha256=worktree_digest())
    math_settings = identity["math_settings"]
    migration_receipt = None
    progress.update(performance_identity=identity.get("performance"), performance_settings=performance)
    step, best, previous = 0, None, None
    if resume or init_checkpoint:
        source = store.recover(resume) if resume else init_checkpoint
        state = load_checkpoint(source, model, optimizer=optimizer, scaler=scaler,
                                schedule=schedule, resume=bool(resume), source_label_config=dataset.data_cfg)
        if state["quality_target"] != config["model"]["quality_target"]:
            raise ValueError("Cannot relabel checkpoint quality target")
        if resume:
            validate_resume_config(state["training_config"], config)
            previous = state.get("runtime_state")
            if state.get("stage") != "public_pretrain" or state.get("experiment") != config.get("experiment"):
                raise ValueError("Resume experiment or stage differs")
            if not previous or previous.get("schema_version") != 1 or previous["identity"] != identity:
                raise ValueError("Resume requires matching complete runtime/code/data identity")
            if state["scheduler_state_dict"]["global_step"] != state["global_step"]:
                raise ValueError("Resume scheduler step differs")
            validate_resume_progress(state, budget)
            if previous.get("performance_migration"):
                from suction20.pretrain_runtime import verify_performance_migration
                migration_receipt = verify_performance_migration(run_dir, previous["performance_migration"], identity)
            step, best = state["global_step"], state.get("best_metric")
    cap = int(train_cfg["max_optimizer_steps"])
    if stop_at_step is not None:
        cap = min(cap, stop_at_step)
    if cap < step:
        raise ValueError("Stop target precedes resumed step")
    pause_active = pause_at_step is not None and step < pause_at_step
    execution = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                 "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                 "gpu_uuid": normalize_gpu_uuid(torch.cuda.get_device_properties(device).uuid) if device.type == "cuda" else None,
                 "math_settings": math_settings, "runtime": config.get("runtime", {}),
                 "model_execution": getattr(model, "execution_settings", {}),
                 "geometry_pipeline_depth": pipeline_depth,
                 "precision": precision.as_dict(),
                 "optimizer_execution": {"name": "AdamW", "fused_requested": opt_cfg.get("fused", False),
                                          "fused_effective": bool(optimizer.defaults.get("fused"))}}
    if migration_receipt:
        execution["performance_diagnostic_window"] = {
            "start_step": migration_receipt["global_step"],
            "end_step_exclusive": migration_receipt["global_step"] + 50}
    if device.type == "cuda":
        # NVML can report host PIDs outside our container PID namespace. The
        # supervisor may identify a sole NVML process only after this positive
        # acknowledgement that our own context is live on the assigned device.
        torch.cuda.synchronize(device)
        progress.update(cuda_context_ready=True, gpu_uuid=execution["gpu_uuid"])
    log = AttemptLog(run_dir, progress.state["attempt_id"], identity={"contract": identity,
                     "execution": execution, "resolved_config": config},
                     previous=(previous or {}).get("ledger"), recovery=store.recovery)
    if not resume:
        if (run_dir / "runtime.json").exists():
            if read_json(run_dir / "runtime.json")["identity"] != identity:
                log.close()
                raise ValueError("Initial run identity differs from existing immutable metadata")
        else:
            atomic_json(run_dir / "resolved_config.json", config)
            atomic_json(run_dir / "source_config_frozen.json", dataset.data_cfg)
            atomic_json(run_dir / "runtime.json", {"identity": identity, "execution": execution})
    pending = (previous or {}).get("pending_validation")
    validations = dict((previous or {}).get("validations", {}))
    for reference in validations.values():
        if file_digest(run_dir / reference["path"]) != reference["sha256"]:
            log.close()
            raise ValueError("Committed validation result checksum mismatch")
    skips = int((previous or {}).get("consecutive_skips", 0))
    last_completed_validation = (previous or {}).get("last_completed_validation")
    completed_validation_nodes = list((previous or {}).get("completed_validation_nodes", []))
    log.event("resume" if resume else "attempt_started", global_step=step, budget=budget, runtime_profile=profile,
              checkpoint=store.current, recovery=store.recovery)
    progress.update(global_step=step, effective_samples_completed=step * effective_batch,
                    last_completed_validation=last_completed_validation, completed_validation_nodes=completed_validation_nodes)
    migration = (previous or {}).get("performance_migration")
    diagnostic_until_step = migration_receipt["global_step"] + 50 if migration_receipt else None
    if diagnostic_until_step is not None:
        log.event("performance_diagnostic_window", start_step=migration_receipt["global_step"],
                  end_step_exclusive=diagnostic_until_step,
                  policy="first 50 successful updates after explicit migration; timing only")
    saved_step, saved_time = step, time.monotonic()

    def save(*, retain=False, promote_best=False, reason="cadence"):
        nonlocal saved_step, saved_time
        checkpoint_started = time.monotonic()
        log.event("checkpoint_started", global_step=step, reason=reason)
        progress.update(phase="checkpointing", global_step=step)
        runtime_state = dict(schema_version=1, identity=identity, ledger=log.snapshot(),
                             pending_validation=pending, validations=validations, consecutive_skips=skips,
                             budget=budget, effective_samples_completed=step * effective_batch,
                             last_completed_validation=last_completed_validation, completed_validation_nodes=completed_validation_nodes)
        if migration:
            runtime_state["performance_migration"] = migration
        def writer(path):
            save_checkpoint(path, model, optimizer, scaler, schedule, global_step=step,
                            config=config, source_label_config=dataset.data_cfg, best_metric=best,
                            runtime_state=runtime_state, code_revision=identity["code_revision"],
                            code_worktree_sha256=identity["worktree_sha256"])
        prefix_node = (store.manifest or {}).get("nodes", {}).get(str(budget["prefix_step"]))
        # Keep the paused prefix's original checkpoint/hash across new attempts.
        # The first attempt may refresh its node through the normal pause save,
        # but a resume's initial save must not replace that recovery evidence.
        retain_prefix = step == budget["prefix_step"] and (not prefix_node or prefix_node["attempt_id"] == log.attempt_id)
        analysis = dict(schema_version=1, global_step=step, attempt_id=log.attempt_id, identity=identity,
                        ledger=runtime_state["ledger"], validations=validations, best=best, budget=budget,
                        counters={"effective_samples_completed": step * effective_batch, "consumed_samples": log.frames,
                                  "skipped_samples": log.skipped * effective_batch, "optimizer_attempts": log.attempts},
                        last_completed_validation=last_completed_validation, completed_validation_nodes=completed_validation_nodes)
        if migration:
            analysis["performance_migration"] = migration
        entry = store.publish(writer, step=step, attempt_id=log.attempt_id, pending=pending,
                              best_step=best["global_step"] if best else None,
                              retain=retain or retain_prefix, promote_best=promote_best, analysis=analysis)
        saved_step, saved_time = step, time.monotonic()
        log.event("checkpoint_published", global_step=step, reason=reason, checkpoint=entry["path"],
                  sha256=entry["sha256"], elapsed_seconds=time.monotonic() - checkpoint_started)
        progress.update(durable_checkpoint=entry, global_step=step, effective_samples_completed=step * effective_batch,
                        last_completed_validation=last_completed_validation, completed_validation_nodes=completed_validation_nodes)

    def pause():
        log.event("pause_requested", global_step=step)
        save(reason="pause")
        progress.update(phase="paused", global_step=step)
        log.event("paused", global_step=step, effective_samples_completed=step * effective_batch)
        return step

    class ValidationPaused(Exception):
        pass

    def validate_pending():
        nonlocal pending, best, last_completed_validation
        from suction20.validation import evaluate_model
        grant = None
        if validation_gate:
            log.event("validation_wait", global_step=step)
            grant = progress.validation_grant(store.current, step)
            if grant is None:
                return False
        validation_started = time.monotonic()
        log.event("validation_started", global_step=step, transaction_id=pending["id"])
        result_path = run_dir / "validations" / f"{step:06d}-{pending['id']}.json"
        progress.update(phase="validating", validation={"step": step, "completed_frames": 0,
                                                      "total_frames": None})
        if result_path.exists():
            result = read_json(result_path)
            if result["transaction_id"] != pending["id"] or result["identity"] != identity:
                raise ValueError("Validation transaction identity mismatch")
            metrics = result["metrics"]
        else:
            rng = capture_rng()
            was_training = model.training
            try:
                def report(completed, total, sample_uid):
                    progress.update(progressed=completed > 0, phase="validating",
                                    validation={"step": step, "completed_frames": completed,
                                                "total_frames": total, "sample_uid": sample_uid})
                    if progress.pause_requested():
                        raise ValidationPaused()
                metrics = evaluate_model(model, config, device,
                            split="smoke_val" if config["data"].get("smoke_val_frame_keys") else "val",
                            progress=report)
            except ValidationPaused:
                log.event("validation_paused", global_step=step)
                return False
            finally:
                restore_rng(rng)
                model.train(was_training)
            expected_frames = len(config["data"].get("smoke_val_frame_keys") or range(320))
            if len(metrics["frames"]) != expected_frames:
                raise ValueError("Validation frame count differs from fixed protocol")
            atomic_json(result_path, {"transaction_id": pending["id"], "identity": identity,
                                      "experiment": config.get("experiment"), "global_step": step,
                                      "metrics": metrics})
        candidate = (metrics["instance_Top1_Q20_at_0_4"], metrics["AP_R20"])
        if any(value is None or not np.isfinite(value) for value in candidate):
            raise ValueError("Validation selection metrics must be finite")
        improved = best is None or candidate > tuple(best["values"])
        if improved:
            best = {"values": candidate, "global_step": step}
        validations[str(step)] = {"path": str(result_path.relative_to(run_dir)),
                                  "sha256": file_digest(result_path)}
        last_completed_validation = dict(step=step, effective_samples=step * effective_batch,
                                         frame_count=len(metrics["frames"]), result=validations[str(step)],
                                         metrics={"instance_Top1_Q20_at_0_4": candidate[0], "AP_R20": candidate[1]})
        completed_validation_nodes[:] = [node for node in completed_validation_nodes if node["step"] != step]
        completed_validation_nodes.append({key: last_completed_validation[key] for key in ("step", "effective_samples", "frame_count")})
        log.event("validation_completed", global_step=step, result=validations[str(step)],
                  metrics=last_completed_validation["metrics"], improved=improved, elapsed_seconds=time.monotonic() - validation_started)
        pending = None
        save(retain=True, promote_best=improved, reason="post_validation")  # one commit for best, result, and pending-clear
        if grant:
            atomic_json(progress.directory / "validation.done.json", grant)
        progress.update(phase="training", validation=None, progressed=True)
        return True

    scheduler = config.get("scheduler", {})
    parameters = [p for p in model.parameters() if p.requires_grad]
    probe = parameters[0]
    optimizer.zero_grad(set_to_none=True)
    loader, loader_iter = None, None
    geometry_pipeline = GeometryPipeline() if pipeline_depth == 2 else None
    active_profiler, profile_done = None, False
    consumed_count = log.frames
    try:
        if performance:
            from suction20.prefetch import make_ticket_loader, consume_tickets, close_ticket_loader
            from suction20.cpu_geometry import CpuGeometry, SamplerTargetsCPU
            from suction20.run_supervisor import process_identity
            loader = make_ticket_loader(dataset, schedule, start_sequence_id=consumed_count,
                microbatch=microbatch, voxel_size_m=config["model"].get("voxel_size_m", .002),
                loader_config=performance["loader"], seed=config.get("sampling_seed", 20261110))
            loader_iter = iter(loader)
            progress.update(loader_worker_identities=[process_identity(worker.pid)
                            for worker in getattr(loader_iter, "_workers", [])])
        save(reason="resume" if resume else "initial")
        while True:
            if progress.pause_requested() or (pause_active and step >= pause_at_step):
                return pause()
            if pending is not None:
                if pending["step"] != step:
                    raise ValueError("Pending validation step differs from checkpoint")
                if not validate_pending():
                    return pause()
            if step >= cap:
                save(retain=True, reason="final")
                log.event("completed", global_step=step, effective_samples_completed=step * effective_batch)
                progress.update(phase="completed", global_step=step, progressed=True)
                return step
            progress.update(phase="training", global_step=step)
            started = time.perf_counter()
            started_unix = time.time()
            started_monotonic = time.monotonic()
            started_utc = datetime.now(timezone.utc).isoformat()
            sampler_before = schedule.state_dict()
            amp_scale_before = scaler.get_scale()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            detailed_timing_active = bool(performance and (performance["detailed_cpu_timing"] or
                (diagnostic_until_step is not None and step < diagnostic_until_step)))
            grouping = getattr(model, "contact_grouping", None)
            if grouping is not None:
                grouping.detailed_cpu_timing = detailed_timing_active
            progress.update(performance_diagnostics={"active": detailed_timing_active,
                            "until_step_exclusive": diagnostic_until_step})
            lr = learning_rate(step, peak=opt_cfg.get("lr_peak", .001),
                               start=scheduler.get("warmup_start_lr", 1e-5),
                               final=scheduler.get("final_lr", 1e-5),
                               warmup=scheduler.get("warmup_optimizer_steps", 2880),
                               total=scheduler.get("total_optimizer_steps", 57600))
            for group in optimizer.param_groups:
                group["lr"] = lr
            totals, consumed, frame_stats, frame_details = {}, [], [], []
            loss_accumulator = UpdateLossAccumulator(accumulation) if fresh_execution else None
            data_seconds, collate_seconds, loader_wait_s = 0., 0., 0.
            production_timing = {}
            timing = {name: 0. for name in ("data", "collate", "h2d_host", "forward_host", "loss_host", "backward_host", "optimizer_host", "diagnostics_host")}
            before = probe.detach().clone()
            pipeline_timing = {"depth": pipeline_depth, "geometry_job_wall_s": 0.,
                               "geometry_wait_host_s": 0.}
            timing["geometry_wait_host"] = 0.
            profile_trace = None
            if profile_update is not None and not profile_done and step + 1 == profile_update:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                active_profiler = torch.profiler.profile(activities=activities, record_shapes=True)
                active_profiler.start()
                profile_done = True

            def prepare_microbatch(micro_index):
                nonlocal data_seconds, collate_seconds, loader_wait_s
                samples, micro_frames, cpu_kwargs = [], [], {}
                if loader_iter is not None:
                    waiting = time.perf_counter()
                    packed = next(loader_iter)
                    loader_wait_s += time.perf_counter() - waiting
                    tickets = packed["tickets"]
                    production = packed.get("production_timing", {})
                    data_seconds += production.get("prepare_total_s", 0.)
                    collate_seconds += production.get("collate_total_s", 0.)
                    for name, seconds in production.items():
                        production_timing[name] = production_timing.get(name, 0.) + seconds
                    for index, ticket in enumerate(tickets):
                        offsets = packed["inputs"]["point_offsets"]
                        detail = dict(sequence_id=ticket.sequence_id, sample_uid=list(ticket.frame_key), augmentation_seed=ticket.augmentation_seed,
                            microbatch_index=micro_index, point_count=int(offsets[index + 1] - offsets[index]),
                            consumed_sampler_epoch=ticket.sample_epoch)
                        micro_frames.append(detail)
                        frame_details.append(detail)
                        consumed.append(list(ticket.frame_key))
                    cpu_kwargs = {"cpu_geometry": CpuGeometry.from_inputs(packed["inputs"]),
                                  "sampler_targets_cpu": SamplerTargetsCPU.from_targets(packed["targets"])}
                else:
                    for _ in range(microbatch):
                        key, aug_seed = schedule.next()
                        normalized_key = frame_key(key, config["data"].get("camera", "realsense"))
                        index = key_to_index[normalized_key]
                        loading_started = time.perf_counter()
                        samples.append(dataset.get_sample(index, aug_seed))
                        data_seconds += time.perf_counter() - loading_started
                        consumed.append(list(normalized_key))
                        detail = dict(sequence_id=log.frames + len(consumed) - 1, sample_uid=list(normalized_key), augmentation_seed=aug_seed,
                                      microbatch_index=micro_index, point_count=len(samples[-1]["inputs"]["xyz_cam_m"]),
                                      consumed_sampler_epoch=schedule.epoch)
                        micro_frames.append(detail)
                        frame_details.append(detail)
                    collate_started = time.perf_counter()
                    packed = collate_public(samples, config["model"].get("voxel_size_m", .002))
                    collate_seconds += time.perf_counter() - collate_started
                regime_epoch = sampling_epoch(budget, step, tickets[-1].sample_epoch if loader_iter is not None else schedule.epoch)
                model.set_epoch(regime_epoch)
                for frame_index, detail in enumerate(micro_frames):
                    detail["voxel_count"] = int((packed["inputs"]["voxel_coords"][:, 0] == frame_index).sum())
                stat = {"microbatch_index": micro_index, "sampling_regime_epoch": regime_epoch, "point_count": len(packed["inputs"]["xyz_cam_m"]),
                                    "voxel_count": len(packed["inputs"]["voxel_coords"])}
                frame_stats.append(stat)
                if loader_iter is not None:
                    stat.update(worker_pid=packed.get("worker_pid"),
                        cache_info=packed.get("cache_info"), production_timing=packed.get("production_timing"))
                timer = time.perf_counter()
                upload = select_training_gpu_inputs(packed["inputs"]) if performance else packed["inputs"]
                inputs = move_to_device(upload, device, non_blocking=bool(performance))
                targets = move_to_device(packed["targets"], device, non_blocking=bool(performance))
                timing["h2d_host"] += time.perf_counter() - timer
                payload = {"packed": packed, "inputs": inputs, "targets": targets,
                           "micro_frames": micro_frames, "stat": stat,
                           "tickets": tickets if loader_iter is not None else None,
                           "cpu_kwargs": cpu_kwargs}
                if geometry_pipeline is not None:
                    timer = time.perf_counter()
                    with precision.autocast(), torch.profiler.record_function("geometry.gpu_prepare"):
                        prepared = model.prepare_forward(inputs, sampler_mode="train", targets=targets, **cpu_kwargs)
                    timing["forward_host"] += time.perf_counter() - timer
                    payload["prepared"] = prepared
                    return payload, prepared.geometry_job
                return payload

            if geometry_pipeline is not None:
                microbatches = geometry_pipeline.iter_update(accumulation, prepare_microbatch)
            else:
                microbatches = (GeometryCompletion(prepare_microbatch(index)) for index in range(accumulation))
            for work in microbatches:
                payload = work.payload
                packed, inputs, targets = payload["packed"], payload["inputs"], payload["targets"]
                micro_frames, stat = payload["micro_frames"], payload["stat"]
                timer = time.perf_counter()
                if geometry_pipeline is not None:
                    with precision.autocast(), torch.profiler.record_function("geometry.gpu_finish"):
                        prediction = model.finish_forward(payload["prepared"], geometry_result=work.geometry_result)
                    stat["geometry_pipeline"] = work.timing
                    pipeline_timing["geometry_job_wall_s"] += work.timing["geometry_job_wall_s"]
                    pipeline_timing["geometry_wait_host_s"] += work.timing["geometry_wait_host_s"]
                    timing["geometry_wait_host"] += work.timing["geometry_wait_host_s"]
                else:
                    with precision.autocast(), torch.profiler.record_function("model.forward"):
                        prediction = model(inputs, sampler_mode="train", targets=targets, **payload["cpu_kwargs"])
                timing["forward_host"] += time.perf_counter() - timer
                seed_counts = prediction["seed_valid"].sum(dim=-1).detach().cpu().tolist()
                if not isinstance(seed_counts, list):
                    seed_counts = [seed_counts]
                if len(seed_counts) != len(micro_frames):
                    raise ValueError("Prediction frame count differs from packed microbatch")
                for detail, count in zip(micro_frames, seed_counts):
                    detail["valid_seed_count"] = int(count)
                stat["valid_seed_count"] = sum(int(count) for count in seed_counts)
                if hasattr(model, "last_grouping_timings"):
                    stat["grouping_timings"] = dict(model.last_grouping_timings)
                timer = time.perf_counter()
                loss = source_losses(prediction, targets, inputs,
                                     quality_target=config["model"]["quality_target"],
                                     beta=config.get("loss", {}).get("smooth_l1_beta", .05),
                                     weights=config.get("loss", {}).get("weights"))
                timing["loss_host"] += time.perf_counter() - timer
                if loss_accumulator is None and not torch.isfinite(loss["total"]):
                    raise FloatingPointError("Non-finite training loss; no optimizer update performed")
                timer = time.perf_counter()
                with torch.profiler.record_function("model.backward"):
                    scaler.scale(loss["total"] / accumulation).backward()
                if loss_accumulator is not None:
                    loss_accumulator.add(loss)
                else:
                    for name, value in loss.items():
                        totals[name] = totals.get(name, 0.) + float(value.detach()) / accumulation
                timing["backward_host"] += time.perf_counter() - timer
                timer = time.perf_counter()
                if "seed_point_index" in prediction and "quality_valid" in packed["targets"]:
                    stat["supervised_frame_counts"] = source_supervision_counts(prediction, packed)
                else:
                    stat["supervised_frame_counts"] = None
                timing["diagnostics_host"] += time.perf_counter() - timer
                if loader_iter is not None:
                    consumed_count = consume_tickets(schedule, payload["tickets"], consumed_count)
                # Drop the completed graph before the generator fills its next slot.
                del work, payload, packed, prediction, loss, inputs, targets
            if loss_accumulator is not None:
                timer = time.perf_counter()
                totals = loss_accumulator.result()
                timing["diagnostics_host"] += time.perf_counter() - timer
                if not np.isfinite(totals["total"]):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Non-finite training loss; no optimizer update performed")
            timer = time.perf_counter()
            updated, grad_norm = optimizer_update(optimizer, scaler, parameters, opt_cfg.get("grad_clip_norm", 10.))
            timing["optimizer_host"] += time.perf_counter() - timer
            if active_profiler is not None:
                active_profiler.stop()
                profile_trace = f"geometry-profile-step{step + 1}-{log.attempt_id}.json"
                active_profiler.export_chrome_trace(str(run_dir / profile_trace))
                active_profiler = None
            timing.update(data=data_seconds, collate=collate_seconds, loader_wait_s=loader_wait_s)
            if updated:
                step += 1
                skips = 0
            else:
                skips += 1
            row = dict(schema_version="suction20.update.v3", run_id=run_dir.name,
                       contract_version=budget["contract_version"], runtime_profile=profile,
                       global_step=step, updated=updated, loss=totals,
                       microbatch_size=microbatch, gradient_accumulation_steps=accumulation, effective_batch_size=effective_batch,
                       effective_samples_completed=step * effective_batch,
                       consumed_samples=log.frames + len(consumed), skipped_samples=(log.skipped + int(not updated)) * effective_batch,
                       sampler_state_before=sampler_before, sampler_state_after=schedule.state_dict(),
                       frame_details=frame_details,
                       started_utc=started_utc, ended_utc=datetime.now(timezone.utc).isoformat(),
                       started_unix=started_unix, ended_unix=time.time(),
                       started_monotonic=started_monotonic, ended_monotonic=time.monotonic(),
                       timing_seconds=timing, production_timing=production_timing,
                       geometry_pipeline=pipeline_timing, profile_trace=profile_trace,
                       loader_wait_s=loader_wait_s, performance_identity=identity.get("performance"),
                       performance_diagnostics={"active": detailed_timing_active,
                           "until_step_exclusive": diagnostic_until_step},
                       timing_scope="Host intervals with incidental synchronization; excludes logging/checkpoint/validation; producer data/collate and background geometry overlap training; loader_wait_s and geometry_wait_host are exposed waits; not isolated GPU kernel times",
                       memory_scope="CUDA peaks reset at each accumulation attempt; RSS is process lifetime maximum",
                       grad_norm_scope="unscaled pre-clipping total gradient norm",
                       amp_scale_before=amp_scale_before, amp_scale_after=scaler.get_scale(),
                       precision=precision.as_dict(), loss_readback="per_update" if fresh_execution else "per_microbatch",
                       skip_reason=None if updated else "nonfinite_gradients",
                       grad_norm=grad_norm if np.isfinite(grad_norm) else None, gradients_finite=bool(np.isfinite(grad_norm)), lr=lr,
                       parameter_probe_delta=float((probe.detach() - before).norm()), frames=consumed,
                       frame_stats=frame_stats, amp_scale=scaler.get_scale(),
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       elapsed_seconds=time.perf_counter() - started,
                       data_seconds=data_seconds, collate_seconds=collate_seconds,
                       peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                       gpu_allocated_bytes=torch.cuda.memory_allocated(device) if device.type == "cuda" else None,
                       gpu_reserved_bytes=torch.cuda.memory_reserved(device) if device.type == "cuda" else None,
                       gpu_memory_unavailable_reason=None if device.type == "cuda" else "cpu_device")
            log.append(row)
            progress.update(global_step=step, effective_samples_completed=step * effective_batch,
                            consumed_samples=row["consumed_samples"], skipped_samples=row["skipped_samples"],
                            progressed=True, phase="training",
                            last_update={key: row[key] for key in
                                         ("data_seconds", "collate_seconds", "loader_wait_s", "elapsed_seconds", "updated")})
            print(json.dumps({"attempt_id": log.attempt_id, "global_step": step,
                              "updated": updated, "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
            if skips >= 10:
                save()
                raise FloatingPointError("Ten consecutive skipped optimizer updates")
            validate_every = train_cfg.get("validate_every_optimizer_steps", 5760)
            if updated and (step % validate_every == 0 or step == cap):
                pending = {"step": step, "id": uuid.uuid4().hex}
                save(reason="pre_validation")
            elif step - saved_step >= budget["checkpoint_interval_updates"] or time.monotonic() - saved_time >= 900:
                save()
    except Exception as error:
        log.event("failed", global_step=step, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if active_profiler is not None:
            active_profiler.stop()
        if geometry_pipeline is not None:
            geometry_pipeline.close()
        if loader is not None:
            close_ticket_loader(loader, loader_iter)
            progress.update(loader_worker_identities=[])
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--performance-config", help="Independent strict performance sidecar; frozen profile stays unchanged")
    parser.add_argument("--stop-at-step", type=int, help="Legacy bounded smoke stop; includes final validation")
    parser.add_argument("--pause-at-step", type=int, help="Pause at a successful update boundary without extra validation")
    parser.add_argument("--control-dir")
    parser.add_argument("--validation-gate", action="store_true")
    parser.add_argument("--attempt-id")
    parser.add_argument("--profile-update", type=int, help="Capture one update CPU/CUDA trace; use a separate diagnostic run, not a throughput window")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--gpu-memory-limit-gib", type=float)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--resume")
    initial.add_argument("--init-checkpoint")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    config = override_paths(load_config(args.config), args)
    config.setdefault("runtime", {})["raw_config_sha256"] = file_digest(args.config)
    if args.performance_config:
        config = attach_performance(config, args.performance_config)
    for key in ("cpu_threads", "gpu_memory_limit_gib"):
        value = getattr(args, key)
        if value is not None:
            if value <= 0:
                parser.error(f"--{key.replace('_', '-')} must be positive")
            config.setdefault("runtime", {})[key] = value
    train(config, args.run_dir, require_device(args.device), resume=args.resume,
          init_checkpoint=args.init_checkpoint, stop_at_step=args.stop_at_step,
          pause_at_step=args.pause_at_step, control_dir=args.control_dir,
          validation_gate=args.validation_gate, attempt_id=args.attempt_id, profile_update=args.profile_update)
    status = read_json(Path(args.control_dir or Path(args.run_dir) / "control") / "status.json")
    if status["phase"] == "paused":
        sys.exit(75)


if __name__ == "__main__":
    main()
