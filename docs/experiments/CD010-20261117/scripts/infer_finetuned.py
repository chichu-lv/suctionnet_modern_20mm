"""Observation-only inference for the fixed E5 target-finetuned checkpoint.

Writes a separate 401-frame result tree. Physical q/q0 remain untouched;
the trained task-success score selects and ranks NMS poses.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from infer_target import (asset, atomic_json, case_key, clean_metadata, existing_cases,
                          file_record, model_observation, read_test_rows, render_overlay,
                          require_compatible_identity, sample_seed, validate_observation)

SCHEMA = "suction20.target-finetuned.case.v1"
RANKING = dict(score_field="task_success_score", proposal_score_field="q0",
               score_semantics="learned_completed_pick_and_transfer_score_not_calibrated_probability",
               physical_quality_field="q", multiply_by_q=False)
NOTICE = ("E5 target finetune, step 1000; candidates are ranked by task_success_score. "
          "q and q0 retain source-domain physical proxy semantics. Task score is not a calibrated "
          "real-world success probability. Inference inputs contain only observed geometry, instance "
          "masks and approximate gravity; original planned actions and manual outcomes are display "
          "metadata only. Historical planned contact poses are not measured executed contact poses. "
          "No physical collision or new-action robot success evaluation is performed.")


def predict_with_logits(model, sample, config, device):
    """Capture raw logits from the same forward used by the frozen predictor."""
    from suction20.validation import predict_sample
    captured = {}

    def capture(_module, _inputs, output):
        valid = output["seed_valid"][0].bool()
        captured["logit"] = output["task_success_logit"][0][valid].detach().float().cpu().numpy()

    handle = model.register_forward_hook(capture)
    try:
        result = predict_sample(model, sample, config, device)
    finally:
        handle.remove()
    result["task_success_logit"] = captured["logit"]
    # Keep the frozen predictor's sigmoid precision and ranking unchanged.
    result["ranking_score"] = result["task_success_score"].copy()
    return result


def initialize(args):
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch
    from suction20.runner import make_model, require_device
    from suction20.training import load_checkpoint, load_config
    from suction20.validation import checkpoint_inference_config, use_task_success_ranking
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (state.get("stage") != "target_finetune" or state.get("target_mode") != "attempt_success"
            or state.get("global_step") != 1000 or not state.get("task_success_head_present")):
        raise ValueError("Expected step1000 attempt_success target-finetune checkpoint with trained head")
    config = checkpoint_inference_config(load_config(args.config), state)
    if config["experiment"] != "E5" or not use_task_success_ranking(state, config):
        raise ValueError("Expected E5 with task-success ranking enabled")
    source = state["source_label_config"]
    if not np.isclose(source["radius_m"], .020):
        raise ValueError("Expected frozen 20mm cup radius")
    torch.set_num_threads(args.threads)
    device = require_device(args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = make_model(config, source, device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    return config, source, model, device


def render_task_overlay(row, root, meta, prediction, selected, path):
    # Existing glyphs retain their truthful q labels; a banner explicitly names
    # the task ranking and its top score, avoiding a renamed physical q.
    result = render_overlay(row, root, meta, prediction, selected, path)
    if result["status"] == "rendered":
        from PIL import Image, ImageDraw
        with Image.open(path) as image:
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        score = f"{float(prediction['task_success_score'][selected[0]]):.4f}" if len(selected) else "none"
        draw.rectangle((0, 0, image.width, 30), fill=(15, 24, 32))
        draw.text((8, 8), f"Finetuned E5 step1000 | rank: task_success_score | top1={score} | q labels: physical proxy", fill="white")
        image.save(path, quality=92)
    result["ranking_score_field"] = "task_success_score"
    return result


def run(args):
    import numpy as np
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress = dict(status="preflight", total=401, completed=0, errors=0,
                    started_unix=time.time(), pid=os.getpid())

    def publish(**fields):
        progress.update(fields, updated_unix=time.time())
        atomic_json(output / "progress.json", progress)

    publish()
    try:
        rows = read_test_rows(args.manifest)
        if len(rows) != 401:
            raise ValueError("Expected exactly401 held-out observations")
        for row in rows:
            validate_observation(row, args.manifest.parent)
        publish(status="loading_model")
        config, source, model, device = initialize(args)
        identity = dict(schema_version=SCHEMA, identity_spec="declared_paths_config_v1",
                        checkpoint=file_record(args.checkpoint),
                        checkpoint_step=1000, checkpoint_stage="target_finetune", target_mode="attempt_success",
                        manifest=file_record(args.manifest), config=file_record(args.config),
                        config_definition=clean_metadata(config), source_label_config=clean_metadata(source),
                        worker=file_record(__file__), baseline_helper=file_record(Path(__file__).with_name("infer_target.py")),
                        ranking=RANKING)
        previous = output / "identity.json"
        if previous.exists():
            require_compatible_identity(json.loads(previous.read_text()), identity)
        atomic_json(previous, identity)
        import torch
        from suction20.target_dataset import load_target_observation
        from suction20.validation import nms_prediction
        atomic_json(output / "inference_config.json", config)
        summaries = []
        archived = existing_cases(output)
        publish(status="running")
        for row in rows:
            started = time.perf_counter()
            case_path = archived.get(row["sample_id"], output / "cases" / f"{case_key(row)}.json")
            key = case_path.stem
            npz_path = case_path.with_suffix(".npz")
            publish(current_sample=row["sample_id"])
            try:
                assets = {name: file_record(asset(args.manifest.parent, row["observation"][name]))
                          for name in ("depth_path", "instance_mask_path", "path", "rgb_path")
                          if row["observation"].get(name)}
                if case_path.exists():
                    case = json.loads(case_path.read_text())
                    require_compatible_identity(case["identity"], identity)
                    if (case["sample_id"] != row["sample_id"]
                            or clean_metadata(case["observation"]) != row["observation"]
                            or not npz_path.is_file()):
                        raise ValueError("Existing sample/observation declaration differs or NPZ is missing")
                else:
                    seed = sample_seed(row["sample_id"])
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    sample = load_target_observation(model_observation(row["observation"]), args.manifest.parent,
                        config.get("target_data", {}).get("normal_estimation"), source_label_config=source)
                    load_s = time.perf_counter() - started
                    before = time.perf_counter()
                    prediction = predict_with_logits(model, sample, config, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_s = time.perf_counter() - before
                    if any(not np.isfinite(v).all() for v in prediction.values() if isinstance(v, np.ndarray)):
                        raise ValueError("Nonfinite prediction")
                    nms = nms_prediction(prediction, config, task_success=True)
                    scene = np.asarray(nms["scene_indices"], np.int64)
                    per_instance = np.asarray(nms["per_instance_indices"], np.int64)
                    ranks = {int(index): rank for rank, index in enumerate(scene, 1)}
                    per_set = set(per_instance.tolist())
                    shape = sample["meta"].get("image_shape") or row["observation"].get("image_size_hw")
                    candidates = []
                    for i in range(len(prediction["q"])):
                        index = int(prediction["source_indices"][i])
                        uv = None
                        if sample["meta"]["source_index_kind"] == "pixel" and shape and 0 <= index < shape[0]*shape[1]:
                            uv = [index % shape[1], index // shape[1]]
                        candidate = {name: float(prediction[name][i]) for name in (
                            "q", "q0", "task_success_logit", "task_success_score", "ranking_score")}
                        candidate.update(prediction_index=i, point_cam_m=prediction["point_cam_m"][i].tolist(),
                            normal_out_cam=prediction["normal_out_cam"][i].tolist(),
                            instance_id=int(prediction["instance_id"][i]), source_index=index, source_pixel_uv=uv,
                            scene_rank=ranks.get(i), per_instance_nms=i in per_set)
                        candidates.append(candidate)
                    rendering = render_task_overlay(row, args.manifest.parent, sample["meta"], prediction,
                                                     scene, output / "images" / f"{key}.jpg")
                    case = dict(schema_version=SCHEMA, sample_id=row["sample_id"], split="test", identity=identity,
                        assets=assets, seed=seed, seed_spec="sample_utf8_seedsequence_v1",
                        candidates=candidates, rendering=rendering, ranking=RANKING,
                        source_index_kind=sample["meta"]["source_index_kind"], observation=row["observation"],
                        metadata={k:v for k,v in row.items() if k != "observation"}, notice=NOTICE,
                        timing=dict(load_seconds=load_s, inference_seconds=inference_s,
                                    total_seconds=time.perf_counter()-started))
                    npz_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = npz_path.with_name(npz_path.name + f".{os.getpid()}.tmp")
                    with tmp.open("wb") as stream:
                        np.savez_compressed(stream, **prediction, nms_scene_indices=scene,
                            nms_per_instance_indices=per_instance,
                            source_index_kind=np.asarray(sample["meta"]["source_index_kind"]))
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(tmp, npz_path)
                    atomic_json(case_path, case)
                top = next((c for c in case["candidates"] if c["scene_rank"] == 1), None)
                summaries.append(dict(sample_id=row["sample_id"], candidate_count=len(case["candidates"]),
                    top1_task_success_score=top["task_success_score"] if top else None, timing=case["timing"]))
                (output / "errors" / f"{key}.json").unlink(missing_ok=True)
                progress["completed"] += 1
                publish(last_frame_seconds=time.perf_counter()-started)
            except Exception as exc:
                atomic_json(output / "errors" / f"{key}.json", dict(sample_id=row["sample_id"],
                    error=str(exc), traceback=traceback.format_exc()))
                progress["errors"] += 1
                publish(last_error=str(exc))
        atomic_json(output / "summary.json", dict(identity=identity, ranking=RANKING, notice=NOTICE,
                    total=len(rows), completed=progress["completed"], errors=progress["errors"], frames=summaries))
        publish(status="completed" if not progress["errors"] else "completed_with_errors", current_sample=None)
        return int(bool(progress["errors"]))
    except Exception as exc:
        publish(status="failed", last_error=str(exc))
        atomic_json(output / "fatal_error.json", dict(error=str(exc), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "repo", "checkpoint", "config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in ("manifest", "repo", "checkpoint", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
