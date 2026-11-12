"""Inference reads observations and checkpoint physics; no label directory.

Use --frames for public data or --observation-manifest for JSONL records with
sample_id and observation only. Target checkpoints should use their run's
resolved_config.json so the model, preprocessing and ranking settings match.
Manifest asset paths resolve relative to the manifest. Output keys use
sample_<URL-escaped sample_id>/... and retain point/pixel source_indices.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from suction20.datasets import PublicObservationDataset
from suction20.target_dataset import load_target_observation
from suction20.runner import add_runtime_arguments, make_model, override_paths, require_device
from suction20.training import load_checkpoint, load_config
from suction20.validation import (checkpoint_inference_config, nms_prediction,
                                  predict_sample, use_task_success_ranking)


def read_observation_manifest(path):
    """Validate lightweight descriptors before loading any observations."""
    rows, identities = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Observation manifest line {line_number} must be an object")
            identity = row.get("sample_id")
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"Observation manifest line {line_number} requires sample_id")
            if identity in identities:
                raise ValueError(f"Duplicate observation sample_id: {identity}")
            if not isinstance(row.get("observation"), dict):
                raise ValueError(f"Observation manifest line {line_number} requires observation object")
            identities.add(identity)
            # Never pass labels/attempt outcomes from extra record fields into
            # observation loading or prediction.
            rows.append(("sample_" + quote(identity, safe=""), row["observation"]))
    if not rows:
        raise ValueError("Observation manifest contains no records")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True,
                        help="Model config; use the run's resolved_config.json for target checkpoints")
    observations = parser.add_mutually_exclusive_group(required=True)
    observations.add_argument("--frames", help="Comma-separated scene:frame keys, e.g.9:0,19:8")
    observations.add_argument("--observation-manifest",
                              help="JSONL with sample_id + observation; no source or label root needed")
    parser.add_argument("--output", required=True)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    config = override_paths(load_config(args.config), args)
    device = require_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_inference_config(config, state)
    task = use_task_success_ranking(state, config)
    if args.observation_manifest:
        manifest = Path(args.observation_manifest).resolve()
        rows = read_observation_manifest(manifest)
        normal_estimation = config.get("target_data", {}).get("normal_estimation")
        samples = ((prefix, load_target_observation(observation, manifest.parent, normal_estimation,
                                                    source_label_config=state["source_label_config"]))
                   for prefix, observation in rows)
        count = len(rows)
    else:
        keys = [tuple(map(int, item.split(":"))) for item in args.frames.split(",")]
        dataset = PublicObservationDataset(config["data"]["source_root"], keys,
                                           data_cfg=state["source_label_config"],
                                           instance_source=config["data"].get("instance_mask_root") or "gt",
                                           camera=config["data"].get("camera", "realsense"))
        samples = (("scene_%04d_%s_%04d" % sample["meta"]["sample_uid"], sample) for sample in dataset)
        count = len(keys)
    model = make_model(config, state["source_label_config"], device)
    load_checkpoint(args.checkpoint, model)
    result = {}
    for prefix, sample in samples:
        prediction = predict_sample(model, sample, config, device)
        selected = nms_prediction(prediction, config, task_success=task)["scene_indices"]
        for name, value in prediction.items():
            if isinstance(value, np.ndarray):
                result[f"{prefix}/{name}"] = value[selected]
        score = prediction["task_success_score"] if task else prediction["q"]
        result[f"{prefix}/suction_api"] = np.column_stack((score[selected], prediction["normal_out_cam"][selected], prediction["point_cam_m"][selected]))
        if args.observation_manifest:
            result[f"{prefix}/source_index_kind"] = np.asarray(sample["meta"]["source_index_kind"])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    print(f"Saved {count} observation frame predictions to {path}")


if __name__ == "__main__":
    main()
