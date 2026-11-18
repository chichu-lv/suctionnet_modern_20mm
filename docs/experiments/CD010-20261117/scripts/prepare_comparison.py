"""Build display-only planned-pose comparisons and sampled RGB-D clouds.

Reads frozen E5 case JSONs and raw observations; never runs the model. Cloud
records are little-endian float32 x,y,z,r,g,b,instance_id,source_pixel_index.
Positions are camera metres, RGB is 0..1, and source_pixel_index = v*width+u.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from infer_target import asset, atomic_json, case_key, sha256


SCHEMA = "suction20.target-comparison.v1"
NOTICE = ("Geometric differences between historical planned pipeline contacts and "
          "selected E5 predictions; not accuracy, measured execution error or suction "
          "success. Original algorithm score and E5 q have different meanings. "
          "Camera geometry reproduces the approved approximate production calibration.")
FIELDS = ["x", "y", "z", "r", "g", "b", "instance_id", "source_pixel_index"]
SOURCE_CONFIG = Path("/home/databeyond/data/datasets/pose/suctionnet-r20-v1/"
                     "full-20261110/output/config_frozen.json")


def vector(value, size):
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    return result if result.shape == (size,) and np.isfinite(result).all() else None


def project(point, intrinsic):
    point = vector(point, 3)
    if point is None or point[2] <= 0:
        return None
    return [float(point[0] * intrinsic[0, 0] / point[2] + intrinsic[0, 2]),
            float(point[1] * intrinsic[1, 1] / point[2] + intrinsic[1, 2])]


def pose_key(pose):
    return pose.get("object_id"), pose.get("instance_id")


def compare_poses(case, intrinsic):
    """Join metadata by explicit identity; never infer pose/action association."""
    metadata = case.get("metadata", {})
    camera_poses = metadata.get("planned_pose_cam") or []
    source_poses = metadata.get("planned_poses") or []
    identities = Counter(pose_key(pose) for pose in camera_poses)
    score_key = case.get("ranking", {}).get("score_field", "q")
    if score_key not in ("q", "task_success_score"):
        raise ValueError(f"Unsupported ranking score: {score_key}")
    originals, comparisons = [], []
    for index, pose in enumerate(camera_poses):
        key = pose_key(pose)
        matches = [source for source in source_poses if pose_key(source) == key]
        complete_key = all(value is not None for value in key)
        match_status = ("missing_identity" if not complete_key else
                        "matched" if len(matches) == 1 else
                        "ambiguous" if matches else "missing")
        source = matches[0] if match_status == "matched" else {}
        original = dict(pose, original_index=index, metadata_match_status=match_status,
                        metadata_match_count=len(matches),
                        duplicate_camera_identity=identities[key] > 1,
                        source_pixel_uv=source.get("suction_uv"),
                        projected_pixel_uv=project(pose.get("point_cam_m"), intrinsic),
                        original_score=source.get("score"),
                        is_published_live_target=source.get("is_published_live_target"),
                        is_live_best=source.get("is_live_best"),
                        actual_execution_verified=False, label_valid=False)
        originals.append(original)
        comparison = dict(original_index=index, instance_id=pose.get("instance_id"),
                          status="unavailable", selection_scope=None,
                          distance_mm=None, outward_axis_angle_deg=None, pixel_delta_px=None)
        comparisons.append(comparison)
        instance_id = pose.get("instance_id")
        if instance_id is None or instance_id <= 0:
            comparison["status"] = "missing_instance_identity"
            continue
        if identities[key] > 1 or match_status == "ambiguous":
            comparison["status"] = "ambiguous_original_identity"
            continue
        candidates = [candidate for candidate in case.get("candidates", [])
                      if candidate.get("instance_id") == instance_id]
        ranked = [candidate for candidate in candidates if candidate.get("scene_rank") is not None]
        if ranked:
            selected = min(ranked, key=lambda candidate: (candidate["scene_rank"], candidate["prediction_index"]))
            scope = "best_scene_rank_same_instance"
        else:
            candidates = [candidate for candidate in candidates
                          if isinstance(candidate.get(score_key), (int, float)) and math.isfinite(candidate[score_key])]
            if not candidates:
                comparison["status"] = "no_same_instance_candidate"
                continue
            selected = min(candidates, key=lambda candidate: (-candidate[score_key], candidate["prediction_index"]))
            scope = f"highest_{score_key}_same_instance_all_candidates"
        comparison.update({key: selected.get(key) for key in (
            "prediction_index", "scene_rank", "instance_id", "q", "task_success_score", "point_cam_m",
            "normal_out_cam", "source_pixel_uv")})
        comparison["selection_scope"] = scope
        original_point, prediction_point = vector(pose.get("point_cam_m"), 3), vector(selected.get("point_cam_m"), 3)
        original_axis, prediction_axis = vector(pose.get("normal_out_cam"), 3), vector(selected.get("normal_out_cam"), 3)
        if any(value is None for value in (original_point, prediction_point, original_axis, prediction_axis)):
            comparison["status"] = "invalid_pose_geometry"
            continue
        norm_product = np.linalg.norm(original_axis) * np.linalg.norm(prediction_axis)
        if norm_product < 1e-12:
            comparison["status"] = "invalid_pose_axis"
            continue
        comparison.update(status="compared", distance_mm=float(np.linalg.norm(original_point-prediction_point)*1000),
                          outward_axis_angle_deg=float(np.degrees(np.arccos(np.clip(
                              np.dot(original_axis, prediction_axis)/norm_product, -1, 1)))))
        original_uv = original["projected_pixel_uv"]
        prediction_uv = project(prediction_point, intrinsic)
        comparison["projected_pixel_uv"] = prediction_uv
        # Both points are projected through the SAME recorded pinhole K. This
        # does not assume the source pipeline's stored suction_uv was rectified.
        if original_uv is not None and prediction_uv is not None:
            comparison["pixel_delta_px"] = float(np.linalg.norm(np.asarray(original_uv)-prediction_uv))
            comparison["pixel_delta_scope"] = "both_camera_positions_projected_with_same_pinhole_K"
    return originals, comparisons


def evenly_spaced(values, count):
    """Select deterministic bin midpoints in source-pixel order, without repeats."""
    if count >= len(values):
        return values
    indices = ((np.arange(count, dtype=np.int64)*2+1)*len(values)) // (2*count)
    return values[indices]


def sample_pixels(valid, mask, maximum):
    pixels = np.flatnonzero(valid.reshape(-1))
    ids = mask.reshape(-1)[pixels]
    foreground = pixels[ids > 0]
    background = pixels[ids == 0]
    target = min(maximum, len(pixels))
    fg_count = min(len(foreground), max(round(target*.6), target-len(background)))
    bg_count = min(len(background), target-fg_count)
    # Each instance is a separate stratum; proportional allocation preserves
    # coverage while foreground receives a greater sampling density overall.
    labels, counts = np.unique(mask.reshape(-1)[foreground], return_counts=True)
    quotas = counts.astype(np.float64)*fg_count/max(1, len(foreground))
    allocations = np.floor(quotas).astype(int)
    remainder = fg_count-int(allocations.sum())
    order = np.argsort(-(quotas-allocations), kind="stable")
    allocations[order[:remainder]] += 1
    selected = [evenly_spaced(background, bg_count)]
    for label, count in zip(labels, allocations):
        selected.append(evenly_spaced(foreground[mask.reshape(-1)[foreground] == label], int(count)))
    return np.sort(np.concatenate(selected))


def prepare_case(case_path, prepared, output, results, source_config, source_identity, maximum, shared_cloud_root=None):
    case = json.loads(case_path.read_text())
    key = case_key(case)
    if key != case_path.stem:
        raise ValueError(f"Case filename does not match sample identity: {case_path}")
    observation = case["observation"]
    if observation.get("projection_convention") != "pinhole":
        raise ValueError(f"Unsupported projection for {key}")
    intrinsic = np.asarray(observation["intrinsics"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
        raise ValueError(f"Invalid intrinsics for {key}")
    originals, comparisons = compare_poses(case, intrinsic)
    if shared_cloud_root is not None:
        # Reuse the immutable observed cloud; only model candidates and pairing change.
        reference_path = shared_cloud_root / "comparison" / f"{key}.json"
        reference = json.loads(reference_path.read_text())
        source_case = json.loads((shared_cloud_root / "cases" / case_path.name).read_text())
        if (reference["sample_id"] != case["sample_id"] or
                source_case["asset_sha256"] != case["asset_sha256"] or
                source_case["observation"] != observation or
                source_case.get("metadata") != case.get("metadata")):
            raise ValueError(f"Shared cloud source identity/geometry mismatch: {key}")
        result = copy.deepcopy(reference)
        cloud = result["cloud"]
        binary = (shared_cloud_root / cloud["url"]).resolve()
        if binary.stat().st_size != cloud["count"] * cloud["stride_bytes"]:
            raise ValueError(f"Shared cloud size mismatch: {binary}")
        cloud["url"] = Path(os.path.relpath(binary, results)).as_posix()
        cloud["shared_immutable_observation"] = True
        result.update(originals=originals, comparisons=comparisons, notice=NOTICE,
                      ranking_score=case.get("ranking", {}).get("score_field", "q"))
        result["provenance"].update(case_json_sha256=sha256(case_path),
                                     checkpoint_sha256=case["identity"]["checkpoint_sha256"],
                                     shared_cloud_comparison_sha256=sha256(reference_path))
        atomic_json(output / f"{key}.json", result)
        return result
    arrays = {}
    hashes = {}
    for field in ("depth_path", "rgb_path", "instance_mask_path"):
        path = asset(prepared, observation[field])
        hashes[field] = sha256(path)
        if hashes[field] != case["asset_sha256"][field]:
            raise ValueError(f"Frozen input digest mismatch: {path}")
        with Image.open(path) as image:
            arrays[field] = np.array(image.convert("RGB") if field == "rgb_path" else image)
    depth, rgb, mask = (arrays[field] for field in ("depth_path", "rgb_path", "instance_mask_path"))
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3) or mask.shape != depth.shape:
        raise ValueError(f"RGB/depth/mask shape mismatch for {key}")
    scale = float(observation["depth_scale_m"])
    minimum = float(observation.get("depth_min_m", source_config["depth_min_m"]))
    maximum_depth = float(observation.get("depth_max_m", source_config["depth_max_m"]))
    if not all(math.isfinite(x) for x in (scale, minimum, maximum_depth)) or scale <= 0 or not 0 <= minimum < maximum_depth:
        raise ValueError(f"Invalid depth scale/range for {key}")
    z = depth.astype(np.float64)*scale
    valid = np.isfinite(z) & (z > 0) & (z >= minimum) & (z <= maximum_depth)
    pixels = sample_pixels(valid, mask, maximum)
    height, width = depth.shape
    v, u = np.divmod(pixels, width)
    zz = z[v, u]
    xyz = np.stack(((u-intrinsic[0, 2])*zz/intrinsic[0, 0],
                    (v-intrinsic[1, 2])*zz/intrinsic[1, 1], zz), axis=1)
    packed = np.empty((len(pixels), len(FIELDS)), dtype="<f4")
    packed[:, :3] = xyz
    packed[:, 3:6] = rgb[v, u]/255.
    packed[:, 6] = mask[v, u]
    packed[:, 7] = pixels
    if np.any(packed[:, 7].astype(np.int64) != pixels):
        raise ValueError("Source pixel IDs exceed exact float32 integer storage")
    # Check the actual browser-precision coordinates, not just float64 helpers.
    projected = np.column_stack((packed[:, 0].astype(float)/packed[:, 2]*intrinsic[0, 0]+intrinsic[0, 2],
                                 packed[:, 1].astype(float)/packed[:, 2]*intrinsic[1, 1]+intrinsic[1, 2]))
    roundtrip_error = float(np.max(np.linalg.norm(projected-np.column_stack((u, v)), axis=1))) if len(pixels) else 0.
    if roundtrip_error > .001:
        raise ValueError(f"Source-pixel roundtrip failed: {roundtrip_error}")
    binary = output/f"{key}.bin"
    temporary = binary.with_suffix(".bin.tmp")
    packed.tofile(temporary)
    temporary.replace(binary)
    cloud = dict(url=binary.relative_to(results).as_posix(), count=len(pixels), stride=8, stride_bytes=32,
                 dtype="float32", endian="little", fields=FIELDS, rgb_range=[0, 1], coordinate_units="metres",
                 source_pixel_formula="v*width+u", source_image_size_hw=[height, width],
                 bounds=dict(min=xyz.min(axis=0).tolist(), max=xyz.max(axis=0).tolist()) if len(pixels) else None,
                 full_valid_count=int(valid.sum()), foreground_count=int(np.count_nonzero(packed[:, 6])),
                 background_count=int(np.count_nonzero(packed[:, 6] == 0)), maximum_points=maximum,
                 sampling="deterministic source-pixel bins within instance strata; 60% foreground budget, rest background; redistribute unused budget",
                 source_pixel_roundtrip_max_px=roundtrip_error, sha256=sha256(binary))
    result = dict(schema_version=SCHEMA, case_key=key, sample_id=case["sample_id"],
                  original_status="missing" if not originals else "single" if len(originals) == 1 else "multiple",
                  original_count=len(originals), original_source_pose_count=len(case.get("metadata", {}).get("planned_poses", [])),
                  originals=originals, comparisons=comparisons, cloud=cloud,
                  camera=dict(intrinsics=intrinsic.tolist(), image_size_hw=[height, width],
                              coordinate_frame=observation.get("coordinate_frame"), projection_convention="pinhole",
                              depth_scale_m=scale, depth_min_m=minimum, depth_max_m=maximum_depth,
                              depth_clip_source="case observation override, otherwise frozen source config"),
                  provenance=dict(case_json_sha256=sha256(case_path), asset_sha256=hashes,
                                  source_config=source_identity, checkpoint_sha256=case["identity"]["checkpoint_sha256"],
                                  planned_pose_source="metadata.planned_pose_cam", original_metadata_join="object_id+instance_id",
                                  geometry_provenance=case.get("metadata", {}).get("geometry_provenance")), notice=NOTICE)
    atomic_json(output/f"{key}.json", result)
    return result


def quantiles(values):
    if not values:
        return None
    return dict(zip(("min", "p10", "median", "p90", "p95", "max"),
                    np.quantile(values, [0, .1, .5, .9, .95, 1]).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Default: RESULTS/comparison; must be below RESULTS")
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--shared-cloud-root", type=Path,
                        help="Reuse existing baseline cases/comparison BINs without rereading raw RGB-D")
    args = parser.parse_args()
    if not 1 <= args.max_points <= 35000:
        parser.error("--max-points must be between 1 and 35000")
    results = args.results.resolve()
    output = (args.output or results/"comparison").resolve()
    if not output.is_relative_to(results) or output == results:
        parser.error("Output must be a subdirectory of results")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.source_config.read_text())
    identity = dict(path=str(args.source_config.resolve()), sha256=sha256(args.source_config),
                    depth_min_m=config["depth_min_m"], depth_max_m=config["depth_max_m"])
    cases = sorted((results/"cases").glob("*.json"))
    if not cases:
        parser.error("No existing case JSON files found")
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def prepare(path):
            return prepare_case(path, args.prepared, output, results, config, identity, args.max_points,
                                args.shared_cloud_root.resolve() if args.shared_cloud_root else None)
        for index, result in enumerate(pool.map(prepare, cases), 1):
            completed.append(result)
            if index % 25 == 0 or index == len(cases):
                print(f"Prepared {index}/{len(cases)} frames", flush=True)
    pairs = [pair for result in completed for pair in result["comparisons"]]
    compared = [pair for pair in pairs if pair["status"] == "compared"]
    summary = dict(schema_version=SCHEMA, frames=len(completed),
                   original_availability=dict(Counter(result["original_status"] for result in completed)),
                   original_poses=sum(result["original_count"] for result in completed),
                   metadata_matches=dict(Counter(pose["metadata_match_status"] for result in completed for pose in result["originals"])),
                   comparison_status=dict(Counter(pair["status"] for pair in pairs)),
                   selection_scope=dict(Counter(pair["selection_scope"] for pair in compared)),
                   same_instance_distance_mm=quantiles([pair["distance_mm"] for pair in compared]),
                   outward_axis_angle_deg=quantiles([pair["outward_axis_angle_deg"] for pair in compared]),
                   pixel_delta_px=quantiles([pair["pixel_delta_px"] for pair in compared if pair["pixel_delta_px"] is not None]),
                   angle_formula="degrees(acos(clamp(dot(a,b)/(norm(a)*norm(b)),-1,1))); oriented outward axes, no abs(dot)",
                   sampled_points=sum(result["cloud"]["count"] for result in completed),
                   maximum_roundtrip_error_px=max(result["cloud"]["source_pixel_roundtrip_max_px"] for result in completed),
                   source_config=identity, generator_sha256=sha256(__file__), notice=NOTICE)
    atomic_json(output/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
