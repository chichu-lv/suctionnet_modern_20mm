"""Fixed pretrained E5 inference on geometry-ready target test observations.

Uses the same observation/prediction APIs as the frozen repository's infer.py.
No training, physical evaluator, or outcome-derived model inputs are used.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

CHECKPOINT_SHA256 = "5bc2c14c014fbf23cbef48062a0dac523ea243b6c18e540bb5af022ff7d0ea3a"
CHECKPOINT_STEP = 2880
SCHEMA = "suction20.target-baseline.case.v1"
NOTICE = ("Pretrained E5; no finetuning. q/q0 are predicted source-domain quality scores, "
          "not probabilities of completed_pick_and_transfer. Original manual outcomes describe "
          "historical actions, not these predicted candidates. No physical collision truth or "
          "own-scene suction success metric is available. This production release uses offline "
          "approximate gravity opposite the belt normal and a numerically reproduced production "
          "calibration, not independently verified physical calibration. Historical planned poses "
          "are not measured executed contact poses.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def asset(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def case_key(row):
    return hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()


def read_test_rows(manifest):
    rows, identities = [], set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or not row["sample_id"].strip():
            raise ValueError(f"Line {number}: sample_id required")
        if row["sample_id"] in identities:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        identities.add(row["sample_id"])
        if row.get("split") not in ("train", "val", "test"):
            raise ValueError(f"Line {number}: explicit train/val/test split required")
        if row["split"] == "test":
            rows.append(row)
    if not rows:
        raise ValueError("Manifest contains no test frames")
    return rows


def validate_observation(row, root):
    """Fail before CUDA initialization; never infer scale/gravity from defaults."""
    obs = row.get("observation", {})
    if row.get("geometry_ready") is not True:
        raise ValueError("geometry_ready=true required after explicit calibration preparation")
    if obs.get("format") not in ("rgbd", "pointcloud_npz"):
        raise ValueError("Observation must be geometry-ready rgbd or pointcloud_npz")
    gravity = obs.get("g_cam")
    if not isinstance(gravity, list) or len(gravity) != 3 or not all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in gravity) or sum(x*x for x in gravity) <= 0:
        raise ValueError("Explicit finite nonzero g_cam required")
    paths = ["path"] if obs["format"] == "pointcloud_npz" else ["depth_path", "instance_mask_path"]
    if obs["format"] == "rgbd":
        scale = obs.get("depth_scale_m")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Explicit positive depth_scale_m (metres per raw unit) required")
        if obs.get("projection_convention") != "pinhole":
            raise ValueError("Frozen RGB-D loader requires explicit projection_convention=pinhole")
    if obs["format"] == "rgbd" or obs.get("projection_convention") == "pinhole":
        import numpy as np
        intrinsic = np.asarray(obs.get("intrinsics"), dtype=float)
        if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1])
                or intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0):
            raise ValueError("intrinsics must be a finite zero-skew pinhole 3x3 K")
    if obs.get("rgb_path"):
        paths.append("rgb_path")
    for key in paths:
        if not isinstance(obs.get(key), str) or not asset(root, obs[key]).is_file():
            raise ValueError(f"Missing observation asset: {key}")


def model_observation(obs):
    # Explicit allowlist: outcomes, planned poses and review metadata cannot leak.
    keys = ("format", "depth_path", "instance_mask_path", "depth_scale_m", "intrinsics",
            "depth_min_m", "depth_max_m", "g_cam", "path")
    return {key: obs[key] for key in keys if key in obs}


def render_overlay(row, root, sample_meta, prediction, selected, destination):
    """Draw source-pixel centers and projected 20mm circles in predicted tangent planes."""
    import numpy as np
    from PIL import Image, ImageDraw
    obs = row["observation"]
    if not obs.get("rgb_path"):
        return dict(status="unavailable", reason="No RGB image supplied")
    with Image.open(asset(root, obs["rgb_path"])) as source:
        image = source.convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = destination.with_name(destination.stem + "_rgb.jpg")
    image.save(original, quality=90)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    shape = sample_meta.get("image_shape") or obs.get("image_size_hw")
    pixels_ok = (sample_meta.get("source_index_kind") == "pixel" and shape is not None
                 and list(shape) == [height, width])
    projection_ok = obs.get("projection_convention") == "pinhole" and "intrinsics" in obs
    if shape is not None and list(shape) != [height, width]:
        projection_ok = False
    intrinsic = np.asarray(obs["intrinsics"], float) if projection_ok else None

    def project(points):
        points = np.asarray(points)
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 1e-6):
            return None
        uvw = points @ intrinsic.T
        return uvw[:, :2] / uvw[:, 2:]

    centers = outlines = axes = 0
    for rank, index in enumerate(selected, 1):
        point = np.asarray(prediction["point_cam_m"][index], float)
        axis = np.asarray(prediction["normal_out_cam"][index], float)
        center = None
        if pixels_ok:
            pixel = int(prediction["source_indices"][index])
            if 0 <= pixel < width * height:
                center = np.array([pixel % width, pixel // width], float)
        color = (255, 220, 40) if rank == 1 else (50, 235, 180)
        if projection_ok and np.isfinite(axis).all() and np.linalg.norm(axis) > 1e-8:
            axis /= np.linalg.norm(axis)
            reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
            tangent = np.cross(axis, reference)
            tangent /= np.linalg.norm(tangent)
            bitangent = np.cross(axis, tangent)
            angles = np.linspace(0, 2 * np.pi, 65)
            rim = point + .020 * (np.cos(angles)[:, None] * tangent + np.sin(angles)[:, None] * bitangent)
            projected_rim = project(rim)
            if projected_rim is not None:
                draw.line([tuple(p) for p in projected_rim], fill=color, width=2)
                outlines += 1
            arrow = project(np.stack([point, point + .030 * axis]))
            if arrow is not None:
                draw.line([tuple(p) for p in arrow], fill=(255, 90, 100), width=2)
                end = arrow[1]
                draw.ellipse((end[0]-2, end[1]-2, end[0]+2, end[1]+2), fill=(255, 90, 100))
                axes += 1
        if center is not None:
            u, v = center
            draw.ellipse((u-3, v-3, u+3, v+3), fill=color)
            draw.text((u+5, v+3), f"{rank} q={float(prediction['q'][index]):.3f}", fill=color)
            centers += 1
    image.save(destination, quality=92)
    return dict(status="rendered", overlay=destination.name, rgb=original.name,
                source_pixel_centers=centers, projected_cup_outlines=outlines, projected_axes=axes,
                projection_convention="pinhole; distortion not applied" if projection_ok else "unavailable",
                radius_m=.020, axis_length_m=.030, axis_direction="outward normal", image_shape=[height, width])


def write_gallery(output, rows):
    escape = lambda value: html.escape(str(value), quote=True)
    cards = []
    for row in rows:
        key = case_key(row)
        case_path = output / "cases" / f"{key}.json"
        case = json.loads(case_path.read_text()) if case_path.exists() else None
        manual = json.dumps(row.get("raw_review", row.get("manual_outcome")), ensure_ascii=False)
        body = f"<h2>{escape(row['sample_id'])}</h2><p>Original manual label: {escape(manual)}</p>"
        if row.get("geometry_provenance"):
            body += ("<details><summary>Geometry provenance</summary><pre>" +
                     escape(json.dumps(row["geometry_provenance"], ensure_ascii=False, indent=2)) + "</pre></details>")
        if case:
            rendering = case["rendering"]
            if rendering.get("overlay"):
                body += (f'<a href="images/{escape(rendering["rgb"])}">Original RGB</a> '
                         f'<a href="cases/{key}.json">JSON</a> <a href="cases/{key}.npz">All candidates NPZ</a>'
                         f'<img loading="lazy" src="images/{escape(rendering["overlay"])}" alt="Predicted suction poses">')
            else:
                body += f"<p>Overlay unavailable: {escape(rendering.get('reason'))}</p>"
            body += "<table><tr><th>Rank</th><th>Instance</th><th>q</th><th>q0</th><th>Source pixel</th></tr>"
            for candidate in sorted(case["candidates"], key=lambda c: c["scene_rank"] or math.inf):
                if candidate["scene_rank"] is not None:
                    values = (candidate["scene_rank"], candidate["instance_id"], f"{candidate['q']:.5f}",
                              f"{candidate['q0']:.5f}", candidate["source_pixel_uv"])
                    body += "<tr>" + "".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>"
            body += "</table>"
        else:
            error_path = output / "errors" / f"{key}.json"
            reason = json.loads(error_path.read_text()).get("error") if error_path.exists() else "Pending"
            body += f"<p>{escape(reason)}</p>"
        cards.append("<article>" + body + "</article>")
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                '<title>Pretrained E5 target test baseline</title><style>body{font:16px system-ui;background:#101820;color:#eee;'
                'margin:24px}article{border:1px solid #567;padding:16px;margin:24px 0}img{display:block;max-width:100%;'
                'height:auto;margin:12px 0}td,th{padding:4px 12px;text-align:left}a{color:#8ed7ff}h2{overflow-wrap:anywhere}'
                'pre{white-space:pre-wrap;overflow-wrap:anywhere}'
                '</style><h1>Pretrained E5 · target test observations</h1><p>' + escape(NOTICE) + '</p><p>'
                'Yellow: top rank; green: other NMS candidates; pink: outward axis. Cup radius: 20 mm. '
                'Projection uses declared pinhole K without distortion; visibility and collision are not evaluated.</p>'
                + "".join(cards) + '</html>')
    temporary = output / f"index.html.{os.getpid()}.tmp"
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, output / "index.html")


def initialize_model(args):
    """Load the real fixed model; optional waiting never substitutes a placeholder."""
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch
    from suction20.performance import attach_performance
    from suction20.runner import make_model
    from suction20.training import load_checkpoint, load_config
    from suction20.validation import checkpoint_inference_config
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from fixed E5-best SHA256")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = attach_performance(load_config(args.config), args.performance)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("global_step") != CHECKPOINT_STEP:
        raise ValueError("Expected E5-best checkpoint step 2880")
    config = checkpoint_inference_config(config, state)
    if config["experiment"] != "E5":
        raise ValueError("Only pretrained E5 is supported")
    source_config = state["source_label_config"]
    if not np.isclose(source_config["radius_m"], .020):
        raise ValueError("Checkpoint cup radius differs from 20mm")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = make_model(config, source_config, device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    return config, source_config, model, device


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    progress = dict(status="preflight", total=0, completed=0, errors=0, started_unix=time.time(), pid=os.getpid())

    def publish(**updates):
        progress.update(updates, updated_unix=time.time())
        atomic_json(output / "progress.json", progress)

    publish()
    rows = []
    try:
        initialized = None
        if args.wait_manifest_seconds:
            publish(status="loading_model")
            initialized = initialize_model(args)
            deadline = time.monotonic() + args.wait_manifest_seconds
            publish(status="waiting_for_manifest", wait_deadline_unix=time.time()+args.wait_manifest_seconds)
            while not manifest.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Ready manifest was not published within {args.wait_manifest_seconds}s")
                time.sleep(min(1., max(0., deadline-time.monotonic())))
                publish()
            publish(status="preflight")
        rows = read_test_rows(manifest)
        publish(total=len(rows))
        for row in rows:
            try:
                validate_observation(row, manifest.parent)
            except Exception as exc:
                atomic_json(output / "errors" / f"{case_key(row)}.json",
                            dict(sample_id=row["sample_id"], stage="preflight", error=str(exc)))
                progress["errors"] += 1
        if progress["errors"]:
            publish(status="blocked", last_error="Unready test geometry; no observations inferred")
            write_gallery(output, rows)
            return 2
        if args.preflight_only:
            publish(status="preflight_passed")
            return 0
        if sha256(args.checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint differs from fixed E5-best SHA256")
        identity = dict(checkpoint_sha256=CHECKPOINT_SHA256, config_sha256=sha256(args.config),
                        performance_sha256=sha256(args.performance), manifest_sha256=sha256(manifest),
                        worker_sha256=sha256(__file__), checkpoint_step=CHECKPOINT_STEP, schema_version=SCHEMA)
        identity["repo_source_sha256"] = {name: sha256(args.repo / name) for name in (
            "suction20/target_dataset.py", "suction20/validation.py", "suction20/training.py",
            "suction20/performance.py", "suction20/runner.py", "models/suction20net.py")}
        saved_identity = output / "identity.json"
        if saved_identity.exists() and json.loads(saved_identity.read_text()) != identity:
            raise ValueError("Output identity differs; use a new output directory")
        atomic_json(saved_identity, identity)
        if initialized is None:
            initialized = initialize_model(args)
        config, source_config, model, device = initialized
        import numpy as np
        import torch
        from suction20.target_dataset import load_target_observation
        from suction20.validation import nms_prediction, predict_sample
        atomic_json(output / "inference_config.json", config)
        publish(status="running")
        for row in rows:
            key = case_key(row)
            case_path = output / "cases" / f"{key}.json"
            npz_path = case_path.with_suffix(".npz")
            publish(current_sample=row["sample_id"])
            started = time.perf_counter()
            try:
                asset_hashes = {name: sha256(asset(manifest.parent, row["observation"][name]))
                                for name in ("depth_path", "instance_mask_path", "path", "rgb_path")
                                if row["observation"].get(name)}
                if case_path.exists():
                    previous = json.loads(case_path.read_text())
                    if previous["identity"] != identity or previous["asset_sha256"] != asset_hashes or not npz_path.is_file():
                        raise ValueError("Existing case identity/assets differ or NPZ is missing")
                    progress["completed"] += 1
                    publish()
                    continue
                seed = int.from_bytes(hashlib.sha256(("42:" + row["sample_id"]).encode()).digest()[:4], "big")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                sample = load_target_observation(model_observation(row["observation"]), manifest.parent,
                            config.get("target_data", {}).get("normal_estimation"), source_label_config=source_config)
                load_seconds = time.perf_counter() - started
                before = time.perf_counter()
                prediction = predict_sample(model, sample, config, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds = time.perf_counter() - before
                if any(not np.isfinite(value).all() for value in prediction.values() if isinstance(value, np.ndarray)):
                    raise ValueError("Nonfinite prediction")
                nms = nms_prediction(prediction, config)
                scene = np.asarray(nms["scene_indices"], np.int64)
                per_instance = np.asarray(nms["per_instance_indices"], np.int64)
                rendering = render_overlay(row, manifest.parent, sample["meta"], prediction, scene,
                                           output / "images" / f"{key}.jpg")
                ranks = {int(index): rank for rank, index in enumerate(scene, 1)}
                per_instance_set = set(per_instance.tolist())
                shape = sample["meta"].get("image_shape") or row["observation"].get("image_size_hw")
                candidates = []
                for i in range(len(prediction["q"])):
                    source_index = int(prediction["source_indices"][i])
                    pixel = None
                    if sample["meta"]["source_index_kind"] == "pixel" and shape is not None:
                        h, w = shape
                        if 0 <= source_index < h*w:
                            pixel = [source_index % w, source_index // w]
                    candidates.append(dict(prediction_index=i, q=float(prediction["q"][i]), q0=float(prediction["q0"][i]),
                        point_cam_m=prediction["point_cam_m"][i].tolist(), normal_out_cam=prediction["normal_out_cam"][i].tolist(),
                        instance_id=int(prediction["instance_id"][i]), source_index=source_index, source_pixel_uv=pixel,
                        scene_rank=ranks.get(i), per_instance_nms=i in per_instance_set))
                case = dict(schema_version=SCHEMA, sample_id=row["sample_id"], split="test", identity=identity,
                            asset_sha256=asset_hashes, seed=seed, candidates=candidates, rendering=rendering,
                            source_index_kind=sample["meta"]["source_index_kind"], observation=row["observation"],
                            metadata={k: v for k, v in row.items() if k != "observation"}, notice=NOTICE,
                            timing=dict(load_seconds=load_seconds, inference_seconds=inference_seconds,
                                        total_seconds=time.perf_counter()-started))
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = npz_path.with_name(npz_path.name + f".{os.getpid()}.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **prediction, nms_scene_indices=scene,
                                        nms_per_instance_indices=per_instance,
                                        source_index_kind=np.asarray(sample["meta"]["source_index_kind"]))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, npz_path)
                atomic_json(case_path, case)
                error_path = output / "errors" / f"{key}.json"
                error_path.unlink(missing_ok=True)
                progress["completed"] += 1
                publish(last_frame_seconds=time.perf_counter()-started)
            except Exception as exc:
                atomic_json(output / "errors" / f"{key}.json", dict(sample_id=row["sample_id"],
                            stage="inference", error=str(exc), traceback=traceback.format_exc()))
                progress["errors"] += 1
                publish(last_error=str(exc))
        write_gallery(output, rows)
        publish(status="completed" if not progress["errors"] else "completed_with_errors", current_sample=None)
        return 0 if not progress["errors"] else 1
    except Exception as exc:
        publish(status="failed", last_error=str(exc))
        atomic_json(output / "fatal_error.json", dict(error=str(exc), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "repo", "checkpoint", "config", "performance", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true", help="Validate manifest readiness without model/CUDA imports")
    parser.add_argument("--wait-manifest-seconds", type=int, default=0,
                        help="Load real model first, then wait up to this many seconds for atomic ready manifest publication")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.wait_manifest_seconds < 0 or args.wait_manifest_seconds > 600:
        parser.error("--wait-manifest-seconds must be between 0 and 600")
    if args.wait_manifest_seconds and args.preflight_only:
        parser.error("--wait-manifest-seconds cannot be combined with --preflight-only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
