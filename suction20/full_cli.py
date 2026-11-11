"""End-to-end SuctionNet-20mm data generation CLI (G0 -> G2 + stats).

Run `python -m suction20.full_cli --help` from the project root.
The full workflow requires user-downloaded GraspNet/SuctionNet dataset.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from .full_data import (
    config_from_file, add_pca_extras, load_seal, load_dense_points,
    compute_scene, generate_view, dataset_stats, scene_frame0,
)
from .patch_official import create_variant, create_wrench_g0_variant
from .core import compare_labels, validate_pair, write_json


def range_spec(spec: str, max_exclusive: int) -> list[int]:
    """Supports 'all', '0,2,4', '0:100' (exclusive), '0-99' inclusive."""
    if spec == "all":
        return list(range(max_exclusive))
    result = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            a, b = item.split(":", 1)
            result.extend(range(int(a), int(b)))
        elif "-" in item:
            a, b = item.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(item))
    result = sorted(set(result))
    if not result or result[0] < 0 or result[-1] >= max_exclusive:
        raise ValueError(f"IDs out of range [0,{max_exclusive}): {spec}")
    return result


def checkpoint(out: Path, stage: str, result: dict):
    write_json(out / "reports" / (stage + ".json"), result)
    print(json.dumps({"stage": stage, **result}, ensure_ascii=False, indent=2), flush=True)


def seal_run(dataset: Path, upstream: Path, out: Path, objects: list[int], mode: str,
             g0_limit_mean: float, g0_limit_max: float, resume: bool = True) -> Path:
    """Run patched ORIGINAL seal_annotate for 10 and 20 mm on SAME candidates.

    G0 10mm comparisons must be reviewed if exact dataset version differs.
    Does not delete any zero-score or colliding candidate.
    """
    if mode not in ("poisson", "origin"):
        raise ValueError("mode must be poisson or origin")
    if mode == "poisson":
        dense = dataset / "dense_point_clouds"
        missing = [i for i in objects if not (dense / f"{i:03d}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing dense .npz for objects {missing[:12]}. Use scripts/pack_dense.py if needed.")
    source = upstream / "seal_annotate.py"
    labels = {}
    for radius in (10, 20):
        dest_script = out / "scripts" / f"seal_r{radius}.py"
        create_variant(source, dest_script, radius, objects)
        target = out / f"seal_r{radius}" / mode
        expected = [target / f"{i:03d}_labels.npz" for i in objects]
        if not (resume and all(p.is_file() for p in expected)):
            cmd = [sys.executable, str(dest_script),
                   "--model_root", str(dataset / "models"),
                   "--save_root", str(out / f"seal_r{radius}"),
                   "--voxel_size", "0.005"]
            if mode == "poisson":
                cmd += ["--poisson_root", str(dataset / "dense_point_clouds")]
            print("EXEC", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        labels[radius] = target
        for p in expected:
            if not p.is_file():
                raise FileNotFoundError(f"Official Seal script did not create {p}")
    rows, g0_errors, g1_errors = [], [], []
    for i in objects:
        ref = dataset / "seal_label" / f"{i:03d}_seal.npz"
        p10 = labels[10] / f"{i:03d}_labels.npz"
        p20 = labels[20] / f"{i:03d}_labels.npz"
        if not ref.exists():
            raise FileNotFoundError(f"Missing published G0 reference {ref}")
        g0 = compare_labels(ref, p10)
        g1 = validate_pair(p10, p20)
        rows.append({"object_id": i, "g0": g0, "g1": g1})
        if not g1["same_candidate_set"]:
            g1_errors.append(i)
        # G0 is an exact regression gate by default: every original candidate,
        # its Seal score and its collision status must agree with the released label.
        identical_candidates = (g0["matched_count"] == g0["reference_count"]
                                == g0["candidate_count"])
        identical_scores = (
            g0["score_abs_mean"] is not None and g0["score_abs_max"] is not None
            and g0["score_abs_mean"] <= g0_limit_mean
            and g0["score_abs_max"] <= g0_limit_max
        )
        identical_collision = g0.get("collision_disagreement_rate") in (None, 0.0)
        if not (identical_candidates and identical_scores and identical_collision):
            g0_errors.append(i)
    result = {"object_ids": objects, "mode": mode, "g0_mismatch_objects": g0_errors,
              "g1_mismatch_objects": g1_errors, "objects": rows,
              "g0_score_mean_limit": g0_limit_mean,
              "g0_score_max_limit": g0_limit_max,
              "g0_exact_by_default": g0_limit_mean == 0 and g0_limit_max == 0,
              "note": "G0 requires identical candidates, Seal scores and available collision labels; investigate nonzero differences."}
    write_json(out / "reports" / "g0_g1_full.json", result)
    if g1_errors:
        raise RuntimeError(f"G1 candidate identity failed: {g1_errors}")
    if g0_errors:
        raise RuntimeError(f"G0 10mm reproduction differs for {g0_errors}; review reports/g0_g1_full.json before stage2")
    return labels[20]


def wrench_g0_replay(dataset: Path, upstream: Path, out: Path,
                     camera: str, mode: str, scenes: list[int],
                     reference_root: Path | None = None,
                     max_error: float = 0.0) -> dict:
    """Replay original 10mm offline Wrench for G0 and compare published arrays.

    The copied upstream script changes only the outer scene loop. It retains
    upstream coordinate calculations and sqrt(2)*r*k*pi exactly; this is
    independent of the corrected 20mm dual Wrench stage.
    """
    r10 = out / "seal_r10" / mode
    if not r10.is_dir():
        raise FileNotFoundError(f"G0 10mm Seal directory not found: {r10}")
    ref_dir = reference_root or (dataset / "wrench_label")
    ref_dir = Path(ref_dir)
    if not ref_dir.is_dir():
        raise FileNotFoundError(
            f"G0 published 10mm Wrench directory not found: {ref_dir}; "
            "pass --g0-wrench-reference to specify it"
        )
    script = out / "scripts" / "wrench_r10_official.py"
    manifest = create_wrench_g0_variant(upstream / "wrench_annotate.py", script, scenes)
    dest = out / "wrench_r10_annot"
    dest.mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(upstream.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, str(script), "--dataset_root", str(dataset),
        "--seal_dir", str(r10), "--save_dir", str(dest),
        "--camera", camera,
    ]
    log = out / "reports" / "g0_wrench_upstream.log"
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(cmd, cwd=str(upstream), env=env,
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    rows = []
    failures = []
    for scene in scenes:
        name = f"{scene}_{camera}_wrench_scores.npz"
        generated = dest / name
        released_names = (
            f"{scene:04d}_wrench.npz", f"{scene}_wrench.npz",
            name, f"{scene:04d}_{camera}_wrench_scores.npz"
        )
        old_path = next(
            (ref_dir / x for x in released_names if (ref_dir / x).is_file()),
            ref_dir / name
        )
        if not old_path.is_file() or not generated.is_file():
            failures.append(
                f"{scene}: missing 10mm Wrench reference or generated file; "
                f"checked {[str(ref_dir / x) for x in released_names]}"
            )
            continue
        with np.load(old_path, allow_pickle=False) as original, np.load(generated, allow_pickle=False) as reproduced:
            old_keys = sorted(k for k in original.files if k.startswith("arr_"))
            new_keys = sorted(k for k in reproduced.files if k.startswith("arr_"))
            if not old_keys or old_keys != new_keys:
                failures.append(f"{scene}: empty or mismatched arr_N scene-object keys")
                continue
            for key in old_keys:
                a = np.asarray(original[key], np.float64)
                b = np.asarray(reproduced[key], np.float64)
                if a.shape != b.shape:
                    failures.append(f"{scene}/{key}: candidate length mismatch")
                    continue
                diff = np.abs(a - b)
                err = float(diff.max()) if diff.size else 0.0
                rows.append({"scene": scene, "object_array": key,
                             "count": int(diff.size), "max_abs_error": err,
                             "mean_abs_error": float(diff.mean()) if diff.size else 0.0})
                if err > max_error:
                    failures.append(f"{scene}/{key}: max Wrench error {err} > {max_error}")
    result = {"camera": camera, "r10_mm": 10,
              "threshold_formula": "sqrt(2) * k * r * pi",
              "scenes": scenes, "max_allowed_error": max_error,
              "comparison": rows, "mismatches": failures,
              "upstream_log": str(log)}
    write_json(out / "reports" / "g0_wrench.json", result)
    if failures:
        raise RuntimeError(f"G0 10mm Wrench regression failed: {failures[:6]}; see reports/g0_wrench.json")
    return result


def seal_extras(dataset: Path, seal_root: Path, out: Path, cfg: dict, objects: list[int], resume=True):
    dest = out / "seal_extras_r20"
    dest.mkdir(parents=True, exist_ok=True)
    info = []
    for obj_id in objects:
        target = dest / f"{obj_id:03d}.npz"
        if resume and target.is_file():
            continue
        seal = load_seal(seal_root / f"{obj_id:03d}_labels.npz")
        points = load_dense_points(dataset, obj_id, 0)  # raw dense points for PCA
        result = add_pca_extras(seal, points, cfg)
        np.savez_compressed(target, **result, indices=seal["indices"])
        info.append({"object_id": obj_id, "count": len(seal["points"]),
                     "normal_valid": int(result["normal_valid"].sum())})
    return {"generated": info, "directory": str(dest)}


def fit_normalization(dataset: Path, seal_root: Path, cfg: dict, out: Path,
                      train_scene_ids=range(100), quantile=(1., 99.)) -> dict:
    objects = set()
    for sc in train_scene_ids:
        cam = dataset / "scenes" / f"scene_{sc:04d}" / cfg["camera"]
        if not cam.exists():
            continue
        for rec in scene_frame0(dataset, sc, cfg["camera"]):
            objects.add(rec["obj_id"])
    if not objects:
        raise ValueError("No train-scene objects; cannot fit normalization without data")
    values = []
    for obj in sorted(objects):
        a = load_seal(seal_root / f"{obj:03d}_labels.npz")["scores"]
        values.append(a)
    all_scores = np.concatenate(values)
    lo, hi = np.percentile(all_scores, quantile)
    if hi <= lo:
        raise ValueError("Degenerate Seal scores in train split")
    cfg["seal_norm_min"] = float(lo)
    cfg["seal_norm_max"] = float(hi)
    result = {"seal_norm_min": float(lo), "seal_norm_max": float(hi),
              "objects_train_only": sorted(objects), "quantiles": quantile,
              "n_candidates": int(sum(len(v) for v in values))}
    write_json(out / "reports" / "seal_train_statistics.json", result)
    write_json(out / "config_frozen.json", cfg)
    return result


def perform(args):
    data = Path(args.dataset).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_from_file(args.config)
    scenes = range_spec(args.scenes, 190)
    frames = range_spec(args.frames, 256)
    objects = range_spec(args.objects, 88)
    seal_root = Path(args.seal_root).resolve() if args.seal_root else out / "seal_r20" / args.mode
    command = args.command
    run_all = command == "run-all"

    if command in ("seal", "run-all"):
        if not args.upstream:
            raise ValueError("--upstream is required for Seal stage")
        seal_root = seal_run(data, Path(args.upstream), out, objects, args.mode,
                             args.g0_limit_mean, args.g0_limit_max, not args.overwrite)
        checkpoint(out, "stage1_seal", {"seal_root": str(seal_root), "count": len(objects)})
    if command in ("wrench-g0", "run-all"):
        if not args.upstream:
            raise ValueError("--upstream required for G0 Wrench replay")
        selected = range_spec(args.g0_wrench_scenes, 190)
        result = wrench_g0_replay(
            data, Path(args.upstream).resolve(), out, cfg["camera"], args.mode,
            selected, Path(args.g0_wrench_reference).resolve()
            if args.g0_wrench_reference else None, args.g0_wrench_max_error)
        checkpoint(out, "stage0_wrench_regression", result)
    if command in ("extras", "run-all"):
        result = seal_extras(data, seal_root, out, cfg, objects, not args.overwrite)
        checkpoint(out, "stage1_extras", result)
    if command in ("fit-norm", "run-all"):
        # Stable train-only fixed bounds; save frozen config used by subsequent stages.
        stats = fit_normalization(data, seal_root, cfg, out)
        checkpoint(out, "fit_norm", stats)
    if command in ("scene", "run-all"):
        rows = []
        for scene in scenes:
            print("Stage2 scene", scene, flush=True)
            rows.append(compute_scene(data, seal_root, out, scene, cfg, not args.overwrite))
        checkpoint(out, "stage2_scenes", {"scenes": len(rows), "items": rows})
    if command in ("views", "run-all"):
        totals = {"frames": 0, "points": 0, "matched": 0, "unmatched_fg": 0, "skipped": 0}
        for scene in scenes:
            for frame in frames:
                row = generate_view(data, seal_root, out, scene, frame, cfg, not args.overwrite)
                if row.get("skipped"):
                    totals["skipped"] += 1
                else:
                    totals["frames"] += 1
                    for k in ("points", "matched", "unmatched_fg"):
                        totals[k] += row.get(k, 0)
            print("Stage3 views scene complete", scene, flush=True)
        checkpoint(out, "stage3_views", totals)
    if command in ("stats", "run-all"):
        stats = dataset_stats(out, cfg, scenes, frames)
        checkpoint(out, "stage4_g2_audit", stats)
        if stats["g2_errors"]:
            raise RuntimeError(f"G2 view GT failures: {len(stats['g2_errors'])}")
    write_json(out / "reports" / "last_run.json",
               {"utc": datetime.now(timezone.utc).isoformat(), "stage": command,
                "radius_m": cfg["radius_m"], "dataset": str(data), "seal_root": str(seal_root),
                "scenes": args.scenes, "frames": args.frames, "objects": args.objects,
                "note": "Physical ground-truth validity requires G0/G1/G2 visual + real-world review."})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for stage in ("seal", "wrench-g0", "extras", "fit-norm", "scene", "views", "stats", "run-all"):
        sp = sub.add_parser(stage)
        sp.add_argument("--dataset", required=True, help="SuctionNet dataset root")
        sp.add_argument("--output", required=True, help="Output root (never overwrite dataset root)")
        sp.add_argument("--config", default=None, help="JSON configuration; config_frozen.json for stages 2+")
        sp.add_argument("--upstream", default=None, help="Official suction_annotation clone, required for seal")
        sp.add_argument("--seal-root", default=None, help="Existing .../seal_r20/poisson or /origin")
        sp.add_argument("--mode", choices=("origin", "poisson"), default="poisson")
        sp.add_argument("--objects", default="all", help="0,2,4 / 0-87 / all")
        sp.add_argument("--scenes", default="all", help="0,2,4 / 0:100 / all")
        sp.add_argument("--frames", default="all", help="0,1,2 / 0:256 / all")
        sp.add_argument("--overwrite", action="store_true")
        sp.add_argument("--g0-limit-mean", type=float, default=0.0,
                        help="Maximum allowed mean absolute Seal score error at G0; default exact")
        sp.add_argument("--g0-limit-max", type=float, default=0.0,
                        help="Maximum allowed per-candidate Seal score error at G0; default exact")
        sp.add_argument("--g0-wrench-scenes", default="0,1,2",
                        help="Scene IDs for the original 10mm Wrench G0 replay")
        sp.add_argument("--g0-wrench-reference", default=None,
                        help="Original 10mm wrench_label directory; defaults to DATASET/wrench_label")
        sp.add_argument("--g0-wrench-max-error", type=float, default=0.0,
                        help="Max absolute difference vs published 10mm Wrench; default exact")
    args = p.parse_args()
    perform(args)


if __name__ == "__main__":
    main()
