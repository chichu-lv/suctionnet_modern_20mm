"""SuctionNet-20mm offline dataset pipeline: geometry, scene and per-frame GT.

Implements the V2 data-engineering specification. Physical calibration is tracked separately.
Unit convention: all XYZ lengths in meters, normals unit-length.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree
from PIL import Image


DEFAULT_CONFIG = {
    "camera": "realsense", "radius_m": 0.020, "voxel_m": 0.005,
    "instance_mask_root": None,
    "wrench_k": 15.6,
    "default_baseline_mass_kg": 1.0,
    "gravity_m_s2": 9.8, "depth_min_m": 0.2, "depth_max_m": 1.2,
    "normal_gate_deg": 30.0, "match_radius_m": 0.005,
    "seal_norm_min": 0.0, "seal_norm_max": 1.0,
    "pca_radius_m": 0.020, "pca_min_points": 12,
    "pca_max_curvature": 0.15,
    "custom_lip_tolerance_m": 0.004,
    "custom_surface_percentile": 95.0,
    "self_height_threshold_m": 0.004,
    "table_width_m": 1.0, "table_depth_m": 1.0,
    "table_bottom_m": -0.02, "table_grid_m": 0.005,
    "obstacle_voxel_m": 0.003,
    "envelope_segments_m": [
        [-0.002, 0.025, 0.022], [0.025, 0.070, 0.028],
        [0.070, 0.150, 0.028]
    ],
    "batch_candidates": 64,
    "mass_by_object": {}, "center_of_mass_by_object": {},
    "vacuum_delta_pa": None, "friction_mu": None,
    "measured_torque_capacity_nm": None,
}


def config_from_file(path: str | Path | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        user_cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(user_cfg) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(user_cfg)
    if cfg["seal_norm_max"] <= cfg["seal_norm_min"]:
        raise ValueError("seal_norm_max must be greater than seal_norm_min")
    if cfg["radius_m"] <= 0 or cfg["match_radius_m"] <= 0:
        raise ValueError("Geometry radii must be positive")
    if cfg["instance_mask_root"] is not None and not isinstance(cfg["instance_mask_root"], str):
        raise TypeError("instance_mask_root must be a directory path or null")
    return cfg


def to_unit(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms < 1e-9):
        raise ValueError("Found zero/invalid surface normal")
    return x / norms


def transform_points(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.asarray(x) @ T[:3, :3].T + T[:3, 3]


def load_seal(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing 20mm seal: {path}")
    with np.load(path, allow_pickle=False) as f:
        o = {k: f[k] for k in f.files}
    for k in ("points", "normals", "scores", "collision", "indices"):
        if k not in o:
            raise ValueError(f"{path}: missing {k}")
    n = len(o["points"])
    for k in ("normals", "scores", "collision", "indices"):
        if len(o[k]) != n:
            raise ValueError(f"{path}: {k} candidate count mismatch")
    o["points"] = np.asarray(o["points"], np.float64).reshape(-1, 3)
    o["normals"] = to_unit(o["normals"])
    o["scores"] = np.asarray(o["scores"], np.float32).reshape(-1)
    o["collision"] = np.asarray(o["collision"], bool).reshape(-1)
    o["indices"] = np.asarray(o["indices"], np.int32).reshape(-1)
    if len(np.unique(o["indices"])) != n:
        raise ValueError(f"{path}: duplicate original indices")
    if not np.all(np.isfinite(o["scores"])):
        raise ValueError(f"{path}: nonfinite seal")
    return o


def voxel_filter(points: np.ndarray, voxel_m: float) -> np.ndarray:
    p = np.asarray(points, np.float32).reshape(-1, 3)
    if not len(p) or voxel_m <= 0:
        return p
    grid = np.floor(p / voxel_m).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return p[np.sort(idx)]


def load_dense_points(dataset_root: Path, obj_id: int, voxel_m: float) -> np.ndarray:
    dense = dataset_root / "dense_point_clouds" / f"{obj_id:03d}.npz"
    if dense.is_file():
        with np.load(dense, allow_pickle=False) as o:
            return voxel_filter(o["points"], voxel_m)
    # Fallback uses mesh vertices only; quality/reliability must be audited.
    import trimesh
    mesh_path = dataset_root / "models" / f"{obj_id:03d}" / "nontextured.ply"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    return voxel_filter(np.asarray(mesh.vertices), voxel_m)


def add_pca_extras(seal: dict, dense_points: np.ndarray, cfg: dict) -> dict:
    """Independent experimental PCA normals / planarity score (not official Seal)."""
    points, normals = seal["points"], seal["normals"]
    tree = cKDTree(dense_points)
    gt_normal = normals.astype(np.float32).copy()
    valid = np.zeros(len(points), dtype=bool)
    custom = np.zeros(len(points), dtype=np.float32)
    for i, ids in enumerate(tree.query_ball_point(points, cfg["pca_radius_m"])):
        if len(ids) < cfg["pca_min_points"]:
            continue
        q = dense_points[ids].astype(np.float64)
        q -= q.mean(axis=0)
        covariance = (q.T @ q) / len(q)
        eig, vec = np.linalg.eigh(covariance)
        if eig[0] / max(eig.sum(), 1e-12) > cfg["pca_max_curvature"]:
            continue
        n = vec[:, 0]
        if n @ normals[i] < 0:
            n = -n
        residual = np.abs(q @ n)
        err = np.percentile(residual, cfg["custom_surface_percentile"])
        gt_normal[i] = n
        custom[i] = max(0, 1 - err / cfg["custom_lip_tolerance_m"])
        valid[i] = not bool(seal["collision"][i]) and seal["scores"][i] > 0
    return {"pca_normals": gt_normal, "normal_valid": valid,
            "seal_custom": custom}


def scene_frame0(dataset_root: Path, scene_id: int, camera: str, *,
                 table=None, camera_poses=None) -> list[dict]:
    """Camera-0 metadata -> table/world poses, ordered as source .mat instances.
    This order is asserted against object_id_list, matching official arr_i order.
    """
    root = dataset_root / "scenes" / f"scene_{scene_id:04d}"
    cam = root / camera
    mat = loadmat(cam / "meta" / "0000.mat")
    cls = np.asarray(mat["cls_indexes"]).reshape(-1).astype(int) - 1
    poses = np.asarray(mat["poses"], dtype=float)
    if poses.ndim == 2:
        poses = poses[:, :, None]
    if poses.shape[:2] != (3, 4) or poses.shape[2] != len(cls):
        raise ValueError(f"scene {scene_id}: metadata poses/class mismatch")
    list_file = root / "object_id_list.txt"
    if list_file.is_file():
        listed = [int(t) for t in list_file.read_text().split()]
        if listed != cls.tolist():
            raise ValueError(f"scene {scene_id}: object_id_list and meta cls_indexes order differ; cannot align arr_i safely")
    # The training adapter already loaded these small, static arrays. Reuse them
    # on the first records-cache miss as well as on later scene-cache hits.
    if table is None:
        table = np.load(cam / "cam0_wrt_table.npy")
    if camera_poses is None:
        camera_poses = np.load(cam / "camera_poses.npy")
    T_world_cam0 = table @ camera_poses[0]
    result = []
    for i, obj_id in enumerate(cls):
        T = np.eye(4)
        T[:3, :4] = poses[:, :, i]
        result.append({"instance_row": i, "obj_id": int(obj_id),
                       "T_world_obj": T_world_cam0 @ T})
    return result


def scene_table_cloud(cfg: dict) -> np.ndarray:
    step = cfg["table_grid_m"]
    xs = np.arange(-cfg["table_width_m"] / 2, cfg["table_width_m"] / 2 + 1e-9, step)
    ys = np.arange(-cfg["table_depth_m"] / 2, cfg["table_depth_m"] / 2 + 1e-9, step)
    xx, yy = np.meshgrid(xs, ys)
    top = np.zeros_like(xx)
    bottom = np.full_like(xx, cfg["table_bottom_m"])
    return np.concatenate([np.stack([xx, yy, top], axis=-1).reshape(-1, 3),
                           np.stack([xx, yy, bottom], axis=-1).reshape(-1, 3)]).astype(np.float32)


def wrench_scores_dual(points_w: np.ndarray, normals_w: np.ndarray,
                       center_w: np.ndarray, mass_kg: float, cfg: dict) -> dict:
    """Save both Wrench thresholds from a shared tangential-torque calculation.

    wrench_annot: upstream suction_annotation convention: sqrt(2)*r*k*pi.
    wrench_eval: suctionnetAPI evaluation convention: r*k*pi.
    Both use the same poses/normals (transformed to the table frame).
    """
    g = np.array([0., 0., -cfg["gravity_m_s2"] * mass_kg], dtype=np.float64)
    tau = np.cross(center_w[None, :] - points_w, g)
    tangent_tau = tau - np.sum(tau * normals_w, axis=1, keepdims=True) * normals_w
    torque = np.linalg.norm(tangent_tau, axis=1)
    threshold_eval = float(cfg["wrench_k"]) * float(cfg["radius_m"]) * np.pi
    if threshold_eval <= 0:
        raise ValueError("Wrench threshold must be positive")
    eval_scores = (1. - np.minimum(1., torque / threshold_eval)).astype(np.float32)
    annot_scores = (1. - np.minimum(1., torque / (threshold_eval * np.sqrt(2.)))).astype(np.float32)
    return {"wrench_eval": eval_scores, "wrench_annot": annot_scores}


def wrench_scores(points_w: np.ndarray, normals_w: np.ndarray, center_w: np.ndarray,
                  mass_kg: float, cfg: dict) -> np.ndarray:
    """Compatibility wrapper: training and GT use the evaluation convention."""
    return wrench_scores_dual(points_w, normals_w, center_w, mass_kg, cfg)["wrench_eval"]


def measured_physics_wrench(points_w: np.ndarray, normals_w: np.ndarray,
                            center_w: np.ndarray, mass_kg: float, cfg: dict):
    """Optional calibrated load margin, not a SuctionNet baseline.

    Requires vacuum delta-pressure, coefficient of friction and measured
    bending torque limit. Without measured parameters, return NaN and invalid.
    """
    necessary = (cfg.get("vacuum_delta_pa"), cfg.get("friction_mu"),
                 cfg.get("measured_torque_capacity_nm"))
    if any(x is None for x in necessary):
        return np.full(len(points_w), np.nan, np.float32), np.zeros(len(points_w), bool)
    pressure, friction, tau_max = [float(x) for x in necessary]
    if min(pressure, friction, tau_max) <= 0:
        raise ValueError("Calibrated vacuum, friction and bending capacity must be positive")
    fmax = pressure * np.pi * cfg["radius_m"] ** 2
    g = np.array([0., 0., -mass_kg * cfg["gravity_m_s2"]])
    normal_load = np.maximum(0, -(normals_w @ g))
    shear = np.linalg.norm(g - (normals_w @ g)[:, None] * normals_w, axis=1)
    tau = np.cross(center_w - points_w, g)
    tau = tau - np.sum(tau * normals_w, axis=1, keepdims=True) * normals_w
    torque_magnitude = np.linalg.norm(tau, axis=1)
    capacity_margins = np.stack([1 - normal_load / fmax,
                                 1 - shear / (friction * fmax),
                                 1 - torque_magnitude / tau_max], axis=1)
    score = np.clip(capacity_margins.min(axis=1), 0, 1).astype(np.float32)
    return score, np.ones(len(points_w), bool)


def collision_mask_for_instance(points_w: np.ndarray, normals_w: np.ndarray,
                                scene_tree: cKDTree, scene_points: np.ndarray,
                                obstacle_instance: np.ndarray, instance_row: int,
                                self_collision: np.ndarray, cfg: dict) -> np.ndarray:
    """Segmented cup/mount/sweep envelope; all candidates, no prefilter.
    KDTree is built on all object + table geometry. Candidate chunks limit memory.
    """
    segments = np.asarray(cfg["envelope_segments_m"], float)
    radius_search = math.hypot(float(segments[:, :2].max()), float(segments[:, 2].max()))
    out = np.asarray(self_collision, dtype=bool).copy()
    chunk = int(cfg["batch_candidates"])
    for start in range(0, len(points_w), chunk):
        end = min(start + chunk, len(points_w))
        lists = scene_tree.query_ball_point(points_w[start:end], radius_search)
        for j, neighbors in enumerate(lists):
            i = start + j
            if out[i] or not neighbors:
                continue
            rel = scene_points[neighbors] - points_w[i]
            z = rel @ normals_w[i]
            r2 = np.maximum(0., np.sum(rel**2, axis=1) - z**2)
            r = np.sqrt(r2)
            same = obstacle_instance[neighbors] == instance_row
            # Cup rim can touch the target at z ~ 0. Reject protrusions above lip compliance.
            if np.any(same & (z > cfg["self_height_threshold_m"]) &
                      (z <= segments[-1, 1]) & (r <= segments[0, 2])):
                out[i] = True
                continue
            for zmin, zmax, seg_radius in segments:
                # For own object, only above defined lip/cup contact clearance.
                own = same & (z > max(float(zmin), cfg["self_height_threshold_m"]))
                ext = ~same
                if np.any((own | ext) & (z >= zmin) & (z <= zmax) & (r <= seg_radius)):
                    out[i] = True
                    break
    return out


def compute_scene(dataset_root: Path, seal_root: Path, out_root: Path,
                  scene_id: int, cfg: dict, resume: bool = True) -> dict:
    camera = cfg["camera"]
    wpath = out_root / "wrench_label_r20" / f"{scene_id:04d}_{camera}_wrench_scores.npz"
    cpath = out_root / "collision_label_r20" / f"{scene_id:04d}_{camera}_collision.npz"
    mpath = out_root / "scene_meta_r20" / f"{scene_id:04d}_{camera}.json"
    if resume and wpath.exists() and cpath.exists() and mpath.exists():
        meta = json.loads(mpath.read_text())
        if meta.get("wrench_schema") != "annot_eval_dual_v1":
            raise ValueError("Old scene Wrench schema found; regenerate with --overwrite")
        with np.load(wpath, allow_pickle=False) as old:
            required = [f"{prefix}_{i}" for i in range(len(meta["objects"]))
                        for prefix in ("wrench_annot", "wrench_eval")]
            if not all(key in old.files for key in required):
                raise ValueError("Scene Wrench file lacks annot/eval arrays; regenerate with --overwrite")
        return meta
    records = scene_frame0(dataset_root, scene_id, camera)
    table = scene_table_cloud(cfg)
    clouds, obj_rows, result = [table], [-np.ones(len(table), dtype=np.int32)], []
    object_data = []
    for record in records:
        obj_id = record["obj_id"]
        seal = load_seal(seal_root / f"{obj_id:03d}_labels.npz")
        dense = load_dense_points(dataset_root, obj_id, cfg["obstacle_voxel_m"])
        T = record["T_world_obj"]
        clouds.append(transform_points(dense, T).astype(np.float32))
        obj_rows.append(np.full(len(dense), record["instance_row"], dtype=np.int32))
        object_data.append((record, seal))
    obstacle_points = np.concatenate(clouds, axis=0)
    obstacle_rows = np.concatenate(obj_rows, axis=0)
    tree = cKDTree(obstacle_points)
    w_arrays, c_arrays = {}, {}
    for row, (record, seal) in enumerate(object_data):
        T = record["T_world_obj"]
        p = transform_points(seal["points"], T)
        n = to_unit(seal["normals"] @ T[:3, :3].T)
        # Use all candidate center positions as in upstream empirical baseline.
        center = p.mean(axis=0)
        dual = wrench_scores_dual(p, n, center, cfg["default_baseline_mass_kg"], cfg)
        wrench_eval, wrench_annot = dual["wrench_eval"], dual["wrench_annot"]
        mass = cfg["mass_by_object"].get(str(record["obj_id"]))
        center_obj = cfg["center_of_mass_by_object"].get(str(record["obj_id"]))
        if mass is not None and center_obj is not None:
            physical, phy_valid = measured_physics_wrench(
                p, n, transform_points(np.asarray(center_obj), T), float(mass), cfg)
        else:
            physical = np.full(len(p), np.nan, dtype=np.float32)
            phy_valid = np.zeros(len(p), bool)
        collision = collision_mask_for_instance(p, n, tree, obstacle_points, obstacle_rows,
                        record["instance_row"], seal["collision"], cfg)
        orient = (1 - np.arccos(np.clip((-n) @ np.array([0., 0., -1.]), -1, 1))/np.pi).astype(np.float32)
        w_arrays[f"wrench_annot_{row}"] = wrench_annot
        w_arrays[f"wrench_eval_{row}"] = wrench_eval
        w_arrays[f"arr_{row}"] = wrench_eval  # legacy-compatible name: evaluation score
        w_arrays[f"physics_{row}"] = physical
        w_arrays[f"physics_valid_{row}"] = phy_valid
        w_arrays[f"orient_{row}"] = orient
        c_arrays[f"arr_{row}"] = collision
        result.append({"row": row, "obj_id": record["obj_id"], "n": len(p),
                       "collision_ratio": float(collision.mean()) if len(p) else None,
                       "wrench_eval_mean": float(wrench_eval.mean()) if len(p) else None,
                       "wrench_annot_mean": float(wrench_annot.mean()) if len(p) else None})
    wpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(wpath, **w_arrays)
    np.savez_compressed(cpath, **c_arrays)
    manifest = {"scene_id": scene_id, "camera": camera, "objects": result,
                "schema_version": "r20-v2.1",
                "radius_m": float(cfg["radius_m"]),
                "envelope_segments_m": cfg["envelope_segments_m"],
                "wrench_schema": "annot_eval_dual_v1",
                "wrench_eval_threshold": "k * r * pi",
                "wrench_annot_threshold": "sqrt(2) * k * r * pi",
                "wrench_eval_is_training_target": True,
                "physics_label_note": "Both Wrench fields use transformed normals; physics field valid only with measured inputs",
                "table_pose_source": "meta/0000.mat + camera_poses[0] + cam0_wrt_table",
                 "instance_id_convention": "1-based scene object row; 0=background"}
    mpath.write_text(json.dumps(manifest, indent=2))
    return manifest


def resolve_instance_ids(class_ids: np.ndarray, scene_records: list[dict],
                         dataset_root: Path, scene_id: int, frame_id: int,
                         cfg: dict, source_indices: np.ndarray,
                         image_shape: tuple[int, int]) -> np.ndarray:
    """Resolve instance_row + 1 as unique instance ID; background = 0.

    Class labels are not instance labels. If classes repeat, an external
    1-based row-index mask is required at
    <instance_mask_root>/scene_XXXX/<camera>/FFFF.png.
    """
    classes = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    row_class = {int(rec["instance_row"]) + 1: int(rec["obj_id"]) + 1
                 for rec in scene_records}
    mask_root = cfg.get("instance_mask_root")
    if mask_root is None:
        if len(set(row_class.values())) != len(row_class):
            raise ValueError(f"Scene {scene_id}: repeated class needs instance_mask_root")
        unknown = set(np.unique(classes).tolist()) - {0} - set(row_class.values())
        if unknown:
            raise ValueError(f"Scene {scene_id}, frame {frame_id}: unknown class IDs: {sorted(unknown)}")
        ids = np.zeros(len(classes), dtype=np.int32)
        for inst_id, cls_id in row_class.items():
            ids[classes == cls_id] = inst_id
    else:
        filename = (Path(mask_root) / f"scene_{scene_id:04d}" / cfg["camera"]
                    / f"{frame_id:04d}.png")
        if not filename.is_file():
            raise FileNotFoundError(f"Missing instance mask: {filename}")
        mask = np.asarray(Image.open(filename))
        if mask.ndim != 2 or mask.shape != tuple(image_shape):
            raise ValueError(f"Instance mask/depth shape mismatch: {filename}")
        ids = mask.reshape(-1)[np.asarray(source_indices, dtype=np.int64)].astype(np.int32)
        if np.any((ids < 0) | (ids > len(scene_records))):
            raise ValueError(f"Instance IDs outside scene object row range: {filename}")
        if np.any(((classes == 0) & (ids != 0)) | ((classes > 0) & (ids == 0))):
            raise ValueError(f"Instance foreground does not match class mask: {filename}")
        for inst_id, cls_id in row_class.items():
            if np.any((ids == inst_id) & (classes != cls_id)):
                raise ValueError(f"Instance {inst_id} does not match class {cls_id}: {filename}")
    return ids


def read_frame(dataset_root: Path, scene_id: int, frame_id: int, cfg: dict):
    camera = cfg["camera"]
    base = dataset_root / "scenes" / f"scene_{scene_id:04d}" / camera
    depth = np.array(Image.open(base / "depth" / f"{frame_id:04d}.png"))
    segmentation = np.array(Image.open(base / "label" / f"{frame_id:04d}.png"))
    meta = loadmat(base / "meta" / f"{frame_id:04d}.mat")
    K = np.asarray(meta["intrinsic_matrix"], dtype=float)
    scale = float(np.asarray(meta["factor_depth"]).reshape(-1)[0])
    if scale <= 0 or depth.shape != segmentation.shape:
        raise ValueError("Bad depth scale or segmentation shape")
    z = depth.astype(float) / scale
    keep = np.isfinite(z) & (z >= cfg["depth_min_m"]) & (z <= cfg["depth_max_m"])
    yy, xx = np.nonzero(keep)
    zz = z[yy, xx]
    xyz = np.stack([(xx-K[0, 2])*zz/K[0, 0],
                    (yy-K[1, 2])*zz/K[1, 1], zz], axis=1).astype(np.float32)
    ids = segmentation[yy, xx].astype(np.int32)
    source_indices = (yy * depth.shape[1] + xx).astype(np.int32)
    # Organized finite difference normals, sign toward camera. Invalid at edges.
    pmap = np.zeros(depth.shape + (3,), dtype=np.float32)
    pmap[yy, xx] = xyz
    normal = np.zeros_like(pmap)
    valid_n = np.zeros(depth.shape, bool)
    if min(depth.shape) >= 3:
        dy = pmap[2:, 1:-1] - pmap[:-2, 1:-1]
        dx = pmap[1:-1, 2:] - pmap[1:-1, :-2]
        raw_n = np.cross(dx, dy)
        ok = keep[1:-1, 1:-1] & keep[2:, 1:-1] & keep[:-2, 1:-1] & keep[1:-1, 2:] & keep[1:-1, :-2]
        mag = np.linalg.norm(raw_n, axis=-1)
        ok &= mag > 1e-8
        raw_n = raw_n / np.maximum(mag[..., None], 1e-8)
        # disallow normals across large depth discontinuity (10 mm)
        ok &= (abs(dy[..., 2]) < 0.01) & (abs(dx[..., 2]) < 0.01)
        normal[1:-1, 1:-1] = raw_n
        valid_n[1:-1, 1:-1] = ok
    n = normal[yy, xx]
    n_valid = valid_n[yy, xx]
    # Face sensor: cross(dx,dy) is +camera z; invert toward sensor.
    towards_cam = np.sum(n * -xyz, axis=1) < 0
    n[towards_cam] *= -1
    return xyz, n, n_valid, ids, source_indices, depth.shape


def match_instance(cam_points: np.ndarray, cam_normals: np.ndarray, normal_valid: np.ndarray,
                  cand_points: np.ndarray, cand_normals: np.ndarray, cfg: dict):
    """Nearest valid normal-matched candidate; NEVER rank by suction score."""
    tree = cKDTree(cand_points)
    radius = cfg["match_radius_m"]
    cosine_gate = np.cos(np.deg2rad(cfg["normal_gate_deg"]))
    matched = np.full(len(cam_points), -1, np.int32)
    distance = np.full(len(cam_points), np.inf, np.float32)
    # Fast common case; fall back to ball query if nearest candidate fails normal gate.
    dd, ii = tree.query(cam_points, k=1, distance_upper_bound=radius)
    ok = np.isfinite(dd)
    ok &= normal_valid
    good = ok.copy()
    good[ok] = np.sum(cam_normals[ok] * cand_normals[ii[ok]], axis=1) >= cosine_gate
    matched[good] = ii[good]
    distance[good] = dd[good]
    for row in np.where(~good & normal_valid)[0]:
        candidates = tree.query_ball_point(cam_points[row], radius)
        if not candidates:
            continue
        candidates = np.asarray(candidates, dtype=np.int32)
        candidates = candidates[np.sum(cam_normals[row] * cand_normals[candidates], axis=1) >= cosine_gate]
        if not len(candidates):
            continue
        dd2 = np.linalg.norm(cand_points[candidates] - cam_points[row], axis=1)
        j = int(np.argmin(dd2))
        matched[row] = candidates[j]
        distance[row] = dd2[j]
    return matched, distance


def generate_view(dataset_root: Path, seal_root: Path, out_root: Path,
                  scene_id: int, frame_id: int, cfg: dict, resume=True) -> dict:
    camera = cfg["camera"]
    target = out_root / "scenes" / f"scene_{scene_id:04d}" / camera / "graspness_r20_v2" / f"{frame_id:04d}.npz"
    manifest_path = out_root / "scene_meta_r20" / f"{scene_id:04d}_{camera}.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("wrench_schema") != "annot_eval_dual_v1":
        raise ValueError("Stage-2 scene labels lack dual Wrench fields; rerun scene stage")
    if resume and target.is_file():
        with np.load(target, allow_pickle=False) as old:
            required = ("gt_wrench_eval", "gt_wrench_annot", "gt_normal_pca", "normals_raw")
            if any(k not in old.files for k in required):
                raise ValueError("Outdated view fields; rerun with --overwrite")
        return {"scene_id": scene_id, "frame_id": frame_id, "skipped": True}
    frame0 = scene_frame0(dataset_root, scene_id, camera)
    cam = dataset_root / "scenes" / f"scene_{scene_id:04d}" / camera
    T_world_cam = np.load(cam / "cam0_wrt_table.npy") @ np.load(cam / "camera_poses.npy")[frame_id]
    T_cam_world = np.linalg.inv(T_world_cam)
    xyz, normals, normal_ok, segmentation, idx, image_shape = read_frame(
        dataset_root, scene_id, frame_id, cfg)
    instance_ids = resolve_instance_ids(
        segmentation, frame0, dataset_root, scene_id, frame_id, cfg, idx, image_shape)
    N = len(xyz)
    out = {
        "xyz": xyz, "source_indices": idx, "class_id": segmentation,
        "instance_id": instance_ids,
        "objectness": (segmentation > 0).astype(np.uint8),
        "gt_valid": np.zeros(N, bool),
        "normal_valid": np.zeros(N, bool),
        "collision_flag": np.zeros(N, bool),  # unknown points have gt_valid=False; flag not evaluated
        "gt_seal_raw": np.zeros(N, np.float32),
        "gt_seal_norm": np.zeros(N, np.float32),
        "gt_seal_custom": np.zeros(N, np.float32),
        "gt_wrench": np.zeros(N, np.float32),         # alias of gt_wrench_eval
        "gt_wrench_eval": np.zeros(N, np.float32),    # main training target
        "gt_wrench_annot": np.zeros(N, np.float32),   # upstream annotation convention
        "gt_orient": np.zeros(N, np.float32),
        "gt_A": np.zeros(N, np.float32),
        "gt_B": np.zeros(N, np.float32),
        "gt_normal": np.zeros((N, 3), np.float32),       # original candidate axis
        "gt_normal_pca": np.zeros((N, 3), np.float32),   # optional fitted geometry
        "normal_pca_valid": np.zeros(N, bool),
        "normals_raw": normals.astype(np.float32),       # measured depth normals
        "normals_raw_valid": normal_ok.astype(bool),
        "candidate_id": np.full(N, -1, np.int32),
        "candidate_row": np.full(N, -1, np.int32),
        "match_distance": np.full(N, np.inf, np.float32),
        "schema_version": np.array("r20-v2.1"),
    }
    wpath = out_root / "wrench_label_r20" / f"{scene_id:04d}_{camera}_wrench_scores.npz"
    cpath = out_root / "collision_label_r20" / f"{scene_id:04d}_{camera}_collision.npz"
    wfile = np.load(wpath, allow_pickle=False)
    cfile = np.load(cpath, allow_pickle=False)
    try:
        for rec in frame0:
            obj = rec["obj_id"]
            row = rec["instance_row"]
            pix = np.where(instance_ids == row + 1)[0]
            if not len(pix):
                continue
            seal = load_seal(seal_root / f"{obj:03d}_labels.npz")
            if (len(seal["points"]) != len(wfile[f"wrench_eval_{row}"])
                    or len(seal["points"]) != len(wfile[f"wrench_annot_{row}"])
                    or len(seal["points"]) != len(cfile[f"arr_{row}"])):
                raise ValueError(f"scene {scene_id}: object {obj} candidate index count mismatch")
            Tc_o = T_cam_world @ rec["T_world_obj"]
            cp = transform_points(seal["points"], Tc_o)
            cn = to_unit(seal["normals"] @ Tc_o[:3, :3].T)
            match, dist = match_instance(xyz[pix], normals[pix], normal_ok[pix], cp, cn, cfg)
            local_valid = match >= 0
            dest = pix[local_valid]
            src = match[local_valid]
            if not len(dest):
                continue
            out["gt_valid"][dest] = True
            out["candidate_row"][dest] = src
            out["candidate_id"][dest] = seal["indices"][src]
            out["match_distance"][dest] = dist[local_valid]
            raw = seal["scores"][src]
            normalized = np.clip((raw - cfg["seal_norm_min"]) / (cfg["seal_norm_max"]-cfg["seal_norm_min"]), 0, 1)
            wrench_eval = wfile[f"wrench_eval_{row}"][src]
            wrench_annot = wfile[f"wrench_annot_{row}"][src]
            collision = cfile[f"arr_{row}"][src]
            out["collision_flag"][dest] = collision
            out["gt_seal_raw"][dest] = raw
            out["gt_seal_norm"][dest] = normalized
            out["gt_wrench"][dest] = wrench_eval
            out["gt_wrench_eval"][dest] = wrench_eval
            out["gt_wrench_annot"][dest] = wrench_annot
            out["gt_orient"][dest] = wfile[f"orient_{row}"][src]
            out["gt_A"][dest] = normalized * (~collision)
            out["gt_B"][dest] = normalized * wrench_eval * (~collision)
            # Physical labels were computed for the original candidate direction.
            out["gt_normal"][dest] = cn[src]
            out["normal_valid"][dest] = (~collision) & (normalized > 0.2)
            extras_path = out_root / "seal_extras_r20" / f"{obj:03d}.npz"
            if extras_path.is_file():
                with np.load(extras_path, allow_pickle=False) as f:
                    e = {k: f[k] for k in f.files}
                if len(e["pca_normals"]) != len(seal["points"]):
                    raise ValueError("PCA extras candidate mismatch")
                out["gt_seal_custom"][dest] = e["seal_custom"][src]
                out["gt_normal_pca"][dest] = to_unit(e["pca_normals"][src] @ Tc_o[:3, :3].T)
                out["normal_pca_valid"][dest] = e["normal_valid"][src]
    finally:
        wfile.close()
        cfile.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **out)
    counts = {"scene_id": scene_id, "frame_id": frame_id, "points": N,
              "fg": int((out["objectness"] > 0).sum()),
              "matched": int(out["gt_valid"].sum()),
              "valid_zero_gtB": int((out["gt_valid"] & (out["gt_B"] == 0)).sum()),
              "unmatched_fg": int(((out["objectness"] > 0) & (~out["gt_valid"])).sum())}
    return counts


def dataset_stats(root: Path, cfg: dict, scene_ids: Iterable[int],
                  frames: Iterable[int]) -> dict:
    stats = {"camera": cfg["camera"], "schema_version": "r20-v2.1",
             "frame_count": 0, "points": 0, "fg": 0, "matched": 0,
             "valid_positive": 0, "valid_zero": 0, "g2_errors": []}
    for scene in scene_ids:
        for frame in frames:
            path = root / "scenes" / f"scene_{scene:04d}" / cfg["camera"] / "graspness_r20_v2" / f"{frame:04d}.npz"
            if not path.is_file():
                stats["g2_errors"].append(f"Missing {path}")
                continue
            with np.load(path, allow_pickle=False) as o:
                p = o["xyz"]; valid = o["gt_valid"]; cls = o["class_id"]
                n = len(p)
                for key in ("gt_A", "gt_B", "gt_wrench_eval", "gt_wrench_annot",
                            "gt_wrench", "candidate_id", "gt_normal", "gt_normal_pca",
                            "normal_pca_valid", "normals_raw", "normals_raw_valid", "source_indices"):
                    if len(o[key]) != n:
                        stats["g2_errors"].append(f"Shape mismatch {path}: {key}")
                if np.any(valid & (o["candidate_id"] < 0)):
                    stats["g2_errors"].append(f"Valid label missing candidate ID {path}")
                if np.any(valid & (cls <= 0)):
                    stats["g2_errors"].append(f"Background point with valid GT {path}")
                inst = o["instance_id"]
                if len(inst) != n:
                    stats["g2_errors"].append(f"Instance ID shape mismatch {path}")
                elif np.any(valid & (inst <= 0)):
                    stats["g2_errors"].append(f"Valid GT without scene-instance ID {path}")
                if np.any(valid & (o["candidate_row"] < 0)):
                    stats["g2_errors"].append(f"Valid GT without candidate row {path}")
                if np.any((~valid) & (o["candidate_id"] != -1)):
                    stats["g2_errors"].append(f"Invalid GT carrying candidate ID {path}")
                if np.any(valid & (np.abs(o["gt_wrench"] - o["gt_wrench_eval"]) > 1e-6)):
                    stats["g2_errors"].append(f"Primary Wrench differs from eval convention {path}")
                if np.any(valid & (o["gt_wrench_annot"] + 1e-6 < o["gt_wrench_eval"])):
                    stats["g2_errors"].append(f"Annotation Wrench unexpectedly below eval Wrench {path}")
                if np.any(valid & (np.abs(o["gt_B"] - o["gt_A"] * o["gt_wrench_eval"]) > 1e-6)):
                    stats["g2_errors"].append(f"GT-B does not use evaluation Wrench {path}")
                if np.any(valid & ((~np.isfinite(o["gt_wrench_eval"])) |
                                    (~np.isfinite(o["gt_wrench_annot"])))):
                    stats["g2_errors"].append(f"Nonfinite Wrench score on valid GT {path}")
                stats["frame_count"] += 1
                stats["points"] += n
                stats["fg"] += int((cls > 0).sum())
                stats["matched"] += int(valid.sum())
                stats["valid_positive"] += int((valid & (o["gt_B"] > 0)).sum())
                stats["valid_zero"] += int((valid & (o["gt_B"] == 0)).sum())
    stats["matched_fraction_of_foreground"] = stats["matched"] / max(stats["fg"], 1)
    stats["valid_zero_fraction"] = stats["valid_zero"] / max(stats["matched"], 1)
    return stats
