"""Read-only, arbitrary-pose R20 evaluation against original GT geometry.

Seal uses the original 72 open sectors in *object* coordinates. It is never
looked up from cached candidate scores, and its cylinder has no height cutoff.
The scene envelope and dual L2 wrench conventions reuse the producer helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .full_data import (
    collision_mask_for_instance, load_seal, scene_frame0, scene_table_cloud,
    to_unit, transform_points, voxel_filter, wrench_scores_dual,
)


def seal_at_pose(dense_points, point, normal, radius_m=0.020,
                 chunk_points=262144):
    """Return (raw Seal, original self collision) for one object-frame pose.

    Chunking only bounds temporary memory; all dense points participate. The
    original Seal self-height is 5 mm, distinct from envelope self clearance.
    """
    p = np.asarray(point, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    if p.shape != (3,) or n.shape != (3,) or not np.isfinite([p, n]).all():
        raise ValueError("point and normal must be finite vectors of length 3")
    if radius_m <= 0 or chunk_points < 1:
        raise ValueError("radius and chunk size must be positive")
    n = to_unit(n)
    y = np.array([n[1], -n[0], 0.])
    if np.linalg.norm(y) < 1e-15:
        y = np.array([0., -1., 0.])
    y = to_unit(y)
    x = to_unit(np.cross(y, n))
    basis = np.column_stack([x, y, n])
    # Arithmetic order matches the production bins (not linspace).
    edges = np.array([2 * np.pi * i / 72 - np.pi for i in range(72)] + [np.pi])
    heights = np.full(72, -np.inf)
    self_collision = False
    for start in range(0, len(dense_points), chunk_points):
        local = (np.asarray(dense_points[start:start + chunk_points], np.float64) - p) @ basis
        rho = np.linalg.norm(local[:, :2], axis=1)
        self_collision |= bool(np.any((rho < radius_m) & (local[:, 2] > .005)))
        ring = local[(rho > radius_m - .1 * radius_m) & (rho < radius_m + .1 * radius_m)]
        angle = np.arctan2(ring[:, 1], ring[:, 0])
        bins = np.searchsorted(edges, angle, side="right") - 1
        inside = (bins >= 0) & (bins < 72)
        bins, angle, ring = bins[inside], angle[inside], ring[inside]
        # Exclude both boundaries, including exactly -pi and +pi.
        inside = (angle > edges[bins]) & (angle < edges[bins + 1])
        np.maximum.at(heights, bins[inside], ring[inside, 2])
    if not np.isfinite(heights).all():
        return 0., self_collision
    theta = edges[:-1] + np.pi / 72
    vertices = np.column_stack([radius_m * np.cos(theta), radius_m * np.sin(theta), heights])
    delta = vertices - np.roll(vertices, 1, axis=0)
    deform = np.min(np.linalg.norm(delta[:, :2], axis=1) / np.linalg.norm(delta, axis=1))
    fit = np.exp(-100000 * np.mean((heights - heights.mean()) ** 2))
    return float(fit * deform), self_collision


@dataclass
class SceneEvalContext:
    scene_id: int
    camera: str
    frozen_cfg: dict
    records: list[dict]
    world_camera: np.ndarray
    dense_points: dict[int, np.ndarray]
    centers_world: dict[int, np.ndarray]
    obstacle_points: np.ndarray
    obstacle_rows: np.ndarray
    obstacle_tree: cKDTree


def load_scene_eval_context(dataset_root, label_root, scene_id, camera,
                            frozen_cfg=None):
    """Load original dense geometry and fixed scene physics, without writes.

    ``dataset_root`` is the extracted Dataset root containing scenes/ and
    dense_point_clouds/. Missing dense inputs fail rather than changing Seal to
    mesh vertices or network voxels. Historical extra frozen keys are allowed.
    """
    dataset_root, label_root = Path(dataset_root), Path(label_root)
    if frozen_cfg is None:
        frozen_cfg = json.loads((label_root / "config_frozen.json").read_text())
    cfg = dict(frozen_cfg)
    if cfg["camera"] != camera:
        raise ValueError("camera differs from frozen label configuration")
    if cfg["radius_m"] != .020 or cfg["seal_norm_max"] <= cfg["seal_norm_min"]:
        raise ValueError("R20 requires 20 mm radius and increasing frozen normalization bounds")
    records = scene_frame0(dataset_root, int(scene_id), camera)
    cam = dataset_root / "scenes" / f"scene_{scene_id:04d}" / camera
    world_camera = np.load(cam / "cam0_wrt_table.npy") @ np.load(cam / "camera_poses.npy")
    table = scene_table_cloud(cfg)
    clouds, rows = [table], [np.full(len(table), -1, np.int32)]
    dense_points, centers = {}, {}
    for rec in records:
        obj_id, row, transform = rec["obj_id"], rec["instance_row"], rec["T_world_obj"]
        if obj_id not in dense_points:
            path = dataset_root / "dense_point_clouds" / f"{obj_id:03d}.npz"
            with np.load(path, allow_pickle=False) as data:
                dense = np.asarray(data["points"], dtype=np.float64)
            if dense.ndim != 2 or dense.shape[1] != 3 or not len(dense) or not np.isfinite(dense).all():
                raise ValueError(f"invalid original Seal dense geometry: {path}")
            dense_points[obj_id] = dense
        obstacle = voxel_filter(dense_points[obj_id], cfg["obstacle_voxel_m"])
        clouds.append(transform_points(obstacle, transform).astype(np.float32))
        rows.append(np.full(len(obstacle), row, np.int32))
        seal = load_seal(label_root / "seal_r20" / "poisson" / f"{obj_id:03d}_labels.npz")
        centers[row] = transform_points(seal["points"], transform).mean(axis=0)
    obstacle_points, obstacle_rows = np.concatenate(clouds), np.concatenate(rows)
    return SceneEvalContext(int(scene_id), camera, cfg, records, world_camera,
                            dense_points, centers, obstacle_points, obstacle_rows,
                            cKDTree(obstacle_points))


def evaluate_poses(context, frame_id, point_cam_m, normal_out_cam, gt_instance_row):
    """Recompute actual poses; GT rows are zero-based, background is -1.

    Background has zero quality, and collision=False denotes no evaluated
    target geometry there. Invalid foreground poses fail loudly. No gt_valid
    gate or candidate lookup is used.
    """
    points, normals = np.asarray(point_cam_m, np.float64), np.asarray(normal_out_cam, np.float64)
    rows = np.asarray(gt_instance_row)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or rows.shape != (len(points),):
        raise ValueError("expected points/normals [K,3] and gt_instance_row [K]")
    if not np.issubdtype(rows.dtype, np.integer) or np.any((rows < -1) | (rows >= len(context.records))):
        raise ValueError("GT instance row is outside this scene")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("poses must be finite")
    if frame_id < 0 or frame_id >= len(context.world_camera):
        raise ValueError("frame_id is outside camera poses")
    result = {key: np.zeros(len(points), np.float64) for key in ("raw_seal", "S", "W", "W_annot", "Q20")}
    result["collision"] = np.zeros(len(points), bool)
    T_world_cam = context.world_camera[frame_id]
    cfg = context.frozen_cfg
    for rec in context.records:
        row = rec["instance_row"]
        ids = np.flatnonzero(rows == row)
        if not len(ids):
            continue
        # Invert the actual metadata matrix, which may only approximately be
        # orthogonal after original storage. Keep world normals normalized.
        T_obj_cam = np.linalg.inv(rec["T_world_obj"]) @ T_world_cam
        p_obj = transform_points(points[ids], T_obj_cam)
        n_obj = to_unit(normals[ids] @ T_obj_cam[:3, :3].T)
        raw_and_self = [seal_at_pose(context.dense_points[rec["obj_id"]], p, n, cfg["radius_m"])
                        for p, n in zip(p_obj, n_obj)]
        raw, self_collision = zip(*raw_and_self)
        p_world = transform_points(points[ids], T_world_cam)
        n_world = to_unit(normals[ids] @ T_world_cam[:3, :3].T)
        wrench = wrench_scores_dual(p_world, n_world, context.centers_world[row],
                                    cfg["default_baseline_mass_kg"], cfg)
        collision = collision_mask_for_instance(
            p_world, n_world, context.obstacle_tree, context.obstacle_points,
            context.obstacle_rows, row, np.array(self_collision), cfg)
        result["raw_seal"][ids] = raw
        result["S"][ids] = np.clip((np.array(raw) - cfg["seal_norm_min"]) /
                                      (cfg["seal_norm_max"] - cfg["seal_norm_min"]), 0, 1)
        result["W"][ids] = wrench["wrench_eval"]
        result["W_annot"][ids] = wrench["wrench_annot"]
        result["collision"][ids] = collision
    result["Q20"] = result["S"] * result["W"] * ~result["collision"]
    return result
