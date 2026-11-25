#!/usr/bin/env python3
"""v2 synthetic contract checks; no learned model, teacher or timing benchmark.

The older geometry_contract module remains a coordinate/schema utility. Its
tangent raster is not part of the v2 B default. All lengths here are metres.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

from geometry_contract import project, surface_frame, validate_label_npz

QUALITY_CONTRACT = "gt_INST20_pose_v2"
LABEL_SCHEMA = "suction20.instance_inst20.v2"
RADII_M = np.array([0, 5, 9, 10, 14, 17, 18, 20, 22], dtype=float) / 1000


def _unit(value):
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) <= 1e-12:
        raise ValueError("Expected a finite nonzero 3-vector")
    return v / np.linalg.norm(v)


def camera_chart_frame(normal_out):
    """Two fixed Rodrigues charts; no camera x/y argmin selection.

    Default reference is -Z with tangents +X,-Y. Near antipodal +Z, use
    reference +Z and tangents +X,+Y. The chart boundary is NOT continuous.
    The supplied directed normal is never flipped.
    """
    n = _unit(normal_out)
    alternate = 1 - n[2] <= 1e-4
    a = np.array([0., 0., 1. if alternate else -1.])
    e1 = np.array([1., 0., 0.])
    e2 = np.cross(a, e1)
    v = np.cross(a, n)
    skew = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    R = np.eye(3) + skew + skew @ skew / (1 + a @ n)
    return np.column_stack((R @ e1, R @ e2, n)), ("positive_z" if alternate else "negative_z")


def projected_rings(point, normal_out, K, phase=0., angles=72):
    """Project ideal tangent-plane samples. Visibility/depth validity is separate.

    Shape [9,72,2]. Radius-zero coordinates repeat only for this transparent
    reference; a production sampler reads the centre once then broadcasts.
    """
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(phase):
        raise ValueError("Invalid point or angular phase")
    if not isinstance(angles, int) or angles <= 0:
        raise ValueError("angles must be a positive integer")
    frame, chart = camera_chart_frame(normal_out)
    theta = 2 * np.pi * (np.arange(angles) + .5) / angles + phase
    direction = np.cos(theta)[:, None] * frame[:, 0] + np.sin(theta)[:, None] * frame[:, 1]
    points = p + RADII_M[:, None, None] * direction[None]
    uv, valid = project(points.reshape(-1, 3), K)
    return {"points": points, "uv": uv.reshape(9, angles, 2),
            "projection_valid": valid.reshape(9, angles), "chart": chart}


def organized_instance_normals(point_map, instance_keep):
    """Baseline organized_normals formula with a single-instance keep mask.

    Symmetric +/-1 pixel differences, opposite-neighbour axial Z jump <10mm,
    cross-product norm >1e-8; not a 15mm metric smoothing operator.
    """
    p, keep = np.asarray(point_map, np.float32), np.asarray(instance_keep, bool)
    if p.shape != keep.shape + (3,) or keep.ndim != 2:
        raise ValueError("Expected HxWx3 point_map and HxW instance_keep")
    if not np.isfinite(p[keep]).all():
        raise ValueError("Kept points must be finite")
    normal, valid = np.zeros_like(p), np.zeros(keep.shape, bool)
    if min(keep.shape) >= 3:
        dy = p[2:, 1:-1] - p[:-2, 1:-1]
        dx = p[1:-1, 2:] - p[1:-1, :-2]
        raw = np.cross(dx, dy)
        mag = np.linalg.norm(raw, axis=-1)
        ok = (keep[1:-1, 1:-1] & keep[2:, 1:-1] & keep[:-2, 1:-1]
              & keep[1:-1, 2:] & keep[1:-1, :-2] & (mag > 1e-8)
              & (abs(dy[..., 2]) < .01) & (abs(dx[..., 2]) < .01))
        normal[1:-1, 1:-1] = raw / np.maximum(mag[..., None], 1e-8)
        valid[1:-1, 1:-1] = ok
    normal[np.sum(normal * -p, axis=-1) < 0] *= -1
    normal[~valid] = 0
    return normal, valid


def _cell_order(x, y):
    morton = sum(((x >> bit) & 1) << (2 * bit) |
                 ((y >> bit) & 1) << (2 * bit + 1) for bit in range(4))
    return sum(((morton >> bit) & 1) << (7 - bit) for bit in range(8))


def stratified_queries(instance_mask, eligible):
    """16x16 full-mask-bbox cells; nearest eligible pixel per cell, then ID.

    Cell boundaries use pixel outer edges. Returned IDs are v*image_width+u.
    Bit-reversed Morton ordering makes train prefixes deterministic. It does
    not promise uniform metric coverage or select using teacher quality.
    """
    mask, eligible = np.asarray(instance_mask, bool), np.asarray(eligible, bool)
    if mask.ndim != 2 or mask.shape != eligible.shape:
        raise ValueError("Expected equal HxW masks")
    if np.any(eligible & ~mask):
        raise ValueError("Eligible pixels must belong to the instance")
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return np.empty(0, dtype=np.int64)
    left, top = xx.min() - .5, yy.min() - .5
    width, height = xx.max() - xx.min() + 1, yy.max() - yy.min() + 1
    y, x = np.nonzero(eligible)
    ids = y * mask.shape[1] + x
    gx = np.clip(np.floor((x - left) * 16 / width).astype(int), 0, 15)
    gy = np.clip(np.floor((y - top) * 16 / height).astype(int), 0, 15)
    chosen = []
    for cy in range(16):
        for cx in range(16):
            rows = np.flatnonzero((gx == cx) & (gy == cy))
            if not len(rows):
                continue
            center = [left + (cx + .5) * width / 16, top + (cy + .5) * height / 16]
            distance2 = (x[rows] - center[0]) ** 2 + (y[rows] - center[1]) ** 2
            winner = rows[np.lexsort((ids[rows], distance2))[0]]
            chosen.append((_cell_order(cx, cy), int(ids[winner])))
    return np.array([value for _, value in sorted(chosen)], dtype=np.int64)


def contact_scalars(points, point, normal, gravity):
    """Gauge-free metric descriptors rho,h,d_t dot g for finite support points."""
    p = np.asarray(points, np.float64)
    center = np.asarray(point, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or center.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(center).all():
        raise ValueError("Invalid support or query points")
    n, g = _unit(normal), _unit(gravity)
    d = p - center
    h = d @ n
    tangent = d - h[:, None] * n
    return np.column_stack((np.linalg.norm(tangent, axis=1), h, tangent @ g))


def angular_gap(points, point, frame):
    """Maximum circular angular gap / 2pi; empty=0 with separate valid=False."""
    local = (np.asarray(points, float) - np.asarray(point, float)) @ np.asarray(frame, float)
    valid = np.linalg.norm(local[:, :2], axis=1) > 1e-12
    if not valid.any():
        return 0., False
    angle = np.sort(np.mod(np.arctan2(local[valid, 1], local[valid, 0]), 2 * np.pi))
    gaps = np.diff(np.r_[angle, angle[0] + 2 * np.pi])
    return float(gaps.max() / (2 * np.pi)), True


def validate_inst_labels(path):
    """Require v2 identity, split self flags, joint mask and both SW/INST products."""
    summary = validate_label_npz(path)
    with np.load(path, allow_pickle=False) as file:
        required = ("schema_version", "quality_contract", "collision_self_seal",
                    "collision_self_envelope", "collision_self", "q_inst")
        if any(key not in file for key in required):
            raise ValueError("Missing v2 instance label fields")
        if file["schema_version"].shape != () or str(file["schema_version"]) != LABEL_SCHEMA:
            raise ValueError("Unexpected label schema")
        if file["quality_contract"].shape != () or str(file["quality_contract"]) != QUALITY_CONTRACT:
            raise ValueError("Unexpected quality contract; do not reinterpret old SW labels")
        valid = np.asarray(file["label_valid"])
        flags = [np.asarray(file[key]) for key in required[2:5]]
        for flag in flags:
            if flag.shape != valid.shape or flag.dtype != np.bool_:
                raise ValueError("Self collision flags must be bool [K]")
        if not np.array_equal(flags[2][valid], (flags[0] | flags[1])[valid]):
            raise ValueError("collision_self must OR the two target-self components")
        q = np.asarray(file["q_inst"])
        if q.shape != valid.shape or q.dtype.kind not in "fiu" or np.isinf(q).any() or not np.isfinite(q[valid]).all():
            raise ValueError("Invalid q_inst values or shape")
        expected = file["q_sw"][valid] * (~flags[2][valid])
        if np.any((q[valid] < 0) | (q[valid] > 1)) or not np.allclose(q[valid], expected, atol=1e-6, rtol=1e-5):
            raise ValueError("q_inst must equal SW times target-self clearance")
    return {**summary, "quality_contract": QUALITY_CONTRACT,
            "scope": "declared_v2_labels_only_not_geometry_teacher_validation"}


def _labels():
    return {"schema_version": np.array(LABEL_SCHEMA), "quality_contract": np.array(QUALITY_CONTRACT),
            "points": np.array([[0, 0, .6], [.01, 0, .6], [np.nan] * 3]),
            "normal_out": np.array([[0, 0, -1], [0, 0, -1], [np.nan] * 3]),
            "seal": np.array([.9, .9, np.nan]), "wrench": np.array([.8, .8, np.nan]),
            "q_sw": np.array([.72, .72, np.nan]), "q_inst": np.array([.72, 0, np.nan]),
            "collision_self_seal": np.array([False, True, False]),
            "collision_self_envelope": np.array([False, False, False]),
            "collision_self": np.array([False, True, False]),
            "label_valid": np.array([True, True, False])}


def selftest():
    checks = []
    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(name)
    def reject(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    n1 = _unit([.199999, .200001, -np.sqrt(.919999999998)])
    n2 = _unit([.200001, .199999, -np.sqrt(.919999999998)])
    old1, old2 = surface_frame(n1), surface_frame(n2)
    old_jump = np.degrees(np.arccos(np.clip(old1[:, 0] @ old2[:, 0], -1, 1)))
    check("old_argmin_frame_counterexample_exceeds_90deg", 92 < old_jump < 93)
    frame1, _ = camera_chart_frame(n1)
    frame2, _ = camera_chart_frame(n2)
    check("fixed_chart_has_no_xy_tie_jump", np.linalg.norm(frame1-frame2) < 1e-5)
    for n in ([0, 0, -1], [0, 0, 1], [.3, .7, .6], [1, 0, 0]):
        frame, _ = camera_chart_frame(n)
        check("chart_SO3_and_directed_axis_" + str(n),
              np.allclose(frame.T @ frame, np.eye(3)) and np.isclose(np.linalg.det(frame), 1)
              and np.allclose(frame[:, 2], _unit(n)))
    K = np.array([[600, 0, 320], [0, 500, 240], [0, 0, 1.]])
    center = np.array([.01, -.02, .6])
    rings = projected_rings(center, n1, K)
    delta = rings["points"] - center
    check("ring_samples_remain_at_exact_tangent_radii", np.allclose(np.linalg.norm(delta, axis=-1), RADII_M[:, None]))
    check("ring_plane_is_conditioned_on_supplied_axis", np.allclose(delta @ n1, 0, atol=1e-12))
    check("projection_finite_for_positive_depth", rings["projection_valid"].all() and np.isfinite(rings["uv"]).all())
    reflected = projected_rings(center, n1, K, phase=2*np.pi/72)
    check("one_sector_phase_is_cyclic_shift", np.allclose(reflected["uv"], np.roll(rings["uv"], -1, axis=1)))
    support = rings["points"][7, :17]
    scalars1 = contact_scalars(support, center, n1, [0, 1, 0])
    scalars2 = contact_scalars(support, center, n2, [0, 1, 0])
    check("gauge_free_scalars_continuous_across_old_tie", np.max(abs(scalars1-scalars2)) < 1e-6)
    theta = .731
    rotated = frame1 @ np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    check("maximum_angular_gap_is_basis_rotation_invariant", np.isclose(angular_gap(support, center, frame1)[0], angular_gap(support, center, rotated)[0]))
    x = np.arange(72, dtype=float) ** 2
    def shared_pool(v):
        response = .25*np.roll(v, 1) + .5*v + .25*np.roll(v, -1)
        return np.array([response.mean(), response.max()])
    check("shared_circular_filter_pool_is_shift_invariant", np.allclose(shared_pool(x), shared_pool(np.roll(x, 18))))
    yy, xx = np.mgrid[:65, :65]
    point_map = np.stack(((xx-32)*.001, (yy-32)*.001, np.full(xx.shape, .6)), axis=-1)
    keep = np.ones(xx.shape, bool)
    normals, normal_valid = organized_instance_normals(point_map, keep)
    check("organized_plane_normal_and_border_mask", normal_valid.sum() == 63**2 and np.allclose(normals[normal_valid], [0, 0, -1]))
    keep[32, 32] = False
    _, normal_valid = organized_instance_normals(point_map, keep)
    check("organized_normal_respects_instance_boundary", not normal_valid[32, 32] and not normal_valid[32, 31] and not normal_valid[31, 32])
    step = point_map.copy()
    step[:, 33:, 2] += .02
    _, normal_valid = organized_instance_normals(step, np.ones_like(keep))
    check("organized_depth_step_invalidates_crossing_differences", not normal_valid[:, 32:34].any())
    mask = np.ones((64, 64), bool)
    ids = stratified_queries(mask, mask)
    check("stratified_grid_has_256_unique_queries", len(ids) == 256 and len(np.unique(ids)) == 256)
    check("bit_reversed_morton_prefix_spans_quadrants", np.array_equal(ids[:4], [65, 2113, 97, 2145]))
    check("training128_is_fixed_evaluation_prefix", len(ids[:128]) == 128 and np.array_equal(ids, stratified_queries(mask, mask)))
    check("empty_candidate_mask_returns_empty", len(stratified_queries(mask, np.zeros_like(mask))) == 0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.npz"
        labels = _labels()
        np.savez_compressed(path, **labels)
        check("v2_self_collision_and_masked_unknown_labels_valid", validate_inst_labels(path)["valid_rows"] == 2)
        bad = {**labels, "q_inst": np.array([.72, .72, np.nan])}
        np.savez_compressed(path, **bad)
        reject("self_collision_cannot_keep_high_quality", lambda: validate_inst_labels(path))
        bad = {**labels, "collision_self": np.array([False, False, False])}
        np.savez_compressed(path, **bad)
        reject("self_components_must_OR", lambda: validate_inst_labels(path))
        bad = {**labels, "quality_contract": np.array("gt_SW20_pose_v1")}
        np.savez_compressed(path, **bad)
        reject("old_SW_contract_not_reinterpreted", lambda: validate_inst_labels(path))
        bad = {**labels, "q_inst": np.array([np.nan, 0, np.nan])}
        np.savez_compressed(path, **bad)
        reject("unknown_q_inst_cannot_be_valid", lambda: validate_inst_labels(path))
    logits = np.array([-2., 1.])
    sigmoid = lambda value: 1 / (1 + np.exp(-value))
    check("probability_mixture_not_mean_logit_sigmoid", not np.isclose(sigmoid(logits).mean(), sigmoid(logits.mean())))
    return {"ok": True, "synthetic_only": True, "checks_passed": len(checks), "checks": checks,
            "old_basis_jump_degrees": float(old_jump), "python": platform.python_version(), "numpy": np.__version__}


def demo(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ["synthetic_instance_labels.npz", "projected_rings.npz", "revision_summary.json"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Use a fresh output directory")
    np.savez_compressed(out / names[0], **_labels())
    K = np.array([[600, 0, 320], [0, 500, 240], [0, 0, 1.]])
    rings = projected_rings([.01, -.02, .6], [.2, .1, -1.], K)
    np.savez_compressed(out / names[1], **rings)
    summary = {"ok": True, "synthetic_only": True, "ring_shape": list(rings["uv"].shape),
               "radius_zero_optimized_unique_samples_per_query": 577,
               "labels": validate_inst_labels(out / names[0]), "files": names,
               "scope": "No CNN, grid_sample engine, real teacher, data pipeline or latency measured"}
    (out / names[2]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    run = sub.add_parser("demo")
    run.add_argument("--output-dir", required=True)
    labels = sub.add_parser("check-labels")
    labels.add_argument("path")
    args = parser.parse_args()
    result = selftest() if args.command == "selftest" else (demo(args.output_dir) if args.command == "demo" else validate_inst_labels(args.path))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
