#!/usr/bin/env python3
"""Executable synthetic geometry and data-contract reference; not a model trainer.

Coordinates/depth are metres in one rectified camera frame. RGB is float [0, 1].
An outward normal points away from the material. Historical axes are never
camera-flipped; only callers explicitly setting orient_proposal=True may do so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import tempfile

import numpy as np


def _vectors(value, dim, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != dim:
        raise ValueError(f"{name} must have shape [N,{dim}]")
    return value


def _intrinsics(K):
    K = np.asarray(K, dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0
            or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])
            or abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12):
        raise ValueError("K must be a finite rectified pinhole matrix with fx,fy>0 and zero skew")
    return K


def project(points_m, K):
    """Return (uv[N,2], valid[N]); invalid pixels are NaN, z must be positive."""
    p, K = _vectors(points_m, 3, "points_m"), _intrinsics(K)
    valid = np.isfinite(p).all(1) & (p[:, 2] > 0)
    uv = np.full((len(p), 2), np.nan)
    uv[valid, 0] = K[0, 0] * p[valid, 0] / p[valid, 2] + K[0, 2]
    uv[valid, 1] = K[1, 1] * p[valid, 1] / p[valid, 2] + K[1, 2]
    return uv, valid


def unproject(uv, depth_m, K):
    """Return (points[N,3], valid[N]); depth is axial camera Z, not ray length."""
    uv, K = _vectors(uv, 2, "uv"), _intrinsics(K)
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.shape != (len(uv),):
        raise ValueError("depth_m must have shape [N]")
    valid = np.isfinite(uv).all(1) & np.isfinite(depth) & (depth > 0)
    p = np.full((len(uv), 3), np.nan)
    p[valid, 0] = (uv[valid, 0] - K[0, 2]) * depth[valid] / K[0, 0]
    p[valid, 1] = (uv[valid, 1] - K[1, 2]) * depth[valid] / K[1, 1]
    p[valid, 2] = depth[valid]
    return p, valid


def reference_affine(K, z_ref_m, center_uv, size=128, resolution_m=0.002):
    """Original pixels -> reference-plane pixels; size=128 has centre 63.5.

    Metric scale is exact only on Z=z_ref, not on a general tilted/deep surface.
    Separate x/y scales account for fx!=fy. No image content is interpolated here.
    """
    K = _intrinsics(K)
    center = np.asarray(center_uv, dtype=np.float64)
    if (center.shape != (2,) or not np.isfinite(center).all() or size <= 0
            or not np.isfinite([z_ref_m, resolution_m]).all()
            or z_ref_m <= 0 or resolution_m <= 0):
        raise ValueError("Invalid reference-plane affine parameters")
    gamma = z_ref_m / (np.array([K[0, 0], K[1, 1]]) * resolution_m)
    A = np.eye(3)
    A[0, 0], A[1, 1] = gamma
    A[:2, 2] = (size - 1) / 2 - gamma * center
    return A


def affine_points(uv, A):
    uv = _vectors(uv, 2, "uv")
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (3, 3) or not np.isfinite(A).all() or not np.allclose(A[2], [0, 0, 1]):
        raise ValueError("A must be a finite affine 3x3 matrix")
    return uv @ A[:2, :2].T + A[:2, 2]


def grid_coordinates(uv, width, height):
    """Pixel-centre coordinates -> PyTorch grid coordinates, align_corners=False."""
    uv = _vectors(uv, 2, "uv")
    if width <= 0 or height <= 0:
        raise ValueError("Grid size must be positive")
    return 2 * (uv + 0.5) / np.array([width, height]) - 1


def _unit(vector, name):
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) <= 1e-12:
        raise ValueError(f"{name} must be a finite nonzero 3-vector")
    return v / np.linalg.norm(v)


def surface_frame(normal_out, point_cam_m=None, *, orient_proposal=False):
    """Return right-handed R=[u,v,n]; tie order for the reference axis is x,y,z.

    orient_proposal=True optionally enforces n dot (-point)>=0. Default False
    preserves the supplied directed axis, including recorded historical actions.
    """
    n = _unit(normal_out, "normal_out")
    if orient_proposal:
        p = np.asarray(point_cam_m, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Proposal orientation requires a finite camera point")
        if n @ (-p) < 0:
            n = -n
    reference = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = _unit(np.cross(reference, n), "tangent")
    v = np.cross(n, u)
    return np.column_stack((u, v, n))


def tool_pose(point_cam_m, normal_out):
    """4x4 camera-from-tool contact pose; tool +Z approaches along -normal_out.

    Roll is a deterministic convention, not an inferred physical suction DOF.
    TCP/flange offsets and robot extrinsics must be applied outside this helper.
    """
    p = np.asarray(point_cam_m, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("point_cam_m must be a finite 3-vector")
    T = np.eye(4)
    T[:3, :3] = surface_frame(-_unit(normal_out, "normal_out"))
    T[:3, 3] = p
    return T


def tangent_raster(points_m, rgb, normals_out, query_point_m, query_normal_out,
                   source_point_ids, *, size=64, resolution_m=0.001, max_height_m=0.040):
    """Deterministic tangent-plane raster with observed holes preserved.

    X[10,H,W]: RGB[0:3], height[3] metres, local normal[4:7], actual selected
    point u/v[7:9] metres, validity[9]. Empty cells are zero, with source ID -1.
    Local interval is [-size*s/2,size*s/2). Select min(abs(height)), then min ID.
    This is observed point selection, not complete-surface or seal reconstruction.
    """
    points, rgb = _vectors(points_m, 3, "points_m"), _vectors(rgb, 3, "rgb")
    normals = _vectors(normals_out, 3, "normals_out")
    ids = np.asarray(source_point_ids)
    if len(rgb) != len(points) or len(normals) != len(points):
        raise ValueError("Point/RGB/normal lengths differ")
    if (ids.shape != (len(points),) or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids > np.iinfo(np.int64).max)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("source_point_ids must be unique nonnegative integers [N]")
    if (not isinstance(size, (int, np.integer)) or size <= 0
            or not np.isfinite([resolution_m, max_height_m]).all()
            or resolution_m <= 0 or max_height_m <= 0):
        raise ValueError("Raster dimensions/resolution/height must be positive")
    query = np.asarray(query_point_m, dtype=np.float64)
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("query_point_m must be a finite 3-vector")
    R = surface_frame(query_normal_out)
    local = (points - query) @ R
    normal_length = np.linalg.norm(normals, axis=1)
    # Zero normals encode an unreliable observed normal, not absent depth/RGB.
    # Such points remain geometry support; query axes are validated separately.
    valid = (np.isfinite(points).all(1) & np.isfinite(rgb).all(1)
             & np.isfinite(normals).all(1))
    if np.any((rgb[np.isfinite(rgb)] < 0) | (rgb[np.isfinite(rgb)] > 1)):
        raise ValueError("RGB must use floating values in [0,1], not 0..255")
    half = size * resolution_m / 2
    valid &= ((np.abs(local[:, 2]) <= max_height_m)
              & (local[:, 0] >= -half) & (local[:, 0] < half)
              & (local[:, 1] >= -half) & (local[:, 1] < half))
    rows = np.flatnonzero(valid)
    col = np.clip(np.floor((local[rows, 0] + half) / resolution_m), 0, size - 1).astype(np.int64)
    row = np.clip(np.floor((local[rows, 1] + half) / resolution_m), 0, size - 1).astype(np.int64)
    cell = row * size + col
    order = np.lexsort((ids[rows], np.abs(local[rows, 2]), cell))
    ordered_cell = cell[order]
    first = np.r_[True, ordered_cell[1:] != ordered_cell[:-1]] if len(order) else np.zeros(0, bool)
    selected, flat = rows[order[first]], ordered_cell[first]
    X = np.zeros((10, size * size), dtype=np.float32)
    selected_ids = np.full(size * size, -1, dtype=np.int64)
    if len(selected):
        unit_normal = np.zeros_like(normals[selected])
        good_normal = normal_length[selected] > 1e-12
        unit_normal[good_normal] = normals[selected][good_normal] / normal_length[selected][good_normal, None]
        local_normal = unit_normal @ R
        X[:, flat] = np.column_stack((rgb[selected], local[selected, 2], local_normal,
                                     local[selected, :2], np.ones(len(selected)))).T
        selected_ids[flat] = ids[selected]
    return {"X": X.reshape(10, size, size), "selected_source_ids": selected_ids.reshape(size, size),
            "valid_mask": (selected_ids >= 0).reshape(size, size), "surface_frame": R,
            "query_point_m": query}


def annulus_masks(u_m, v_m):
    """core [0,10), inner [10,18), lip [18,22] mm; boundaries are metric."""
    radius = np.hypot(np.asarray(u_m), np.asarray(v_m))
    return {"core": np.isfinite(radius) & (radius < .010),
            "inner": (radius >= .010) & (radius < .018),
            "lip": (radius >= .018) & (radius <= .022)}


def angular_coverage(u_m, v_m, valid=None, sectors=72):
    """Occupancy of observed lip sectors; explicitly not a physical seal score."""
    u, v = np.broadcast_arrays(np.asarray(u_m), np.asarray(v_m))
    if sectors <= 0:
        raise ValueError("sectors must be positive")
    keep = annulus_masks(u, v)["lip"]
    if valid is not None:
        keep &= np.broadcast_to(np.asarray(valid, dtype=bool), keep.shape)
    angle = np.mod(np.arctan2(v[keep], u[keep]), 2 * np.pi)
    index = np.floor(angle * sectors / (2 * np.pi)).astype(int) % sectors
    counts = np.bincount(index, minlength=sectors)
    return {"counts": counts, "occupied": counts > 0, "fraction": float(np.mean(counts > 0)),
            "meaning": "observed_lip_occupancy_not_seal_label"}


def validate_label_npz(path):
    """Check SW labels at the supplied poses. NaN is allowed ONLY on mask-false rows.

    Infinity is never accepted. Mask-false finite values remain unsupervised.
    No mask is inferred, no label is imputed, and no quality is divided by C.
    """
    required = ("points", "normal_out", "seal", "wrench", "q_sw", "label_valid")
    with np.load(path, allow_pickle=False) as file:
        if any(key not in file for key in required):
            raise ValueError("Missing required label arrays: " + ",".join(required))
        data = {key: np.asarray(file[key]) for key in required}
    valid = data["label_valid"]
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError("label_valid must be bool [K]")
    count = len(valid)
    for key in required[:-1]:
        value = data[key]
        shape = (count, 3) if key in ("points", "normal_out") else (count,)
        if value.shape != shape or value.dtype.kind not in "fiu":
            raise ValueError(f"{key} must be a numeric array of shape {shape}")
        if np.isinf(value).any() or not np.isfinite(value[valid]).all():
            raise ValueError(f"Nonfinite valid {key}, or infinity anywhere")
    if np.any(np.abs(np.linalg.norm(data["normal_out"][valid], axis=1) - 1) > 1e-4):
        raise ValueError("Valid normal_out rows must be unit vectors within 1e-4")
    for key in ("seal", "wrench", "q_sw"):
        if np.any((data[key][valid] < 0) | (data[key][valid] > 1)):
            raise ValueError(f"Valid {key} must lie in [0,1]")
    if not np.allclose(data["q_sw"][valid], data["seal"][valid] * data["wrench"][valid],
                       atol=1e-6, rtol=1e-5):
        raise ValueError("q_sw != seal*wrench at valid poses")
    return {"ok": True, "rows": count, "valid_rows": int(valid.sum()),
            "invalid_rows": int((~valid).sum()), "has_supervision": bool(valid.any()),
            "finite_but_masked_quality_rows": int(np.count_nonzero(~valid & np.isfinite(data["q_sw"]))) }


def validate_manifest(path):
    """Validate declared IDs/hashes, not underlying file bytes or temporal independence."""
    samples, group_split, hash_split, hash_group = set(), {}, {}, {}
    totals = {"train": 0, "val": 0, "test": 0}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {number}: expected a JSON object")
        for key in ("sample_id", "group_id", "observation_sha256", "split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Line {number}: missing/non-string {key}")
        sid, group, sha, split = (row[k] for k in ("sample_id", "group_id", "observation_sha256", "split"))
        if split not in totals or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
            raise ValueError(f"Line {number}: invalid split or SHA-256")
        sha = sha.lower()
        if sid in samples:
            raise ValueError(f"Duplicate sample_id: {sid}")
        if group in group_split and group_split[group] != split:
            raise ValueError(f"Group crosses splits: {group}")
        if sha in hash_split and hash_split[sha] != split:
            raise ValueError(f"Observation SHA-256 crosses splits: {sha}")
        if sha in hash_group and hash_group[sha] != group:
            raise ValueError("The same observation SHA-256 has inconsistent group_id")
        samples.add(sid)
        group_split[group], hash_split[sha], hash_group[sha] = split, split, group
        totals[split] += 1
    if not samples:
        raise ValueError("Manifest is empty")
    return {"ok": True, "rows": len(samples), "groups": len(group_split),
            "observations": len(hash_split), "splits": totals,
            "scope": "declared_hash_and_group_consistency; file bytes are not read"}


def _synthetic():
    point, normal = np.array([.01, -.02, .60]), _unit([.2, .1, -1], "normal")
    R = surface_frame(normal)
    angle = (np.arange(72) + .5) * 2 * np.pi / 72
    local = np.column_stack((.020 * np.cos(angle), .020 * np.sin(angle), np.zeros(72)))
    points = point + local @ R.T
    rgb = np.tile([.3, .6, .9], (72, 1))
    normals = np.tile(normal, (72, 1))
    seal, wrench = np.linspace(.2, 1., 72), np.full(72, .7)
    labels = {"points": np.vstack((points, [np.nan] * 3)),
              "normal_out": np.vstack((normals, [np.nan] * 3)),
              "seal": np.r_[seal, np.nan], "wrench": np.r_[wrench, np.nan],
              "q_sw": np.r_[seal * wrench, np.nan], "label_valid": np.r_[np.ones(72, bool), False]}
    return point, normal, points, rgb, normals, labels


def demo(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    point, normal, points, rgb, normals, labels = _synthetic()
    raster = tangent_raster(points, rgb, normals, point, normal, np.arange(1000, 1072))
    np.savez_compressed(output / "synthetic_labels.npz", **labels)
    np.savez_compressed(output / "tangent_raster.npz", **raster)
    rows = [{"sample_id": f"synthetic-{i}", "group_id": f"group-{i}",
             "observation_sha256": str(i + 1) * 64, "split": split,
             "synthetic_only": True} for i, split in enumerate(("train", "val", "test"))]
    (output / "synthetic_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    coverage = angular_coverage(raster["X"][7], raster["X"][8], raster["valid_mask"])
    report = {"ok": True, "synthetic_only": True, "description": "No robot, model, or real data evaluated",
              "raster_shape": list(raster["X"].shape), "occupied_cells": int(raster["valid_mask"].sum()),
              "lip_sector_fraction": coverage["fraction"], "label_check": validate_label_npz(output / "synthetic_labels.npz"),
              "manifest_check": validate_manifest(output / "synthetic_manifest.jsonl"),
              "files": ["synthetic_labels.npz", "synthetic_manifest.jsonl", "tangent_raster.npz", "demo_summary.json"]}
    (output / "demo_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def selftest():
    checks = []
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejects(fn, name):
        try:
            fn()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError("Expected rejection: " + name)
    K = np.array([[580., 0, 320.], [0, 620., 240.], [0, 0, 1.]])
    p = np.array([[.1, -.2, .6], [-.02, .03, 1.1]])
    uv, valid = project(p, K)
    recovered, okay = unproject(uv, p[:, 2], K)
    check(valid.all() and okay.all() and np.allclose(p, recovered), "project_unproject_roundtrip_fx_ne_fy")
    check(not project([[0, 0, 0]], K)[1][0], "nonpositive_camera_depth_invalid")
    A = reference_affine(K, .5, [320, 240])
    check(np.allclose(affine_points([[320, 240]], A), [[63.5, 63.5]]), "reference_pixel_centre_63_5")
    plane = np.array([[0, 0, .5], [.020, 0, .5], [0, .020, .5]])
    crop = affine_points(project(plane, K)[0], A)
    check(np.allclose(crop[1:] - crop[0], [[10, 0], [0, 10]]), "independent_fx_fy_metric_reference_scale")
    check(np.allclose(grid_coordinates([[63.5, 63.5]], 128, 128), 0), "align_corners_false_grid_centre")
    back = affine_points(crop, np.linalg.inv(A))
    check(np.allclose(back, project(plane, K)[0]), "affine_inverse_roundtrip")
    deep = unproject(back, np.ones(3), K)[0]
    check(np.isclose(np.linalg.norm(deep[1] - deep[0]), .040), "different_depth_is_not_fixed_2mm_per_pixel")
    theta = np.deg2rad(45)
    tilted = np.array([[0, 0, .5], [.02 * np.cos(theta), 0, .5 - .02 * np.sin(theta)]])
    tc = affine_points(project(tilted, K)[0], A)
    check(not np.isclose(np.linalg.norm(tc[1] - tc[0]), 10), "tilted_tangent_circle_is_not_reference_plane_circle")
    for n in ([1, 0, 0], [0, -1, 0], [0, 0, -1], [.1, .2, 1]):
        R = surface_frame(n)
        check(np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1), "surface_SO3_" + str(n))
    check(np.allclose(surface_frame([0, 0, 1], [0, 0, .5])[:, 2], [0, 0, 1]), "historical_axis_is_not_flipped")
    check(np.allclose(surface_frame([0, 0, 1], [0, 0, .5], orient_proposal=True)[:, 2], [0, 0, -1]), "proposal_camera_orientation_is_explicit")
    T = tool_pose([0, 0, .5], [0, 0, 1])
    check(np.allclose(T[:3, 2], [0, 0, -1]) and np.isclose(np.linalg.det(T[:3, :3]), 1), "tool_plus_z_is_negative_outward_normal")
    masks = annulus_masks([0, .010, .018, .022, .023], np.zeros(5))
    check(np.array_equal(masks["core"], [1, 0, 0, 0, 0]) and np.array_equal(masks["inner"], [0, 1, 0, 0, 0])
          and np.array_equal(masks["lip"], [0, 0, 1, 1, 0]), "annulus_metric_boundaries")
    point, normal, points, rgb, normals, labels = _synthetic()
    raster = tangent_raster(points, rgb, normals, point, normal, np.arange(72))
    cover = angular_coverage(raster["X"][7], raster["X"][8], raster["valid_mask"])
    check(cover["occupied"].all(), "all_72_observed_lip_sectors_occupied")
    check(raster["valid_mask"].sum() == 72 and np.all(raster["X"][:, ~raster["valid_mask"]] == 0), "holes_are_preserved_with_zero_features")
    zero_normal = tangent_raster([[.001, 0, .001]], [[.2, .3, .4]], [[0, 0, 0]],
                                 [0, 0, 0], [0, 0, 1], np.array([7]))
    check(zero_normal["valid_mask"].sum() == 1 and np.all(zero_normal["X"][4:7] == 0)
          and zero_normal["selected_source_ids"].max() == 7, "zero_normal_geometry_support_retained")
    tie_points = np.array([[.0001, .0001, .0008], [.0001, .0001, .0008]])
    tie_norm = np.tile([0, 0, 1.], (2, 1))
    kwargs = dict(query_point_m=[0, 0, 0], query_normal_out=[0, 0, 1])
    tie = tangent_raster(tie_points, np.ones((2, 3)), tie_norm, source_point_ids=np.array([9, 3]), **kwargs)
    reverse = tangent_raster(tie_points[::-1], np.ones((2, 3)), tie_norm, source_point_ids=np.array([3, 9]), **kwargs)
    check(np.array_equal(tie["selected_source_ids"], reverse["selected_source_ids"])
          and tie["selected_source_ids"].max() == 3, "cell_tie_min_source_id_and_input_order_invariance")
    tie_points[0, 2] = -.0002
    closer = tangent_raster(tie_points, np.ones((2, 3)), tie_norm, source_point_ids=np.array([9, 3]), **kwargs)
    check(closer["selected_source_ids"].max() == 9, "cell_selection_min_absolute_height_precedes_id")
    with tempfile.TemporaryDirectory() as tmp:
        path, manifest = Path(tmp) / "labels.npz", Path(tmp) / "manifest.jsonl"
        np.savez(path, **labels)
        check(validate_label_npz(path)["invalid_rows"] == 1, "masked_nan_labels_are_allowed")
        for field, value, name in (("seal", np.nan, "valid_nan_rejected"), ("normal_out", [0, 0, 2], "nonunit_valid_axis_rejected"),
                                   ("q_sw", .99, "incorrect_SW_product_rejected")):
            broken = {key: a.copy() for key, a in labels.items()}
            broken[field][0] = value
            np.savez(path, **broken)
            rejects(lambda: validate_label_npz(path), name)
        rows = [{"sample_id": "a", "group_id": "g1", "observation_sha256": "a" * 64, "split": "train"},
                {"sample_id": "b", "group_id": "g2", "observation_sha256": "b" * 64, "split": "val"}]
        def write_rows(value):
            manifest.write_text("".join(json.dumps(r) + "\n" for r in value), encoding="utf-8")
        write_rows(rows)
        check(validate_manifest(manifest)["rows"] == 2, "valid_disjoint_manifest")
        for key, name in (("group_id", "cross_split_group_rejected"), ("observation_sha256", "cross_split_observation_hash_rejected"),
                          ("sample_id", "duplicate_sample_id_rejected")):
            changed = [dict(r) for r in rows]
            changed[1][key] = changed[0][key]
            write_rows(changed)
            rejects(lambda: validate_manifest(manifest), name)
    return {"ok": True, "synthetic_only": True, "checks_passed": len(checks), "checks": checks,
            "python": platform.python_version(), "numpy": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="Run synthetic geometry and contract checks")
    demo_parser = sub.add_parser("demo", help="Write synthetic labels, raster and manifest")
    demo_parser.add_argument("--output-dir", required=True)
    for name in ("check-labels", "check-manifest"):
        sub.add_parser(name).add_argument("path")
    args = parser.parse_args()
    try:
        if args.command == "selftest":
            result = selftest()
        elif args.command == "demo":
            result = demo(args.output_dir)
        elif args.command == "check-labels":
            result = validate_label_npz(args.path)
        else:
            result = validate_manifest(args.path)
    except (ValueError, KeyError, OSError, AssertionError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
