#!/usr/bin/env python3
"""v3 NumPy normal/sampler contract reference; no model or timing benchmark.

Inputs are one rectified camera's raw organized XYZ (metres) and a Boolean
single-instance mask. Normals use a separate smoothed copy; query positions
always retain raw source pixels. The physical INST20 teacher is not changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from revision_contract import (
    _cell_order,
    organized_instance_normals as organized_instance_normals_v2,
    stratified_queries as stratified_queries_v2,
)


NORMAL_SPEC = "organized_masked3x3_span3_v3"
SAMPLER_SPEC = "image_stratified_morton_refill_v3"
DEPTH_JUMP_M = .010
MIN_SMOOTHING_SUPPORT = 5
SPAN_PIXELS = 3
CROSS_NORM_MIN_M2 = 1e-8


def _boolean_image(value, name):
    image = np.asarray(value)
    if image.dtype != np.bool_ or image.ndim != 2 or min(image.shape) < 1:
        raise ValueError(f"{name} must be a nonempty bool HxW image")
    return image


def _masked_xyz_mean3(points, valid):
    """Mean raw XYZ from same-instance 3x3 neighbours near centre axial Z.

    A valid centre is mandatory. Missing values are selected out before any
    summation; zero-times-NaN is never used. This helper does not alter inputs.
    """
    height, width = valid.shape
    safe = np.where(valid[..., None], points, 0.)
    sums = np.zeros_like(safe)
    counts = np.zeros(valid.shape, dtype=np.int16)
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            y0, y1 = max(0, -dv), min(height, height - dv)
            x0, x1 = max(0, -du), min(width, width - du)
            target = np.s_[y0:y1, x0:x1]
            source = np.s_[y0+dv:y1+dv, x0+du:x1+du]
            keep = (valid[target] & valid[source]
                    & (np.abs(safe[source][..., 2] - safe[target][..., 2]) < DEPTH_JUMP_M))
            sums[target] += np.where(keep[..., None], safe[source], 0.)
            counts[target] += keep
    smooth_valid = valid & (counts >= MIN_SMOOTHING_SUPPORT)
    smooth = np.zeros_like(sums)
    smooth[smooth_valid] = sums[smooth_valid] / counts[smooth_valid, None]
    return smooth, smooth_valid, counts


def organized_instance_normals_v3(point_map, instance_keep):
    """Return (normals[H,W,3], normal_valid[H,W], JSON-safe diagnostics).

    Reference arithmetic is float64. instance_keep must select one instance;
    finite XYZ and positive raw Z are enforced as well. Upstream sensor depth
    bounds belong in instance_keep. No invalid depth is inpainted, no +/-1
    fallback is applied, and neither input is modified.
    """
    requested = _boolean_image(instance_keep, "instance_keep")
    points = np.asarray(point_map, dtype=np.float64)
    if points.shape != requested.shape + (3,):
        raise ValueError("point_map must have shape [H,W,3] matching instance_keep")
    finite_positive = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    valid = requested & finite_positive
    safe = np.where(valid[..., None], points, 0.)
    smooth, smooth_valid, counts = _masked_xyz_mean3(points, valid)
    normal = np.zeros_like(points)
    normal_valid = np.zeros(requested.shape, dtype=bool)
    inbounds = np.zeros(requested.shape, dtype=bool)
    path_support_rejects = depth_path_rejects = cross_rejects = 0
    height, width = requested.shape
    if min(requested.shape) >= 2 * SPAN_PIXELS + 1:
        s = SPAN_PIXELS
        centre = np.s_[s:height-s, s:width-s]
        inbounds[centre] = True
        dx = smooth[s:height-s, 2*s:width] - smooth[s:height-s, :width-2*s]
        dy = smooth[2*s:height, s:width-s] - smooth[:height-2*s, s:width-s]
        smoothing_ok = (smooth_valid[centre]
                        & smooth_valid[s:height-s, 2*s:width]
                        & smooth_valid[s:height-s, :width-2*s]
                        & smooth_valid[2*s:height, s:width-s]
                        & smooth_valid[:height-2*s, s:width-s])
        raw_path_ok = np.ones(smoothing_ok.shape, dtype=bool)
        jumps_ok = np.ones_like(raw_path_ok)
        for offset in range(-s, s + 1):
            raw_path_ok &= valid[s:height-s, s+offset:width-s+offset]
            raw_path_ok &= valid[s+offset:height-s+offset, s:width-s]
        for offset in range(-s, s):
            horizontal_jump = (safe[s:height-s, s+offset+1:width-s+offset+1, 2]
                               - safe[s:height-s, s+offset:width-s+offset, 2])
            vertical_jump = (safe[s+offset+1:height-s+offset+1, s:width-s, 2]
                             - safe[s+offset:height-s+offset, s:width-s, 2])
            jumps_ok &= (np.abs(horizontal_jump) < DEPTH_JUMP_M)
            jumps_ok &= (np.abs(vertical_jump) < DEPTH_JUMP_M)
        raw = np.cross(dx, dy)
        magnitude = np.linalg.norm(raw, axis=-1)
        cross_ok = (np.isfinite(raw).all(axis=-1) & np.isfinite(magnitude)
                    & (magnitude > CROSS_NORM_MIN_M2))
        ok = valid[centre] & smoothing_ok & raw_path_ok & jumps_ok & cross_ok
        output = normal[centre]
        output[ok] = raw[ok] / magnitude[ok, None]
        normal_valid[centre] = ok
        path_support_rejects = int(np.count_nonzero(valid[centre] & ~(smoothing_ok & raw_path_ok)))
        depth_path_rejects = int(np.count_nonzero(valid[centre] & smoothing_ok & raw_path_ok & ~jumps_ok))
        cross_rejects = int(np.count_nonzero(valid[centre] & smoothing_ok & raw_path_ok & jumps_ok & ~cross_ok))
    flip = normal_valid & (np.sum(normal * -safe, axis=-1) < 0)
    normal[flip] *= -1
    diagnostics = {
        "normal_spec": NORMAL_SPEC,
        "reference_dtype": "float64",
        "image_shape": list(requested.shape),
        "requested_instance_pixels": int(requested.sum()),
        "raw_valid_pixels": int(valid.sum()),
        "invalid_or_nonpositive_requested_pixels": int(np.count_nonzero(requested & ~finite_positive)),
        "smooth_valid_pixels": int(smooth_valid.sum()),
        "normal_valid_pixels": int(normal_valid.sum()),
        "raw_valid_without_normal_pixels": int(np.count_nonzero(valid & ~normal_valid)),
        "border_raw_valid_pixels": int(np.count_nonzero(valid & ~inbounds)),
        "stencil_or_smoothing_reject_pixels": path_support_rejects,
        "adjacent_depth_path_reject_pixels": depth_path_rejects,
        "degenerate_cross_reject_pixels": cross_rejects,
        "smoothing_support_histogram_0_to_9": np.bincount(counts[valid], minlength=10).tolist(),
        "positions_modified": False,
        "fallback_to_span1": False,
    }
    return normal, normal_valid, diagnostics


def stratified_queries_refill(instance_mask, eligible, max_candidates=256):
    """Return (source_ids[K], diagnostics) using deterministic spatial refill.

    Default production construction uses cap256, then takes training prefixes.
    A smaller cap is accepted for prefix checks; it never changes buckets,
    order or within-cell selection. Source IDs use original width, v*W+u.
    """
    mask = _boolean_image(instance_mask, "instance_mask")
    eligible = _boolean_image(eligible, "eligible")
    if eligible.shape != mask.shape or np.any(eligible & ~mask):
        raise ValueError("eligible must match and be a subset of instance_mask")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, (int, np.integer)) or not 1 <= max_candidates <= 256:
        raise ValueError("max_candidates must be an integer in [1,256]")
    y, x = np.nonzero(eligible)
    ids = (y * mask.shape[1] + x).astype(np.int64)
    selected_ids, selected_cells, selected_rounds = [], [], []
    states = []
    yy, xx = np.nonzero(mask)
    if len(xx):
        left, top = xx.min() - .5, yy.min() - .5
        bbox_width, bbox_height = xx.max() - xx.min() + 1, yy.max() - yy.min() + 1
        gx = np.clip(np.floor((x - left) * 16 / bbox_width).astype(int), 0, 15)
        gy = np.clip(np.floor((y - top) * 16 / bbox_height).astype(int), 0, 15)
        for cy in range(16):
            for cx in range(16):
                rows = np.flatnonzero((gx == cx) & (gy == cy))
                if not len(rows):
                    continue
                xy = np.column_stack((x[rows], y[rows])).astype(np.float64)
                centre = np.array([left + (cx + .5) * bbox_width / 16,
                                   top + (cy + .5) * bbox_height / 16])
                d2 = np.sum((xy - centre) ** 2, axis=1)
                first = int(np.lexsort((ids[rows], d2))[0])
                states.append({"order": _cell_order(cx, cy), "cell": cy * 16 + cx,
                               "xy": xy, "ids": ids[rows], "first": first,
                               "chosen": np.zeros(len(rows), bool),
                               "nearest_distance2": np.full(len(rows), np.inf)})
        states.sort(key=lambda item: item["order"])
    first_pass_count = refill_count = rounds_with_selection = 0
    round_index = 0
    while len(selected_ids) < max_candidates:
        added = 0
        for state in states:
            remaining = np.flatnonzero(~state["chosen"])
            if not len(remaining):
                continue
            if not state["chosen"].any():
                winner = state["first"]
            else:
                nearest = state["nearest_distance2"][remaining]
                tied = remaining[nearest == nearest.max()]
                winner = int(tied[np.argmin(state["ids"][tied])])
            state["chosen"][winner] = True
            distance2 = np.sum((state["xy"] - state["xy"][winner]) ** 2, axis=1)
            state["nearest_distance2"] = np.minimum(state["nearest_distance2"], distance2)
            selected_ids.append(int(state["ids"][winner]))
            selected_cells.append(int(state["cell"]))
            selected_rounds.append(round_index)
            if round_index == 0:
                first_pass_count += 1
            else:
                refill_count += 1
            added += 1
            if len(selected_ids) == max_candidates:
                break
        if not added:
            break
        rounds_with_selection += 1
        round_index += 1
    if not mask.any():
        stop_reason = "empty_instance"
    elif not len(ids):
        stop_reason = "no_eligible_pixels"
    elif len(selected_ids) == max_candidates:
        stop_reason = "budget_reached"
    else:
        stop_reason = "eligible_exhausted"
    diagnostics = {
        "sampler_spec": SAMPLER_SPEC,
        "max_candidates": int(max_candidates),
        "eligible_pixel_count": int(len(ids)),
        "nonempty_cell_count": len(states),
        "first_pass_count": first_pass_count,
        "refill_count": refill_count,
        "refill_rounds": max(0, rounds_with_selection - 1),
        "candidate_count": len(selected_ids),
        "stop_reason": stop_reason,
        "selected_cell_ids": selected_cells,
        "selected_round_index": selected_rounds,
        "metric_uniformity_guaranteed": False,
    }
    return np.asarray(selected_ids, dtype=np.int64), diagnostics


def _plane(size=65, slope_x=0., noise_std_m=0., seed=20261118):
    """Rectified-camera fixture; plane Z=.6+slope_x*X before optional Z noise."""
    yy, xx = np.mgrid[:size, :size]
    cx = cy = (size - 1) / 2
    z = .6 / (1 - slope_x * (xx - cx) / 600.)
    if noise_std_m:
        z = z + np.random.default_rng(seed).normal(0., noise_std_m, z.shape)
    return np.stack(((xx-cx)*z/600., (yy-cy)*z/600., z), axis=-1)


def _noise_observation():
    points = _plane(noise_std_m=.0005)
    keep = np.ones(points.shape[:2], dtype=bool)
    old_n, old_valid = organized_instance_normals_v2(points, keep)
    new_n, new_valid, _ = organized_instance_normals_v3(points, keep)
    common = old_valid & new_valid
    old_angles = np.degrees(np.arccos(np.clip(-old_n[common, 2].astype(float), -1, 1)))
    new_angles = np.degrees(np.arccos(np.clip(-new_n[common, 2], -1, 1)))
    return {"fixture": "65x65_front_plane_Z0.6_fx_fy600_iid_axial_Gaussian_sigma0.0005_seed20261118",
            "common_valid_pixels": int(common.sum()),
            "v2_mean_angle_deg": float(old_angles.mean()),
            "v3_mean_angle_deg": float(new_angles.mean()),
            "v2_p95_angle_deg": float(np.percentile(old_angles, 95)),
            "v3_p95_angle_deg": float(np.percentile(new_angles, 95)),
            "scope": "one_synthetic_noise_fixture_not_sensor_measurement_or_general_noise_reduction_guarantee"}


def selftest():
    checks = []
    def check(name, ok):
        if not bool(ok):
            raise AssertionError(name)
        checks.append(name)
    def reject(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    points = _plane()
    keep = np.ones(points.shape[:2], bool)
    before_points, before_keep = points.tobytes(), keep.tobytes()
    normal, valid, diagnostics = organized_instance_normals_v3(points, keep)
    check("front_plane_camera_facing_unit_normals", np.allclose(normal[valid], [0, 0, -1]))
    check("span3_boundary_is_explicit", valid.sum() == 59**2 and not valid[:3].any() and not valid[:, :3].any())
    check("raw_XYZ_and_keep_are_unchanged", points.tobytes() == before_points and keep.tobytes() == before_keep)
    damaged = points.copy()
    damaged[20, 20] = np.nan
    damaged[30, 30, 2] = np.inf
    damaged[40, 40, 2] = 0.
    n_bad, v_bad, d_bad = organized_instance_normals_v3(damaged, keep)
    check("NaN_Inf_nonpositive_depth_do_not_propagate", np.isfinite(n_bad).all() and not v_bad[20, 20] and not v_bad[30, 30] and not v_bad[40, 40] and d_bad["invalid_or_nonpositive_requested_pixels"] == 3)
    hole = keep.copy()
    hole[32, 32] = False
    _, v_hole, _ = organized_instance_normals_v3(points, hole)
    check("span3_paths_never_jump_over_holes", not v_hole[32, 29:36].any() and not v_hole[29:36, 32].any())
    target_only = keep.copy()
    target_only[:, 32:] = False
    _, v_target, _ = organized_instance_normals_v3(points, target_only)
    check("normal_stencil_does_not_cross_instance_boundary", not v_target[:, 29:].any() and v_target[32, 28])
    sparse = np.zeros_like(keep)
    sparse[32, 32] = sparse[31, 32] = sparse[33, 32] = sparse[32, 31] = True
    _, _, d_four = organized_instance_normals_v3(points, sparse)
    sparse[32, 33] = True
    _, _, d_five = organized_instance_normals_v3(points, sparse)
    check("smoothing_requires_five_of_nine_support_points", d_four["smooth_valid_pixels"] == 0 and d_five["smooth_valid_pixels"] == 1)
    step = points.copy()
    step[:, 33:, 2] += .020
    _, v_step, d_step = organized_instance_normals_v3(step, keep)
    check("adjacent_depth_discontinuity_rejects_crossing_paths", not v_step[32, 30:36].any() and d_step["adjacent_depth_path_reject_pixels"] > 0)
    tilted = _plane(slope_x=2.)
    n_tilt, v_tilt, _ = organized_instance_normals_v3(tilted, keep)
    expected = np.array([2., 0., -1.]) / np.sqrt(5.)
    check("wide_total_depth_span_over_10mm_can_remain_valid", tilted[32, 35, 2] - tilted[32, 29, 2] > .010 and np.max(abs(np.diff(tilted[32, 29:36, 2]))) < .010 and v_tilt[32, 32] and np.allclose(n_tilt[32, 32], expected))
    degenerate = np.broadcast_to([0., 0., .6], points.shape).copy()
    n_zero, v_zero, _ = organized_instance_normals_v3(degenerate, keep)
    check("zero_area_surface_is_invalid_without_NaN", not v_zero.any() and np.isfinite(n_zero).all() and not n_zero.any())
    narrow = np.zeros_like(keep)
    narrow[:, 30:35] = True
    _, v_narrow, _ = organized_instance_normals_v3(points, narrow)
    check("narrow_support_does_not_silently_fallback_to_span1", not v_narrow.any())
    noise = _noise_observation()
    check("fixed_iid_noise_fixture_observed_angle_improves", noise["common_valid_pixels"] > 0 and noise["v3_mean_angle_deg"] < noise["v2_mean_angle_deg"])
    yy, xx = np.mgrid[:128, :128]
    diagonal = abs(xx - yy) <= 4
    old = stratified_queries_v2(diagonal, diagonal)
    ids, sampling = stratified_queries_refill(diagonal, diagonal)
    check("thin_fixture_has_1132_pixels_but_only_46_v2_cells", diagonal.sum() == 1132 and len(old) == 46)
    check("thin_fixture_refills_to_256_unique_source_pixels", len(ids) == 256 and len(np.unique(ids)) == 256 and diagonal.ravel()[ids].all())
    check("refill_first_pass_matches_v2_cell_order", np.array_equal(ids[:46], old) and sampling["first_pass_count"] == 46)
    check("refill_diagnostics_account_for_all_candidates", sampling["refill_count"] == 210 and sampling["candidate_count"] == sampling["first_pass_count"] + sampling["refill_count"] and sampling["refill_rounds"] > 0)
    ids64, _ = stratified_queries_refill(diagonal, diagonal, 64)
    ids128, _ = stratified_queries_refill(diagonal, diagonal, 128)
    check("caps64_128_are_identical_prefixes_of_cap256", np.array_equal(ids64, ids[:64]) and np.array_equal(ids128, ids[:128]))
    ids_repeat, repeat_diagnostics = stratified_queries_refill(diagonal, diagonal)
    check("refill_is_deterministic_including_metadata", np.array_equal(ids, ids_repeat) and sampling == repeat_diagnostics)
    full = np.ones((64, 64), bool)
    full_ids, full_diagnostics = stratified_queries_refill(full, full)
    check("full_grid_keeps_bit_reversed_Morton_quadrant_prefix", np.array_equal(full_ids[:4], [65, 2113, 97, 2145]))
    cell_ids = np.asarray(full_diagnostics["selected_cell_ids"])
    coarse_ids = ((cell_ids // 16) // 2) * 8 + ((cell_ids % 16) // 2)
    check("Morton_prefix64_128_cover_every_2x2_cell_block",
          np.all(np.bincount(coarse_ids[:64], minlength=64) == 1)
          and np.all(np.bincount(coarse_ids[:128], minlength=64) == 2))
    check("full_grid_first_pass_needs_no_refill", np.array_equal(full_ids, stratified_queries_v2(full, full)) and full_diagnostics["refill_count"] == 0)
    few = np.zeros_like(full)
    few[10, 10:17] = True
    few_ids, few_diagnostics = stratified_queries_refill(full, few)
    check("eligible_below_budget_is_exhausted_without_duplicates", len(few_ids) == 7 and len(np.unique(few_ids)) == 7 and set(few_ids) == set(np.flatnonzero(few)) and few_diagnostics["stop_reason"] == "eligible_exhausted")
    empty_ids, empty_diagnostics = stratified_queries_refill(np.zeros_like(full), np.zeros_like(full))
    check("empty_instance_returns_empty_with_reason", len(empty_ids) == 0 and empty_diagnostics["stop_reason"] == "empty_instance")
    no_eligible, no_eligible_diagnostics = stratified_queries_refill(full, np.zeros_like(full))
    check("nonempty_instance_without_eligible_pixels_is_explicit", len(no_eligible) == 0 and no_eligible_diagnostics["stop_reason"] == "no_eligible_pixels")
    reject("eligible_outside_instance_is_rejected", lambda: stratified_queries_refill(np.zeros_like(full), full))
    reject("caps_above_fixed_256_are_rejected", lambda: stratified_queries_refill(full, full, 257))
    return {"ok": True, "synthetic_only": True, "checks_passed": len(checks), "checks": checks,
            "normal_spec": NORMAL_SPEC, "sampler_spec": SAMPLER_SPEC,
            "noise_observation": noise,
            "thin_fixture": {key: value for key, value in sampling.items() if not key.startswith("selected_")},
            "python": platform.python_version(), "numpy": np.__version__,
            "scope": "synthetic_normal_and_sampling_contracts_only_no_teacher_model_or_latency_benchmark"}


def demo(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ["synthetic_normals_v3.npz", "thin_queries_v3.npz", "sampling_summary.json"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Use a fresh output directory")
    points = _plane(noise_std_m=.0005)
    keep = np.ones(points.shape[:2], bool)
    normal, normal_valid, normal_diagnostics = organized_instance_normals_v3(points, keep)
    np.savez_compressed(out / names[0], raw_xyz=points, instance_keep=keep,
                        normals=normal, normal_valid=normal_valid)
    yy, xx = np.mgrid[:128, :128]
    diagonal = abs(xx - yy) <= 4
    ids, sampling = stratified_queries_refill(diagonal, diagonal)
    np.savez_compressed(out / names[1], instance_mask=diagonal, eligible=diagonal,
                        source_ids=ids, train64=ids[:64], train128=ids[:128],
                        cell_ids=np.asarray(sampling["selected_cell_ids"]),
                        round_index=np.asarray(sampling["selected_round_index"]))
    summary = {"ok": True, "synthetic_only": True, "normal_diagnostics": normal_diagnostics,
               "sampling_diagnostics": {key: value for key, value in sampling.items() if not key.startswith("selected_")},
               "noise_observation": _noise_observation(), "files": names,
               "scope": "raw_query_positions_unchanged_no_sensor_data_model_teacher_or_timing"}
    (out / names[2]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", aliases=["self-test"])
    run = sub.add_parser("demo")
    run.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = selftest() if args.command in ("selftest", "self-test") else demo(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
