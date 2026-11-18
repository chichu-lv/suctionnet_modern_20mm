"""Production single-instance organized geometry and deterministic v3 queries.

This module owns its implementation and never imports executable documentation.
Float64 raw positions are immutable; only the normal-estimation copy is smoothed.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scipy.ndimage import label as connected_labels

NORMAL_SPEC = "organized_masked3x3_span3_v3"
SAMPLER_SPEC = "image_stratified_morton_refill_v3"
PRECISION_SPEC = "raw_float64_network_float32_v1"
DEPTH_JUMP_M = .010
MIN_SMOOTHING_SUPPORT = 5
SPAN_PIXELS = 3
CROSS_NORM_MIN_M2 = 1e-8


def _cell_order(x, y):
    morton = sum(((x >> bit) & 1) << (2 * bit) |
                 ((y >> bit) & 1) << (2 * bit + 1) for bit in range(4))
    return sum(((morton >> bit) & 1) << (7 - bit) for bit in range(8))


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


def array_digest(*arrays):
    """Content identity includes dtype and shape, not machine-specific paths."""
    digest = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        digest.update(a.dtype.str.encode())
        digest.update(json.dumps(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def prepare_instance(rgb, depth_m, mask, intrinsics, gravity, *,
                     max_queries=256, depth_min_m=.2, depth_max_m=1.2,
                     min_support=100):
    """Build an unlabeled instance from rectified RGB-D, retaining raw pixels.

    ``instance_mask`` is the full supplied instance, including depth holes.
    ``mask`` is its finite in-range depth support. Missing RGB is zero-filled
    only together with an explicit validity channel. No GT physical geometry
    or teacher quality is accepted by this function.
    """
    instance_mask = _boolean_image(mask, 'mask').copy()
    z = np.asarray(depth_m, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    g = np.asarray(gravity, dtype=np.float64)
    if z.shape != instance_mask.shape:
        raise ValueError('depth_m must match mask HxW')
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('intrinsics must be finite 3x3 with positive focal lengths')
    if not np.allclose(K[2], [0., 0., 1.]) or abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12:
        raise ValueError('rectified zero-skew pinhole intrinsics required')
    if g.shape != (3,) or not np.isfinite(g).all() or np.linalg.norm(g) <= 1e-12:
        raise ValueError('gravity must be a finite nonzero length-3 vector')
    if not np.isfinite([depth_min_m, depth_max_m]).all() or not 0 <= depth_min_m < depth_max_m:
        raise ValueError('depth bounds must be increasing finite metres')
    if isinstance(min_support, bool) or int(min_support) != min_support or min_support < 1:
        raise ValueError('min_support must be a positive integer')
    colors = np.asarray(rgb)
    if colors.shape != z.shape + (3,):
        raise ValueError('RGB must have shape HxWx3')
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.
    elif np.issubdtype(colors.dtype, np.floating):
        colors = colors.astype(np.float32, copy=True)
    else:
        raise ValueError('RGB must be uint8 or floating [0,1]')
    rgb_valid = np.isfinite(colors).all(axis=-1)
    if np.any(rgb_valid & ((colors < 0).any(-1) | (colors > 1).any(-1))):
        raise ValueError('finite floating RGB values must be in [0,1]')
    rgb_valid &= instance_mask
    colors[~rgb_valid] = 0
    # Single-instance caches retain original image coordinates, but do not
    # duplicate scene background or permit it to become a hidden model input.
    z = np.where(instance_mask, z, 0.)
    depth_valid = instance_mask & np.isfinite(z) & (z > 0) & (z >= depth_min_m) & (z <= depth_max_m)
    safe_depth = np.where(depth_valid, z, 0.)
    yy, xx = np.indices(z.shape)
    xyz = np.stack(((xx - K[0, 2]) * safe_depth / K[0, 0],
                    (yy - K[1, 2]) * safe_depth / K[1, 1], safe_depth), axis=-1)
    support_mask = instance_mask & depth_valid
    normals, normal_valid, normal_diagnostics = organized_instance_normals_v3(xyz, support_mask)
    eligible = support_mask & normal_valid
    source_ids, query_diagnostics = stratified_queries_refill(instance_mask, eligible, max_queries)
    support_ids = np.flatnonzero(support_mask.ravel()).astype(np.int64)
    h, w = z.shape
    points = xyz.reshape(-1, 3)[support_ids]
    mu = points.mean(axis=0) if len(points) else np.zeros(3, np.float64)
    component_map, component_count = connected_labels(
        support_mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.int8))
    ys, xs = np.nonzero(instance_mask)
    bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.int64) if len(xs) else np.zeros(4, np.int64)
    edges = bbox.astype(np.float64) - .5 if len(xs) else np.zeros(4, np.float64)
    uv = lambda ids: np.column_stack((ids % w, ids // w)).astype(np.int64)
    result = dict(
        rgb=colors, rgb_valid=rgb_valid, depth_m=z.copy(), depth_valid=depth_valid,
        xyz_map=xyz, instance_mask=instance_mask, mask=support_mask,
        normal_map=normals, normal_valid=normal_valid, eligible_mask=eligible,
        component_map=component_map.astype(np.int32), support_points=points,
        support_rgb=colors.reshape(-1, 3)[support_ids],
        support_rgb_valid=rgb_valid.ravel()[support_ids],
        support_normals=normals.reshape(-1, 3)[support_ids],
        support_normal_valid=normal_valid.ravel()[support_ids],
        support_uv=uv(support_ids), support_source_ids=support_ids,
        support_components=component_map.ravel()[support_ids].astype(np.int32),
        query_points=xyz.reshape(-1, 3)[source_ids],
        query_normals=normals.reshape(-1, 3)[source_ids], query_uv=uv(source_ids),
        query_ids=np.arange(len(source_ids), dtype=np.int64), query_source_ids=source_ids,
        query_components=component_map.ravel()[source_ids].astype(np.int32),
        query_valid=np.ones(len(source_ids), bool),
        query_cell_ids=np.asarray(query_diagnostics['selected_cell_ids'], np.int16),
        query_rounds=np.asarray(query_diagnostics['selected_round_index'], np.int16),
        centroid=mu, intrinsics=K.copy(), gravity=g / np.linalg.norm(g),
        bbox_xyxy=bbox, bbox_pixel_edges=edges,
        image_shape=np.array([h, w], np.int64), trainable=bool(len(points) >= min_support and len(source_ids)),
        normal_spec=NORMAL_SPEC, sampler_spec=SAMPLER_SPEC, precision_spec=PRECISION_SPEC,
        diagnostics=dict(normal=normal_diagnostics, sampling=query_diagnostics,
                         support_count=len(points), component_count=int(component_count),
                         min_support=int(min_support),
                         status='ok' if len(points) >= min_support and len(source_ids) else
                         ('insufficient_support' if len(points) < min_support else 'no_valid_normal')),
    )
    result['query_identity'] = array_digest(result['query_points'], result['query_normals'], source_ids)
    result['observation_identity'] = array_digest(colors, rgb_valid, z, instance_mask, K, result['gravity'])
    return result
