"""Static source-image geometry and per-exposure RGB for route B.

The static object owns NumPy arrays, contains no source-array references, RGB,
query phase or learned features, and can be retained by the bounded worker cache.
Its owner supplies the source identity; shape/profile checks here do not claim
to validate that identity. Camera pixel coordinates remain global after cropping.
"""
from __future__ import annotations

import math

import numpy as np
import torch


STATIC_INPUTS_SPEC = "route_b_static_image_geometry64_rgb_indices_quad_v2"


def bilinear_validity_map(xyz, mask):
    """Four ORIGINAL pixels valid with <=10 mm depth span at each top-left.

    This depends only on immutable image geometry, never on RGB, query phase
    or model weights. A rectangle's last row/column cannot begin a valid quad;
    no geometry outside a compact valid-support rectangle can complete one.
    The query consumer separately checks original-camera bounds before lookup.
    """
    result = torch.zeros_like(mask, dtype=torch.bool)
    if min(mask.shape) < 2:
        return result
    corners = ((slice(None, -1), slice(None, -1)),
               (slice(None, -1), slice(1, None)),
               (slice(1, None), slice(None, -1)),
               (slice(1, None), slice(1, None)))
    valid = torch.stack([mask[y, x] for y, x in corners], -1).all(-1)
    depth = torch.stack([xyz[y, x, 2] for y, x in corners], -1)
    result[:-1, :-1] = valid & ((depth.amax(-1) - depth.amin(-1)) <= .010 + 1e-12)
    return result


def _numpy_owned(value):
    return value.numpy().copy() if torch.is_tensor(value) else value


def _bbox(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return 0, 0, 0, 0
    return int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1


def _roi_at(image, iy, ix, *, top_left):
    """Read a zero-outside ROI at already-clamped original camera pixels."""
    y0, x0 = top_left
    h, w = image.shape[:2]
    if not h or not w:
        return image.new_zeros((*iy.shape, *image.shape[2:]))
    ly, lx = iy - y0, ix - x0
    inside = (ly >= 0) & (ly < h) & (lx >= 0) & (lx < w)
    value = image[ly.clamp(0, h - 1), lx.clamp(0, w - 1)]
    for _ in image.shape[2:]:
        inside = inside.unsqueeze(-1)
    return torch.where(inside, value, torch.zeros_like(value))


def _nearest_indices(uv, image_hw):
    height, width = image_hw
    ix = torch.floor(uv[..., 0] + .5).long()
    iy = torch.floor(uv[..., 1] + .5).long()
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    cy, cx = iy.clamp(0, height - 1), ix.clamp(0, width - 1)
    return cy, cx, inside


def prepare_source_static_inputs(observation, *, min_support_points=100, sample_image_channels=True):
    """Prepare only RGB/phase/weight-independent source data on the CPU.

    Pixel validity, reductions and FP64 geometry use the reference Torch
    operations. Whole-camera XYZ/normal conversion is avoided by first finding
    the instance mask's rectangle; rejected pixels are still exactly zero.
    """
    if not sample_image_channels:
        raise ValueError("Static route-B inputs require sampled image channels")
    if isinstance(min_support_points, bool) or not isinstance(min_support_points, int) or min_support_points < 1:
        raise ValueError("min_support_points must be a positive integer")
    raw_xyz = np.asarray(observation["xyz_map"])
    if raw_xyz.ndim != 3 or raw_xyz.shape[-1] != 3 or min(raw_xyz.shape[:2]) < 1:
        raise ValueError("xyz_map must be a nonempty HxWx3 array")
    height, width = raw_xyz.shape[:2]
    image_hw = (height, width)
    mask = np.asarray(observation["mask"], dtype=bool)
    full_mask = np.asarray(observation.get("instance_mask", observation["mask"]), dtype=bool)
    if mask.shape != image_hw or full_mask.shape != image_hw:
        raise ValueError("mask shape must match xyz_map")
    raw_normals = np.asarray(observation["normal_map"])
    raw_nv = np.asarray(observation["normal_valid"], dtype=bool)
    if raw_normals.shape != raw_xyz.shape or raw_nv.shape != image_hw:
        raise ValueError("normal_map/normal_valid must match xyz_map")

    y0, y1, x0, x1 = _bbox(mask)
    xyz = torch.as_tensor(np.ascontiguousarray(raw_xyz[y0:y1, x0:x1]), dtype=torch.float64)
    valid = torch.as_tensor(np.ascontiguousarray(mask[y0:y1, x0:x1]))
    valid = valid & torch.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    # Trim only after validating depth; the geometry ROI may be smaller than the
    # full instance-mask bbox used for the context and texture branch.
    ry0, ry1, rx0, rx1 = _bbox(valid.numpy())
    xyz, valid = xyz[ry0:ry1, rx0:rx1], valid[ry0:ry1, rx0:rx1]
    y0, x0 = y0 + ry0, x0 + rx0
    y1, x1 = y0 + valid.shape[0], x0 + valid.shape[1]
    xyz = torch.where(valid[..., None], xyz, torch.zeros_like(xyz))
    normals = torch.as_tensor(np.ascontiguousarray(raw_normals[y0:y1, x0:x1]), dtype=torch.float64)
    nv = torch.as_tensor(np.ascontiguousarray(raw_nv[y0:y1, x0:x1]))
    nv = nv & valid & torch.isfinite(normals).all(-1) & (normals.norm(dim=-1) > 1e-8)
    normals = torch.where(nv[..., None], normals, torch.zeros_like(normals))
    gravity = torch.as_tensor(observation["gravity"], dtype=torch.float64).reshape(3)
    if not bool(torch.isfinite(gravity).all()) or float(gravity.norm()) <= 1e-8:
        raise ValueError("gravity must be finite and nonzero")
    gravity = gravity / gravity.norm()
    intrinsics = torch.as_tensor(observation["intrinsics"], dtype=torch.float64)
    if intrinsics.shape != (3, 3) or not bool(torch.isfinite(intrinsics).all()):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if float(intrinsics[0, 0]) <= 0 or float(intrinsics[1, 1]) <= 0:
        raise ValueError("intrinsics require positive focal lengths")
    if not torch.allclose(intrinsics[2], intrinsics.new_tensor((0., 0., 1.)), atol=1e-12, rtol=0):
        raise ValueError("intrinsics must use the standard pinhole homogeneous row")
    points = xyz[valid]
    support_count = len(points)
    centroid = points.mean(0) if len(points) else xyz.new_zeros(3)
    extent = points.amax(0) - points.amin(0) if len(points) else xyz.new_zeros(3)
    z_med = torch.quantile(points[:, 2], .5) if len(points) else xyz.new_zeros(())
    ymin, ymax, xmin, xmax = _bbox(full_mask)
    if not ymax or not xmax:
        ymin = xmin = 0
        ymax = xmax = 1
    box_width, box_height = xmax - xmin, ymax - ymin
    margin_x, margin_y = math.ceil(.1 * box_width), math.ceil(.1 * box_height)
    length = max(box_width + 2 * margin_x, box_height + 2 * margin_y)
    left = (xmin + xmax - 1) / 2 - length / 2
    top = (ymin + ymax - 1) / 2 - length / 2
    scale = 128. / length
    yy, xx = torch.meshgrid(torch.arange(128, dtype=torch.float64),
                            torch.arange(128, dtype=torch.float64), indexing="ij")
    uv_source = torch.stack((left + (xx + .5) / scale, top + (yy + .5) / scale), -1)
    cy, cx, cinside = _nearest_indices(uv_source, image_hw)
    cv = _roi_at(valid, cy, cx, top_left=(y0, x0))
    cxyz = _roi_at(xyz, cy, cx, top_left=(y0, x0))
    cnormals = _roi_at(normals, cy, cx, top_left=(y0, x0))
    context_geometry = torch.cat((((cxyz - centroid) / .1).float(), cnormals.float(),
                                  gravity.float().expand(*cv.shape, 3), cv[..., None].float()), -1)
    context_valid = cv & cinside
    context_geometry = torch.where(context_valid[..., None], context_geometry,
                                   torch.zeros_like(context_geometry)).permute(2, 0, 1)[None].contiguous()

    tx0, ty0 = xmin - 3, ymin - 3
    roi_w, roi_h = box_width + 6, box_height + 6
    pad_w, pad_h = 32 * math.ceil(roi_w / 32), 32 * math.ceil(roi_h / 32)
    ry, rx = torch.meshgrid(torch.arange(pad_h), torch.arange(pad_w), indexing="ij")
    roi_uv = torch.stack((rx + tx0, ry + ty0), -1).to(torch.float64)
    ty, tx, tinside = _nearest_indices(roi_uv, image_hw)
    tv = _roi_at(valid, ty, tx, top_left=(y0, x0))
    tz = _roi_at(xyz[..., 2], ty, tx, top_left=(y0, x0))
    texture_valid = tv & tinside & (rx < roi_w) & (ry < roi_h)
    texture_geometry = torch.stack((((tz - z_med) / .1).float(), tv.float()), -1)
    texture_geometry = torch.where(texture_valid[..., None], texture_geometry,
                                   torch.zeros_like(texture_geometry)).permute(2, 0, 1)[None].contiguous()
    affine = xyz.new_tensor(((scale, 0., -scale * left - .5),
                             (0., scale, -scale * top - .5), (0., 0., 1.)))
    encoded = dict(xyz_map=xyz, mask=valid, normal_map=normals, normal_valid=nv,
                   bilinear_valid_map=bilinear_validity_map(xyz, valid),
                   intrinsics=intrinsics, gravity=gravity, centroid=centroid, extent=extent,
                   support_points=points, context_affine=affine,
                   context_affine_inverse=torch.linalg.inv(affine), intrinsics_context=affine @ intrinsics,
                   texture_origin_uv=xyz.new_tensor((tx0, ty0)), texture_roi_hw=(roi_h, roi_w),
                   texture_padded_hw=(pad_h, pad_w), bbox_xyxy=(xmin, ymin, xmax, ymax),
                   support_count=support_count, support_sufficient=support_count >= min_support_points,
                   instance_valid=support_count >= min_support_points)
    return dict(spec=STATIC_INPUTS_SPEC, min_support_points=min_support_points,
                image_hw=image_hw, top_left=(y0, x0),
                encoded={key: _numpy_owned(value) for key, value in encoded.items()},
                context_geometry=_numpy_owned(context_geometry), texture_geometry=_numpy_owned(texture_geometry),
                context_indices=_numpy_owned(torch.stack((cy, cx), -1)),
                texture_indices=_numpy_owned(torch.stack((ty, tx), -1)),
                context_valid=_numpy_owned(context_valid), texture_valid=_numpy_owned(texture_valid))


def combine_source_images(observation, static_inputs, *, min_support_points):
    """Attach only the current exposure's sampled RGB to immutable geometry."""
    if (static_inputs.get("spec") != STATIC_INPUTS_SPEC
            or static_inputs.get("min_support_points") != min_support_points):
        raise ValueError("Static route-B input profile mismatch")
    raw = np.asarray(observation["rgb"])
    image_hw = tuple(static_inputs["image_hw"])
    if raw.shape != (*image_hw, 3) or np.asarray(observation["xyz_map"]).shape != raw.shape:
        raise ValueError("Static route-B image shape mismatch")
    rgb_valid = observation.get("rgb_valid")
    if rgb_valid is not None:
        rgb_valid = np.asarray(rgb_valid, dtype=bool)
        if rgb_valid.shape != image_hw:
            raise ValueError("rgb_valid shape must match xyz_map")
    mean = torch.tensor((.485, .456, .406), dtype=torch.float32)
    std = torch.tensor((.229, .224, .225), dtype=torch.float32)

    def image_channels(branch):
        indices = static_inputs[branch + "_indices"]
        iy, ix = indices[..., 0], indices[..., 1]
        rgb = torch.as_tensor(np.ascontiguousarray(raw[iy, ix]), dtype=torch.float32)
        finite = torch.isfinite(rgb).all(-1)
        rv = finite if rgb_valid is None else torch.as_tensor(np.ascontiguousarray(rgb_valid[iy, ix]))
        rgb = torch.where((rv & finite)[..., None], rgb, torch.zeros_like(rgb))
        rgb = (rgb - mean) / std
        keep = torch.as_tensor(static_inputs[branch + "_valid"]) & rv
        rgb = torch.where(keep[..., None], rgb, torch.zeros_like(rgb)).permute(2, 0, 1)[None]
        geometry = torch.as_tensor(static_inputs[branch + "_geometry"])
        return torch.cat((rgb, geometry), 1).contiguous().numpy()

    # All dynamic allocations are new. No cached geometry array is mutated.
    return dict(static_inputs["encoded"], context=image_channels("context"), texture=image_channels("texture"),
                geometry_image_hw=image_hw, geometry_top_left=tuple(static_inputs["top_left"]),
                sample_uid=str(observation.get("sample_uid", observation.get("instance_uid", "instance"))))
