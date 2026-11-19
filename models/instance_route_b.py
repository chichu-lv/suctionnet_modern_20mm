"""Route B v3: shared image features and pose-conditioned projected rings.

The network uses only PyTorch operators.  Geometric decisions use float64;
learned tensors use the module dtype (float32 by default).  An encoded item is
one instance, so gradient accumulation does not change the native-ROI padding
contract.  No GT object geometry is read by this module.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from suction20.instance_config import B_FEATURE_FRAME, B_LEGACY_FEATURE_FRAME
from .instance_route_b_preparation import (bilinear_validity_map, combine_source_images,
                                           prepare_source_static_inputs)


RADII_M = (0., .005, .009, .010, .014, .017, .018, .020, .022)
ANGLES = 72
BOUNDARY_EPS_M = 1e-12
GEOMETRY_CONTRACT = "b_v3_supported_projected_rings_float64"
CPU_INPUTS_SPEC = "route_b_cpu_inputs_fp32_geometry64_v1"
TRAINING_INPUTS_SPEC = "route_b_training_roi_fp32_geometry64_quad_v3"
_IMAGE_SOURCE_KEYS = ("xyz_map", "mask", "instance_mask", "rgb", "rgb_valid",
                      "normal_map", "normal_valid", "gravity", "intrinsics")


def _tensor(value, device, dtype):
    if hasattr(value, "strides") and any(stride < 0 for stride in value.strides):
        value = value.copy()
    return torch.as_tensor(value, device=device, dtype=dtype)


def _mlp(widths):
    modules = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        modules.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def _gn(channels):
    return nn.GroupNorm(8, channels, eps=1e-5, affine=True)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, source, target, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(source, target, 3, stride, 1, bias=False)
        self.norm1 = _gn(target)
        self.conv2 = nn.Conv2d(target, target, 3, 1, 1, bias=False)
        self.norm2 = _gn(target)
        self.shortcut = (nn.Identity() if source == target and stride == 1 else
                         nn.Sequential(nn.Conv2d(source, target, 1, stride, bias=False), _gn(target)))

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(h)) + self.shortcut(x))


class ResNet18GN(nn.Module):
    """Standard ResNet-18 feature stages, 13 input channels and GN8 throughout."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(13, 64, 7, 2, 3, bias=False), _gn(64), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        self.layer1 = nn.Sequential(_BasicBlock(64, 64), _BasicBlock(64, 64))
        self.layer2 = nn.Sequential(_BasicBlock(64, 128, 2), _BasicBlock(128, 128))
        self.layer3 = nn.Sequential(_BasicBlock(128, 256, 2), _BasicBlock(256, 256))
        self.layer4 = nn.Sequential(_BasicBlock(256, 512, 2), _BasicBlock(512, 512))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        l1 = self.layer1(self.stem(x))
        l2 = self.layer2(l1)
        return l1, l2, self.layer4(self.layer3(l2))


def masked_mean_max(features, mask):
    """Pool the last dimension, preserving zero for an empty support set."""
    # Accumulate reductions in FP32 even when the surrounding network uses AMP.
    if features.dtype in (torch.float16, torch.bfloat16):
        features = features.float()
    m = mask.unsqueeze(-2)
    safe = torch.where(m, features, torch.zeros_like(features))
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    mean = safe.sum(-1) / count.to(features.dtype)
    maximum = torch.where(m, safe, torch.full_like(safe, -torch.inf)).amax(-1)
    maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


class MaskedRingEncoder(nn.Module):
    """A region's three radii -> 72 angular features -> masked mean/max."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(75, 16, 1, bias=False),
                                    nn.Conv1d(16, 16, 3, bias=False),
                                    nn.Conv1d(16, 16, 3, bias=False)])
        self.norms = nn.ModuleList([nn.LayerNorm(16, eps=1e-5) for _ in range(3)])

    def forward(self, tokens, angle_valid):
        if tokens.shape[0] == 0:
            return tokens.new_empty((0, 32))
        h = tokens
        m = angle_valid[:, None, :]
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = torch.where(m, h, torch.zeros_like(h))
            if i:
                h = torch.cat((h[..., -1:], h, h[..., :1]), dim=-1)
            h = conv(h)
            h = F.relu(norm(h.transpose(1, 2))).transpose(1, 2)
            h = torch.where(m, h, torch.zeros_like(h))
        return masked_mean_max(h, angle_valid)


def fixed_z_frames(normals, *, feature_frame=B_FEATURE_FRAME):
    """Two-chart Rodrigues basis; input normals must already be finite/unit.

    The current chart switches at nz > .5, keeping the Rodrigues denominator
    at least .5.  The legacy chart is explicit for saved model compatibility.
    Both choices retain a discontinuous in-plane basis at the chart seam;
    changing the policy therefore changes the feature sampling contract.
    """
    n = normals.to(torch.float64)
    if feature_frame == B_FEATURE_FRAME:
        positive = n[..., 2] > .5
    elif feature_frame == B_LEGACY_FEATURE_FRAME:
        positive = (1. - n[..., 2]) <= 1e-4
    else:
        raise ValueError("Unsupported route-B feature_frame")
    a = torch.zeros_like(n)
    a[..., 2] = torch.where(positive, torch.ones_like(n[..., 2]), -torch.ones_like(n[..., 2]))
    b1 = torch.zeros_like(n)
    b1[..., 0] = 1.
    b2 = torch.zeros_like(n)
    b2[..., 1] = torch.where(positive, torch.ones_like(n[..., 2]), -torch.ones_like(n[..., 2]))
    v = torch.cross(a, n, dim=-1)
    c = (a * n).sum(-1, keepdim=True)

    def rotate(b):
        return b + torch.cross(v, b, dim=-1) + torch.cross(v, torch.cross(v, b, dim=-1), dim=-1) / (1 + c)

    return rotate(b1), rotate(b2), positive


def ring_directions(normals, phase, *, feature_frame=B_FEATURE_FRAME):
    e1, e2, chart = fixed_z_frames(normals, feature_frame=feature_frame)
    theta = 2 * math.pi * (torch.arange(ANGLES, device=normals.device, dtype=torch.float64) + .5) / ANGLES
    theta = theta[None, :] + phase[:, None]
    cosine, sine = theta.cos()[..., None], theta.sin()[..., None]
    radial = e1[:, None, :] * cosine + e2[:, None, :] * sine
    tangent = -e1[:, None, :] * sine + e2[:, None, :] * cosine
    return radial, tangent, chart


def projected_ring_gate(observed_points, observed_valid, points, normals, phase,
                        *, feature_frame=B_FEATURE_FRAME, _directions=None):
    """Float64 v3 contact gates, with raw observations retained independently."""
    radial, angular, chart = (ring_directions(normals, phase, feature_frame=feature_frame)
                              if _directions is None else _directions)
    delta = torch.where(observed_valid[..., None], observed_points, points[:, None, None, :]) - points[:, None, None, :]
    height = (delta * normals[:, None, None, :]).sum(-1)
    tangent = delta - height[..., None] * normals[:, None, None, :]
    rho = torch.linalg.vector_norm(tangent, dim=-1)
    tangential = (tangent * angular[:, None]).sum(-1)
    signed_radial = (tangent * radial[:, None]).sum(-1)
    snapped = rho
    for boundary in (0., .010, .018, .022):
        snapped = torch.where((snapped - boundary).abs() <= BOUNDARY_EPS_M,
                              torch.full_like(snapped, boundary), snapped)
    region = torch.cat(((snapped[:, :3] >= 0) & (snapped[:, :3] < .010),
                        (snapped[:, 3:6] >= .010) & (snapped[:, 3:6] < .018),
                        (snapped[:, 6:] >= .018) & (snapped[:, 6:] <= .022)), dim=1) & observed_valid
    radii = rho.new_tensor(RADII_M)[None, :, None]
    radial_error = rho - radii
    radial_ok = radial_error.abs() <= .003 + BOUNDARY_EPS_M
    tangential_ok = tangential.abs() <= .003 + BOUNDARY_EPS_M
    positive_radial = signed_radial > 0
    contact = observed_valid & region & radial_ok & tangential_ok & positive_radial
    centre = observed_valid[:, 0, 0] & region[:, 0, 0] & (rho[:, 0, 0] <= .003 + BOUNDARY_EPS_M)
    contact = torch.cat((centre[:, None, None].expand(-1, 1, ANGLES), contact[:, 1:]), dim=1)
    return {"observed_valid": observed_valid, "contact_valid": contact, "height_m": height,
            "rho_m": rho, "radial_error_m": radial_error, "tangential_offset_m": tangential,
            "signed_radial_m": signed_radial, "in_region": region, "radial_ok": radial_ok,
            "tangential_ok": tangential_ok, "positive_radial": positive_radial,
            "radial_directions": radial, "chart_positive_z": chart}


def ring_support_summary(contact, observed, height):
    """Interleaved nine (coverage,gap) pairs followed by three raw summaries."""
    count = contact.sum(-1).to(height.dtype)
    doubled = torch.cat((contact, contact), dim=-1)
    index = torch.arange(2 * ANGLES, device=contact.device).expand_as(doubled)
    last_valid = torch.where(doubled, index, -torch.ones_like(index)).cummax(-1).values
    gap = torch.where(doubled, torch.zeros_like(index), index - last_valid).amax(-1).clamp_max(ANGLES)
    support = torch.stack((count / ANGLES, gap.to(height.dtype) / ANGLES), -1).flatten(1)
    obs = observed.reshape(-1, 3, 3 * ANGLES)
    con = contact.reshape(-1, 3, 3 * ANGLES)
    raw_h = torch.where(observed, height, torch.zeros_like(height)).reshape(-1, 3, 3 * ANGLES)
    obs_count = obs.sum(-1).to(height.dtype)
    reject = (obs & ~con).sum(-1).to(height.dtype) / obs_count.clamp_min(1)
    pos = torch.where(obs, raw_h, torch.zeros_like(raw_h)).amax(-1).clamp_min(0) / .040
    neg = torch.where(obs, -raw_h, torch.zeros_like(raw_h)).amax(-1).clamp_min(0) / .040
    raw = torch.stack((obs_count / (3 * ANGLES), reject, pos, neg), dim=-1).flatten(1)
    return support, raw


def sample_feature_at_indices(feature, uv):
    """Bilinear sampling at feature-index coordinates, no hidden stride shift."""
    # Geometry stays FP64 up to this sampling boundary.  A BF16 grid would
    # quantize source locations, and CPU grid_sample has no BF16 kernel.
    if feature.dtype in (torch.float16, torch.bfloat16):
        feature = feature.float()
    h, w = feature.shape[-2:]
    can_touch = (torch.isfinite(uv).all(-1) & (uv[..., 0] > -1) & (uv[..., 0] < w)
                 & (uv[..., 1] > -1) & (uv[..., 1] < h))
    # A finite float64 coordinate can overflow on the FP32 network boundary.
    # Use finite, fully out-of-image placeholders before the dtype conversion;
    # NaN/Inf grid values can poison backward even after output masking.
    safe_uv = torch.where(can_touch[..., None], uv, torch.full_like(uv, -2.))
    grid = torch.stack((2 * (safe_uv[..., 0] + .5) / w - 1,
                        2 * (safe_uv[..., 1] + .5) / h - 1), dim=-1)
    shape = uv.shape[:-1]
    grid = grid.reshape(1, -1, 1, 2).to(feature.dtype)
    out = F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out[0, :, :, 0].transpose(0, 1).reshape(*shape, feature.shape[1])


def _nearest_map(image, uv):
    """Nearest pixel centres, floor(x+.5) ties, and explicit out-of-image mask."""
    h, w = image.shape[:2]
    finite = torch.isfinite(uv).all(-1)
    # Keep an invalid grid finite before integer conversion or feature sampling.
    safe_uv = torch.where(finite[..., None], uv, torch.zeros_like(uv))
    ix = torch.floor(safe_uv[..., 0] + .5).to(torch.long)
    iy = torch.floor(safe_uv[..., 1] + .5).to(torch.long)
    inside = finite & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    values = image[iy.clamp(0, h - 1), ix.clamp(0, w - 1)]
    return values, inside, iy * w + ix, torch.stack((ix, iy), dim=-1)


def _geometry_at_pixels_many(encoded, keys, iy, ix):
    """Reuse camera/ROI indices when reading several maps at the same pixels."""
    image = encoded[keys[0]]
    h, w = encoded.get("geometry_image_hw", image.shape[:2])
    cy, cx = iy.clamp(0, h - 1), ix.clamp(0, w - 1)
    y0, x0 = encoded.get("geometry_top_left", (0, 0))
    rh, rw = image.shape[:2]
    if (rh, rw) == (h, w) and (y0, x0) == (0, 0):
        return {key: encoded[key][cy, cx] for key in keys}
    if not rh or not rw:
        return {key: encoded[key].new_zeros((*iy.shape, *encoded[key].shape[2:])) for key in keys}
    ly, lx = cy - y0, cx - x0
    inside_roi = (ly >= 0) & (ly < rh) & (lx >= 0) & (lx < rw)
    ly, lx = ly.clamp(0, rh - 1), lx.clamp(0, rw - 1)
    result = {}
    for key in keys:
        values = encoded[key][ly, lx]
        mask = inside_roi.reshape((*inside_roi.shape, *((1,) * (encoded[key].ndim - 2))))
        result[key] = torch.where(mask, values, torch.zeros_like(values))
    return result


def _geometry_at_pixels(encoded, key, iy, ix):
    """Read compact geometry using ORIGINAL camera indices and edge clamping."""
    return _geometry_at_pixels_many(encoded, (key,), iy, ix)[key]


def _nearest_geometry_many(encoded, keys, uv):
    """Nearest original pixel, independent of the stored geometry rectangle."""
    h, w = encoded.get("geometry_image_hw", encoded[keys[0]].shape[:2])
    finite = torch.isfinite(uv).all(-1)
    safe_uv = torch.where(finite[..., None], uv, torch.zeros_like(uv))
    ix = torch.floor(safe_uv[..., 0] + .5).long()
    iy = torch.floor(safe_uv[..., 1] + .5).long()
    inside = finite & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    return (_geometry_at_pixels_many(encoded, keys, iy, ix), inside,
            iy * w + ix, torch.stack((ix, iy), -1))


def _nearest_geometry(encoded, key, uv):
    values, inside, source_id, nearest_uv = _nearest_geometry_many(encoded, (key,), uv)
    return values[key], inside, source_id, nearest_uv


def prepare_shared_images(observation: Mapping, *, device="cpu", feature_dtype=torch.float32,
                          min_support_points=100, sample_image_channels=False):
    """Build context13 and native texture5 without changing the raw geometry.

    RGB photometric augmentation, when requested, must already have been
    applied once to observation['rgb']; both branches then use that same RGB.
    Support counts finite positive-depth instance pixels, independently of
    normal validity or an optional flattened observation['support_points'].
    """
    if isinstance(min_support_points, bool) or not isinstance(min_support_points, int) or min_support_points < 1:
        raise ValueError("min_support_points must be a positive integer")
    xyz = _tensor(observation["xyz_map"], device, torch.float64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or min(xyz.shape[:2]) < 1:
        raise ValueError("xyz_map must be a nonempty HxWx3 array")
    h, w = xyz.shape[:2]
    mask = _tensor(observation["mask"], device, torch.bool)
    full_mask = _tensor(observation.get("instance_mask", observation["mask"]), device, torch.bool)
    if mask.shape != (h, w) or full_mask.shape != (h, w):
        raise ValueError("mask shape must match xyz_map")
    valid = mask & torch.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    xyz = torch.where(valid[..., None], xyz, torch.zeros_like(xyz))
    rgb = _tensor(observation["rgb"], device, feature_dtype)
    if rgb.shape != (h, w, 3):
        raise ValueError("rgb must be HxWx3 in [0,1]")
    rgb_valid = _tensor(observation.get("rgb_valid", torch.isfinite(rgb).all(-1)), device, torch.bool)
    rgb = torch.where((valid & rgb_valid & torch.isfinite(rgb).all(-1))[..., None], rgb, torch.zeros_like(rgb))
    normals = _tensor(observation["normal_map"], device, torch.float64)
    nv = _tensor(observation["normal_valid"], device, torch.bool)
    if normals.shape != xyz.shape or nv.shape != valid.shape:
        raise ValueError("normal_map/normal_valid must match xyz_map")
    nv = nv & valid & torch.isfinite(normals).all(-1) & (normals.norm(dim=-1) > 1e-8)
    normals = torch.where(nv[..., None], normals, torch.zeros_like(normals))
    gravity = _tensor(observation["gravity"], device, torch.float64).reshape(3)
    if not bool(torch.isfinite(gravity).all()) or float(gravity.norm()) <= 1e-8:
        raise ValueError("gravity must be finite and nonzero")
    gravity = gravity / gravity.norm()
    intrinsics = _tensor(observation["intrinsics"], device, torch.float64)
    if intrinsics.shape != (3, 3) or not bool(torch.isfinite(intrinsics).all()):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if float(intrinsics[0, 0]) <= 0 or float(intrinsics[1, 1]) <= 0:
        raise ValueError("intrinsics require positive focal lengths")
    if not torch.allclose(intrinsics[2], intrinsics.new_tensor((0., 0., 1.)), atol=1e-12, rtol=0):
        raise ValueError("intrinsics must use the standard pinhole homogeneous row")
    points = xyz[valid]
    support_count = len(points)
    support_sufficient = support_count >= min_support_points
    centroid = points.mean(0) if len(points) else xyz.new_zeros(3)
    extent = points.amax(0) - points.amin(0) if len(points) else xyz.new_zeros(3)
    nonzero = torch.nonzero(full_mask, as_tuple=False)
    if len(nonzero):
        ymin, xmin = nonzero.amin(0).tolist()
        ymax, xmax = nonzero.amax(0).tolist()
    else:
        xmin = ymin = xmax = ymax = 0
    width, height = xmax - xmin + 1, ymax - ymin + 1
    margin_x, margin_y = math.ceil(.1 * width), math.ceil(.1 * height)
    length = max(width + 2 * margin_x, height + 2 * margin_y)
    left = (xmin + xmax) / 2 - length / 2
    top = (ymin + ymax) / 2 - length / 2
    scale = 128. / length
    mean = rgb.new_tensor((.485, .456, .406))
    std = rgb.new_tensor((.229, .224, .225))
    rgb_norm = (rgb - mean) / std
    rgb_norm = torch.where(rgb_valid[..., None], rgb_norm, torch.zeros_like(rgb_norm))
    yy, xx = torch.meshgrid(torch.arange(128, device=device, dtype=torch.float64),
                            torch.arange(128, device=device, dtype=torch.float64), indexing="ij")
    uv_source = torch.stack((left + (xx + .5) / scale, top + (yy + .5) / scale), -1)
    x0, y0 = xmin - 3, ymin - 3
    roi_w, roi_h = width + 6, height + 6
    pad_w, pad_h = 32 * math.ceil(roi_w / 32), 32 * math.ceil(roi_h / 32)
    ry, rx = torch.meshgrid(torch.arange(pad_h, device=device), torch.arange(pad_w, device=device), indexing="ij")
    roi_uv = torch.stack((rx + x0, ry + y0), -1).to(torch.float64)
    # Channel formation is pointwise. In the CPU worker gather exactly the
    # same nearest source pixels first, retaining original full-frame bounds.
    if sample_image_channels:
        cv, inside, _, idx = _nearest_map(valid, uv_source)
        cy, cx = idx[..., 1].clamp(0, h-1), idx[..., 0].clamp(0, w-1)
        crgb, cxyz, cnormals = rgb_norm[cy, cx], xyz[cy, cx], normals[cy, cx]
        tv, texture_inside, _, idx = _nearest_map(valid, roi_uv)
        ty, tx = idx[..., 1].clamp(0, h-1), idx[..., 0].clamp(0, w-1)
        trgb, tz = rgb_norm[ty, tx], xyz[..., 2][ty, tx]
    else:
        cv, crgb, cxyz, cnormals = valid, rgb_norm, xyz, normals
        tv, trgb, tz = valid, rgb_norm, xyz[..., 2]
    channels = torch.cat((crgb, ((cxyz - centroid) / .1).to(feature_dtype),
                          cnormals.to(feature_dtype), gravity.to(feature_dtype).expand(*cv.shape, 3),
                          cv[..., None].to(feature_dtype)), dim=-1)
    context = torch.where(cv[..., None], channels, torch.zeros_like(channels))
    z_med = torch.quantile(points[:, 2], .5) if len(points) else xyz.new_zeros(())
    native = torch.cat((trgb, ((tz - z_med) / .1)[..., None].to(feature_dtype),
                        tv[..., None].to(feature_dtype)), -1)
    texture = torch.where(tv[..., None], native, torch.zeros_like(native))
    if not sample_image_channels:
        context, inside, _, _ = _nearest_map(context, uv_source)
        texture, texture_inside, _, _ = _nearest_map(texture, roi_uv)
    context = torch.where(inside[..., None], context, torch.zeros_like(context)).permute(2, 0, 1)[None]
    roi_valid = texture_inside & (rx < roi_w) & (ry < roi_h)
    texture = torch.where(roi_valid[..., None], texture, torch.zeros_like(texture)).permute(2, 0, 1)[None]
    affine = xyz.new_tensor(((scale, 0., -scale * left - .5),
                             (0., scale, -scale * top - .5), (0., 0., 1.)))
    return {"context": context.contiguous(), "texture": texture.contiguous(),
            "xyz_map": xyz, "mask": valid, "normal_map": normals, "normal_valid": nv,
            "intrinsics": intrinsics, "gravity": gravity, "centroid": centroid, "extent": extent,
            "support_points": points, "context_affine": affine, "context_affine_inverse": torch.linalg.inv(affine),
            "intrinsics_context": affine @ intrinsics,
            "texture_origin_uv": xyz.new_tensor((x0, y0)), "texture_roi_hw": (roi_h, roi_w),
            "texture_padded_hw": (pad_h, pad_w), "bbox_xyxy": (xmin, ymin, xmax + 1, ymax + 1),
            "support_count": support_count, "support_sufficient": support_sufficient,
            "instance_valid": support_sufficient,
            "sample_uid": str(observation.get("sample_uid", observation.get("instance_uid", "instance"))),
            "observation": observation}


def prepare_source_phase_cpu(observation):
    """Prepare immutable pose identities in a loader worker, without a step.

    Sampling can prefetch beyond the current optimizer update.  Consequently
    this cache contains pose bytes only: the consuming model supplies the
    actual update/seed later.  It is not a file checksum or content validator.
    """
    points = np.asarray(observation["query_points"], dtype=np.float64)
    normals = np.asarray(observation["query_normals"], dtype=np.float64)
    if points.ndim != 2 or points.shape[-1] != 3 or normals.shape != points.shape:
        raise ValueError("Canonical query points/normals must have matching Qx3 shapes")
    poses = np.ascontiguousarray(np.concatenate((points, normals), axis=-1))
    result = dict(observation)
    result["_route_b_phase_poses"] = {
        "points": points.copy(), "normals": normals.copy(),
        "pose_bytes": tuple(pose.tobytes() for pose in poses),
    }
    return result


def prepare_source_inputs_cpu(observation, *, min_support_points=100, sample_image_channels=False,
                              static_inputs=None):
    """Prepare a single immutable loader ticket using only CPU/NumPy outputs.

    The original arrays remain read-only for this ticket's lifetime. A copied
    observation replacing any image/calibration array invalidates the cache.
    NumPy payloads avoid CUDA or shared torch-storage lifetime dependencies in
    spawned workers. No update or model weights are captured here.
    """
    result = prepare_source_phase_cpu(observation)
    if sample_image_channels:
        if static_inputs is None:
            static_inputs = prepare_source_static_inputs(result, min_support_points=min_support_points)
        prepared = combine_source_images(result, static_inputs, min_support_points=min_support_points)
    else:
        if static_inputs is not None:
            raise ValueError("Static route-B inputs require sampled image channels")
        prepared = prepare_shared_images(result, device="cpu", feature_dtype=torch.float32,
                                         min_support_points=min_support_points)
        prepared.pop("observation")
        prepared = {key: value.numpy() if torch.is_tensor(value) else value
                    for key, value in prepared.items()}
    result["_route_b_inputs"] = {
        "spec": CPU_INPUTS_SPEC, "min_support_points": min_support_points,
        "sources": {key: result[key] for key in _IMAGE_SOURCE_KEYS if key in result},
        "encoded": prepared,
    }
    return result


# Prepared geometry is exactly zero outside the valid instance. Preserve the
# camera pixel indices (including off-image and invalid pixels) when decoding.
_GEOMETRY_MAPS = ("xyz_map", "mask", "normal_map", "normal_valid")
_TRAINING_ENCODED_KEYS = frozenset((
    "context", "texture", "xyz_map", "mask", "normal_map", "normal_valid", "bilinear_valid_map", "intrinsics", "gravity",
    "centroid", "extent", "support_points", "context_affine", "context_affine_inverse", "intrinsics_context",
    "texture_origin_uv", "texture_roi_hw", "texture_padded_hw", "bbox_xyxy", "support_count",
    "support_sufficient", "instance_valid", "sample_uid"))


def compact_prepared_source_item(item):
    """Private training wire format, after augmentation and source validation."""
    obs = item["observation"]
    prepared = obs["_route_b_inputs"]
    encoded = dict(prepared["encoded"])
    if "geometry_image_hw" in encoded:
        image_hw = encoded.pop("geometry_image_hw")
        y0, x0 = encoded.pop("geometry_top_left")
    else:
        mask = encoded["mask"]
        image_hw = mask.shape
        ys = np.flatnonzero(mask.any(axis=1))
        xs = np.flatnonzero(mask.any(axis=0))
        y0, y1 = (int(ys[0]), int(ys[-1])+1) if len(ys) else (0, 0)
        x0, x1 = (int(xs[0]), int(xs[-1])+1) if len(xs) else (0, 0)
        for key in _GEOMETRY_MAPS:
            encoded[key] = np.ascontiguousarray(encoded[key][y0:y1, x0:x1])
    if "bilinear_valid_map" not in encoded:
        encoded["bilinear_valid_map"] = bilinear_validity_map(
            torch.as_tensor(encoded["xyz_map"]), torch.as_tensor(encoded["mask"])).numpy()
    compact = {key: obs[key] for key in ("query_points", "query_normals",
               "sample_uid", "instance_uid", "_route_b_phase_poses") if key in obs}
    if "support_points" in obs:
        compact["source_support_count"] = len(obs["support_points"])
    elif "source_support_count" in obs:
        compact["source_support_count"] = obs["source_support_count"]
    compact["_route_b_training_inputs"] = {
        "spec": TRAINING_INPUTS_SPEC,
        "min_support_points": prepared["min_support_points"], "encoded": encoded,
        "image_hw": tuple(image_hw), "top_left": (y0, x0)}
    _validate_training_wire(compact["_route_b_training_inputs"])
    return dict(item, observation=compact)


def _validate_training_wire(wire):
    """Validate the private CPU transfer schema without rescanning file data."""
    expected = {"spec", "min_support_points", "encoded", "image_hw", "top_left"}
    if not isinstance(wire, Mapping) or set(wire) != expected or wire.get("spec") != TRAINING_INPUTS_SPEC:
        raise ValueError("Compact route-B wire schema mismatch")
    threshold = wire["min_support_points"]
    if isinstance(threshold, bool) or not isinstance(threshold, Integral) or threshold < 1:
        raise ValueError("Compact route-B support threshold must be positive")
    def integers(value, length, name):
        if (not isinstance(value, (tuple, list)) or len(value) != length
                or any(isinstance(v, bool) or not isinstance(v, Integral) for v in value)):
            raise ValueError("Malformed compact route-B " + name)
        return tuple(int(v) for v in value)
    height, width = integers(wire["image_hw"], 2, "image size")
    y0, x0 = integers(wire["top_left"], 2, "geometry origin")
    if min(height, width) < 1 or not (0 <= y0 <= height and 0 <= x0 <= width):
        raise ValueError("Compact route-B ROI is outside the image")
    encoded = wire["encoded"]
    if not isinstance(encoded, Mapping) or set(encoded) != _TRAINING_ENCODED_KEYS:
        raise ValueError("Compact route-B encoded fields mismatch")
    def array(key, shape, dtype):
        value = encoded[key]
        valid_type = isinstance(value, np.ndarray) or (torch.is_tensor(value) and value.device.type == "cpu")
        wanted_dtype = getattr(torch, dtype) if torch.is_tensor(value) else np.dtype(dtype)
        if (not valid_type or value.dtype != wanted_dtype or len(value.shape) != len(shape)
                or any(want is not None and actual != want for actual, want in zip(value.shape, shape))):
            raise ValueError("Malformed compact route-B array: " + key)
        return value
    mask = array("mask", (None, None), "bool")
    rh, rw = mask.shape
    if (not rh) != (not rw) or y0 + rh > height or x0 + rw > width:
        raise ValueError("Compact route-B ROI shape is outside the image")
    array("xyz_map", (rh, rw, 3), "float64")
    array("normal_map", (rh, rw, 3), "float64")
    array("normal_valid", (rh, rw), "bool")
    array("bilinear_valid_map", (rh, rw), "bool")
    array("context", (1, 13, 128, 128), "float32")
    ph, pw = integers(encoded["texture_padded_hw"], 2, "texture padding")
    th, tw = integers(encoded["texture_roi_hw"], 2, "texture ROI")
    if min(th, tw) < 1 or ph != 32 * math.ceil(th / 32) or pw != 32 * math.ceil(tw / 32):
        raise ValueError("Compact route-B texture padding mismatch")
    array("texture", (1, 5, ph, pw), "float32")
    for key in ("intrinsics", "context_affine", "context_affine_inverse", "intrinsics_context"):
        array(key, (3, 3), "float64")
    for key in ("gravity", "centroid", "extent"):
        array(key, (3,), "float64")
    array("texture_origin_uv", (2,), "float64")
    points = array("support_points", (None, 3), "float64")
    count = encoded["support_count"]
    if isinstance(count, bool) or not isinstance(count, Integral) or count != len(points):
        raise ValueError("Compact route-B support count mismatch")
    for key in ("support_sufficient", "instance_valid"):
        if not isinstance(encoded[key], (bool, np.bool_)) or bool(encoded[key]) != (count >= threshold):
            raise ValueError("Compact route-B support validity mismatch")
    xmin, ymin, xmax, ymax = integers(encoded["bbox_xyxy"], 4, "instance bbox")
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError("Compact route-B instance bbox is outside the image")
    if not isinstance(encoded["sample_uid"], str):
        raise ValueError("Compact route-B sample identity must be text")


def _restore_training_encoded(wire, device):
    _validate_training_wire(wire)
    encoded = {key: torch.as_tensor(value).to(device, non_blocking=True)
               if isinstance(value, np.ndarray) or torch.is_tensor(value) else value
               for key, value in wire["encoded"].items()}
    encoded["geometry_image_hw"] = tuple(wire["image_hw"])
    encoded["geometry_top_left"] = tuple(wire["top_left"])
    return encoded


class InstanceSuction2DFastV3(nn.Module):
    """The route-B architecture with a shared encode / arbitrary-pose score API."""

    route = "b"
    architecture_id = "instance_route_b_v3_support100_chart_policy_v1"
    quality_contract = "gt_INST20_pose_v2"

    def __init__(self, *, query_chunk=64, phase_seed=42, training_phase=True,
                 min_support_points=100, feature_frame=B_FEATURE_FRAME):
        super().__init__()
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        if isinstance(min_support_points, bool) or not isinstance(min_support_points, int) or min_support_points < 1:
            raise ValueError("min_support_points must be a positive integer")
        if feature_frame not in (B_FEATURE_FRAME, B_LEGACY_FEATURE_FRAME):
            raise ValueError("Unsupported route-B feature_frame")
        self.query_chunk = int(query_chunk)
        self.phase_seed = int(phase_seed)
        self.training_phase = bool(training_phase)
        self.min_support_points = min_support_points
        self.feature_frame = feature_frame
        self.successful_update = 0
        self.execution_precision = "fp32"
        self.dense_query_chunk = None
        self.compile_mode = None
        # Ordinary dictionaries do not register compiled wrappers as additional
        # submodules: checkpoint keys and parameter groups remain unchanged.
        self._compiled_dense = {}
        self.encoder = ResNet18GN()
        self.midlevel_query_projections = nn.ModuleList([nn.Conv2d(64, 32, 1, bias=True),
                                                        nn.Conv2d(128, 32, 1, bias=True)])
        self.global_projection = nn.Sequential(nn.Linear(1024, 128), nn.ReLU())
        self.texture_stem = nn.Sequential(nn.Conv2d(5, 16, 3, 1, 1, bias=False), _gn(16), nn.ReLU(),
                                          nn.Conv2d(16, 16, 3, 1, 1, groups=16, bias=False), _gn(16), nn.ReLU(),
                                          nn.Conv2d(16, 16, 1, bias=False), _gn(16), nn.ReLU())
        self.ring_encoders = nn.ModuleList([MaskedRingEncoder() for _ in range(3)])
        self.query_fusion = nn.Sequential(nn.Linear(330, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.output_heads = nn.ModuleDict({name: _mlp((128, 64, 1))
                                           for name in ("q_inst", "seal", "wrench", "ell", "task_logit")})
        nn.init.zeros_(self.output_heads["ell"][-1].weight)
        nn.init.constant_(self.output_heads["ell"][-1].bias, math.log(.1))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def feature_dtype(self):
        return next(self.parameters()).dtype

    def configure_execution(self, *, precision="fp32", compile_mode=None, dense_query_chunk=None):
        """Select explicit execution options without altering stored weights.

        Geometry and image-coordinate sampling stay outside autocast.  Only
        neural blocks opt into BF16.  Compilation is lazy and optional; it is
        not a promise of speedup for any GPU or shape distribution.
        """
        if precision not in ("fp32", "bf16_dense"):
            raise ValueError("Route B execution precision must be fp32 or bf16_dense")
        if compile_mode not in (None, "default", "reduce-overhead", "max-autotune-no-cudagraphs"):
            raise ValueError("Unsupported route B compile_mode")
        if (dense_query_chunk is not None and
                (isinstance(dense_query_chunk, bool) or not isinstance(dense_query_chunk, Integral)
                 or dense_query_chunk < 1)):
            raise ValueError("dense_query_chunk must be a positive integer or None")
        self.execution_precision = precision
        self.dense_query_chunk = int(dense_query_chunk) if dense_query_chunk is not None else None
        if compile_mode != self.compile_mode:
            self._compiled_dense.clear()
            if compile_mode is not None:
                if not hasattr(torch, "compile"):
                    raise RuntimeError("This PyTorch build does not support torch.compile")
                modules = {"encoder": self.encoder, "query_fusion": self.query_fusion,
                           **{f"ring_{i}": module for i, module in enumerate(self.ring_encoders)},
                           **{f"head_{name}": module for name, module in self.output_heads.items()}}
                self._compiled_dense.update({name: torch.compile(module, mode=compile_mode)
                                             for name, module in modules.items()})
            self.compile_mode = compile_mode
        return self

    def _network_autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.training and self.execution_precision == "bf16_dense")

    def _network_call(self, name, module, *args):
        return self._compiled_dense.get(name, module)(*args)

    def architecture_metadata(self):
        """Inspectable feature and geometry semantics, without execution sizes."""
        return {
            "architecture_id": self.architecture_id, "route": self.route,
            "quality_contract": self.quality_contract,
            "encoder": "ResNet18GN", "input_channels": 13,
            "texture_input_channels": 5, "texture_feature_channels": 16,
            "global_feature_channels": 128, "query_feature_channels": 64,
            "ring_feature_channels": 96, "support_summary_channels": 18,
            "raw_summary_channels": 12, "query_descriptor_channels": 12,
            "fusion_channels": 330, "ring_token_channels": 25,
            "ring_encoder_input_channels": 75,
            "minimum_support_points": self.min_support_points,
            "support_count_semantics": "finite positive-depth instance pixels; independent of normal validity",
            "geometry_contract": GEOMETRY_CONTRACT, "geometry_dtype": "float64",
            "feature_frame": self.feature_frame,
            "feature_frame_switch": ("nz > 0.5" if self.feature_frame == B_FEATURE_FRAME
                                     else "1 - nz <= 1e-4"),
            "feature_frame_seam_continuous": False,
            "feature_frame_seam_semantics": "in-plane chart transition can rotate the finite angular sample grid",
            "radii_m": list(RADII_M), "angles": ANGLES,
            "unique_projected_slots": 1 + (len(RADII_M) - 1) * ANGLES,
            "broadcast_projected_slots": len(RADII_M) * ANGLES,
            "encoder_normalization": "GroupNorm(8)",
            "texture_normalization": "GroupNorm(8)",
            "ring_normalization": "LayerNorm(16) per angle across channels; masked before and after each layer",
            "center_summary_semantics": {
                "zero_radius_is_center_support_not_circumference_coverage": True,
                "center_broadcast_counts_as_one_radius_weight": True,
                "center_core_raw_coverage_weight": 1 / 3,
                "unique_source_pixels_per_radius_deduplicates_center": True,
            },
            "scale_semantics": "ell=log_laplace_scale; log_variance=log(2)+2*ell",
        }

    def set_training_context(self, update, seed=None):
        self.successful_update = int(update)
        if seed is not None:
            self.phase_seed = int(seed)

    def parameter_groups(self):
        decoder = list(self.midlevel_query_projections.parameters())
        decoder += list(self.global_projection.parameters()) + list(self.texture_stem.parameters())
        return {"encoder": list(self.encoder.parameters()), "decoder": decoder,
                "contact": list(self.ring_encoders.parameters()) + list(self.query_fusion.parameters()),
                "heads": list(self.output_heads.parameters())}

    def encode_instance(self, observation):
        return self.encode_instances([observation])[0]

    def _prepare_encoded(self, observation):
        wire = observation.get("_route_b_training_inputs")
        if wire is not None:
            if (not isinstance(wire, Mapping) or self.feature_dtype != torch.float32
                    or wire.get("min_support_points") != self.min_support_points):
                raise ValueError("Compact training inputs do not match the model preparation profile")
            prepared = _restore_training_encoded(wire, self.device)
            prepared["observation"] = observation
            return prepared
        cache = observation.get("_route_b_inputs")
        source_keys = {key for key in _IMAGE_SOURCE_KEYS if key in observation}
        if (cache is not None and self.feature_dtype == torch.float32
                and cache.get("spec") == CPU_INPUTS_SPEC
                and cache.get("min_support_points") == self.min_support_points
                and set(cache.get("sources", {})) == source_keys
                and all(cache["sources"][key] is observation[key] for key in source_keys)):
            prepared = {key: torch.as_tensor(value).to(self.device, non_blocking=True)
                        if isinstance(value, np.ndarray) or torch.is_tensor(value) else value
                        for key, value in cache["encoded"].items()}
            prepared["observation"] = observation
            prepared["sample_uid"] = str(observation.get("sample_uid", observation.get("instance_uid", "instance")))
            return prepared
        return prepare_shared_images(observation, device=self.device, feature_dtype=self.feature_dtype,
                                     min_support_points=self.min_support_points)

    def encode_instances(self, observations):
        """Batch independent context encodings and exact-size texture ROIs."""
        if not observations:
            return []
        with torch.autocast(device_type=self.device.type, enabled=False):
            prepared = [self._prepare_encoded(obs) for obs in observations]
        context = torch.cat([item["context"] for item in prepared])
        with self._network_autocast():
            l1, l2, l4 = self._network_call("encoder", self.encoder, context)
            mask = F.adaptive_max_pool2d(context[:, -1:], l4.shape[-2:])[:, 0].flatten(1) > 0
            global_features = self.global_projection(masked_mean_max(l4.flatten(2), mask))
            levels = [module(x) for module, x in zip(self.midlevel_query_projections, (l1, l2))]
        groups = {}
        # Split each learned batch once. Independent slices give every instance
        # a SliceBackward that allocates a gradient for the entire source batch.
        query_features = zip(*(level.split(1) for level in levels))
        for i, (item, g, query) in enumerate(zip(prepared, global_features.split(1), query_features)):
            item['global_feature'] = g if item['instance_valid'] else g * 0
            item['query_features'] = query
            groups.setdefault(item['texture'].shape[-2:], []).append(i)
        # Padding dissimilar ROIs to a common size would change GroupNorm statistics.
        for indices in groups.values():
            with self._network_autocast():
                features = self.texture_stem(torch.cat([prepared[i]['texture'] for i in indices]))
            for index, feature in zip(indices, features.split(1)):
                prepared[index]['texture_feature'] = feature
        return prepared

    def forward_instances(self, observations):
        """One context batch, exact ROI buckets, then cross-instance query MLPs.

        ``query_chunk`` bounds each instance's geometric construction;
        ``dense_query_chunk`` bounds total neural queries across instances.
        Keeping these separate avoids treating 4096 as a per-instance geometry
        budget when the actual source query count is only 128.
        """
        encoded = self.encode_instances(observations)
        if not encoded:
            return []
        prepared, owners = [], []
        results = [None] * len(observations)
        for index, (item, obs) in enumerate(zip(encoded, observations)):
            points = _tensor(obs["query_points"], self.device, torch.float64)
            normals = _tensor(obs["query_normals"], self.device, torch.float64)
            if points.ndim != 2 or points.shape[-1] != 3 or normals.shape != points.shape:
                raise ValueError("p and n must have matching Qx3 shapes")
            if not len(points):
                results[index] = self._empty_scores(points, normals)
                continue
            phase = self._canonical_phase(obs)
            with torch.autocast(device_type=self.device.type, enabled=False):
                for start in range(0, len(points), self.query_chunk):
                    stop = start + self.query_chunk
                    prepared.append(self._prepare_score_chunk(item, points[start:stop], normals[start:stop],
                                                              phase[start:stop]))
                    owners.append(index)
        if not prepared:
            return results
        outputs = self._score_prepared_many(prepared)
        for index in range(len(results)):
            parts = [part for owner, part in zip(owners, outputs) if owner == index]
            if parts:
                results[index] = self._concatenate_scores(parts)
        return results

    def _canonical_phase(self, observation):
        """Generate legacy phase from CPU source poses at the consumed step."""
        if not self.training or not self.training_phase:
            return torch.zeros(len(observation["query_points"]), device=self.device, dtype=torch.float64)
        cache = observation.get("_route_b_phase_poses")
        points, normals = observation["query_points"], observation["query_normals"]
        # Tensor callers may reside on CUDA; the general arbitrary-pose API
        # retains its explicit CPU fallback. Dataset/worker inputs are NumPy.
        if torch.is_tensor(points) or torch.is_tensor(normals):
            encoded = {"sample_uid": str(observation.get("sample_uid", observation.get("instance_uid", "instance")))}
            return self._phase(encoded, _tensor(points, self.device, torch.float64),
                               _tensor(normals, self.device, torch.float64))
        if (cache is None or not np.array_equal(cache["points"], points, equal_nan=True)
                or not np.array_equal(cache["normals"], normals, equal_nan=True)):
            cache = prepare_source_phase_cpu(observation)["_route_b_phase_poses"]
        uid = str(observation.get("sample_uid", observation.get("instance_uid", "instance")))
        prefix = f"{self.phase_seed}|{self.successful_update}|{uid}|".encode()
        values = [int.from_bytes(hashlib.blake2b(prefix + pose, digest_size=8).digest(), "little")
                  / (1 << 64) * 2 * math.pi for pose in cache["pose_bytes"]]
        return torch.tensor(values, device=self.device, dtype=torch.float64)

    def _phase(self, encoded, points, normals):
        if not self.training or not self.training_phase:
            return points.new_zeros(len(points))
        values = []
        prefix = f"{self.phase_seed}|{self.successful_update}|{encoded['sample_uid']}|".encode()
        # A stable pose identity also covers historical queries outside the
        # canonical candidate list, independently of chunking and call order.
        poses = torch.cat((points, normals), -1).detach().cpu().contiguous().numpy()
        for pose in poses:
            digest = hashlib.blake2b(prefix + pose.tobytes(), digest_size=8).digest()
            values.append((int.from_bytes(digest, "little") / (1 << 64)) * 2 * math.pi)
        return points.new_tensor(values)

    @staticmethod
    def _project(points, intrinsics):
        homogeneous = points @ intrinsics.T
        okay = torch.isfinite(homogeneous).all(-1) & (points[..., 2] > 0)
        denominator = torch.where(okay, homogeneous[..., 2], torch.ones_like(homogeneous[..., 2]))
        uv = homogeneous[..., :2] / denominator[..., None]
        return torch.where(okay[..., None], uv, torch.zeros_like(uv)), okay

    def _query_support(self, encoded, points, finite):
        if not encoded["instance_valid"] or not len(points):
            return torch.zeros(len(points), device=points.device, dtype=torch.bool)
        # Canonical queries are recognized at their immutable source pixels;
        # only arbitrary off-surface/history queries need a full distance scan.
        uv, projected = self._project(points, encoded["intrinsics"])
        sampled, inside, _, nearest_uv = _nearest_geometry(encoded, "xyz_map", uv)
        mask = _geometry_at_pixels(encoded, "mask", nearest_uv[..., 1], nearest_uv[..., 0])
        exact = finite & projected & inside & mask & ((sampled - points).norm(dim=-1) <= BOUNDARY_EPS_M)
        unresolved = torch.nonzero(finite & ~exact, as_tuple=False).flatten()
        valid = exact.clone()
        if len(unresolved):
            p = points[unresolved]
            minimum = points.new_full((len(p),), torch.inf)
            support = encoded["support_points"]
            for start in range(0, len(support), 8192):
                dist2 = (p[:, None, :] - support[None, start:start + 8192, :]).square().sum(-1)
                minimum = torch.minimum(minimum, dist2.amin(-1))
            valid[unresolved] = minimum <= (.005 + BOUNDARY_EPS_M) ** 2
        return valid

    def _query_semantics(self, encoded, points):
        uv, okay = self._project(points, encoded["intrinsics"])
        a = encoded["context_affine"]
        ctx = uv * a.diag()[:2] + a[:2, 2]
        f1 = sample_feature_at_indices(encoded["query_features"][0], ctx / 4)
        f2 = sample_feature_at_indices(encoded["query_features"][1], ctx / 8)
        return torch.where(okay[:, None], torch.cat((f1, f2), -1), torch.zeros_like(torch.cat((f1, f2), -1)))

    def _read_slots(self, encoded, unique_points):
        uv, projected = self._project(unique_points, encoded["intrinsics"])
        nearest_maps, inside, source_id, nearest_uv = _nearest_geometry_many(encoded, _GEOMETRY_MAPS, uv)
        actual, mask = nearest_maps["xyz_map"], nearest_maps["mask"]
        normal, nv = nearest_maps["normal_map"], nearest_maps["normal_valid"]
        observed = projected & inside & mask
        actual = torch.where(observed[..., None], actual, torch.zeros_like(actual))
        normal = torch.where((observed & nv)[..., None], normal, torch.zeros_like(normal))
        # Bilinear features are allowed only when all four ORIGINAL pixels
        # belong to this instance and span <=10mm in observed depth.
        floor_uv = torch.floor(uv).long()
        h, w = encoded.get("geometry_image_hw", encoded["mask"].shape)
        if "bilinear_valid_map" in encoded:
            # Worker geometry is immutable. Phase changes the queried pixel,
            # not the four-corner depth rule stored at that original pixel.
            fx, fy = floor_uv[..., 0], floor_uv[..., 1]
            bilinear_ok = _geometry_at_pixels(encoded, "bilinear_valid_map", fy, fx)
            bilinear_ok &= (fx >= 0) & (fx < w - 1) & (fy >= 0) & (fy < h - 1)
            bilinear_ok &= projected & observed
        else:
            # Public encoded callers can mutate raw maps between queries.
            # Preserve their live-depth behavior instead of caching a rule.
            bilinear_ok = projected.clone()
            zs = []
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                cx, cy = floor_uv[..., 0] + dx, floor_uv[..., 1] + dy
                valid_corner = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
                valid_corner = valid_corner & _geometry_at_pixels(encoded, "mask", cy, cx)
                bilinear_ok &= valid_corner
                zs.append(_geometry_at_pixels(encoded, "xyz_map", cy, cx)[..., 2])
            depth = torch.stack(zs, -1)
            bilinear_ok &= (depth.amax(-1) - depth.amin(-1)) <= .010 + BOUNDARY_EPS_M
            bilinear_ok &= observed
        tex = encoded["texture_feature"]
        tex_uv = uv - encoded["texture_origin_uv"]
        bilinear = sample_feature_at_indices(tex, tex_uv)
        nearest_roi = nearest_uv - encoded["texture_origin_uv"].long()
        tx = nearest_roi[..., 0].clamp(0, tex.shape[-1] - 1)
        ty = nearest_roi[..., 1].clamp(0, tex.shape[-2] - 1)
        nearest = tex[0, :, ty, tx].movedim(0, -1)
        features = torch.where(bilinear_ok[..., None], bilinear, nearest)
        features = torch.where(observed[..., None], features, torch.zeros_like(features))
        return {"points": actual, "observed": observed, "normal": normal,
                "normal_valid": nv & observed, "texture": features,
                "source_ids": torch.where(observed, source_id, -torch.ones_like(source_id)),
                "bilinear_valid": bilinear_ok, "uv": uv}

    def _contact_inputs(self, encoded, points, normals, phase):
        """Construct pose-conditioned tokens and diagnostics in FP64 geometry."""
        directions = ring_directions(normals, phase, feature_frame=self.feature_frame)
        radial = directions[0]
        radii = points.new_tensor(RADII_M[1:])
        ring_points = points[:, None, None, :] + radii[None, :, None, None] * radial[:, None, :, :]
        unique_points = torch.cat((points[:, None, :], ring_points.flatten(1, 2)), 1)
        unique = self._read_slots(encoded, unique_points)

        def expand(x):
            tail = x.shape[2:]
            centre = x[:, :1].reshape(len(points), 1, 1, *tail).expand(-1, 1, ANGLES, *tail)
            return torch.cat((centre, x[:, 1:].reshape(len(points), 8, ANGLES, *tail)), 1)

        slots = {key: expand(value) for key, value in unique.items()}
        gates = projected_ring_gate(slots["points"], slots["observed"], points, normals, phase,
                                    feature_frame=self.feature_frame, _directions=directions)
        normal_alignment = (slots["normal"] * normals[:, None, None, :]).sum(-1)
        normal_alignment = torch.where(slots["normal_valid"], normal_alignment, torch.zeros_like(normal_alignment))
        gravity = encoded["gravity"]
        gravity_radial = (gates["radial_directions"] * gravity).sum(-1)[:, None, :].expand(-1, 9, -1)
        gravity_normal = (normals * gravity).sum(-1)[:, None, None].expand(-1, 9, ANGLES)
        geometry = torch.stack((gates["height_m"] / .040, gates["radial_error_m"] / .022,
                                gates["tangential_offset_m"] / .022, normal_alignment,
                                slots["normal_valid"].to(points.dtype), slots["observed"].to(points.dtype),
                                gates["in_region"].to(points.dtype), gravity_radial, gravity_normal), -1)
        tokens = torch.cat((slots["texture"], geometry.to(slots["texture"].dtype)), -1)
        contact = gates["contact_valid"]
        tokens = torch.where(contact[..., None], tokens, torch.zeros_like(tokens))
        support, raw = ring_support_summary(contact, slots["observed"], gates["height_m"])
        # Sorting only the fixed 72 source IDs per radius provides an exact
        # independent-pixel count without treating repeated slots as new data.
        ids = slots["source_ids"].sort(-1).values
        fresh = torch.cat((torch.ones_like(ids[..., :1], dtype=torch.bool), ids[..., 1:] != ids[..., :-1]), -1)
        unique_count = (fresh & (ids >= 0)).sum(-1)
        uv = slots["uv"]
        physical_width_pixels = (uv[:, 8] - uv[:, 6]).norm(dim=-1)
        outer_observed = slots["observed"][:, 6] & slots["observed"][:, 8]
        underresolved = ((physical_width_pixels < 2) & outer_observed).sum(-1) / outer_observed.sum(-1).clamp_min(1)
        obs_count = slots["observed"].sum(-1).clamp_min(1)
        tangent_reject = (slots["observed"] & ~gates["tangential_ok"]).sum(-1) / obs_count
        opposite_reject = (slots["observed"] & ~gates["positive_radial"]).sum(-1) / obs_count
        # Radius zero has no angular or half-plane test; do not present its
        # undefined radial direction as a rejected observation diagnostic.
        tangent_reject = torch.cat((torch.zeros_like(tangent_reject[:, :1]), tangent_reject[:, 1:]), -1)
        opposite_reject = torch.cat((torch.zeros_like(opposite_reject[:, :1]), opposite_reject[:, 1:]), -1)
        diagnostics = {"contact_valid": contact, "observed_valid": slots["observed"],
                       "height_m": gates["height_m"], "rho_m": gates["rho_m"],
                       "chart_positive_z": gates["chart_positive_z"],
                       "source_ids": slots["source_ids"], "unique_source_pixels_per_radius": unique_count,
                       "bilinear_fallback_count": (slots["observed"] & ~slots["bilinear_valid"]).sum((1, 2)),
                       "radial_gate_rejection_fraction": (slots["observed"] & ~gates["radial_ok"]).sum(-1) / obs_count,
                       "tangential_gate_rejection_fraction": tangent_reject,
                       "opposite_angle_gate_rejection_fraction": opposite_reject,
                       "native_underresolved_fraction": underresolved, "phase": phase,
                       "support_summary": support, "raw_summary": raw}
        return tokens, contact, support, raw, diagnostics

    def _encode_contact_tokens(self, tokens, contact):
        ring_features = []
        with self._network_autocast():
            regions = zip(self.ring_encoders, tokens.split(3, dim=1), contact.split(3, dim=1))
            for i, (encoder, region_tokens, region_contact) in enumerate(regions):
                region_tokens = region_tokens.permute(0, 1, 3, 2).reshape(len(tokens), 75, ANGLES)
                ring_features.append(self._network_call(f"ring_{i}", encoder, region_tokens,
                                                        region_contact.any(1)))
        return torch.cat(ring_features, -1)

    def contact_features(self, encoded, points, normals, phase):
        """Return learned ring features and auditable raw/contact diagnostics."""
        with torch.autocast(device_type=self.device.type, enabled=False):
            tokens, contact, support, raw, diagnostics = self._contact_inputs(encoded, points, normals, phase)
        return self._encode_contact_tokens(tokens, contact), support, raw, diagnostics

    def _prepare_score_chunk(self, encoded, points, normals, phase):
        finite = torch.isfinite(points).all(-1) & (points[:, 2] > 0) & torch.isfinite(normals).all(-1)
        norm = torch.where(torch.isfinite(normals), normals, torch.zeros_like(normals)).norm(dim=-1)
        finite &= norm > 1e-8
        safe_points = torch.where(finite[:, None], points, points.new_tensor((0., 0., 1.)))
        normal = torch.where(finite[:, None], normals / norm.clamp_min(1e-8)[:, None], normals.new_tensor((0., 0., -1.)))
        valid = finite & self._query_support(encoded, safe_points, finite)
        # Unsupported finite inputs can be arbitrarily far away.  Clear them
        # before the float32 MLP, rather than masking a possible Inf afterwards.
        feature_points = torch.where(valid[:, None], safe_points, points.new_tensor((0., 0., 1.)))
        feature_normals = torch.where(valid[:, None], normal, normals.new_tensor((0., 0., -1.)))
        fq = self._query_semantics(encoded, feature_points)
        tokens, contact, support, raw, diagnostics = self._contact_inputs(encoded, feature_points,
                                                                        feature_normals, phase)
        descriptor = torch.cat(((feature_points - encoded["centroid"]) / .1, feature_normals,
                                encoded["gravity"].expand(len(points), -1), encoded["extent"].expand(len(points), -1) / .1), -1)
        return {"tokens": tokens, "contact": contact, "fq": fq, "support": support, "raw": raw,
                "descriptor": descriptor.to(fq.dtype),
                "global_feature": encoded["global_feature"].expand(len(points), -1),
                "valid": valid, "points_cam_m": points,
                "normals_out_cam": torch.where(finite[:, None], normal, torch.zeros_like(normal)),
                "diagnostics": {**diagnostics, "fq": fq,
                    "support_count": torch.full_like(valid, encoded["support_count"], dtype=torch.long),
                    "support_sufficient": torch.full_like(valid, encoded["support_sufficient"]),
                    "query_input_valid": finite, "query_support_valid": valid}}

    def _score_prepared_many(self, prepared):
        """Merge query rows only after each instance's geometric isolation."""
        if not prepared:
            return []
        lengths = [len(item["valid"]) for item in prepared]
        keys = ("tokens", "contact", "global_feature", "fq", "support", "raw", "descriptor")
        batch = {key: torch.cat([item[key] for item in prepared]) for key in keys}
        chunk = self.dense_query_chunk or self.query_chunk
        predictions, fusions = [], []
        # One split node assembles each source gradient once, including a short
        # final chunk, instead of zero-filling the whole query batch per chunk.
        chunks = zip(*(batch[key].split(chunk) for key in keys))
        for values in chunks:
            part = dict(zip(keys, values))
            ring = self._encode_contact_tokens(part["tokens"], part["contact"])
            fq = part["fq"]
            fusion = torch.cat((part["global_feature"].to(fq.dtype), fq, ring.to(fq.dtype),
                                part["support"].to(fq.dtype), part["raw"].to(fq.dtype),
                                part["descriptor"]), -1)
            with self._network_autocast():
                latent = self._network_call("query_fusion", self.query_fusion, fusion)
                heads = {name: self._network_call(f"head_{name}", head, latent).squeeze(-1)
                         for name, head in self.output_heads.items()}
            heads = {name: value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
                     for name, value in heads.items()}
            # Loss scale arithmetic and output probabilities are deliberately
            # FP32. Only the learned convolutions/linear blocks are autocast.
            with torch.autocast(device_type=self.device.type, enabled=False):
                for name in ("q_inst", "seal", "wrench"):
                    heads[name] = heads[name].sigmoid()
                heads["ell"] = heads["ell"].clamp(-6., 1.)
            predictions.append(heads)
            fusions.append(fusion)
        merged = {name: torch.cat([part[name] for part in predictions]) for name in predictions[0]}
        fusion = torch.cat(fusions)
        instance_predictions = {name: value.split(lengths) for name, value in merged.items()}
        results = []
        for index, (item, instance_fusion) in enumerate(zip(prepared, fusion.split(lengths))):
            valid = item["valid"]
            result = {name: torch.where(valid, values[index], torch.zeros_like(values[index]))
                      for name, values in instance_predictions.items()}
            result.update(log_scale=result["ell"], log_variance=math.log(2.) + 2 * result["ell"],
                          valid=valid, query_valid=valid, points_cam_m=item["points_cam_m"],
                          normals_out_cam=item["normals_out_cam"],
                          diagnostics={**item["diagnostics"], "fusion": instance_fusion})
            results.append(result)
        return results

    def _score_chunk(self, encoded, points, normals, phase):
        with torch.autocast(device_type=self.device.type, enabled=False):
            prepared = self._prepare_score_chunk(encoded, points, normals, phase)
        return self._score_prepared_many([prepared])[0]

    def _empty_scores(self, points, normals):
        dtype = torch.float32 if self.feature_dtype in (torch.float16, torch.bfloat16) else self.feature_dtype
        empty = torch.empty((0,), device=self.device, dtype=dtype)
        valid = torch.empty((0,), device=self.device, dtype=torch.bool)
        return {**{key: empty for key in ("q_inst", "seal", "wrench", "ell", "log_scale", "log_variance", "task_logit")},
                "valid": valid, "query_valid": valid, "points_cam_m": points,
                "normals_out_cam": normals, "diagnostics": {}}

    @staticmethod
    def _concatenate_scores(parts):
        result = {key: torch.cat([part[key] for part in parts], 0)
                  for key in parts[0] if key != "diagnostics"}
        result["diagnostics"] = {key: torch.cat([part["diagnostics"][key] for part in parts], 0)
                                 for key in parts[0]["diagnostics"]}
        return result

    def score_at_pose(self, encoded, p, n, query_batch_index=None, *, phase=None, query_chunk=None):
        points = _tensor(p, self.device, torch.float64)
        normals = _tensor(n, self.device, torch.float64)
        if points.ndim != 2 or points.shape[-1] != 3 or normals.shape != points.shape:
            raise ValueError("p and n must have matching Qx3 shapes")
        if query_batch_index is not None:
            batch = _tensor(query_batch_index, self.device, torch.long)
            if batch.shape != (len(points),) or bool((batch != 0).any()):
                raise ValueError("An encoded item is one instance; query_batch_index must be all zero")
        if not len(points):
            return self._empty_scores(points, normals)
        if phase is None:
            phase = self._phase(encoded, points, normals)
        else:
            phase = _tensor(phase, self.device, torch.float64)
            if phase.ndim == 0:
                phase = phase.expand(len(points))
            if phase.shape != (len(points),) or not bool(torch.isfinite(phase).all()):
                raise ValueError("phase must be finite scalar or Q-vector")
        chunk = self.query_chunk if query_chunk is None else int(query_chunk)
        if chunk <= 0:
            raise ValueError("query_chunk must be positive")
        with torch.autocast(device_type=self.device.type, enabled=False):
            prepared = [self._prepare_score_chunk(encoded, points[i:i + chunk], normals[i:i + chunk], phase[i:i + chunk])
                        for i in range(0, len(points), chunk)]
        return self._concatenate_scores(self._score_prepared_many(prepared))

    def forward(self, observation, points=None, normals=None):
        canonical = points is None and normals is None
        if points is None:
            points = observation["query_points"]
        if normals is None:
            normals = observation["query_normals"]
        phase = self._canonical_phase(observation) if canonical else None
        return self.score_at_pose(self.encode_instance(observation), points, normals, phase=phase)


InstanceRouteB = InstanceSuction2DFastV3
