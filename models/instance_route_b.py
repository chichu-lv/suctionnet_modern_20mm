"""Route B v3: shared image features and pose-conditioned projected rings.

The network uses only PyTorch operators.  Geometric decisions use float64;
learned tensors use the module dtype (float32 by default).  An encoded item is
one instance, so gradient accumulation does not change the native-ROI padding
contract.  No GT object geometry is read by this module.
"""
from __future__ import annotations

import hashlib
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


RADII_M = (0., .005, .009, .010, .014, .017, .018, .020, .022)
ANGLES = 72
BOUNDARY_EPS_M = 1e-12
GEOMETRY_CONTRACT = "b_v3_supported_projected_rings_float64"


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


def fixed_z_frames(normals):
    """Two-chart Rodrigues basis; input normals must already be finite/unit."""
    n = normals.to(torch.float64)
    positive = (1. - n[..., 2]) <= 1e-4
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


def ring_directions(normals, phase):
    e1, e2, chart = fixed_z_frames(normals)
    theta = 2 * math.pi * (torch.arange(ANGLES, device=normals.device, dtype=torch.float64) + .5) / ANGLES
    theta = theta[None, :] + phase[:, None]
    radial = e1[:, None, :] * theta.cos()[..., None] + e2[:, None, :] * theta.sin()[..., None]
    tangent = -e1[:, None, :] * theta.sin()[..., None] + e2[:, None, :] * theta.cos()[..., None]
    return radial, tangent, chart


def projected_ring_gate(observed_points, observed_valid, points, normals, phase):
    """Float64 v3 contact gates, with raw observations retained independently."""
    radial, angular, chart = ring_directions(normals, phase)
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


def prepare_shared_images(observation: Mapping, *, device="cpu", feature_dtype=torch.float32):
    """Build context13 and native texture5 without changing the raw geometry.

    RGB photometric augmentation, when requested, must already have been
    applied once to observation['rgb']; both branches then use that same RGB.
    """
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
    channels = torch.cat((rgb_norm, ((xyz - centroid) / .1).to(feature_dtype),
                          normals.to(feature_dtype), gravity.to(feature_dtype).expand(h, w, 3),
                          valid[..., None].to(feature_dtype)), dim=-1)
    channels = torch.where(valid[..., None], channels, torch.zeros_like(channels))
    yy, xx = torch.meshgrid(torch.arange(128, device=device, dtype=torch.float64),
                            torch.arange(128, device=device, dtype=torch.float64), indexing="ij")
    uv_source = torch.stack((left + (xx + .5) / scale, top + (yy + .5) / scale), -1)
    context, inside, _, _ = _nearest_map(channels, uv_source)
    context = torch.where(inside[..., None], context, torch.zeros_like(context)).permute(2, 0, 1)[None]
    z_med = torch.quantile(points[:, 2], .5) if len(points) else xyz.new_zeros(())
    native = torch.cat((rgb_norm, ((xyz[..., 2] - z_med) / .1)[..., None].to(feature_dtype),
                        valid[..., None].to(feature_dtype)), -1)
    native = torch.where(valid[..., None], native, torch.zeros_like(native))
    x0, y0 = xmin - 3, ymin - 3
    roi_w, roi_h = width + 6, height + 6
    pad_w, pad_h = 32 * math.ceil(roi_w / 32), 32 * math.ceil(roi_h / 32)
    ry, rx = torch.meshgrid(torch.arange(pad_h, device=device), torch.arange(pad_w, device=device), indexing="ij")
    roi_uv = torch.stack((rx + x0, ry + y0), -1).to(torch.float64)
    texture, texture_inside, _, _ = _nearest_map(native, roi_uv)
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
            "instance_valid": bool(valid.any()),
            "sample_uid": str(observation.get("sample_uid", observation.get("instance_uid", "instance"))),
            "observation": observation}


class InstanceSuction2DFastV3(nn.Module):
    """The route-B architecture with a shared encode / arbitrary-pose score API."""

    def __init__(self, *, query_chunk=64, phase_seed=42, training_phase=True):
        super().__init__()
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.query_chunk = int(query_chunk)
        self.phase_seed = int(phase_seed)
        self.training_phase = bool(training_phase)
        self.successful_update = 0
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
        encoded = prepare_shared_images(observation, device=self.device, feature_dtype=self.feature_dtype)
        l1, l2, l4 = self.encoder(encoded["context"])
        mask4 = F.adaptive_max_pool2d(encoded["context"][:, -1:], l4.shape[-2:]) > 0
        pooled = masked_mean_max(l4.flatten(2), mask4[:, 0].flatten(1))
        g = self.global_projection(pooled)
        encoded["global_feature"] = g if encoded["instance_valid"] else g * 0
        encoded["query_features"] = (self.midlevel_query_projections[0](l1),
                                     self.midlevel_query_projections[1](l2))
        encoded["texture_feature"] = self.texture_stem(encoded["texture"])
        return encoded

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
        sampled, inside, _, _ = _nearest_map(encoded["xyz_map"], uv)
        mask, _, _, _ = _nearest_map(encoded["mask"], uv)
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
        actual, inside, source_id, nearest_uv = _nearest_map(encoded["xyz_map"], uv)
        mask, _, _, _ = _nearest_map(encoded["mask"], uv)
        normal, _, _, _ = _nearest_map(encoded["normal_map"], uv)
        nv, _, _, _ = _nearest_map(encoded["normal_valid"], uv)
        observed = projected & inside & mask
        actual = torch.where(observed[..., None], actual, torch.zeros_like(actual))
        normal = torch.where((observed & nv)[..., None], normal, torch.zeros_like(normal))
        # Bilinear features are allowed only when all four ORIGINAL pixels
        # belong to this instance and span <=10mm in observed depth.
        floor_uv = torch.floor(uv).long()
        bilinear_ok = projected.clone()
        zs = []
        h, w = encoded["mask"].shape
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            cx, cy = floor_uv[..., 0] + dx, floor_uv[..., 1] + dy
            valid_corner = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            cy, cx = cy.clamp(0, h - 1), cx.clamp(0, w - 1)
            valid_corner = valid_corner & encoded["mask"][cy, cx]
            bilinear_ok &= valid_corner
            zs.append(encoded["xyz_map"][cy, cx, 2])
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

    def contact_features(self, encoded, points, normals, phase):
        """Return learned ring features and auditable raw/contact diagnostics."""
        radial, _, _ = ring_directions(normals, phase)
        radii = points.new_tensor(RADII_M[1:])
        ring_points = points[:, None, None, :] + radii[None, :, None, None] * radial[:, None, :, :]
        unique_points = torch.cat((points[:, None, :], ring_points.flatten(1, 2)), 1)
        unique = self._read_slots(encoded, unique_points)

        def expand(x):
            tail = x.shape[2:]
            centre = x[:, :1].reshape(len(points), 1, 1, *tail).expand(-1, 1, ANGLES, *tail)
            return torch.cat((centre, x[:, 1:].reshape(len(points), 8, ANGLES, *tail)), 1)

        slots = {key: expand(value) for key, value in unique.items()}
        gates = projected_ring_gate(slots["points"], slots["observed"], points, normals, phase)
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
        ring_features = []
        for i, encoder in enumerate(self.ring_encoders):
            zone = slice(3 * i, 3 * i + 3)
            region_tokens = tokens[:, zone].permute(0, 1, 3, 2).reshape(len(points), 75, ANGLES)
            ring_features.append(encoder(region_tokens, contact[:, zone].any(1)))
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
        return torch.cat(ring_features, -1), support, raw, diagnostics

    def _score_chunk(self, encoded, points, normals, phase):
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
        ring, support, raw, diagnostics = self.contact_features(encoded, feature_points, feature_normals, phase)
        descriptor = torch.cat(((feature_points - encoded["centroid"]) / .1, feature_normals,
                                encoded["gravity"].expand(len(points), -1), encoded["extent"].expand(len(points), -1) / .1), -1)
        fusion = torch.cat((encoded["global_feature"].expand(len(points), -1), fq, ring,
                            support.to(fq.dtype), raw.to(fq.dtype), descriptor.to(fq.dtype)), -1)
        latent = self.query_fusion(fusion)
        result = {name: head(latent).squeeze(-1) for name, head in self.output_heads.items()}
        for name in ("q_inst", "seal", "wrench"):
            result[name] = result[name].sigmoid()
        result["ell"] = result["ell"].clamp(-6., 1.)
        result = {key: torch.where(valid, value, torch.zeros_like(value)) for key, value in result.items()}
        result.update(log_scale=result["ell"], log_variance=math.log(2.) + 2 * result["ell"],
                      valid=valid, query_valid=valid, points_cam_m=points,
                      normals_out_cam=torch.where(finite[:, None], normal, torch.zeros_like(normal)),
                      diagnostics={**diagnostics, "fq": fq, "fusion": fusion,
                                   "query_input_valid": finite, "query_support_valid": valid})
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
            empty = next(self.parameters()).new_empty((0,))
            valid = torch.empty((0,), device=self.device, dtype=torch.bool)
            return {**{key: empty for key in ("q_inst", "seal", "wrench", "ell", "log_scale", "log_variance", "task_logit")},
                    "valid": valid, "query_valid": valid, "points_cam_m": points,
                    "normals_out_cam": normals, "diagnostics": {}}
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
            parts = [self._score_chunk(encoded, points[i:i + chunk], normals[i:i + chunk], phase[i:i + chunk])
                     for i in range(0, len(points), chunk)]
        result = {key: torch.cat([part[key] for part in parts], 0) for key in parts[0] if key != "diagnostics"}
        result["diagnostics"] = {key: torch.cat([part["diagnostics"][key] for part in parts], 0)
                                 for key in parts[0]["diagnostics"]}
        return result

    def forward(self, observation, points=None, normals=None):
        if points is None:
            points = observation["query_points"]
        if normals is None:
            normals = observation["query_normals"]
        return self.score_at_pose(self.encode_instance(observation), points, normals)


InstanceRouteB = InstanceSuction2DFastV3
