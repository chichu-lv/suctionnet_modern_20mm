"""Complete physical-region queries followed by bounded differentiable encoding.

Geometry selection runs on CPU and never reads supervision. Neural feature
selection remains differentiable and uses every raw point selected by the
physical query, even when that point shares a network voxel with another object.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from .candidate_sampler import fps_indices
from .heads import chunk_apply, mlp

ZONE_NAMES = ('core', 'lip', 'cup', 'mount', 'approach')
ZONE_LIMITS = {'core': 32, 'lip': 64, 'cup': 32, 'mount': 32, 'approach': 32}


@dataclass
class SpatialIndex:
    xyz: np.ndarray
    instance: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray
    frame_rows: list
    trees: list

    @classmethod
    def from_inputs(cls, inputs):
        xyz = inputs['xyz_cam_m'].detach().float().cpu().numpy().astype(np.float64)
        batch = inputs['batch_index'].detach().cpu().numpy()
        rows = [np.flatnonzero(batch == b) for b in range(len(inputs['g_cam']))]
        return cls(xyz, inputs['instance_id_input'].detach().cpu().numpy(),
                   inputs['normal_obs'].detach().float().cpu().numpy(),
                   inputs['normal_obs_valid'].detach().cpu().numpy(), rows,
                   [cKDTree(xyz[r]) if len(r) else None for r in rows])

    def ball(self, frame, center, radius):
        tree = self.trees[frame]
        if tree is None:
            return np.empty(0, np.int64)
        # Expand only the covering query by rounding epsilon. Exact zone filters
        # below still decide membership using the specified metric boundaries.
        local = tree.query_ball_point(center, radius + 1e-8)
        return np.sort(self.frame_rows[frame][local])


class ContactGrouping(nn.Module):
    def __init__(self, frozen_config, *, chunk_size=32, checkpointed=True):
        super().__init__()
        if not np.isclose(frozen_config.get('radius_m', np.nan), .020, rtol=0, atol=1e-9):
            raise ValueError('Contact grouping requires frozen 20mm contact radius')
        segments = np.asarray(frozen_config.get('envelope_segments_m', []), np.float64)
        if segments.shape != (3, 3) or not np.isfinite(segments).all():
            raise ValueError('Frozen configuration must contain three [z_min,z_max,radius] segments')
        if np.any(segments[:, 0] >= segments[:, 1]) or np.any(segments[:, 2] <= 0):
            raise ValueError('Invalid frozen collision envelope geometry')
        if not np.allclose(segments[1:, 0], segments[:-1, 1], rtol=0, atol=1e-9):
            raise ValueError('Frozen envelope segments must have adjacent axial boundaries')
        self.segments = segments
        self.self_ignore = float(frozen_config['self_height_threshold_m'])
        if not np.isfinite(self.self_ignore):
            raise ValueError('Invalid frozen own-object clearance')
        self.limits = dict(ZONE_LIMITS)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError('pose_chunk_size must be positive')
        self.checkpointed = checkpointed
        self.local_projection = mlp((256, 64))
        self.point_mlp = mlp((71, 64, 64))
        self.zone_projections = nn.ModuleDict({name: mlp((130, 64)) for name in ZONE_NAMES})
        self.last_timings = {}

    @torch.no_grad()
    def query(self, index, points, normals, instances, batch, valid, *, random_roll=False):
        """Return raw indices/masks, seven metric descriptors, pre-limit counts."""
        points = points.detach().float().cpu().numpy().astype(np.float64)
        normals = normals.detach().float().cpu().numpy().astype(np.float64)
        instances = instances.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        valid = valid.detach().cpu().numpy()
        rolls = (torch.rand(len(points)).numpy() * (2 * math.pi) if random_roll else
                 np.zeros(len(points)))
        result = {
            name: {'indices': np.zeros((len(points), limit), np.int64),
                   'mask': np.zeros((len(points), limit), bool),
                   'geometry': np.zeros((len(points), limit, 7), np.float32),
                   'counts': np.zeros(len(points), np.int64)}
            for name, limit in self.limits.items()}
        sectors = np.zeros((len(points), 8), np.int64)
        for q in np.flatnonzero(valid):
            n = normals[q] / np.linalg.norm(normals[q])
            e = np.eye(3)[np.argmin(np.abs(n))]
            u = np.cross(e, n)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            c, s = np.cos(rolls[q]), np.sin(rolls[q])
            u, v = c * u + s * v, -s * u + c * v
            contact_rows = index.ball(batch[q], points[q], math.hypot(.022, .008))
            for zone, name in enumerate(ZONE_NAMES):
                if zone < 2:
                    rows, r, h, z_mid = contact_rows, .022, .008, 0.
                else:
                    z_min, z_max, r = self.segments[zone - 2]
                    z_mid = (z_min + z_max) / 2
                    h = max(abs(z_min), abs(z_max))
                    rows = index.ball(batch[q], points[q] + z_mid * n,
                                      math.hypot(r, (z_max - z_min) / 2))
                if not len(rows):
                    continue
                d = index.xyz[rows] - points[q]
                x, y, z = d @ u, d @ v, d @ n
                rho = np.hypot(x, y)
                same = index.instance[rows] == instances[q]
                if name == 'core':
                    keep = (rho <= .018) & (np.abs(z) <= .008) & same
                elif name == 'lip':
                    keep = (rho > .018) & (rho <= .022) & (np.abs(z) <= .008) & same
                else:
                    lower = z >= z_min if name == 'cup' else z > z_min
                    keep = (rho <= r) & lower & (z <= z_max) & (~same | (z > self.self_ignore))
                rows = rows[keep]
                x, y, z, rho, same = x[keep], y[keep], z[keep], rho[keep], same[keep]
                data = result[name]
                data['counts'][q] = len(rows)
                if not len(rows):
                    continue
                if name == 'lip':
                    sector = np.floor((np.arctan2(y, x) + math.pi) * (8 / (2 * math.pi))).astype(np.int64) % 8
                    sectors[q] = np.bincount(sector, minlength=8)
                # Start FPS nearest the zone's axial midpoint, ties by raw index.
                distance = x*x + y*y + (z - z_mid)**2
                first = rows[np.argmin(distance)]
                if len(rows) > self.limits[name]:
                    chosen = fps_indices(index.xyz, rows, self.limits[name], first=first)
                    positions = np.searchsorted(rows, chosen)
                else:
                    chosen, positions = rows, np.arange(len(rows))
                obs_valid = index.normal_valid[chosen]
                normal_dot = np.where(obs_valid, index.normals[chosen] @ n, 0.)
                geom = np.column_stack((x[positions]/r, y[positions]/r, z[positions]/h,
                                        rho[positions]/r, normal_dot, same[positions], obs_valid))
                data['indices'][q, :len(chosen)] = chosen
                data['mask'][q, :len(chosen)] = True
                data['geometry'][q, :len(chosen)] = geom
        result['lip_sector_counts'] = sectors
        return result

    def forward(self, encoded, points, normals, instances, batch, valid, *, random_roll=False):
        started = time.perf_counter()
        if 'spatial_index' not in encoded:
            encoded['spatial_index'] = SpatialIndex.from_inputs(encoded)
        built = time.perf_counter()
        query = self.query(encoded['spatial_index'], points, normals.detach(), instances, batch,
                           valid, random_roll=random_roll)
        queried = time.perf_counter()
        device = encoded['F_local'].device
        local = encoded['F_local']
        # Project just the union of selected points; this is exactly the same
        # pointwise projection as projecting the entire raw scene first.
        used = np.unique(np.concatenate([query[name]['indices'][query[name]['mask']]
                                         for name in ZONE_NAMES]))
        used_rows = torch.as_tensor(used, dtype=torch.long, device=device)
        projected = chunk_apply(self.local_projection, local[used_rows],
                                checkpointed=self.checkpointed and self.training)
        # Extra zero feature is the only padding row; no real point is duplicated.
        projected = torch.cat((projected, projected.new_zeros((1, 64))))
        prepared = {}
        for name in ZONE_NAMES:
            q = query[name]
            positions = np.full(q['indices'].shape, len(used), np.int64)
            positions[q['mask']] = np.searchsorted(used, q['indices'][q['mask']])
            prepared[name] = {
                'indices': torch.as_tensor(positions, device=device),
                'mask': torch.as_tensor(q['mask'], device=device),
                'geometry': torch.as_tensor(q['geometry'], device=device),
                'counts': torch.as_tensor(q['counts'], device=device)}
        sectors = torch.as_tensor(query['lip_sector_counts'], device=device)
        transferred = time.perf_counter()
        features = {}
        for name, q in prepared.items():
            def encode_zone(rows, mask, geometry, count, name=name):
                p = torch.cat((projected[rows], geometry.to(projected)), -1)
                p = self.point_mlp(p)
                # Pool and log counts in FP32 even under AMP.
                p = p.float()
                maximum = p.masked_fill(~mask[..., None], -torch.inf).max(1).values
                maximum = torch.where(mask.any(1)[:, None], maximum, torch.zeros_like(maximum))
                mean = (p * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
                stats = torch.stack((count.float().log1p(), (count > 0).float()), -1)
                output = self.zone_projections[name](torch.cat((maximum, mean, stats), -1))
                return output * (count > 0)[:, None]
            features[name] = chunk_apply(encode_zone, q['indices'], q['mask'], q['geometry'], q['counts'],
                                         chunk_size=self.chunk_size,
                                         checkpointed=self.checkpointed and self.training)
        finished = time.perf_counter()
        self.last_timings = {'spatial_index_seconds': built-started,
                             'query_cpu_seconds': queried-built,
                             'feature_projection_and_transfer_seconds': transferred-queried,
                             'feature_encoding_enqueue_seconds': finished-transferred}
        # CUDA feature encoding is asynchronous; the enqueue timing is explicitly
        # named and is not reported as synchronized GPU kernel latency.
        encoded['grouping_timings'] = dict(self.last_timings)
        return {**features, 'lip_sector_log_counts': sectors.float().log1p(),
                'lip_sector_counts': sectors,
                'zone_counts': torch.stack([prepared[n]['counts'] for n in ZONE_NAMES], -1)}
