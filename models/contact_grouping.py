"""Complete physical-region queries followed by bounded differentiable encoding.

Geometry selection runs on CPU and never reads supervision. Neural feature
selection remains differentiable and uses every raw point selected by the
physical query, even when that point shares a network voxel with another object.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from .candidate_sampler import fps_indices
from .contact_ops import native_contact_query
from .heads import chunk_apply, mlp
from .spatial_index import SpatialIndex

ZONE_NAMES = ('core', 'lip', 'cup', 'mount', 'approach')
ZONE_LIMITS = {'core': 32, 'lip': 64, 'cup': 32, 'mount': 32, 'approach': 32}


class ContactGrouping(nn.Module):
    def __init__(self, frozen_config, *, chunk_size=32, checkpointed=True, fps_fn=fps_indices,
                 query_chunk_size=None, query_workers=1, detailed_cpu_timing=False, spatial_index=None,
                 contact_query_backend='numpy', chunk_input_mode='slice',
                 geometry_threads=2, geometry_precision='float64'):
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
        self.chunk_input_mode = chunk_input_mode
        self.fps_fn = fps_fn
        self.spatial_index_config = dict(spatial_index or {})
        if geometry_precision not in ('float32', 'float64'):
            raise ValueError('geometry_precision must be float32 or float64')
        if type(geometry_threads) is not int or not 1 <= geometry_threads <= 256:
            raise ValueError('geometry_threads must be an integer in 1..256')
        if geometry_precision == 'float32' and (
                self.spatial_index_config.get('backend') != 'uniform_grid_cpu' or
                contact_query_backend != 'native_cpu'):
            raise ValueError('float32 contact geometry requires the native CPU grid path')
        self.geometry_precision = geometry_precision
        self.geometry_threads = geometry_threads
        if geometry_precision != 'float64':
            self.spatial_index_config['precision'] = geometry_precision
        if contact_query_backend not in ('numpy', 'native_cpu'):
            raise ValueError('Unsupported contact query backend')
        self.contact_query_backend = contact_query_backend
        self.query_chunk_size = query_chunk_size
        self.query_workers = query_workers
        self.detailed_cpu_timing = detailed_cpu_timing
        if query_chunk_size is not None and (type(query_chunk_size) is not int or not 1 <= query_chunk_size <= 64):
            raise ValueError('query_chunk_size must be between 1 and 64')
        if type(query_workers) is not int or not 1 <= query_workers <= 4:
            raise ValueError('query_workers must be between 1 and 4')
        self.last_query_timings = {}
        self.local_projection = mlp((256, 64))
        self.point_mlp = mlp((71, 64, 64))
        self.zone_projections = nn.ModuleDict({name: mlp((130, 64)) for name in ZONE_NAMES})
        self.last_timings = {}

    @torch.no_grad()
    def query_reference(self, index, points, normals, instances, batch, valid, *, random_roll=False):
        """Return raw indices/masks, seven metric descriptors, pre-limit counts."""
        started = time.perf_counter()
        points = points.detach().float().cpu().numpy().astype(np.float64)
        normals = normals.detach().float().cpu().numpy().astype(np.float64)
        instances = instances.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        valid = valid.detach().cpu().numpy()
        transferred = time.perf_counter()
        detailed = self.detailed_cpu_timing
        stats = dict(pose_d2h_host_s=transferred-started, valid_candidates=int(valid.sum()),
                     ball_calls=4*int(valid.sum()), ball_api_calls=0, ball_returned_points=0,
                     fps_calls=0, fps_input_points=0)
        if detailed:
            stats.update(ball_query_s=0., region_filter_s=0., fps_s=0., descriptor_pack_s=0.)

        def ball(frame, center, radius):
            timer = time.perf_counter() if detailed else 0.
            rows = index.ball(frame, center, radius)
            stats['ball_api_calls'] += int(index.frame_has_points(frame))
            stats['ball_returned_points'] += len(rows)
            if detailed:
                stats['ball_query_s'] += time.perf_counter() - timer
            return rows

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
            timer = time.perf_counter() if detailed else 0.
            n = normals[q] / np.linalg.norm(normals[q])
            e = np.eye(3)[np.argmin(np.abs(n))]
            u = np.cross(e, n)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            c, s = np.cos(rolls[q]), np.sin(rolls[q])
            u, v = c * u + s * v, -s * u + c * v
            if detailed:
                stats['region_filter_s'] += time.perf_counter() - timer
            contact_rows = ball(batch[q], points[q], math.hypot(.022, .008))
            for zone, name in enumerate(ZONE_NAMES):
                if zone < 2:
                    rows, r, h, z_mid = contact_rows, .022, .008, 0.
                else:
                    z_min, z_max, r = self.segments[zone - 2]
                    z_mid = (z_min + z_max) / 2
                    h = max(abs(z_min), abs(z_max))
                    rows = ball(batch[q], points[q] + z_mid * n,
                                      math.hypot(r, (z_max - z_min) / 2))
                if not len(rows):
                    continue
                timer = time.perf_counter() if detailed else 0.
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
                    if detailed:
                        stats['region_filter_s'] += time.perf_counter() - timer
                    continue
                if name == 'lip':
                    sector = np.floor((np.arctan2(y, x) + math.pi) * (8 / (2 * math.pi))).astype(np.int64) % 8
                    sectors[q] = np.bincount(sector, minlength=8)
                # Start FPS nearest the zone's axial midpoint, ties by raw index.
                distance = x*x + y*y + (z - z_mid)**2
                if detailed:
                    stats['region_filter_s'] += time.perf_counter() - timer
                timer = time.perf_counter() if detailed else 0.
                first = rows[np.argmin(distance)]
                if len(rows) > self.limits[name]:
                    stats['fps_calls'] += 1
                    stats['fps_input_points'] += len(rows)
                    chosen = self.fps_fn(index.xyz, rows, self.limits[name], first=first, presorted=True)
                    positions = np.searchsorted(rows, chosen)
                else:
                    chosen, positions = rows, np.arange(len(rows))
                if detailed:
                    stats['fps_s'] += time.perf_counter() - timer
                timer = time.perf_counter() if detailed else 0.
                obs_valid = index.normal_valid[chosen]
                normal_dot = np.where(obs_valid, index.normals[chosen] @ n, 0.)
                geom = np.column_stack((x[positions]/r, y[positions]/r, z[positions]/h,
                                        rho[positions]/r, normal_dot, same[positions], obs_valid))
                data['indices'][q, :len(chosen)] = chosen
                data['mask'][q, :len(chosen)] = True
                data['geometry'][q, :len(chosen)] = geom
                if detailed:
                    stats['descriptor_pack_s'] += time.perf_counter() - timer
        result['lip_sector_counts'] = sectors
        self.last_query_timings = stats
        return result

    @torch.no_grad()
    def query(self, index, points, normals, instances, batch, valid, *, random_roll=False):
        """Four-ball queries, via the P2 NumPy path or fused native CPU path."""
        started = time.perf_counter()
        dtype = np.float32 if self.geometry_precision == 'float32' else np.float64
        points = np.ascontiguousarray(points.detach().float().cpu().numpy(), dtype=dtype)
        normals = np.ascontiguousarray(normals.detach().float().cpu().numpy(), dtype=dtype)
        instances = instances.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        valid = valid.detach().cpu().numpy()
        transferred = time.perf_counter()
        # Exactly one draw for all input poses, before frame grouping/valid masks.
        rolls = (torch.rand(len(points)).numpy() * (2 * math.pi) if random_roll else
                 np.zeros(len(points)))
        result = {name: {'indices': np.zeros((len(points), limit), np.int64),
                         'mask': np.zeros((len(points), limit), bool),
                         'geometry': np.zeros((len(points), limit, 7), np.float32),
                         'counts': np.zeros(len(points), np.int64)}
                  for name, limit in self.limits.items()}
        sectors = np.zeros((len(points), 8), np.int64)
        selected = np.flatnonzero(valid)
        axes, tangent_u, tangent_v = (np.zeros_like(points) for _ in range(3))
        detailed = self.detailed_cpu_timing
        native_query = self.contact_query_backend == 'native_cpu'
        stats = dict(pose_d2h_host_s=transferred-started, valid_candidates=len(selected),
                     ball_calls=4*len(selected), ball_api_calls=0, ball_returned_points=0,
                     fps_calls=0, fps_input_points=0)
        if detailed and not native_query:
            stats.update(ball_query_s=0., region_filter_s=0., fps_s=0., descriptor_pack_s=0.)
        timer = time.perf_counter() if detailed else 0.
        for q in selected:
            n = normals[q] / np.linalg.norm(normals[q])
            e = np.eye(3)[np.argmin(np.abs(n))]
            u = np.cross(e, n)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            c, s = np.cos(rolls[q]), np.sin(rolls[q])
            axes[q], tangent_u[q], tangent_v[q] = n, c*u + s*v, -s*u + c*v
        if detailed:
            stats['pose_frame_setup_s' if native_query else 'region_filter_s'] = time.perf_counter() - timer

        if native_query:
            result['lip_sector_counts'] = sectors
            stats.update(native_contact_query(index, points, axes, tangent_u, tangent_v,
                         instances, batch, selected, self.segments, self.self_ignore,
                         result, detailed=detailed, workers=self.geometry_threads,
                         precision=self.geometry_precision))
            self.last_query_timings = stats
            return result

        def chunks():
            if self.query_chunk_size is None:
                # Original per-candidate SciPy schedule remains the baseline.
                for q in selected:
                    yield batch[q], np.asarray([q]), False
            else:
                for frame in np.unique(batch[selected]):
                    queries = selected[batch[selected] == frame]
                    for start in range(0, len(queries), self.query_chunk_size):
                        yield frame, queries[start:start+self.query_chunk_size], True

        for frame, queries, batched in chunks():
            timer = time.perf_counter() if detailed else 0.
            centers = np.empty((len(queries), 4, 3), np.float64)
            radii = np.empty((len(queries), 4), np.float64)
            centers[:, 0], radii[:, 0] = points[queries], math.hypot(.022, .008)
            for zone, (z_min, z_max, radius) in enumerate(self.segments, start=1):
                centers[:, zone] = points[queries] + ((z_min+z_max)/2) * axes[queries]
                radii[:, zone] = math.hypot(radius, (z_max-z_min)/2)
            if batched:
                balls = index.balls(frame, centers.reshape(-1, 3), radii.reshape(-1), workers=self.query_workers)
                stats['ball_api_calls'] += int(index.frame_has_points(frame))
            else:
                balls = [index.ball(frame, center, radius)
                         for center, radius in zip(centers[0], radii[0])]
                stats['ball_api_calls'] += 4 * int(index.frame_has_points(frame))
            stats['ball_returned_points'] += sum(map(len, balls))
            if detailed:
                stats['ball_query_s'] += time.perf_counter() - timer
            for local_q, q in enumerate(queries):
                n, u, v = axes[q], tangent_u[q], tangent_v[q]
                contact = None
                for zone, name in enumerate(ZONE_NAMES):
                    timer = time.perf_counter() if detailed else 0.
                    if zone < 2:
                        rows, r, h, z_mid = balls[4*local_q], .022, .008, 0.
                    else:
                        rows = balls[4*local_q+zone-1]
                        z_min, z_max, r = self.segments[zone-2]
                        z_mid = (z_min+z_max)/2
                        h = max(abs(z_min), abs(z_max))
                    if not len(rows):
                        if detailed:
                            stats['region_filter_s'] += time.perf_counter() - timer
                        continue
                    if zone == 1:
                        x, y, z, rho, same = contact
                    else:
                        d = index.xyz[rows] - points[q]
                        x, y, z = d @ u, d @ v, d @ n
                        rho = np.hypot(x, y)
                        same = index.instance[rows] == instances[q]
                        if zone == 0:
                            contact = x, y, z, rho, same
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
                    if len(rows) and name == 'lip':
                        sector = np.floor((np.arctan2(y, x) + math.pi) * (8 / (2 * math.pi))).astype(np.int64) % 8
                        sectors[q] = np.bincount(sector, minlength=8)
                    # Preserve the original axial midpoint and operation ordering.
                    distance = x*x + y*y + (z-z_mid)**2
                    if detailed:
                        stats['region_filter_s'] += time.perf_counter() - timer
                    if not len(rows):
                        continue
                    timer = time.perf_counter() if detailed else 0.
                    first = rows[np.argmin(distance)]
                    if len(rows) > self.limits[name]:
                        stats['fps_calls'] += 1
                        stats['fps_input_points'] += len(rows)
                        chosen = self.fps_fn(index.xyz, rows, self.limits[name], first=first, presorted=True)
                        positions = np.searchsorted(rows, chosen)
                    else:
                        chosen, positions = rows, np.arange(len(rows))
                    if detailed:
                        stats['fps_s'] += time.perf_counter() - timer
                    timer = time.perf_counter() if detailed else 0.
                    obs_valid = index.normal_valid[chosen]
                    normal_dot = np.where(obs_valid, index.normals[chosen] @ n, 0.)
                    geom = np.column_stack((x[positions]/r, y[positions]/r, z[positions]/h,
                                            rho[positions]/r, normal_dot, same[positions], obs_valid))
                    data['indices'][q, :len(chosen)] = chosen
                    data['mask'][q, :len(chosen)] = True
                    data['geometry'][q, :len(chosen)] = geom
                    if detailed:
                        stats['descriptor_pack_s'] += time.perf_counter() - timer
            # Do not retain a previous block while allocating the next ball lists.
            del balls
        result['lip_sector_counts'] = sectors
        self.last_query_timings = stats
        return result

    def forward(self, encoded, points, normals, instances, batch, valid, *, random_roll=False):
        started = time.perf_counter()
        existing_index = encoded.get('spatial_index')
        backend = (existing_index.backend if existing_index is not None
                   else self.spatial_index_config.get('backend', 'kdtree'))
        build_field = 'index_build_s' if backend == 'uniform_grid_cpu' else 'tree_build_s'
        setup_timings = {'geometry_cpu_setup_s': 0., build_field: 0.,
                         'spatial_index_backend': backend}
        if 'spatial_index' not in encoded:
            geometry = encoded.get('cpu_geometry')
            encoded['spatial_index'] = (SpatialIndex.from_cpu_geometry(geometry, **self.spatial_index_config) if geometry is not None
                                        else SpatialIndex.from_inputs(encoded, **self.spatial_index_config))
            setup_timings = encoded['spatial_index'].setup_timings
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
                                input_mode=self.chunk_input_mode,
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
                                         input_mode=self.chunk_input_mode,
                                         checkpointed=self.checkpointed and self.training)
        finished = time.perf_counter()
        self.last_timings = {**setup_timings, **self.last_query_timings,
                             'spatial_index_seconds': built-started,
                             'query_cpu_seconds': queried-built,
                             'feature_projection_and_transfer_seconds': transferred-queried,
                             'feature_encoding_enqueue_seconds': finished-transferred}
        # CUDA feature encoding is asynchronous; the enqueue timing is explicitly
        # named and is not reported as synchronized GPU kernel latency.
        encoded['grouping_timings'] = dict(self.last_timings)
        return {**features, 'lip_sector_log_counts': sectors.float().log1p(),
                'lip_sector_counts': sectors,
                'zone_counts': torch.stack([prepared[n]['counts'] for n in ZONE_NAMES], -1)}
