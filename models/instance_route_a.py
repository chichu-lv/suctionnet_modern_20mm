"""Route A: single-instance sparse 3D/RGB scoring at exact contact poses.

The default backbone is the verified 12-channel SpConv ResUNet14. A caller may
inject a backbone for explicit unit tests; there is no automatic CPU surrogate.
Observation geometry is fixed float64 CPU data. Learned tensors remain on the
model device and retain their complete gradient graph across pose chunks.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .backbones.spconv_resunet14 import SpconvResUNet14
from .instance_route_a_geometry import (
    BOUNDARY_ROUNDOFF_M, HEIGHT_LIMIT_M, ZONE_BOUNDS, ZONE_SAMPLES,
    NeighborIndex, association_diagnostics, contact_coordinates,
    deterministic_fps, lip_summary, numpy_array, voxel_representatives,
)


def _mlp(widths, *, output=False):
    # New route-A MLPs follow the design: bias + ReLU, no LN or dropout.
    layers = []
    for index, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right, bias=True))
        if not (output and index == len(widths) - 2):
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SingletonRunningBatchNorm1d(nn.BatchNorm1d):
    """Use frozen running moments only for an otherwise invalid singleton.

    The ordinary path is exactly BatchNorm1d, including its momentum and batch
    counter behavior. Parameter/buffer names are unchanged for checkpointing.
    """
    singleton_fallback_count = 0

    def forward(self, value):
        if self.training and value.numel() // self.num_features <= 1:
            if self.running_mean is None or self.running_var is None:
                raise RuntimeError('singleton running-stats policy requires tracked BatchNorm moments')
            self.singleton_fallback_count += 1
            return F.batch_norm(value, self.running_mean, self.running_var,
                                self.weight, self.bias, training=False, momentum=0., eps=self.eps)
        return super().forward(value)


def _install_singleton_policy(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d) and not isinstance(child, SingletonRunningBatchNorm1d):
            if not child.track_running_stats:
                raise ValueError('Route-A singleton policy requires track_running_stats=True')
            replacement = SingletonRunningBatchNorm1d(
                child.num_features, eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats,
                device=child.running_mean.device, dtype=child.running_mean.dtype)
            replacement.load_state_dict(child.state_dict())
            replacement.train(child.training)
            if child.affine:
                replacement.weight.requires_grad_(child.weight.requires_grad)
                replacement.bias.requires_grad_(child.bias.requires_grad)
            setattr(module, name, replacement)
        else:
            _install_singleton_policy(child)


def _interpolate(features, association, *, chunk=2048):
    rows, weights = association['indices'], association['weights']
    if not len(rows):
        return features.new_empty((0, features.shape[-1]))
    if not len(features):
        return features.new_zeros((len(rows), features.shape[-1]))
    output = []
    for start in range(0, len(rows), chunk):
        index = torch.as_tensor(np.maximum(rows[start:start + chunk], 0),
                                device=features.device, dtype=torch.long)
        weight = torch.as_tensor(weights[start:start + chunk],
                                 device=features.device, dtype=features.dtype)
        output.append((features[index] * weight[..., None]).sum(1))
    return torch.cat(output, dim=0)


def _support(observation):
    points = numpy_array(observation['support_points'], np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('support_points must be finite [M,3] original camera points')
    count = len(points)
    rgb = numpy_array(observation['support_rgb'], np.float64)
    normals = numpy_array(observation['support_normals'], np.float64)
    normal_valid = numpy_array(observation['support_normal_valid'], bool)
    components = numpy_array(observation['support_components'])
    if components.dtype.kind not in 'iu' or np.any(components < 0):
        raise ValueError('Support components must be nonnegative integer IDs')
    components = components.astype(np.int64, copy=False)
    if rgb.shape != points.shape or normals.shape != points.shape:
        raise ValueError('Raw RGB/normals must follow support row identity')
    if normal_valid.shape != (count,) or components.shape != (count,):
        raise ValueError('Support validity/components must have shape [M]')
    if 'support_source_ids' in observation:
        source_ids = numpy_array(observation['support_source_ids'])
        if source_ids.dtype.kind not in 'iu':
            raise ValueError('Support source IDs must be integers')
        source_ids = source_ids.astype(np.int64, copy=False)
    else:
        uv = numpy_array(observation['support_uv'], np.int64)
        if uv.shape != (count, 2):
            raise ValueError('support_uv must be [M,2] in u,v order')
        source_ids = uv[:, 1] * int(observation['rgb'].shape[1]) + uv[:, 0]
    if source_ids.shape != (count,) or np.any(source_ids < 0):
        raise ValueError('Support source IDs must be nonnegative [M]')
    rgb_valid = numpy_array(observation.get('support_rgb_valid', np.isfinite(rgb).all(1)), bool)
    if rgb_valid.shape != (count,):
        raise ValueError('support_rgb_valid must have shape [M]')
    if not np.isfinite(rgb[rgb_valid]).all() or np.any((rgb[rgb_valid] < 0) | (rgb[rgb_valid] > 1)):
        raise ValueError('Valid RGB must be finite in [0,1]')
    if not np.isfinite(normals[normal_valid]).all():
        raise ValueError('Valid observed normals must be finite')
    length = np.linalg.norm(normals[normal_valid], axis=-1)
    if np.any(length <= 1e-12) or not np.isfinite(length).all():
        raise ValueError('Valid observed normals must be nonzero')
    normals = normals.copy()
    normals[normal_valid] /= length[:, None]
    normals[~normal_valid] = 0.
    rgb = rgb.copy()
    rgb[~rgb_valid] = 0.
    order = np.argsort(source_ids, kind='stable')
    points, rgb, normals, normal_valid, rgb_valid, source_ids, components = (
        value[order] for value in (points, rgb, normals, normal_valid, rgb_valid, source_ids, components))
    duplicate = source_ids[1:] == source_ids[:-1]
    if np.any(duplicate):
        for value in (points, rgb, normals):
            if np.any(np.max(np.abs(value[1:] - value[:-1]), axis=1)[duplicate] > BOUNDARY_ROUNDOFF_M):
                raise ValueError('One source ID cannot identify different support attributes')
        if any(np.any((value[1:] != value[:-1])[duplicate]) for value in
               (normal_valid, rgb_valid, components)):
            raise ValueError('One source ID cannot identify conflicting masks/components')
        unique = np.r_[True, ~duplicate]
        points, rgb, normals, normal_valid, rgb_valid, source_ids, components = (
            value[unique] for value in (points, rgb, normals, normal_valid, rgb_valid, source_ids, components))
    return points, rgb, normals, normal_valid, rgb_valid, source_ids, components


def _instance_geometry(observation, backbone_points, min_support_points):
    points, rgb, normals, normal_valid, rgb_valid, source_ids, components = _support(observation)
    centroid = points.mean(0) if len(points) else np.zeros(3, np.float64)
    supplied_centroid = numpy_array(observation.get('centroid', centroid), np.float64)
    if supplied_centroid.shape != (3,) or not np.allclose(supplied_centroid, centroid, rtol=0, atol=1e-9):
        raise ValueError('centroid must be the full visible support mean, not a CAD/teacher centre')
    gravity = numpy_array(observation['gravity'], np.float64)
    if (gravity.shape != (3,) or not np.isfinite(gravity).all()
            or not np.isfinite(np.linalg.norm(gravity)) or np.linalg.norm(gravity) <= 1e-12):
        raise ValueError('gravity must be a finite nonzero camera direction')
    gravity = gravity / np.linalg.norm(gravity)
    source_index = NeighborIndex(points, source_ids, components)
    representatives, voxel_keys = voxel_representatives(points, source_ids, centroid)
    chosen = deterministic_fps(points[representatives], source_ids[representatives], backbone_points)
    selected = representatives[chosen]
    keys = voxel_keys[chosen]
    sufficient = len(points) >= min_support_points
    if not sufficient:
        selected = np.empty(0, np.int64)
        keys = np.empty((0, 3), np.int64)
    backbone_index = NeighborIndex(points[selected], source_ids[selected], components[selected])
    association = backbone_index.associate(points, components)
    return dict(points_np=points, rgb_np=rgb, normals_np=normals, normal_valid_np=normal_valid,
                rgb_valid_np=rgb_valid, source_ids_np=source_ids, components_np=components,
                centroid_np=centroid, gravity_np=gravity, source_index=source_index,
                backbone_index=backbone_index, raw_association=association,
                raw_feature_valid_np=association['valid'], backbone_raw_indices=selected,
                backbone_voxel_keys=keys, support_sufficient=sufficient)


def _contact_geometry(encoded, points, normals, components, valid):
    count = len(points)
    slots = sum(ZONE_SAMPLES)
    indices = np.zeros((count, slots), np.int64)
    masks = np.zeros((count, slots), bool)
    descriptor = np.zeros((count, slots, 8), np.float64)
    raw_count = np.zeros((count, 3), np.int64)
    summary = np.zeros((count, 8), np.float64)
    angular_support = np.zeros(count, np.int64)
    for row in np.flatnonzero(valid):
        support = encoded['source_index'].support_ball(points[row], components[row])
        rho, height, tangent, zones = contact_coordinates(encoded['points_np'][support], points[row], normals[row])
        summary[row], angular_support[row] = lip_summary(height[zones[2]], tangent[zones[2]], normals[row])
        offset = 0
        for zone, (mask, budget) in enumerate(zip(zones, ZONE_SAMPLES)):
            local = np.flatnonzero(mask)
            raw_count[row, zone] = len(local)
            rows = support[local]
            chosen = deterministic_fps(encoded['points_np'][rows], encoded['source_ids_np'][rows], budget)
            rows, local = rows[chosen], local[chosen]
            selected_slice = slice(offset, offset + len(rows))
            indices[row, selected_slice] = rows
            masks[row, selected_slice] = True
            outer_radius = ZONE_BOUNDS[zone][1]
            observed_normal = encoded['normals_np'][rows]
            descriptor[row, selected_slice] = np.column_stack((
                rho[local] / outer_radius, height[local] / HEIGHT_LIMIT_M,
                observed_normal @ normals[row], tangent[local] @ encoded['gravity_np'] / outer_radius,
                observed_normal @ encoded['gravity_np'], encoded['raw_feature_valid_np'][rows],
                encoded['normal_valid_np'][rows], encoded['rgb_valid_np'][rows]))
            offset += budget
    return indices, masks, descriptor, raw_count, summary, angular_support


def _query_geometry(encoded, points, normals, pose_valid):
    safe_points = np.where(pose_valid[:, None], points, 0.)
    anchors, anchor_distance = encoded['source_index'].nearest_anchor(safe_points)
    valid = pose_valid & (anchors >= 0) & encoded['support_sufficient']
    # Invalid poses are never allowed to introduce Inf/NaN into the dense
    # graph before a final mask; 0 * Inf in a backward pass is still NaN.
    safe_points = np.where(valid[:, None], points, encoded['centroid_np'])
    normals = np.where(valid[:, None], normals, 0.)
    components = np.full(len(points), -1, np.int64)
    components[valid] = encoded['components_np'][anchors[valid]]
    association = encoded['backbone_index'].associate(safe_points, components)
    contact = _contact_geometry(encoded, safe_points, normals, components, valid)
    return safe_points, normals, anchors, anchor_distance, valid, association, contact


def _normalize_queries(points, normals):
    lengths = np.linalg.norm(normals, axis=1)
    pose_valid = (np.isfinite(points).all(1) & np.isfinite(normals).all(1)
                  & np.isfinite(lengths) & (lengths > 1e-12))
    normalized = np.zeros_like(normals)
    normalized[pose_valid] = normals[pose_valid] / lengths[pose_valid, None]
    return pose_valid, normalized


def prepare_source_geometry(observation, *, backbone_points=2048, min_support_points=100, query_chunk_size=32):
    """Prepare final per-instance CPU geometry after query selection and RGB jitter.

    This is transient worker output. Keep it attached to its own observation
    for the upcoming forward pass, never across augmentation or resampling.
    Each worker can handle a different instance with the same model settings.
    """
    geometry = _instance_geometry(observation, backbone_points, min_support_points)
    points = numpy_array(observation['query_points'], np.float64).copy()
    normals = numpy_array(observation['query_normals'], np.float64).copy()
    pose_valid, normalized = _normalize_queries(points, normals)
    chunks = {}
    for start in range(0, len(points), query_chunk_size):
        part = slice(start, start + query_chunk_size)
        chunks[start] = _query_geometry(geometry, points[part], normalized[part], pose_valid[part])
    cache = dict(geometry=geometry, settings=(backbone_points, min_support_points, query_chunk_size),
                 points=points, normals=normals, chunks=chunks)
    return dict(observation, _route_a_geometry=cache)


class InstanceSuction3D(nn.Module):
    """A's 531D query fusion and separate 200D Seal head.

    ``observation`` contains one instance, not a padded batch of objects.
    Microbatch-one gradient accumulation is handled by the training loop.
    ``backbone_points`` changes are explicit and belong in checkpoint identity.
    """
    route = 'a'
    architecture_id = 'InstanceSuction3D.v3.12ch.531fusion.200seal'
    quality_contract = 'gt_INST20_pose_v2'

    def __init__(self, backbone_root=None, *, backbone=None, backbone_points=2048,
                 min_support_points=100, query_chunk_size=32,
                 backbone_coordinate_fastpath=False, singleton_bn_policy='running_stats'):
        super().__init__()
        for name, value in [('backbone_points', backbone_points),
                            ('min_support_points', min_support_points),
                            ('query_chunk_size', query_chunk_size)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if backbone is None and backbone_root is None:
            raise ValueError('Route A requires backbone_root for the verified SpConv backbone; '
                             'an injected backbone is allowed only as an explicit caller choice')
        self.backbone = backbone if backbone is not None else SpconvResUNet14(
            backbone_root, in_channels=12,
            backbone_coordinate_fastpath=backbone_coordinate_fastpath)
        self.backbone_injected = backbone is not None
        if singleton_bn_policy not in {'running_stats', 'error'}:
            raise ValueError('singleton_bn_policy must be running_stats or error')
        self.singleton_bn_policy = singleton_bn_policy
        if singleton_bn_policy == 'running_stats':
            _install_singleton_policy(self.backbone)
        self.backbone_points = backbone_points
        self.min_support_points = min_support_points
        self.query_chunk_size = query_chunk_size
        self.global_projection = _mlp((512, 128, 64))
        self.local_projection = _mlp((256, 64))
        self.local_point_mlp = _mlp((75, 64, 64))
        self.zone_projection = _mlp((130, 64))
        self.query_fusion = _mlp((531, 256, 256))
        self.q_head = _mlp((256, 128, 1), output=True)
        self.seal_head = _mlp((200, 128, 1), output=True)
        self.wrench_head = _mlp((329, 128, 1), output=True)
        self.scale_head = _mlp((256, 128, 1), output=True)
        self.task_head = _mlp((260, 128, 1), output=True)
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.constant_(self.scale_head[-1].bias, math.log(.1))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _tensor(self, value, *, dtype=None):
        return torch.as_tensor(value, device=self.device, dtype=dtype or self.dtype)

    def architecture_metadata(self):
        return {'architecture_id': self.architecture_id, 'route': self.route,
                'quality_contract': self.quality_contract, 'input_channels': 12,
                'fusion_channels': 531, 'seal_input_channels': 200,
                'backbone_points': self.backbone_points, 'voxel_size_m': .002,
                'minimum_support_points': self.min_support_points,
                'geometry_dtype': 'float64', 'backbone_injected': self.backbone_injected,
                'singleton_bn_policy': self.singleton_bn_policy,
                'scale_semantics': 'ell=log_laplace_scale; log_variance=log(2)+2*ell'}

    def prepare_source_geometry(self, observation):
        return prepare_source_geometry(observation, backbone_points=self.backbone_points,
                                       min_support_points=self.min_support_points,
                                       query_chunk_size=self.query_chunk_size)

    def encode_instance(self, observation):
        prepared = observation.get('_route_a_geometry')
        if prepared is None:
            geometry = _instance_geometry(observation, self.backbone_points, self.min_support_points)
        else:
            if prepared['settings'] != (self.backbone_points, self.min_support_points, self.query_chunk_size):
                raise ValueError('Prepared route-A geometry has different model settings')
            geometry = prepared['geometry']
        points, rgb, normals, normal_valid, rgb_valid, source_ids, components = (
            geometry[name] for name in ('points_np', 'rgb_np', 'normals_np', 'normal_valid_np',
                                        'rgb_valid_np', 'source_ids_np', 'components_np'))
        centroid, gravity = geometry['centroid_np'], geometry['gravity_np']
        source_index, backbone_index = geometry['source_index'], geometry['backbone_index']
        selected, keys = geometry['backbone_raw_indices'], geometry['backbone_voxel_keys']
        sufficient = geometry['support_sufficient']
        normalized_rgb = (rgb - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
        normalized_rgb[~rgb_valid] = 0.
        if len(selected) and sufficient:
            offset = keys.min(0)
            nonnegative = keys - offset
            if nonnegative.max() > np.iinfo(np.int32).max - 1:
                raise ValueError('Instance extent overflows int32 voxel coordinates')
            coords = np.column_stack((np.zeros(len(selected), np.int32), nonnegative[:, ::-1])).astype(np.int32)
            inputs = np.concatenate(((points[selected] - centroid) / .1,
                                     normalized_rgb[selected], normals[selected],
                                     np.broadcast_to(gravity, (len(selected), 3))), axis=1)
            # Pad the empty high end to complete all four stride-2 blocks.
            # With a tight odd extent, spconv drops the trailing cells and an
            # occluded instance can lose its entire active set at the bottleneck.
            spatial_shape = ((coords[:, 1:].max(0).astype(np.int64) // 16 + 1) * 16).tolist()
            sparse_inputs = {'spatial_shape': spatial_shape,
                             'voxel_coords': self._tensor(np.ascontiguousarray(coords), dtype=torch.int32),
                             'voxel_features': self._tensor(inputs), 'g_cam': self._tensor(gravity[None])}
            features = self.backbone(sparse_inputs)
            if features.shape != (len(selected), 256):
                raise ValueError('Route-A backbone must preserve input row order and return [N,256]')
            pooled = torch.cat((features.mean(0), features.max(0).values), dim=0)
            context = self.global_projection(pooled)
        else:
            # Invalid small instances cannot exercise a one-active-voxel BN path.
            selected = np.empty(0, np.int64)
            keys = np.empty((0, 3), np.int64)
            offset = np.zeros(3, np.int64)
            sparse_inputs = {'voxel_coords': self._tensor(np.empty((0, 4), np.int32), dtype=torch.int32),
                             'voxel_features': self._tensor(np.empty((0, 12))),
                             'g_cam': self._tensor(gravity[None])}
            features = self._tensor(np.empty((0, 256)))
            context = self._tensor(np.zeros(64))
        association = geometry['raw_association']
        raw_features = _interpolate(features, association)
        feature_valid = self._tensor(association['valid'], dtype=torch.bool)
        projected = self.local_projection(raw_features)
        projected = torch.where(feature_valid[:, None], projected, torch.zeros_like(projected))
        return {'source_index': source_index, 'backbone_index': backbone_index, 'prepared': prepared,
                'points_np': points, 'rgb': self._tensor(normalized_rgb),
                'normals_np': normals, 'normal_valid_np': normal_valid,
                'rgb_valid_np': rgb_valid, 'source_ids_np': source_ids,
                'components_np': components, 'centroid_np': centroid, 'gravity_np': gravity,
                'backbone_features': features, 'raw_projected_features': projected,
                'raw_feature_valid_np': association['valid'], 'context': context,
                'backbone_raw_indices': selected, 'backbone_source_ids': source_ids[selected],
                'backbone_voxel_keys': keys, 'voxel_offset_xyz': offset,
                'sparse_inputs': sparse_inputs, 'support_sufficient': sufficient,
                'association_diagnostics': association_diagnostics(association),
                'query_points': observation.get('query_points', np.empty((0, 3))),
                'query_normals': observation.get('query_normals', np.empty((0, 3))),
                'query_ids': observation.get('query_ids'), 'observation': observation}

    def _contact_batch(self, encoded, points, contact):
        count, slots = len(points), sum(ZONE_SAMPLES)
        indices, masks, descriptor, raw_count, summary, angular_support = contact
        mask = self._tensor(masks, dtype=torch.bool)
        if len(encoded['points_np']):
            index = self._tensor(indices, dtype=torch.long)
            local_features = encoded['raw_projected_features'][index]
            rgb = encoded['rgb'][index]
        else:
            local_features = self._tensor(np.zeros((count, slots, 64)))
            rgb = self._tensor(np.zeros((count, slots, 3)))
        point_input = torch.cat((local_features, rgb, self._tensor(descriptor)), dim=-1)
        point_input = torch.where(mask[..., None], point_input, torch.zeros_like(point_input))
        local = self.local_point_mlp(point_input)
        local = torch.where(mask[..., None], local, torch.zeros_like(local))
        zone_output = []
        offset = 0
        for zone, budget in enumerate(ZONE_SAMPLES):
            values, keep = local[:, offset:offset + budget], mask[:, offset:offset + budget]
            nonempty = keep.any(1)
            mean = values.sum(1) / keep.sum(1, keepdim=True).clamp_min(1)
            maximum = values.masked_fill(~keep[..., None], -torch.inf).max(1).values
            maximum = torch.where(nonempty[:, None], maximum, torch.zeros_like(maximum))
            stats = torch.cat((mean, maximum, self._tensor(np.log1p(raw_count[:, zone]))[:, None],
                               nonempty[:, None].to(mean.dtype)), dim=-1)
            projected = self.zone_projection(stats)
            zone_output.append(torch.where(nonempty[:, None], projected, torch.zeros_like(projected)))
            offset += budget
        return torch.cat(zone_output, -1), self._tensor(summary), raw_count, angular_support

    def _score_chunk(self, encoded, points, normals, pose_valid, geometry=None):
        if geometry is None:
            geometry = _query_geometry(encoded, points, normals, pose_valid)
        safe_points, normals, anchors, anchor_distance, valid, association, contact = geometry
        query_features = _interpolate(encoded['backbone_features'], association)
        zones, lip, raw_counts, angular_support = self._contact_batch(encoded, safe_points, contact)
        context = encoded['context'].expand(len(points), -1)
        delta = self._tensor((safe_points - encoded['centroid_np']) / .1)
        axis = self._tensor(normals)
        gravity = self._tensor(encoded['gravity_np']).expand(len(points), -1)
        dot_gravity = (axis * gravity).sum(-1, keepdim=True)
        descriptor = torch.cat((delta, axis, gravity, dot_gravity), -1)
        query_feature_valid = self._tensor(association['valid'], dtype=torch.bool)
        fused = self.query_fusion(torch.cat((query_features, context, zones, lip, descriptor,
                                             query_feature_valid[:, None].to(query_features.dtype)), -1))
        q = self.q_head(fused).squeeze(-1).sigmoid()
        seal = self.seal_head(torch.cat((zones, lip), -1)).squeeze(-1).sigmoid()
        wrench = self.wrench_head(torch.cat((fused, context, delta, axis, gravity), -1)).squeeze(-1).sigmoid()
        ell = self.scale_head(fused).squeeze(-1).clamp(-6., 1.)
        task = self.task_head(torch.cat((fused, axis, dot_gravity), -1)).squeeze(-1)
        keep = self._tensor(valid, dtype=torch.bool)
        masked = lambda value: torch.where(keep, value, torch.zeros_like(value))
        reasons = ['ok' if ok else ('invalid_pose' if not good else
                   'insufficient_support' if not encoded['support_sufficient'] else 'pose_geometry_invalid')
                   for ok, good in zip(valid, pose_valid)]
        return {'q_inst': masked(q), 'seal': masked(seal), 'wrench': masked(wrench),
                'ell': masked(ell), 'log_scale': masked(ell),
                'log_variance': masked(math.log(2.) + 2 * ell), 'task_logit': masked(task),
                'task_probability': masked(task.sigmoid()), 'valid': keep,
                'pose_geometry_valid': keep, 'query_feature_valid': query_feature_valid & keep,
                'lip_summary': lip, 'region_counts': self._tensor(raw_counts, dtype=torch.long),
                'independent_angular_support': self._tensor(angular_support, dtype=torch.long),
                'anchor_source_ids': self._tensor(np.where(valid, encoded['source_ids_np'][np.maximum(anchors, 0)], -1)
                                                 if len(encoded['source_ids_np']) else np.full(len(points), -1), dtype=torch.long),
                'anchor_distance_m': self._tensor(anchor_distance, dtype=torch.float64),
                'reason': reasons}

    def score_at_pose(self, encoded, p, n, query_batch_index=None):
        points, normals = numpy_array(p, np.float64), numpy_array(n, np.float64)
        if points.shape == (3,):
            points = points[None]
        if normals.shape == (3,):
            normals = normals[None]
        if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
            raise ValueError('Pose inputs must have matching [Q,3] shapes')
        if query_batch_index is not None:
            batch = numpy_array(query_batch_index)
            if batch.shape != (len(points),) or np.any(batch != 0):
                raise ValueError('encode_instance contains one object; query_batch_index must be all zero')
        pose_valid, normalized = _normalize_queries(points, normals)
        prepared = encoded.get('prepared')
        use_cached = (prepared is not None and np.array_equal(points, prepared['points'], equal_nan=True)
                      and np.array_equal(normals, prepared['normals'], equal_nan=True))
        outputs = [self._score_chunk(encoded, points[start:start + self.query_chunk_size],
                                      normalized[start:start + self.query_chunk_size],
                                      pose_valid[start:start + self.query_chunk_size],
                                      prepared['chunks'][start] if use_cached else None)
                   for start in range(0, len(points), self.query_chunk_size)]
        if not outputs:
            # Execute zero-width dense layers to retain well-defined tensor types.
            outputs = [self._score_chunk(encoded, points, normalized, pose_valid)]
        combined = {key: (sum((item[key] for item in outputs), []) if key == 'reason'
                          else torch.cat([item[key] for item in outputs], dim=0)) for key in outputs[0]}
        combined['points'] = self._tensor(points, dtype=torch.float64)
        combined['normal_out'] = self._tensor(normalized, dtype=torch.float64)
        combined['query_ids'] = self._tensor(np.arange(len(points)), dtype=torch.long)
        return combined

    def forward(self, observation, points=None, normals=None):
        encoded = self.encode_instance(observation)
        canonical = points is None and normals is None
        if (points is None) != (normals is None):
            raise ValueError('Provide both pose points and normals, or neither')
        result = self.score_at_pose(encoded, encoded['query_points'] if canonical else points,
                                    encoded['query_normals'] if canonical else normals)
        if canonical and encoded['query_ids'] is not None:
            result['query_ids'] = self._tensor(numpy_array(encoded['query_ids'], np.int64), dtype=torch.long)
        return result

    def parameter_groups(self):
        if hasattr(self.backbone, 'parameter_groups'):
            groups = self.backbone.parameter_groups()
            encoder, decoder = list(groups.get('encoder', [])), list(groups.get('decoder', []))
        else:
            encoder, decoder = list(self.backbone.parameters()), []
        decoder += list(self.global_projection.parameters())
        contact = [parameter for module in (self.local_projection, self.local_point_mlp,
                                           self.zone_projection, self.query_fusion) for parameter in module.parameters()]
        heads = [parameter for module in (self.q_head, self.seal_head, self.wrench_head,
                                         self.scale_head, self.task_head) for parameter in module.parameters()]
        result = {'encoder': encoder, 'decoder': decoder, 'contact': contact, 'heads': heads}
        ids = [id(parameter) for values in result.values() for parameter in values]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters()}:
            raise RuntimeError('Parameter groups must form a disjoint exhaustive partition')
        return result
