"""Shared scene encoder and explicit physical-pose scoring interface."""
from __future__ import annotations

import torch
from torch import nn

from .backbones.spconv_resunet14 import SpconvResUNet14
from .candidate_sampler import CandidateSampler
from .geometry_ops import make_fps_backend
from .heads import AxisHead, chunk_apply, mlp
from .contact_grouping import ContactGrouping, ZONE_LIMITS, ZONE_NAMES
from .instance_fusion import InstanceFusion, instance_blocks


EXPERIMENTS = {
    'E0': (False, False, False), 'E1': (False, False, False),
    'E2': (True, False, False), 'E3': (False, True, False),
    'E4': (True, True, False), 'E5': (True, True, True),
}


class Suction20Net(nn.Module):
    def __init__(self, model_config, frozen_config, backbone=None, *, performance=None):
        super().__init__()
        performance = dict(performance or {})
        self.no_instance_voxel_heads = bool(performance.get('no_instance_voxel_heads', False))
        fps_fn = make_fps_backend(performance.get('fps_backend', 'numpy'))
        self.model_config = dict(model_config)
        self.frozen_config = dict(frozen_config)
        self.experiment = model_config.get('experiment', 'E1')
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f'Unknown model experiment {self.experiment}')
        self.use_instance_fusion, self.use_contact_grouping, self.use_aux_heads = EXPERIMENTS[self.experiment]
        self.quality_target = 'gt_A' if self.experiment == 'E0' else 'gt_B'
        structural = {
            'name': 'suction20net', 'backbone': 'spconv_resunet14',
            'voxel_size_m': .002, 'point_feature_dim': 256, 'instance_feature_dim': 128,
            'normal_parameterization': 'tangent_residual', 'normal_max_tilt_deg': 30.,
            'normal_target': 'gt_normal_original_candidate_axis',
            'physical_geometry_source': 'frozen_label_config',
            'seal_normalization_source': 'frozen_label_config',
            'quality_target': self.quality_target, 'group_samples': ZONE_LIMITS,
            'use_instance_fusion': self.use_instance_fusion,
            'use_contact_grouping': self.use_contact_grouping, 'use_aux_heads': self.use_aux_heads,
        }
        for key, expected in structural.items():
            if key in model_config and model_config[key] != expected:
                raise ValueError(f'{self.experiment} requires {key}={expected!r}')
        self.backbone = backbone if backbone is not None else SpconvResUNet14(
            model_config.get('backbone_root', 'external/graspness_modern'),
            backbone_coordinate_fastpath=performance.get('backbone_coordinate_fastpath', False))
        self.fusion = mlp((393, 256, 256))
        self.axis_head = AxisHead()
        self.q0_head = mlp((256, 128, 1), output=True)
        self.pose_fusion = mlp((590, 256, 256))
        self.q_head = mlp((256, 128, 1), output=True)
        use_success = model_config.get('use_task_success_head',
                                       model_config.get('task_success_head', False))
        self.task_success_head = mlp((260, 128, 1), output=True) if use_success else None
        if self.task_success_head is not None:
            nn.init.zeros_(self.task_success_head[-1].bias)
        self.sampler = CandidateSampler(model_config.get('seed_count', model_config.get('num_seeds', 512)),
                                        fps_fn=fps_fn)
        self.chunk_size = int(model_config.get('dense_chunk_size', 16384))
        self.checkpoint_dense = bool(model_config.get('checkpoint_dense', True))
        if self.chunk_size <= 0 or self.sampler.num_seeds <= 0:
            raise ValueError('dense_chunk_size and seed_count must be positive')
        # Optional modules are instantiated after the common E1 modules, keeping
        # every E1 state-dict name and its initialization sequence unchanged.
        self.instance_module = InstanceFusion() if self.use_instance_fusion else None
        self.contact_grouping = (ContactGrouping(frozen_config,
                                  chunk_size=model_config.get('pose_chunk_size', 32),
                                  checkpointed=self.checkpoint_dense, fps_fn=fps_fn,
                                  query_chunk_size=performance.get('query_chunk_size'),
                                  query_workers=performance.get('query_workers', 1),
                                  spatial_index=performance.get('spatial_index'),
                                  contact_query_backend=performance.get('contact_query_backend', 'numpy'),
                                  detailed_cpu_timing=performance.get('detailed_cpu_timing', False))
                                 if self.use_contact_grouping else None)
        if self.use_aux_heads:
            self.seal_head = mlp((136, 128, 1), output=True)
            self.wrench_head = mlp((396, 128, 1), output=True)
            self.collision_head = mlp((451, 128, 1), output=True)
        self.last_grouping_timings = {}
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _chunks(self, fn, *args):
        return chunk_apply(fn, *args, chunk_size=self.chunk_size,
                           checkpointed=self.checkpoint_dense and self.training)

    def encode_scene(self, batch_input, *, cpu_geometry=None):
        """Only observation inputs enter this method; labels have no encoder path."""
        if len(batch_input['voxel_features']):
            voxel_features = self.backbone(batch_input)
        else:
            # Sparse kernels need an active set. Empty observations still pass
            # through the dense heads, producing graph-connected empty outputs.
            voxel_features = batch_input['voxel_features'].new_empty((0, 256))
        local = voxel_features[batch_input['inverse_map']]
        voxel_heads = self.no_instance_voxel_heads and not self.use_instance_fusion
        frame = (batch_input['voxel_coords'][:, 0].long() if voxel_heads
                 else batch_input['batch_index'])
        gravity = batch_input['g_cam'][frame]
        stats = {}
        if self.instance_module is not None:
            stats = self.instance_module(local, batch_input['xyz_cam_m'], batch_input['batch_index'],
                                         batch_input['instance_id_input'],
                                         checkpointed=self.checkpoint_dense and self.training)
            def fuse(feature, g, xyz, slots):
                blocks = instance_blocks(stats, xyz, slots)
                return self.fusion(torch.cat((feature, *(x.to(feature) for x in blocks), g.to(feature)), -1))
            fused = self._chunks(fuse, local, gravity, batch_input['xyz_cam_m'], stats['point_instance_index'])
        else:
            def fuse(feature, g):
                zero = feature.new_zeros((len(feature), 134))
                return self.fusion(torch.cat((feature, zero, g.to(feature)), -1))
            fused = self._chunks(fuse, voxel_features if voxel_heads else local, gravity)
        fused_key = 'F_fused_voxel' if voxel_heads else 'F_fused'
        return {**batch_input, **stats, 'F_local': local, fused_key: fused,
                'random_group_roll': self.training, 'cpu_geometry': cpu_geometry}

    def predict_dense(self, encoded):
        voxel_heads = 'F_fused_voxel' in encoded
        fused = encoded['F_fused_voxel' if voxel_heads else 'F_fused']
        q0 = self._chunks(lambda x: self.q0_head(x).sigmoid(), fused).squeeze(-1)
        if voxel_heads:
            # Targets and all subsequent loss reductions remain in raw-point space.
            q0 = q0[encoded['inverse_map']]
        axis = self._chunks(self.axis_head, encoded['F_local'], encoded['normal_obs'],
                            encoded['normal_obs_valid'])
        return q0, axis

    @staticmethod
    def gather_fused_features(encoded, safe_anchors):
        """Gather associated raw-point features without expanding voxel fusion."""
        voxel_heads = 'F_fused_voxel' in encoded
        fused = encoded['F_fused_voxel' if voxel_heads else 'F_fused']
        if not len(fused):
            return fused.new_zeros((len(safe_anchors), 256))
        rows = encoded['inverse_map'][safe_anchors] if voxel_heads else safe_anchors
        return fused[rows]

    def sample(self, encoded, q0_all, sampler_mode='inference', targets=None, *, sampler_targets_cpu=None):
        return self.sampler(encoded, q0_all, mode=sampler_mode, targets=targets, epoch=self.epoch,
                            sampler_targets_cpu=sampler_targets_cpu)

    @staticmethod
    @torch.no_grad()
    def _anchors(encoded, points, instance, batch, supplied):
        count = len(points)
        anchor = torch.full((count,), -1, device=points.device, dtype=torch.long)
        if not len(encoded['xyz_cam_m']):
            return anchor
        if supplied is not None:
            supplied = supplied.long()
            bounded = (supplied >= 0) & (supplied < len(encoded['xyz_cam_m']))
            safe = supplied.clamp(0, max(0, len(encoded['xyz_cam_m']) - 1))
            okay = bounded & (encoded['batch_index'][safe] == batch) & (
                encoded['instance_id_input'][safe] == instance)
            anchor[okay] = supplied[okay]
        # Resolve only unassociated executed poses. Never replace their true centers.
        for i in torch.where(anchor < 0)[0].tolist():
            mask = (encoded['batch_index'] == batch[i]) & (encoded['instance_id_input'] == instance[i])
            rows = torch.where(mask)[0]
            if not len(rows) or instance[i] <= 0 or not torch.isfinite(points[i]).all():
                continue
            distance = (encoded['xyz_cam_m'][rows].float() - points[i].float()).square().sum(-1)
            nearest = distance.argmin()
            if distance[nearest] <= .005 ** 2:
                anchor[i] = rows[nearest]
        return anchor

    def score_at_pose(self, encoded, point_cam_m, normal_out_cam, instance_id,
                      proposal_q0=None, feature_point_index=None, pose_batch_index=None):
        if pose_batch_index is None:
            if len(encoded['g_cam']) != 1:
                raise ValueError('Packed explicit poses require pose_batch_index')
            pose_batch_index = torch.zeros_like(instance_id)
        batch = pose_batch_index.long()
        if ((batch < 0) | (batch >= len(encoded['g_cam']))).any():
            raise ValueError('Pose batch index outside encoded batch')
        # All quality consumers see detached, normalized directed physical axes.
        n = normal_out_cam.detach().float()
        norm = n.norm(dim=-1)
        valid_normal = torch.isfinite(n).all(-1) & (norm > 1e-6)
        n = torch.where(valid_normal[:, None], n / norm.clamp_min(1e-6)[:, None], torch.zeros_like(n))
        anchors = self._anchors(encoded, point_cam_m, instance_id, batch, feature_point_index)
        valid = (anchors >= 0) & (instance_id > 0) & valid_normal & torch.isfinite(point_cam_m).all(-1)
        safe = anchors.clamp_min(0)
        fused = self.gather_fused_features(encoded, safe)
        gravity = encoded['g_cam'][batch].to(fused)
        if self.contact_grouping is not None:
            zones = self.contact_grouping(encoded, point_cam_m, n, instance_id, batch, valid,
                                          random_roll=encoded.get('random_group_roll', self.training))
            zone_features = torch.cat([zones[name] for name in ZONE_NAMES] +
                                      [zones['lip_sector_log_counts']], -1).to(fused)
            self.last_grouping_timings = dict(self.contact_grouping.last_timings)
        else:
            zones = None
            zone_features = fused.new_zeros((len(fused), 328))
        h = self.pose_fusion(torch.cat((fused, zone_features, n.to(fused), gravity), -1))
        q = self.q_head(h).squeeze(-1).sigmoid()
        h = h * valid[:, None]
        result = {'h': h, 'pose_features': h, 'q': q * valid,
                  'pose_feature_valid': valid, 'feature_point_index': anchors}
        if zones is not None:
            result['zone_counts'] = zones['zone_counts']
            result['lip_sector_counts'] = zones['lip_sector_counts']
        if self.use_aux_heads:
            slots = (encoded['point_instance_index'][safe] if len(encoded['point_instance_index']) else
                     torch.zeros(len(point_cam_m), dtype=torch.long, device=safe.device))
            blocks = instance_blocks(encoded, point_cam_m, slots)
            blocks = tuple(torch.where(valid[:, None], x.to(h), torch.zeros_like(x).to(h)) for x in blocks)
            local_seed = (encoded['F_local'][safe] if len(encoded['F_local']) else
                          fused.new_zeros((len(fused), 256)))
            result['seal_aux'] = self.seal_head(torch.cat((zones['core'], zones['lip'],
                                      zones['lip_sector_log_counts'].to(h)), -1)).squeeze(-1).sigmoid() * valid
            result['wrench_aux'] = self.wrench_head(torch.cat((h, *blocks, n.to(h), gravity), -1)).squeeze(-1).sigmoid() * valid
            result['collision_free_logit'] = self.collision_head(torch.cat((zones['cup'], zones['mount'],
                                             zones['approach'], local_seed, n.to(h)), -1)).squeeze(-1) * valid
        if self.task_success_head is not None:
            result['task_success_logit'] = self.task_success_head(torch.cat(
                (h, n.to(h), (n * gravity.float()).sum(-1, keepdim=True).to(h)), -1)).squeeze(-1) * valid
        return result

    def forward(self, batch_input, *, sampler_mode='inference', targets=None, cpu_geometry=None,
                sampler_targets_cpu=None):
        encoded = self.encode_scene(batch_input, cpu_geometry=cpu_geometry)
        encoded['random_group_roll'] = self.training and sampler_mode != 'inference'
        q0, axis = self.predict_dense(encoded)
        seeds, valid = self.sample(encoded, q0, sampler_mode, targets, sampler_targets_cpu=sampler_targets_cpu)
        rows = seeds[valid]
        pose = self.score_at_pose(encoded, encoded['xyz_cam_m'][rows], axis[rows],
                                  encoded['instance_id_input'][rows], proposal_q0=q0[rows],
                                  feature_point_index=rows, pose_batch_index=encoded['batch_index'][rows])
        self.last_grouping_timings.update(self.sampler.last_timings)
        result = {'q0_all': q0, 'normal_out_all': axis,
                  'seed_point_index': seeds, 'seed_valid': valid}
        for name in ['q', 'pose_features', 'pose_feature_valid', 'task_success_logit',
                     'seal_aux', 'wrench_aux', 'collision_free_logit', 'zone_counts', 'lip_sector_counts']:
            if name in pose:
                value = pose[name]
                output = value.new_zeros((*valid.shape, *value.shape[1:]))
                output[valid] = value
                result[name] = output
        for name, value in [('point_cam_m', encoded['xyz_cam_m'][rows]), ('normal_out_cam', axis[rows])]:
            output = value.new_zeros((*valid.shape, 3))
            output[valid] = value
            result[name] = output
        return result

    def parameter_groups(self):
        if hasattr(self.backbone, 'parameter_groups'):
            groups = self.backbone.parameter_groups()
        else:
            groups = {'encoder': list(self.backbone.parameters()), 'decoder': []}
        groups = {name: list(params) for name, params in groups.items()}
        groups['decoder'].extend(self.fusion.parameters())
        groups['decoder'].extend(self.pose_fusion.parameters())
        if self.instance_module is not None:
            groups['decoder'].extend(self.instance_module.parameters())
        if self.contact_grouping is not None:
            groups['decoder'].extend(self.contact_grouping.parameters())
        feature_ids = {id(p) for params in groups.values() for p in params}
        groups['heads'] = [p for p in self.parameters() if id(p) not in feature_ids]
        return groups
