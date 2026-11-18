"""Exact-pose INST20 teacher: frozen R20 Seal/Wrench and target-only collision.

No table or other-instance obstacle is loaded or queried. The original dense
object scan and candidate-position centroid remain the authoritative physics.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .eval_r20 import seal_at_pose
from .full_data import (collision_mask_for_instance, load_seal, scene_frame0,
                        to_unit, transform_points, voxel_filter, wrench_scores_dual)
from .instance_geometry import array_digest

LABEL_SCHEMA = 'suction20.instance_inst20.v2'
QUALITY_CONTRACT = 'gt_INST20_pose_v2'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def empty_labels(points, normals, query_ids=None):
    points, normals = np.asarray(points, np.float64), np.asarray(normals, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError('points and normals must have shape [K,3]')
    k = len(points)
    ids = np.arange(k, dtype=np.int64) if query_ids is None else np.asarray(query_ids)
    if ids.shape != (k,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('query_ids must be an aligned integer vector')
    result = {'points': points.copy(), 'normal_out': normals.copy(),
              'query_id': ids.astype(np.int64), 'label_valid': np.zeros(k, bool),
              'schema_version': LABEL_SCHEMA, 'quality_contract': QUALITY_CONTRACT}
    for key in ('raw_seal', 'seal', 'wrench', 'wrench_annot', 'q_sw', 'q_inst'):
        result[key] = np.full(k, np.nan, np.float64)
    for key in ('collision_self_seal', 'collision_self_envelope', 'collision_self'):
        result[key] = np.zeros(k, bool)  # Placeholder only: label_valid controls meaning.
    return result


def validate_labels(labels, *, expected_points=None, expected_normals=None):
    """Reject stale pose caches and false-as-known collision placeholders."""
    if str(labels.get('schema_version')) != LABEL_SCHEMA or str(labels.get('quality_contract')) != QUALITY_CONTRACT:
        raise ValueError('instance label schema/quality contract mismatch')
    p, n = np.asarray(labels['points']), np.asarray(labels['normal_out'])
    valid = np.asarray(labels['label_valid'])
    if p.ndim != 2 or p.shape[1] != 3 or n.shape != p.shape or valid.shape != (len(p),) or valid.dtype != np.bool_:
        raise ValueError('invalid instance label shapes or validity dtype')
    if np.isinf(p).any() or np.isinf(n).any():
        raise ValueError('infinite label pose')
    if not np.isfinite(p[valid]).all() or not np.isfinite(n[valid]).all():
        raise ValueError('valid labels require finite poses')
    if np.any(abs(np.linalg.norm(n[valid], axis=-1) - 1) > 1e-4):
        raise ValueError('valid label axes must be unit vectors')
    for expected, actual in ((expected_points, p), (expected_normals, n)):
        if expected is not None and (np.shape(expected) != actual.shape or not np.array_equal(np.asarray(expected), actual, equal_nan=True)):
            raise ValueError('label cache pose differs from exact observation query')
    for key in ('seal', 'wrench', 'q_sw', 'q_inst'):
        a = np.asarray(labels[key])
        if a.shape != valid.shape or np.isinf(a).any() or not np.isfinite(a[valid]).all():
            raise ValueError(f'invalid label values: {key}')
        if np.any((a[valid] < 0) | (a[valid] > 1)):
            raise ValueError(f'label outside [0,1]: {key}')
    for key in ('collision_self_seal', 'collision_self_envelope', 'collision_self'):
        a = np.asarray(labels[key])
        if a.shape != valid.shape or a.dtype != np.bool_:
            raise ValueError(f'invalid collision field {key}')
    collision = labels['collision_self_seal'] | labels['collision_self_envelope']
    if not np.array_equal(np.asarray(labels['collision_self'])[valid], collision[valid]):
        raise ValueError('collision_self must OR both target-self components')
    if not np.allclose(np.asarray(labels['q_sw'])[valid], (labels['seal'] * labels['wrench'])[valid], atol=1e-6, rtol=1e-5):
        raise ValueError('q_sw must equal Seal times Wrench')
    if not np.allclose(np.asarray(labels['q_inst'])[valid], (labels['q_sw'] * ~collision)[valid], atol=1e-6, rtol=1e-5):
        raise ValueError('q_inst must include target self collision')
    return bool(valid.any())


class InstanceTeacher:
    """Cached, bounded-scene exact-pose physical teacher, never an input feature."""
    def __init__(self, dataset_root, label_root, camera='realsense', frozen_cfg=None,
                 *, scene_cache_size=2, chunk_points=262144):
        self.dataset_root, self.label_root = Path(dataset_root), Path(label_root)
        if frozen_cfg is None:
            frozen_cfg = json.loads((self.label_root / 'config_frozen.json').read_text())
        self.config = dict(frozen_cfg)
        required = ('camera', 'radius_m', 'seal_norm_min', 'seal_norm_max', 'wrench_k',
                    'default_baseline_mass_kg', 'gravity_m_s2', 'obstacle_voxel_m',
                    'self_height_threshold_m', 'envelope_segments_m', 'batch_candidates')
        missing = set(required) - set(self.config)
        if missing:
            raise ValueError(f'frozen teacher configuration missing {sorted(missing)}')
        cfg = self.config
        if cfg['camera'] != camera or cfg['radius_m'] != .020:
            raise ValueError('instance teacher requires matching camera and frozen R20 physics')
        scalars = [cfg[x] for x in required if x not in ('camera', 'envelope_segments_m')]
        segments = np.asarray(cfg['envelope_segments_m'], float)
        if not np.isfinite(scalars).all() or cfg['seal_norm_max'] <= cfg['seal_norm_min']:
            raise ValueError('invalid frozen physics/Seal normalization')
        if segments.ndim != 2 or segments.shape[1] != 3 or not len(segments) or not np.isfinite(segments).all():
            raise ValueError('invalid frozen target envelope')
        if min(cfg['wrench_k'], cfg['default_baseline_mass_kg'], cfg['gravity_m_s2'], cfg['obstacle_voxel_m']) <= 0:
            raise ValueError('physics scales must be positive')
        if scene_cache_size < 1 or chunk_points < 1 or cfg['batch_candidates'] < 1:
            raise ValueError('cache/chunk sizes must be positive')
        self.camera, self.scene_cache_size = camera, int(scene_cache_size)
        self.chunk_points = int(chunk_points)
        self._objects, self._scenes = {}, OrderedDict()
        self.identity = {'quality_contract': QUALITY_CONTRACT, 'schema_version': LABEL_SCHEMA,
                         'physics_sha256': json_digest(cfg),
                         'teacher_source_sha256': file_sha256(__file__),
                         'seal_source_sha256': file_sha256(Path(__file__).with_name('eval_r20.py')),
                         'producer_source_sha256': file_sha256(Path(__file__).with_name('full_data.py'))}

    def _scene(self, scene_id):
        scene_id = int(scene_id)
        if scene_id not in self._scenes:
            base = self.dataset_root / 'scenes' / f'scene_{scene_id:04d}' / self.camera
            table = np.load(base / 'cam0_wrt_table.npy', allow_pickle=False)
            poses = np.load(base / 'camera_poses.npy', allow_pickle=False)
            if table.shape != (4, 4) or poses.ndim != 3 or poses.shape[1:] != (4, 4):
                raise ValueError('invalid camera transforms')
            if not np.isfinite(table).all() or not np.isfinite(poses).all():
                raise ValueError('nonfinite camera transforms')
            recs = scene_frame0(self.dataset_root, scene_id, self.camera, table=table, camera_poses=poses)
            self._scenes[scene_id] = {'records': recs, 'world_camera': table @ poses,
                                     'targets': {}, 'transforms_sha256': array_digest(table, poses,
                                         np.asarray([r['T_world_obj'] for r in recs]))}
            if len(self._scenes) > self.scene_cache_size:
                self._scenes.popitem(last=False)
        self._scenes.move_to_end(scene_id)
        return self._scenes[scene_id]

    def _object(self, obj_id):
        if obj_id not in self._objects:
            dense_path = self.dataset_root / 'dense_point_clouds' / f'{obj_id:03d}.npz'
            seal_path = self.label_root / 'seal_r20' / 'poisson' / f'{obj_id:03d}_labels.npz'
            with np.load(dense_path, allow_pickle=False) as f:
                dense = np.asarray(f['points'], np.float64)
            if dense.ndim != 2 or dense.shape[1] != 3 or not len(dense) or not np.isfinite(dense).all():
                raise ValueError('original dense object geometry must be finite nonempty [M,3]')
            seal = load_seal(seal_path)
            if not len(seal['points']) or not np.isfinite(seal['points']).all():
                raise ValueError('original Seal candidate positions cannot be empty/nonfinite')
            self._objects[obj_id] = {'dense': dense, 'candidate_points': seal['points'],
                'obstacle': voxel_filter(dense, self.config['obstacle_voxel_m']),
                'dense_sha256': file_sha256(dense_path), 'seal_candidates_sha256': file_sha256(seal_path)}
        return self._objects[obj_id]

    def evaluate(self, scene_id, frame_id, instance_row, points, normals, query_ids=None):
        result = empty_labels(points, normals, query_ids)
        scene = self._scene(scene_id)
        if not 0 <= int(frame_id) < len(scene['world_camera']):
            raise ValueError('frame outside camera poses')
        if not 0 <= int(instance_row) < len(scene['records']):
            raise ValueError('GT instance row outside scene')
        rec = scene['records'][int(instance_row)]
        obj = self._object(rec['obj_id'])
        T_world_cam, T_world_obj = scene['world_camera'][int(frame_id)], rec['T_world_obj']
        if instance_row not in scene['targets']:
            # Match old producer arithmetic and 3mm voxelization before transform.
            obstacle = transform_points(obj['obstacle'], T_world_obj).astype(np.float32)
            scene['targets'][instance_row] = (obstacle, cKDTree(obstacle),
                transform_points(obj['candidate_points'], T_world_obj).mean(axis=0))
        obstacle, tree, center = scene['targets'][instance_row]
        p, n = result['points'], result['normal_out']
        valid = np.isfinite(p).all(1) & np.isfinite(n).all(1)
        valid &= abs(np.linalg.norm(n, axis=1) - 1) <= 1e-4
        ids = np.flatnonzero(valid)
        if len(ids):
            T_obj_cam = np.linalg.inv(T_world_obj) @ T_world_cam
            p_obj, n_obj = transform_points(p[ids], T_obj_cam), to_unit(n[ids] @ T_obj_cam[:3, :3].T)
            values = [seal_at_pose(obj['dense'], pi, ni, .020, self.chunk_points) for pi, ni in zip(p_obj, n_obj)]
            raw, seal_self = np.asarray([x[0] for x in values]), np.asarray([x[1] for x in values], bool)
            p_world, n_world = transform_points(p[ids], T_world_cam), to_unit(n[ids] @ T_world_cam[:3, :3].T)
            wrench = wrench_scores_dual(p_world, n_world, center, self.config['default_baseline_mass_kg'], self.config)
            envelope = collision_mask_for_instance(p_world, n_world, tree, obstacle,
                np.full(len(obstacle), int(instance_row), np.int32), int(instance_row),
                np.zeros(len(ids), bool), self.config)
            result['raw_seal'][ids] = raw
            result['seal'][ids] = np.clip((raw - self.config['seal_norm_min']) /
                (self.config['seal_norm_max'] - self.config['seal_norm_min']), 0, 1)
            result['wrench'][ids], result['wrench_annot'][ids] = wrench['wrench_eval'], wrench['wrench_annot']
            result['collision_self_seal'][ids], result['collision_self_envelope'][ids] = seal_self, envelope
            result['collision_self'][ids] = seal_self | envelope
            result['q_sw'][ids] = result['seal'][ids] * result['wrench'][ids]
            result['q_inst'][ids] = result['q_sw'][ids] * ~result['collision_self'][ids]
            result['label_valid'][ids] = True
        result['identity'] = dict(self.identity, dense_sha256=obj['dense_sha256'],
            seal_candidates_sha256=obj['seal_candidates_sha256'],
            transforms_sha256=scene['transforms_sha256'], query_pose_sha256=array_digest(p, n),
            scene_id=int(scene_id), frame_id=int(frame_id), instance_row=int(instance_row), obj_id=rec['obj_id'])
        validate_labels(result)
        return result
