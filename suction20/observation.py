"""Observation-only geometry shared by source and target readers (metres)."""
from dataclasses import dataclass

import numpy as np


def unit_vectors(vectors, valid=None):
    vectors = np.asarray(vectors, dtype=np.float32)
    magnitude = np.linalg.norm(vectors, axis=-1)
    ok = np.isfinite(vectors).all(axis=-1) & (magnitude > 1e-6)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    result = np.zeros_like(vectors)
    result[ok] = vectors[ok] / magnitude[ok, None]
    return result, ok


def organized_normals(point_map, keep):
    """The producer's finite differences, 10mm jump gate and camera-facing sign.

    Invalid rows are zeroed here: they remain scene context but never an axis.
    """
    point_map = np.asarray(point_map, np.float32)
    keep = np.asarray(keep, bool)
    if point_map.shape != keep.shape + (3,):
        raise ValueError("point map and depth mask shapes differ")
    normal = np.zeros_like(point_map)
    valid = np.zeros(keep.shape, bool)
    if min(keep.shape) >= 3:
        dy = point_map[2:, 1:-1] - point_map[:-2, 1:-1]
        dx = point_map[1:-1, 2:] - point_map[1:-1, :-2]
        raw = np.cross(dx, dy)
        mag = np.linalg.norm(raw, axis=-1)
        ok = (keep[1:-1, 1:-1] & keep[2:, 1:-1] & keep[:-2, 1:-1]
              & keep[1:-1, 2:] & keep[1:-1, :-2] & (mag > 1e-8)
              & (abs(dy[..., 2]) < .01) & (abs(dx[..., 2]) < .01))
        normal[1:-1, 1:-1] = raw / np.maximum(mag[..., None], 1e-8)
        valid[1:-1, 1:-1] = ok
    normal[np.sum(normal * -point_map, axis=-1) < 0] *= -1
    normal[~valid] = 0
    return normal, valid


def depth_geometry(depth_m, intrinsic, depth_min_m=.2, depth_max_m=1.2):
    """Return organized XYZ, normal, validity and depth mask from current depth."""
    z = np.asarray(depth_m, np.float64)
    K = np.asarray(intrinsic, np.float64)
    if z.ndim != 2 or K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Expected HxW depth and finite 3x3 camera intrinsics")
    keep = np.isfinite(z) & (z > 0) & (z >= depth_min_m) & (z <= depth_max_m)
    yy, xx = np.nonzero(keep)
    zz = z[yy, xx]
    point_map = np.zeros(z.shape + (3,), np.float32)
    point_map[yy, xx] = np.stack([(xx-K[0, 2])*zz/K[0, 0],
                                (yy-K[1, 2])*zz/K[1, 1], zz], axis=-1)
    normal, valid = organized_normals(point_map, keep)
    return point_map, normal, valid, keep


@dataclass(frozen=True)
class AugmentConfig:
    """Rigid augmentation about gravity; optional pre-normal depth corruption.

    Defaults are the public pretraining policy. Target/replay can explicitly set
    depth_noise_std_m=0 and dropout_max=0. No scale augmentation is supported.
    """
    yaw_max_deg: float = 180.0
    translation_max_m: float = .02
    depth_noise_std_m: float = .0005
    depth_noise_clip_m: float = .0015
    dropout_max: float = .05

    def __post_init__(self):
        if (min(self.yaw_max_deg, self.translation_max_m, self.depth_noise_std_m,
                self.depth_noise_clip_m, self.dropout_max) < 0 or self.dropout_max >= 1):
            raise ValueError("Invalid augmentation magnitude")


def rigid_transform(gravity, rng, config):
    g, ok = unit_vectors(np.asarray(gravity)[None])
    if not ok[0]:
        raise ValueError("Invalid gravity vector")
    g = g[0].astype(np.float64)
    angle = rng.uniform(-config.yaw_max_deg, config.yaw_max_deg) * np.pi / 180
    skew = np.array([[0, -g[2], g[1]], [g[2], 0, -g[0]], [-g[1], g[0], 0]])
    R = np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)
    # Independent table-plane axes, each within the specified 20mm bound.
    basis = np.eye(3)[np.argmin(abs(g))]
    u = np.cross(g, basis)
    u /= np.linalg.norm(u)
    v = np.cross(g, u)
    tx, ty = rng.uniform(-config.translation_max_m, config.translation_max_m, size=2)
    t = tx*u + ty*v
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply_rigid(sample, transform):
    """Apply an explicit rigid coordinate change consistently, in place."""
    T = np.asarray(transform, np.float32)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("Augmentation must be a finite 4x4 rigid transform")
    R, t = T[:3, :3], T[:3, 3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError("Augmentation must be a rigid rotation and translation")
    inputs, targets = sample['inputs'], sample['targets']
    inputs['xyz_cam_m'] = inputs['xyz_cam_m'] @ R.T + t
    inputs['normal_obs'] = inputs['normal_obs'] @ R.T
    inputs['g_cam'] = inputs['g_cam'] @ R.T
    if 'normal_label' in targets:
        targets['normal_label'] = targets['normal_label'] @ R.T
    old = sample['meta'].get('T_aug_from_cam', np.eye(4, dtype=np.float32))
    sample['meta']['T_aug_from_cam'] = T @ old
    return sample


def poses_to_original_camera(points, normals, meta):
    T = np.asarray(meta.get('T_aug_from_cam', np.eye(4)))
    return ((np.asarray(points)-T[:3, 3]) @ T[:3, :3],
            np.asarray(normals) @ T[:3, :3])
