#!/usr/bin/env python3
"""Executable V2-Belt70 geometry contract; NOT a production adapter or teacher.

Only NumPy and the standard library are required. Geometry is in metres and
float64. A plane is [nx, ny, nz, d], with a unit normal toward the object side.
The camera origin must be above the support plane. Rays have z=1, not unit norm.
Obstacle tests use supplied complete scene points; they do not certify that a
point cloud captures every part of the physical scene. No Seal/Wrench is built.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EPS_M = 1e-12
MIN_HEIGHT_M = .002
MAX_DELTA_Z_M = .040
CROP_MARGIN_M = .080
SEGMENTS = ((-.002, .025, .022), (.025, .070, .028))


class ContractError(ValueError):
    """A declared geometry or data contract is not satisfied."""


def _plane(value):
    plane = np.asarray(value, dtype=np.float64)
    if (plane.shape != (4,) or not np.isfinite(plane).all()
            or abs(np.linalg.norm(plane[:3]) - 1.) > 1e-8):
        raise ContractError('plane_cam must be finite [nx,ny,nz,d] with unit normal')
    return plane


def _vectors(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ContractError(f'{name} must have shape [N,3]')
    return array


def _bool_vector(value, size, name):
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype != np.bool_:
        raise ContractError(f'{name} must be a Boolean [N] array')
    return array


def rays_z1(uv, K):
    """Return optical-Z rays K^-1 [u,v,1], preserving the z=1 convention."""
    uv, K = np.asarray(uv, dtype=np.float64), np.asarray(K, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ContractError('uv must be finite [N,2] in the registered camera image')
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0
            or K[1, 1] <= 0 or not np.array_equal(K[2], [0., 0., 1.])
            or abs(np.linalg.det(K)) < 1e-12):
        raise ContractError('K must be a nonsingular pinhole intrinsic matrix')
    return np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T


def backproject_z(uv, depth_z, K):
    """Return (points, valid); nonpositive/nonfinite depth has no 3-D point."""
    rays, depth_z = rays_z1(uv, K), np.asarray(depth_z, dtype=np.float64)
    if depth_z.shape != (len(rays),):
        raise ContractError('depth_z must align with uv')
    valid = np.isfinite(depth_z) & (depth_z > 0)
    points = np.full_like(rays, np.nan)
    points[valid] = depth_z[valid, None] * rays[valid]
    return points, valid


def clamp_hole_depth(depth_z, rays, plane_cam, min_height_m=MIN_HEIGHT_M):
    """Clamp only proposed hole depths along rays; return (depth, valid).

    A ray must point toward the plane and intersect its allowed-height boundary
    at positive optical depth. Parallel, reversed or malformed rays are invalid.
    Callers must NOT use this function to replace observed raw-valid depths.
    """
    rays, plane = _vectors(rays, 'rays'), _plane(plane_cam)
    depth = np.asarray(depth_z, dtype=np.float64)
    if depth.shape != (len(rays),) or not np.isfinite(min_height_m) or min_height_m < 0:
        raise ContractError('invalid hole depth shape or minimum height')
    slope = rays @ plane[:3]
    valid = (np.isfinite(rays).all(1) & (abs(rays[:, 2] - 1.) <= EPS_M)
             & np.isfinite(depth) & (depth > 0) & (slope < -EPS_M))
    limit = np.full(len(rays), np.nan)
    np.divide(min_height_m - plane[3], slope, out=limit, where=valid)
    valid &= np.isfinite(limit) & (limit > 0)
    output = np.full(len(rays), np.nan)
    output[valid] = np.minimum(depth[valid], limit[valid])
    return output, valid


def apply_hole_residual(raw_depth_z, raw_valid, fill_depth_z, rays, plane_cam,
                        delta_z=None, max_delta_z_m=MAX_DELTA_Z_M):
    """Keep every raw-valid value exactly; clamp initial and refined holes.

    Invalid hole rays/fills/residuals receive NaN plus False validity, never a
    fake zero-depth surface. Raw-valid pixels ignore fill and residual values.
    """
    rays = _vectors(rays, 'rays')
    count = len(rays)
    raw = np.asarray(raw_depth_z, dtype=np.float64)
    fill = np.asarray(fill_depth_z, dtype=np.float64)
    keep = _bool_vector(raw_valid, count, 'raw_valid')
    delta = np.zeros(count) if delta_z is None else np.asarray(delta_z, dtype=np.float64)
    if raw.shape != (count,) or fill.shape != (count,) or delta.shape != (count,):
        raise ContractError('depth and residual vectors must align with rays')
    if not np.isfinite(max_delta_z_m) or max_delta_z_m < 0:
        raise ContractError('residual bound must be finite and nonnegative')
    ray_valid = np.isfinite(rays).all(1) & (abs(rays[:, 2] - 1.) <= EPS_M)
    if not (np.isfinite(raw[keep]) & (raw[keep] > 0) & ray_valid[keep]).all():
        raise ContractError('raw-valid entries need positive depth and valid z=1 rays')
    filled, fill_valid = clamp_hole_depth(fill, rays, plane_cam)
    refined, refine_valid = clamp_hole_depth(filled + delta, rays, plane_cam)
    hole_valid = (~keep & fill_valid & refine_valid & np.isfinite(delta)
                  & (abs(delta) <= max_delta_z_m + EPS_M))
    fill_only_valid = ~keep & fill_valid
    filled[~fill_only_valid] = np.nan
    refined[~hole_valid] = np.nan
    filled[keep], refined[keep] = raw[keep], raw[keep]
    return dict(filled_depth_z=filled, refined_depth_z=refined,
                filled_points=filled[:, None] * rays,
                refined_points=refined[:, None] * rays,
                hole_valid=hole_valid, point_valid=keep | hole_valid,
                raw_valid=keep.copy())


def conservative_crop_bounds(raw_depth_z, raw_valid, fill_depth_z, rays, plane_cam,
                             max_delta_z_m=MAX_DELTA_Z_M, margin_m=CROP_MARGIN_M):
    """Bound all target raw points and both clamped hole residual endpoints.

    Expand that entire possible-target AABB by margin_m in all six directions.
    This includes off-axis lateral motion caused by optical-Z residuals. Every
    target hole must have valid endpoints; otherwise no coverage claim is made.
    """
    minimum_margin = float(np.hypot(.070, .028))
    if not np.isfinite(margin_m) or margin_m < minimum_margin:
        raise ContractError('crop margin cannot cover an arbitrary Belt70 orientation')
    base = apply_hole_residual(raw_depth_z, raw_valid, fill_depth_z, rays, plane_cam)
    if not len(base['raw_valid']) or not base['point_valid'].all():
        raise ContractError('cannot bound target holes with unknown surface support')
    points = [base['refined_points'][base['raw_valid']]]
    holes = ~base['raw_valid']
    for sign in (-1., 1.):
        result = apply_hole_residual(raw_depth_z, raw_valid, base['filled_depth_z'],
                                    rays, plane_cam,
                                    np.full(len(holes), sign * max_delta_z_m),
                                    max_delta_z_m=max_delta_z_m)
        if not result['point_valid'].all():
            raise ContractError('one residual endpoint has no valid positive ray depth')
        points.append(result['refined_points'][holes])
    bound_points = np.concatenate(points, axis=0)
    return bound_points.min(0) - margin_m, bound_points.max(0) + margin_m


def c70_at_pose(point, normal, obstacle_points, obstacle_is_target, plane_cam,
                *, scene_geometry_complete):
    """Four finite-envelope point collision terms plus analytic plane contact.

    Cup: rho<=22mm, -2<=z<=25mm. Mount: rho<=28mm, 25<z<=70mm.
    Self points count only at z>4mm. External points have no self exception.
    Plane contact uses each cylinder's minimum signed height (<=0 collides),
    conservatively including the infimum at mount's open lower endpoint.
    Unknown scene coverage is rejected; old unbounded Seal-self is never ORed.
    """
    if scene_geometry_complete is not True:
        raise ContractError('unknown scene geometry cannot be labelled collision-free')
    p, n = np.asarray(point, np.float64), np.asarray(normal, np.float64)
    obstacles, plane = _vectors(obstacle_points, 'obstacle_points'), _plane(plane_cam)
    own = _bool_vector(obstacle_is_target, len(obstacles), 'obstacle_is_target')
    if (p.shape != (3,) or n.shape != (3,) or not np.isfinite(p).all()
            or not np.isfinite(n).all() or abs(np.linalg.norm(n) - 1.) > 1e-8
            or not np.isfinite(obstacles).all()):
        raise ContractError('contact pose and obstacles must be finite; normal must be unit')
    relative = obstacles - p
    z = relative @ n
    rho = np.linalg.norm(relative - z[:, None] * n, axis=1)
    cup = (rho <= .022 + EPS_M) & (z >= -.002 - EPS_M) & (z <= .025 + EPS_M)
    mount = (rho <= .028 + EPS_M) & (z > .025 + EPS_M) & (z <= .070 + EPS_M)
    self_allowed = own & (z > .004 + EPS_M)
    output = dict(C_self_cup=bool(np.any(cup & self_allowed)),
                  C_self_mount=bool(np.any(mount & self_allowed)),
                  C_ext_cup=bool(np.any(cup & ~own)),
                  C_ext_mount=bool(np.any(mount & ~own)))
    cosine = float(np.clip(plane[:3] @ n, -1., 1.))
    base_height = float(plane[:3] @ p + plane[3])
    clearances = [base_height + min(low * cosine, high * cosine)
                  - radius * np.sqrt(max(0., 1. - cosine * cosine))
                  for low, high, radius in SEGMENTS]
    output.update(C_plane=bool(min(clearances) <= EPS_M),
                  plane_clearance_m=float(min(clearances)))
    output['C70'] = any(output[key] for key in
        ('C_self_cup', 'C_self_mount', 'C_ext_cup', 'C_ext_mount', 'C_plane'))
    return output


def validate_frame_schema(frame):
    """Minimal observation schema check; not a production release validator.

    frame uses depth_raw_z_m[H,W], raw_valid[H,W], target_mask[H,W], K[3,3],
    plane_cam[4], g_cam[3], depth_kind='optical_z_m'. Labels stay outside frame.
    """
    required = {'depth_raw_z_m', 'raw_valid', 'target_mask', 'K', 'plane_cam',
                'g_cam', 'depth_kind'}
    if not required <= frame.keys() or frame['depth_kind'] != 'optical_z_m':
        raise ContractError('missing observation fields or incorrect depth convention')
    # A whitelist also rejects supervision aliases such as depth_gt_m; relying
    # on a gt_ prefix blacklist would silently accept those production names.
    if set(frame) - required:
        raise ContractError('reference observations accept only the explicit input whitelist')
    depth = np.asarray(frame['depth_raw_z_m'], np.float64)
    if depth.ndim != 2 or not depth.size:
        raise ContractError('raw optical depth must be nonempty [H,W]')
    for key in ('raw_valid', 'target_mask'):
        mask = np.asarray(frame[key])
        if mask.shape != depth.shape or mask.dtype != np.bool_:
            raise ContractError(f'{key} must be a Boolean image aligned with depth')
    keep = np.asarray(frame['raw_valid'])
    if not (np.isfinite(depth[keep]) & (depth[keep] > 0)).all():
        raise ContractError('raw-valid depth must be finite and positive')
    rays_z1(np.array([[0., 0.]]), frame['K'])
    _plane(frame['plane_cam'])
    gravity = np.asarray(frame['g_cam'], np.float64)
    if (gravity.shape != (3,) or not np.isfinite(gravity).all()
            or abs(np.linalg.norm(gravity) - 1.) > 1e-8):
        raise ContractError('g_cam must be a finite unit vector')
    return True


def validate_split_records(records):
    """Reject duplicate IDs and declared physical provenance crossing splits.

    frame_id must be source-unique, include its scene/setup and omit camera:
    e.g. scene_0000/frame_0016. bottle_id denotes a physical bottle, not a class.
    Optional base_mesh_id binds a synthetic family when that policy is desired.
    This compares declared IDs only; it does not perform content deduplication.
    """
    required = ('sample_id', 'source', 'split', 'group_id', 'frame_id', 'camera_id')
    provenance = ('group_id', 'frame_id', 'capture_id', 'sequence_id',
                  'setup_id', 'bottle_id', 'base_mesh_id')
    samples, owners = set(), {}
    for record in records:
        if any(not isinstance(record.get(k), str) or not record[k].strip() for k in required):
            raise ContractError('record requires nonempty identity and provenance strings')
        if record['split'] not in ('train', 'dev', 'test'):
            raise ContractError('reference split must be train/dev/test')
        sample = (record['source'], record['sample_id'])
        if sample in samples:
            raise ContractError(f'duplicate sample: {sample}')
        samples.add(sample)
        for field in provenance:
            value = record.get(field)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f'{field} must be a nonempty string when declared')
            key = (record['source'], field, value)
            if owners.setdefault(key, record['split']) != record['split']:
                raise ContractError(f'cross-split provenance leakage: {field}={value}')
    return {'record_count': len(samples), 'provenance_key_count': len(owners)}


def self_test():
    """Bounded numeric counterexamples and edge cases, with no file data scan."""
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejected(name, function):
        try:
            function()
        except ContractError:
            checks.append(name)
        else:
            raise AssertionError(name)

    K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    uv = np.array([[820., 240.], [320., 240.], [570., 240.]])
    rays = rays_z1(uv, K)
    points, valid = backproject_z(uv, [.5, .8, .9], K)
    check('offaxis_optical_z_not_unit_ray', np.array_equal(points[0], [.5, 0., .5]))
    check('projection_roundtrip', np.allclose((points @ K.T)[:, :2] / points[:, 2, None], uv))
    plane = np.array([0., 0., -1., 1.])
    offaxis = apply_hole_residual([0.], [False], [.8], rays[:1], plane, [.04])
    check('offaxis_depth_residual_moves_x_and_z', np.allclose(offaxis['refined_points'][0], [.84, 0., .84]))
    raw, keep = np.array([.999, 0., 0.]), np.array([True, False, False])
    result = apply_hole_residual(raw, keep, [np.nan, 1.2, .97], rays, plane, [np.nan, .04, .04])
    check('zero_touch_even_raw_near_belt', result['refined_depth_z'][0] == raw[0])
    check('initial_fill_clamped', abs(result['filled_depth_z'][1] - .998) < EPS_M)
    check('refined_fill_clamped', np.all(result['refined_points'][~keep] @ plane[:3] + plane[3] >= .002 - EPS_M))
    check('raw_mask_immutable', np.array_equal(keep, [True, False, False]))
    _, invalid = clamp_hole_depth([1.], [[1., 0., 1.]], [0., 1., 0., 1.])
    check('parallel_plane_ray_is_invalid', not invalid[0])
    invalid_result = apply_hole_residual([0.], [False], [.8], [[0., 0., 1.]], plane, [.041])
    check('out_of_bound_residual_is_invalid', not invalid_result['point_valid'][0])
    lo, hi = conservative_crop_bounds(raw, keep, [np.nan, 1.2, .97], rays, plane)
    for delta in (-.04, 0., .04):
        test = apply_hole_residual(raw, keep, [np.nan, 1.2, .97], rays, plane, np.full(3, delta))
        check(f'crop_contains_residual_{delta}', bool(np.all(test['refined_points'] >= lo + .08 - EPS_M)
              and np.all(test['refined_points'] <= hi - .08 + EPS_M)))
    farthest = float(np.hypot(.070, .028))
    worst_normal = np.array([.070, 0., .028]) / farthest
    worst_tangent = np.array([.028, 0., -.070]) / farthest
    corner = .070 * worst_normal + .028 * worst_tangent
    check('75mm_is_insufficient', corner[0] > .075 and abs(worst_normal @ worst_tangent) < 1e-12)
    check('80mm_covers_arbitrary_envelope', farthest < .080)
    rejected('75mm_crop_rejected', lambda: conservative_crop_bounds(raw, keep, [np.nan, 1.2, .97], rays, plane, margin_m=.075))

    def collision(xyz, own=False):
        return c70_at_pose([0., 0., 0.], [0., 0., 1.], [xyz], [own],
                           [0., 0., 1., 1.], scene_geometry_complete=True)

    check('70mm_boundary_included', collision([.028, 0., .070])['C_ext_mount'])
    check('above_70mm_excluded', not collision([0., 0., .070 + 1e-9])['C70'])
    check('25mm_is_cup_radius_22', not collision([.027, 0., .025])['C70'])
    check('above_25mm_uses_mount_radius_28', collision([.027, 0., .025 + 1e-9])['C_ext_mount'])
    check('beyond_mount_radius_excluded', not collision([.028 + 1e-9, 0., .04])['C70'])
    check('self_4mm_boundary_exempt', not collision([0., 0., .004], True)['C70'])
    check('self_above_4mm_collides', collision([0., 0., .004 + 1e-9], True)['C_self_cup'])
    check('external_contact_has_no_self_exception', collision([0., 0., -.002])['C_ext_cup'])
    check('old_unbounded_self_not_orred', not collision([0., 0., .100], True)['C70'])
    empty, flags = np.empty((0, 3)), np.empty(0, dtype=bool)
    touching = c70_at_pose([0., 0., .002], [0., 0., 1.], empty, flags, [0., 0., 1., 0.], scene_geometry_complete=True)
    check('analytic_plane_contact', touching['C_plane'])
    tilted = c70_at_pose([0., 0., .027], [1., 0., 0.], empty, flags, [0., 0., 1., 0.], scene_geometry_complete=True)
    check('analytic_tilted_mount_plane_collision', tilted['C_plane'])
    rejected('unknown_colliders_rejected', lambda: c70_at_pose([0., 0., 0.], [0., 0., 1.], empty, flags, plane, scene_geometry_complete=False))
    frame = dict(depth_raw_z_m=np.ones((2, 2)), raw_valid=np.ones((2, 2), bool),
                 target_mask=np.ones((2, 2), bool), K=K, plane_cam=plane,
                 g_cam=[0., 0., 1.], depth_kind='optical_z_m')
    check('observation_schema', validate_frame_schema(frame))
    rejected('gt_not_an_input', lambda: validate_frame_schema({**frame, 'gt_depth': np.ones((2, 2))}))
    rejected('production_gt_alias_not_an_input', lambda: validate_frame_schema({**frame, 'depth_gt_m': np.ones((2, 2))}))
    a = dict(source='fixture', sample_id='a', split='train', group_id='g1',
             frame_id='g1/f0', camera_id='realsense', bottle_id='physical_1')
    b = dict(a, sample_id='b', camera_id='kinect')
    check('dual_camera_same_split', validate_split_records([a, b])['record_count'] == 2)
    rejected('cross_camera_frame_leakage', lambda: validate_split_records([a, dict(b, split='dev', group_id='g2', bottle_id='physical_2')]))
    rejected('bottle_leakage_across_groups', lambda: validate_split_records([a, dict(b, split='dev', group_id='g2', frame_id='g2/f1')]))
    rejected('group_leakage_across_frames', lambda: validate_split_records([a, dict(b, split='dev', frame_id='g1/f2', bottle_id='physical_2')]))
    rejected('duplicate_samples', lambda: validate_split_records([a, a]))
    return dict(status='passed', scope='reference_contract_only_not_production_teacher',
                units='metres', checks_passed=len(checks), checks=checks,
                envelope_max_distance_m=farthest, crop_margin_m=CROP_MARGIN_M,
                limitations=['No dataset adapter, fitting, sampler, Seal or Wrench implementation.',
                    'Point-cloud collisions require declared complete scene geometry.',
                    'Split checks cover declared identity relationships, not file-content duplicates.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.add_argument('--output', type=Path, help='Optional JSON report path')
    args = parser.parse_args(argv)
    report = self_test()
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
