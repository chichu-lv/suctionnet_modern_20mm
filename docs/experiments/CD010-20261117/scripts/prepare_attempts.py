"""Convert reviewed unique published planning commands to target attempt rows.

Human frame review establishes the outcome association; no measured robot
endpoint is available. Frozen geometry, group splits and original data stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from infer_target import atomic_json, clean_metadata, file_record


def read_json(path):
    return clean_metadata(json.loads(Path(path).read_text(encoding='utf-8')))


def canonical(value):
    return json.dumps(clean_metadata(value), sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def exclusion(row):
    review = row['raw_review']
    if type(review['state']) is not bool:
        return 'not_attempted' if review['state'] == 'missed' else 'unmarked'
    comment = review.get('commit', '').strip()
    if any(s in comment for s in ('未执行', '未进行吸取', '漏放')):
        return 'review_says_unexecuted_or_missed'
    if any(s in comment for s in ('模糊', '未显示', '进一步', '可行性？')):
        return 'uncertain_outcome_or_association'
    if any(s in comment.lower() for s in ('rl', '设备故障', '运动受到限制')):
        return 'external_controller_or_equipment_issue'
    if any(s in comment for s in ('吸错东西', '另一个瓶子', '怎么会有这个东西')):
        return 'review_refers_to_other_object'
    arm = review.get('arm')
    if arm not in ('left', 'right'):
        return 'missing_reviewed_arm'
    if ('右臂' if arm == 'left' else '左臂') in comment:
        return 'review_arm_conflict'
    if not row.get('geometry_ready'):
        return 'geometry_unqualified'
    poses = row['planned_poses']
    if not poses:
        return 'no_published_pose'
    if len(poses) != 1 or len(row['planned_pose_cam']) != 1:
        return 'multiple_pose_ambiguous'
    p, pc = poses[0], row['planned_pose_cam'][0]
    if p.get('is_published_live_target') is not True:
        return 'not_published_target'
    if (p['object_id'], p['instance_id']) != (pc['object_id'], pc['instance_id']):
        return 'pose_instance_mismatch'
    if row['instance_masks']['tracker_to_local'].get(str(p['object_id'])) != p['instance_id']:
        return 'mask_instance_mismatch'
    assoc = row['pose_association']
    if (assoc.get('association') != 'exact_source_stamp_with_identity_checks' or
            any(assoc.get('identity_checks', {}).get(k) is not True
                for k in ('timestamp', 'session', 'encoder', 'object_ids'))):
        return 'source_identity_unverified'
    if str(assoc['frame_stamp_ns']) != str(row['source']['record']['stamp_ns']):
        return 'source_stamp_mismatch'
    return None


def pixel_anchor(row, prepared):
    """Only attach a source pixel when original projection, mask and depth agree."""
    obs, pose = row['observation'], row['planned_pose_cam'][0]
    uv = np.asarray(row['planned_poses'][0]['suction_uv'], float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        return None
    u, v = np.rint(uv).astype(int)
    h, w = obs['image_size_hw']
    if not (0 <= u < w and 0 <= v < h) or np.max(np.abs(uv - [u, v])) > .25:
        return None
    with Image.open(prepared / obs['instance_mask_path']) as im:
        if im.getpixel((int(u), int(v))) != pose['instance_id']:
            return None
    with Image.open(prepared / obs['depth_path']) as im:
        z = float(im.getpixel((int(u), int(v)))) * obs['depth_scale_m']
    if not .2 <= z <= 1.2:
        return None
    k = np.asarray(obs['intrinsics'])
    xyz = np.array([(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z])
    if np.linalg.norm(xyz - pose['point_cam_m']) > .005:
        return None
    return int(v*w+u)


def prepare(release, output, remote_release):
    prepared = release / 'prepared'
    inventory = prepared / 'geometry_observations.jsonl'
    rows = [clean_metadata(json.loads(line)) for line in inventory.read_text().splitlines() if line.strip()]
    output.mkdir(parents=True, exist_ok=True)
    reviews, review_sources = {}, {}
    for day in sorted({r['day'] for r in rows}):
        path = release / 'raw' / day / 'commit/result.json'
        reviews[day], review_sources[day] = read_json(path), file_record(path)
    accepted, excluded, counts, anchors = [], [], Counter(), Counter()
    group_split = {}
    for row in rows:
        sid, gid, split = row['sample_id'], row['group_id'], row['split']
        assert group_split.setdefault(gid, split) == split, 'Frozen group split conflict'
        assert reviews[row['day']].get(sid, {'state': None, 'commit': '-', 'arm': None}) == row['raw_review'], sid
        reason = exclusion(row)
        if reason:
            excluded.append(dict(sample_id=sid, group_id=gid, split=split, reason=reason,
                                 raw_review=row['raw_review']))
            continue
        depth = prepared / row['observation']['depth_path']
        raw_path = depth.with_name('suction_result.json')
        raw = read_json(raw_path)
        raw_points = raw.get('points', [])
        # Cross-check original export rather than trusting only the reduced inventory.
        assert len(raw_points) == 1 and raw_points[0]['is_published_live_target'] is True, sid
        assert raw_points[0]['object_id'] == row['planned_pose_cam'][0]['object_id'], sid
        assert raw['raw_trace'].get('live_best_published') is True, sid
        assert raw['raw_trace'].get('live_best_object_ids') == [raw_points[0]['object_id']], sid
        assert str(raw['frame_stamp_ns']) == str(row['source']['record']['stamp_ns']), sid
        pc = row['planned_pose_cam'][0]
        point, normal = np.asarray(pc['point_cam_m']), np.asarray(pc['normal_out_cam'])
        assert point.shape == normal.shape == (3,) and np.isfinite([point, normal]).all(), sid
        assert abs(np.linalg.norm(normal)-1) < 1e-5 and pc['instance_id'] > 0, sid
        obs = {k: copy.deepcopy(row['observation'][k]) for k in
               ('format', 'depth_path', 'depth_scale_m', 'intrinsics', 'instance_mask_path', 'g_cam')}
        obs.update(depth_min_m=.2, depth_max_m=1.2)
        for key in ('depth_path', 'instance_mask_path'):
            obs[key] = str((prepared / obs[key]).resolve())
        attempt = {k: copy.deepcopy(pc[k]) for k in ('point_cam_m', 'normal_out_cam', 'instance_id')}
        attempt.update(success=int(row['raw_review']['state']), label_valid=True,
                       pose_source='planned_pipeline_contact', actual_execution_verified=False)
        pixel = pixel_anchor(row, prepared)
        if pixel is not None:
            attempt['source_index'] = pixel
        anchors['explicit_source_pixel' if pixel is not None else 'same_instance_5mm_fallback'] += 1
        provenance = dict(
            association='inferred_unique_published_command_plus_frame_keyed_human_review',
            actual_execution_verified=False,
            endpoint_measured=False,
            arm=row['raw_review']['arm'],
            raw_review=row['raw_review'],
            original_sample_id=sid,
            object_id=pc['object_id'],
            review_path=f"raw/{row['day']}/commit/result.json",
            review_source=review_sources[row['day']], review_key=sid,
            suction_result_path=str(raw_path.resolve().relative_to(release)),
            suction_result_source=file_record(raw_path),
            frame_stamp_ns=str(raw['frame_stamp_ns']),
            request_id=raw.get('request_id'), identity_checks=raw['identity_checks'],
            mask_source=file_record(prepared / row['observation']['instance_mask_path']),
            depth_source=file_record(depth),
            geometry=row['geometry_provenance'],
            limitation='Human outcome is linked to the sole published plan by frame key; no independent robot acknowledgement, executed TCP/contact measurement or tool calibration.')
        accepted.append(dict(schema_version='suction20.target.v1', sample_id=sid+'/attempt0',
                             group_id=gid, split=split, label_kind='attempt_success',
                             success_definition='completed_pick_and_transfer', observation=obs,
                             attempt=attempt, provenance=provenance))
        counts[f'{split}:success' if attempt['success'] else f'{split}:failure'] += 1
    accepted.sort(key=lambda r: r['sample_id'])
    remote = copy.deepcopy(accepted)
    for row in remote:
        for key in ('depth_path', 'instance_mask_path'):
            row['observation'][key] = str(remote_release / Path(row['observation'][key]).relative_to(release))
    for name, records in (('manifest_attempt_success.local.jsonl', accepted),
                          ('manifest_attempt_success.jsonl', remote), ('excluded.jsonl', excluded)):
        path = output/name
        content = b''.join(canonical(row)+b'\n' for row in records)
        temp = path.with_suffix('.tmp')
        temp.write_bytes(content)
        temp.replace(path)
    summary = dict(schema_version='suction20.target.attempt_preparation.v1',
                   inventory_frames=len(rows), accepted_attempts=len(accepted), excluded_frames=len(excluded),
                   counts=dict(sorted(counts.items())), anchor_counts=dict(anchors),
                   exclusion_counts=dict(sorted(Counter(r['reason'] for r in excluded).items())),
                   source_inventory=file_record(inventory),
                   frozen_groups=file_record(prepared/'manifest_groups.json'),
                   source_reviews=review_sources,
                   outputs={p.name:file_record(p) for p in sorted(output.glob('*.jsonl'))},
                   success_definition='completed_pick_and_transfer',
                   association='inferred_unique_published_command_plus_frame_keyed_human_review',
                   measured_endpoint_available=False,
                   scope='User authorized finetuning on 2026-11-16; frozen release and test split unchanged.')
    atomic_json(output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-release', type=Path, default=Path('/home/ubuntu/data/datasets/pose/target-scenes-v1/20261116'))
    args = parser.parse_args()
    prepare(args.release.resolve(), args.output, args.remote_release)
