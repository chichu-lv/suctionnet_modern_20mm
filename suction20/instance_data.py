"""Manifest-backed RGB-D instances, exact label caches, and recorded attempts.

Source training uses all scenes 0..99. Source test scenes 100..189 can be used
for checkpoint selection only with explicit test-selected result metadata.
No data or geometry is synthesized when an asset is missing.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .full_data import resolve_instance_ids, scene_frame0
from .instance_geometry import (NORMAL_SPEC, SAMPLER_SPEC, PRECISION_SPEC,
                                array_digest, prepare_instance)
from .instance_teacher import (InstanceTeacher, LABEL_SCHEMA, QUALITY_CONTRACT,
                               file_sha256, json_digest, validate_labels)

OBSERVATION_SCHEMA = 'instance_observation_v2'
MANIFEST_VERSION = 'suction20.instance_manifest.v3'
QUERY_FIELDS = ('query_points', 'query_normals', 'query_uv', 'query_ids',
                'query_source_ids', 'query_components', 'query_valid',
                'query_cell_ids', 'query_rounds')
LABEL_ALIASES = {'S': 'seal', 'W': 'wrench', 'Q_inst': 'q_inst', 'valid': 'label_valid'}
TEACHER_IDENTITY_FIELDS = ('quality_contract', 'schema_version', 'physics_sha256',
    'teacher_source_sha256', 'seal_source_sha256', 'producer_source_sha256')


def _teacher_identity(identity):
    """Shared frozen physics/code contract, excluding per-object pose assets."""
    if not isinstance(identity, dict) or any(key not in identity for key in TEACHER_IDENTITY_FIELDS):
        raise ValueError('supervised manifest requires complete frozen teacher identity')
    if identity['quality_contract'] != QUALITY_CONTRACT or identity['schema_version'] != LABEL_SCHEMA:
        raise ValueError('frozen teacher schema/quality mismatch')
    result = {key: identity[key] for key in TEACHER_IDENTITY_FIELDS}
    for key, value in result.items():
        if key.endswith('_sha256') and (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError(f'invalid frozen teacher hash: {key}')
    return result


def source_split(scene_id):
    if isinstance(scene_id, bool) or int(scene_id) != scene_id:
        raise ValueError('scene ID must be integer')
    scene_id = int(scene_id)
    if 0 <= scene_id < 100:
        return 'train', None
    if 100 <= scene_id < 190:
        return 'test', ('seen' if scene_id < 130 else 'similar' if scene_id < 160 else 'novel')
    raise ValueError('SuctionNet source scenes must be in [0,189]')


def source_scene_ids(split):
    if split == 'train':
        return tuple(range(100))
    if split == 'test':
        return tuple(range(100, 190))
    if split == 'all':
        return tuple(range(190))
    raise ValueError('source split must be train, test, or all; no held-out dev split')


def _path(root, value):
    value = Path(value)
    return value if value.is_absolute() else Path(root) / value


def _save_npz(path, value):
    """Atomic, pickle-free array archive with separate JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: item for key, item in value.items() if isinstance(item, np.ndarray)}
    metadata = {key: item for key, item in value.items() if key not in arrays}
    arrays['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return file_sha256(path)


def _read_npz(path, expected_sha256=None):
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f'cache content hash differs from manifest: {path}')
    with np.load(path, allow_pickle=False) as file:
        result = {key: file[key] for key in file.files if key != 'metadata_json'}
        if 'metadata_json' in file:
            meta = json.loads(str(file['metadata_json']))
            if set(meta) & set(result):
                raise ValueError('array/metadata key collision')
            result.update(meta)
    return result


def save_observation(path, observation):
    """Save a complete prepared instance, including its full query sequence."""
    if observation['normal_spec'] != NORMAL_SPEC or observation['sampler_spec'] != SAMPLER_SPEC:
        raise ValueError('only current v3 observations can be stored')
    return _save_npz(path, observation)


def load_observation(path, *, expected_sha256=None):
    """Load an unlabeled prepared cache; physical labels are never required."""
    obs = _read_npz(path, expected_sha256)
    if obs.get('normal_spec') != NORMAL_SPEC or obs.get('sampler_spec') != SAMPLER_SPEC:
        raise ValueError('observation normal/sampler version mismatch')
    if obs.get('precision_spec') != PRECISION_SPEC:
        raise ValueError('observation precision contract mismatch')
    k = len(obs['query_points'])
    if obs['query_points'].shape != (k, 3) or obs['query_normals'].shape != (k, 3):
        raise ValueError('malformed cached queries')
    if any(len(obs[key]) != k for key in QUERY_FIELDS):
        raise ValueError('unaligned query cache fields')
    query_identity = array_digest(obs['query_points'], obs['query_normals'], obs['query_source_ids'])
    if query_identity != obs.get('query_identity'):
        raise ValueError('query identity mismatch')
    if not np.isfinite(obs['query_points']).all() or not np.isfinite(obs['query_normals']).all():
        raise ValueError('cached canonical queries must be finite')
    return obs


def load_labels(path, *, expected_sha256=None, points=None, normals=None):
    labels = _read_npz(path, expected_sha256)
    validate_labels(labels, expected_points=points, expected_normals=normals)
    for alias, source in LABEL_ALIASES.items():
        labels[alias] = labels[source]
    return labels


class InstanceDataset:
    """One visible GT instance per item; failures remain in evaluation counts."""
    def __init__(self, manifest_path, split=None, training_queries=None,
                 require_labels=True, *, verify_hashes=True, cache_size=2):
        self.manifest = Path(manifest_path)
        self.root = self.manifest.parent
        self.manifest_digest = file_sha256(self.manifest)
        if split not in (None, 'train', 'test'):
            raise ValueError('source instance split must be train/test')
        if training_queries is not None and (isinstance(training_queries, bool) or
            not isinstance(training_queries, int) or not 1 <= training_queries <= 256):
            raise ValueError('training_queries must be an integer in [1,256]')
        self.training_queries, self.require_labels = training_queries, bool(require_labels)
        self.verify_hashes, self.cache_size = bool(verify_hashes), int(cache_size)
        if self.cache_size < 0:
            raise ValueError('cache_size cannot be negative')
        self.records, identifiers, identities = [], set(), set()
        teacher_identity, object_assets = None, {}
        for number, line in enumerate(self.manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('schema_version') != OBSERVATION_SCHEMA or row.get('manifest_version') != MANIFEST_VERSION:
                raise ValueError(f'manifest line {number}: schema mismatch')
            if row.get('sample_id') in identifiers:
                raise ValueError('duplicate source instance identity')
            identifiers.add(row['sample_id'])
            expected_split, subset = source_split(row['scene_id'])
            if row.get('split') != expected_split or row.get('test_subset') != subset:
                raise ValueError('manifest source scene split mismatch')
            if row.get('normal_spec') != NORMAL_SPEC or row.get('sampler_spec') != SAMPLER_SPEC:
                raise ValueError('manifest preprocessing contract mismatch')
            if row.get('quality_contract') != QUALITY_CONTRACT or row.get('precision_spec') != PRECISION_SPEC:
                raise ValueError('manifest quality/precision contract mismatch')
            if row.get('camera') not in ('realsense', 'kinect') or not 0 <= int(row['frame_id']) < 256:
                raise ValueError('manifest camera/frame is outside the source protocol')
            if int(row['instance_row']) < 0:
                raise ValueError('source instance_row must be nonnegative')
            identities.add((row.get('normal_spec'), row.get('sampler_spec'), row.get('quality_contract')))
            if split is None or row['split'] == split:
                if require_labels and not row.get('label_cache'):
                    raise ValueError('supervised dataset requires explicit label_cache; no inferred negatives')
                if require_labels:
                    current_teacher = _teacher_identity(row.get('label_identity'))
                    if teacher_identity is not None and current_teacher != teacher_identity:
                        raise ValueError('mixed frozen teacher physics/code contracts in one manifest')
                    teacher_identity = current_teacher
                    label_identity = row['label_identity']
                    for key in ('obj_id', 'scene_id', 'frame_id', 'instance_row'):
                        if label_identity.get(key) != row[key]:
                            raise ValueError(f'teacher identity differs from source instance: {key}')
                    object_key = str(int(row['obj_id']))
                    assets = {key: label_identity.get(key) for key in
                              ('dense_sha256', 'seal_candidates_sha256')}
                    for key, value in assets.items():
                        if (not isinstance(value, str) or len(value) != 64 or
                                any(c not in '0123456789abcdef' for c in value)):
                            raise ValueError(f'invalid frozen object asset hash: {key}')
                    if object_key in object_assets and object_assets[object_key] != assets:
                        raise ValueError('same object ID has mixed dense/Seal assets in one manifest')
                    object_assets[object_key] = assets
                self.records.append(row)
        if len(identities) > 1:
            raise ValueError('mixed label/observation contracts in one manifest')
        self.identity = dict(manifest_sha256=self.manifest_digest, normal_spec=NORMAL_SPEC,
            sampler_spec=SAMPLER_SPEC, precision_spec=PRECISION_SPEC,
            quality_contract=QUALITY_CONTRACT, split=split, training_queries=training_queries,
            teacher_identity=teacher_identity, object_assets=object_assets)
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return copy.deepcopy(self._cache[index])
        record = copy.deepcopy(self.records[index])
        obs = load_observation(_path(self.root, record['observation_npz']),
            expected_sha256=record['observation_sha256'] if self.verify_hashes else None)
        obs['sample_uid'], obs['instance_uid'] = record['sample_id'], record['instance_uid']
        full_count = len(obs['query_points'])
        count = min(full_count, self.training_queries) if self.training_queries is not None else full_count
        labels = None
        if self.require_labels and record.get('label_cache'):
            labels = load_labels(_path(self.root, record['label_cache']),
                expected_sha256=record['label_sha256'] if self.verify_hashes else None)
            if labels.get('identity') != record.get('label_identity'):
                raise ValueError('cached label identity differs from manifest frozen teacher/geometry')
            available = len(labels['points'])
            if available < count and self.require_labels:
                raise ValueError('requested query count exceeds label prefix; use training_queries or build complete labels')
            label_count = min(count, available)
            for key, value in list(labels.items()):
                if isinstance(value, np.ndarray) and value.ndim and len(value) == available:
                    labels[key] = value[:label_count].copy()
            validate_labels(labels, expected_points=obs['query_points'][:label_count],
                            expected_normals=obs['query_normals'][:label_count])
            if not np.array_equal(labels['query_id'], obs['query_ids'][:label_count]):
                raise ValueError('label query IDs differ from observation prefix')
        for key in QUERY_FIELDS:
            obs[key] = obs[key][:count].copy()
        obs['full_query_identity'] = obs['query_identity']
        obs['query_identity'] = array_digest(obs['query_points'], obs['query_normals'], obs['query_source_ids'])
        record['has_supervision'] = bool(labels is not None and labels['label_valid'].any() and obs['trainable'])
        result = {'observation': obs, 'labels': labels, 'record': record}
        if self.cache_size:
            self._cache[index] = copy.deepcopy(result)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result


def read_suctionnet_frame(dataset_root, scene_id, frame_id, camera='realsense', *,
                         instance_mask_root=None):
    """Read official RGB/depth/MAT with strict class-to-instance row resolution."""
    root = Path(dataset_root)
    base = root / 'scenes' / f'scene_{int(scene_id):04d}' / camera
    paths = {name: base / folder / f'{int(frame_id):04d}.{suffix}' for name, folder, suffix in
             [('rgb', 'rgb', 'png'), ('depth', 'depth', 'png'), ('label', 'label', 'png'), ('meta', 'meta', 'mat')]}
    with Image.open(paths['rgb']) as im:
        rgb = np.asarray(im.convert('RGB'), np.uint8).astype(np.float32) / 255.
    with Image.open(paths['depth']) as im:
        depth = np.asarray(im)
    with Image.open(paths['label']) as im:
        classes = np.asarray(im)
    mat = loadmat(paths['meta'])
    scale = float(np.asarray(mat['factor_depth']).reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0 or depth.ndim != 2 or classes.shape != depth.shape or rgb.shape[:2] != depth.shape:
        raise ValueError('invalid source depth scale/image shapes')
    if not np.issubdtype(classes.dtype, np.integer) or np.any(classes < 0):
        raise ValueError('source class labels must be nonnegative integers')
    K = np.asarray(mat['intrinsic_matrix'], np.float64)
    table_path, pose_path = base / 'cam0_wrt_table.npy', base / 'camera_poses.npy'
    table, poses = np.load(table_path, allow_pickle=False), np.load(pose_path, allow_pickle=False)
    if not 0 <= int(frame_id) < len(poses):
        raise ValueError('source frame outside camera pose sequence')
    records = scene_frame0(root, int(scene_id), camera, table=table, camera_poses=poses)
    frame_classes = np.asarray(mat['cls_indexes']).reshape(-1).astype(int) - 1
    if frame_classes.tolist() != [r['obj_id'] for r in records]:
        raise ValueError('frame object order differs from scene frame0')
    pixels = np.arange(depth.size, dtype=np.int64)
    cfg = {'camera': camera, 'instance_mask_root': None if instance_mask_root is None else str(instance_mask_root)}
    instance_ids = resolve_instance_ids(classes.ravel(), records, root, int(scene_id), int(frame_id), cfg,
                                       pixels, depth.shape).reshape(depth.shape)
    T_world_cam = table @ poses[int(frame_id)]
    g = T_world_cam[:3, :3].T @ np.array([0., 0., -1.])
    source_hashes = {key: file_sha256(path) for key, path in paths.items()}
    source_hashes.update(camera_poses=file_sha256(pose_path), cam0_wrt_table=file_sha256(table_path))
    if instance_mask_root is not None:
        source_hashes['instance_mask'] = file_sha256(Path(instance_mask_root) / f'scene_{scene_id:04d}' / camera / f'{frame_id:04d}.png')
    return dict(rgb=rgb, depth_m=depth.astype(np.float64) / scale, instance_ids=instance_ids,
                intrinsics=K, gravity=g, records=records, source_hashes=source_hashes,
                calibration_sha256=array_digest(K, T_world_cam, np.asarray(scale)))


def build_source_instances(dataset_root, label_root, output, *, split='all', camera='realsense',
                           scenes=None, frames=None, with_labels=True, train_queries=128,
                           instance_mask_root=None, depth_min_m=.2, depth_max_m=1.2):
    """Build a deterministic shard; no missing frames/objects silently disappear.

    Full defaults cover 100 training and 90 test scenes, 256 views each. A
    caller-supplied shard is labeled partial in the summary, never as full test.
    """
    allowed = source_scene_ids(split)
    selected_scenes = tuple(allowed if scenes is None else sorted(set(map(int, scenes))))
    selected_frames = tuple(range(256) if frames is None else sorted(set(map(int, frames))))
    if not set(selected_scenes) <= set(allowed) or not set(selected_frames) <= set(range(256)):
        raise ValueError('requested scenes/frames outside declared split')
    if not 1 <= int(train_queries) <= 256:
        raise ValueError('train_queries must be in [1,256]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    teacher = InstanceTeacher(dataset_root, label_root, camera) if with_labels else None
    rows, empty_frames, status_counts = [], [], {}
    for scene_id in selected_scenes:
        row_split, subset = source_split(scene_id)
        for frame_id in selected_frames:
            frame = read_suctionnet_frame(dataset_root, scene_id, frame_id, camera,
                                          instance_mask_root=instance_mask_root)
            visible_ids = [int(x) for x in np.unique(frame['instance_ids']) if x > 0]
            if not visible_ids:
                empty_frames.append([scene_id, frame_id])
            for instance_id in visible_ids:
                instance_row = instance_id - 1
                uid = f'{camera}_s{scene_id:04d}_f{frame_id:04d}_i{instance_row:03d}'
                obs = prepare_instance(frame['rgb'], frame['depth_m'], frame['instance_ids'] == instance_id,
                    frame['intrinsics'], frame['gravity'], depth_min_m=depth_min_m, depth_max_m=depth_max_m)
                obs.update(sample_uid=uid, instance_uid=uid)
                status = obs['diagnostics']['status']
                status_counts[status] = status_counts.get(status, 0) + 1
                obs_rel = Path('observations') / f'{uid}.npz'
                obs_path = output / obs_rel
                if obs_path.exists():
                    old = load_observation(obs_path)
                    if old.get('observation_identity') != obs['observation_identity'] or old['query_identity'] != obs['query_identity']:
                        raise ValueError(f'existing observation identity changed: {obs_path}; use a new output directory')
                    obs_sha = file_sha256(obs_path)
                else:
                    obs_sha = save_observation(obs_path, obs)
                label_rel, label_sha, label_identity, labelled_queries = None, None, None, 0
                if teacher is not None:
                    count = min(len(obs['query_points']), int(train_queries) if row_split == 'train' else 256)
                    label_rel = Path('labels') / f'{uid}.npz'
                    label_path = output / label_rel
                    if label_path.exists():
                        labels = load_labels(label_path, points=obs['query_points'][:count], normals=obs['query_normals'][:count])
                        if any(labels['identity'].get(k) != v for k, v in teacher.identity.items()):
                            raise ValueError('existing teacher cache identity changed; use a new output directory')
                        # Exact geometry assets are checked even when labels are reused.
                        scene = teacher._scene(scene_id)
                        obj = teacher._object(scene['records'][instance_row]['obj_id'])
                        for key, expected in [('dense_sha256', obj['dense_sha256']),
                                              ('seal_candidates_sha256', obj['seal_candidates_sha256']),
                                              ('transforms_sha256', scene['transforms_sha256'])]:
                            if labels['identity'].get(key) != expected:
                                raise ValueError('existing teacher geometry asset identity changed')
                        label_sha = file_sha256(label_path)
                    else:
                        labels = teacher.evaluate(scene_id, frame_id, instance_row,
                            obs['query_points'][:count], obs['query_normals'][:count], obs['query_ids'][:count])
                        label_sha = _save_npz(label_path, labels)
                    label_identity, labelled_queries = labels['identity'], count
                row = dict(schema_version=OBSERVATION_SCHEMA, manifest_version=MANIFEST_VERSION,
                    sample_id=uid, instance_uid=uid, group_id=f'scene_{scene_id:04d}',
                    capture_uid=f'{camera}_s{scene_id:04d}_f{frame_id:04d}',
                    split=row_split, test_subset=subset, scene_id=scene_id, frame_id=frame_id,
                    instance_row=instance_row, obj_id=frame['records'][instance_row]['obj_id'], camera=camera,
                    source_kind='suctionnet_rgbd', observation_npz=obs_rel.as_posix(),
                    observation_sha256=obs_sha, observation_identity=obs['observation_identity'],
                    query_cache=obs_rel.as_posix(), query_identity=obs['query_identity'],
                    label_cache=None if label_rel is None else label_rel.as_posix(), label_sha256=label_sha,
                    label_identity=label_identity, labelled_queries=labelled_queries,
                    normal_spec=NORMAL_SPEC, sampler_spec=SAMPLER_SPEC, precision_spec=PRECISION_SPEC,
                    quality_contract=QUALITY_CONTRACT, source_hashes=frame['source_hashes'],
                    calibration_sha256=frame['calibration_sha256'], support_count=len(obs['support_points']),
                    query_count=len(obs['query_points']), trainable=obs['trainable'], status=status)
                rows.append(row)
    manifest = output / f'manifest_{split}.jsonl'
    temporary = manifest.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows))
    os.replace(temporary, manifest)
    summary = dict(manifest_version=MANIFEST_VERSION, manifest=manifest.name,
        manifest_sha256=file_sha256(manifest), camera=camera, split=split,
        scene_ids=list(selected_scenes), frame_ids=list(selected_frames),
        complete_requested_split=selected_scenes == allowed and selected_frames == tuple(range(256)),
        frame_count=len(selected_scenes) * len(selected_frames), visible_instance_count=len(rows),
        empty_gt_frames=empty_frames, status_counts=status_counts, train_queries=int(train_queries),
        test_queries=256, labels_computed=bool(with_labels), quality_contract=QUALITY_CONTRACT,
        normal_spec=NORMAL_SPEC, sampler_spec=SAMPLER_SPEC, selection_policy='test_selected_if_used_for_best')
    manifest.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return manifest, summary


class InstanceAttemptDataset:
    """Adapt existing planned-action RGB-D records without replacing action p/n.

    Point-cloud-only records need their organized RGB-D and calibration assets;
    this adapter never invents images to make route B appear supported.
    """
    def __init__(self, manifest_path, split='train', *, cache_size=2):
        if split not in ('train', 'val', 'test'):
            raise ValueError('target split must be train/val/test')
        self.manifest, self.root = Path(manifest_path), Path(manifest_path).parent
        self.manifest_digest = file_sha256(self.manifest)
        groups, ids, observations = {}, set(), OrderedDict()
        definitions = set()
        for line in self.manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('schema_version') != 'suction20.target.v1' or row.get('label_kind') != 'attempt_success':
                raise ValueError('target records must use suction20.target.v1 / attempt_success')
            sample_id, group = row['sample_id'], row['group_id']
            if sample_id in ids:
                raise ValueError('duplicate attempt ID')
            ids.add(sample_id)
            if group in groups and groups[group] != row['split']:
                raise ValueError('target group crosses splits')
            groups[group] = row['split']
            obs, attempt = row['observation'], row['attempt']
            if obs.get('format') != 'rgbd' or not obs.get('rgb_path'):
                raise ValueError('instance routes require organized RGB-D including rgb_path; pointcloud-only target needs original assets')
            if not isinstance(attempt['instance_id'], int) or attempt['instance_id'] <= 0:
                raise ValueError('planned attempt requires a positive predicted instance_id')
            if not isinstance(attempt['label_valid'], bool) or (attempt['label_valid'] and attempt['success'] not in (0, 1)):
                raise ValueError('invalid attempt success/label validity')
            if row.get('success_definition'):
                definitions.add(row['success_definition'])
            key = (str(_path(self.root, obs['depth_path']).resolve()), attempt['instance_id'])
            if key not in observations:
                observations[key] = dict(observation=obs, rows=[], split=row['split'], group_id=group,
                    instance_id=attempt['instance_id'], sample_id=json_digest({'observation': obs, 'instance_id': attempt['instance_id']}))
            item = observations[key]
            if item['observation'] != obs or item['split'] != row['split'] or item['group_id'] != group:
                raise ValueError('repeated target observation has conflicting calibration/split/group')
            item['rows'].append(row)
        if len(definitions) != 1:
            raise ValueError('attempt dataset requires exactly one success_definition')
        self.success_definition = next(iter(definitions))
        self.records = [r for r in observations.values() if r['split'] == split]
        # The old target manifest does not require hashes for raw RGB-D assets.
        # Bind actual bytes once here so resume detects same-path replacements.
        asset_hashes = {}
        for record in self.records:
            for field in ('rgb_path', 'depth_path', 'instance_mask_path'):
                path = _path(self.root, record['observation'][field])
                key = str(path.resolve())
                if key not in asset_hashes:
                    asset_hashes[key] = file_sha256(path)
        # Paths are tracked for checking, but identity is content-based and
        # remains portable when a relative-manifest tree is moved intact.
        assets_identity = json_digest({record['sample_id']: {
            field: asset_hashes[str(_path(self.root, record['observation'][field]).resolve())]
            for field in ('rgb_path', 'depth_path', 'instance_mask_path')}
            for record in self.records})
        self.identity = dict(manifest_sha256=self.manifest_digest, split=split, normal_spec=NORMAL_SPEC,
            sampler_spec=SAMPLER_SPEC, precision_spec=PRECISION_SPEC, success_definition=self.success_definition,
            observation_assets_sha256=assets_identity, pose_source='planned_pipeline_contact')
        self._asset_hashes = asset_hashes
        self.cache_size, self._cache = int(cache_size), OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return copy.deepcopy(self._cache[index])
        record = copy.deepcopy(self.records[index])
        source = record['observation']
        for field in ('rgb_path', 'depth_path', 'instance_mask_path'):
            path = _path(self.root, source[field])
            if file_sha256(path) != self._asset_hashes[str(path.resolve())]:
                raise ValueError('target observation changed after dataset identity was frozen')
        with Image.open(_path(self.root, source['rgb_path'])) as im:
            rgb = np.asarray(im.convert('RGB'), np.uint8)
        with Image.open(_path(self.root, source['depth_path'])) as im:
            depth = np.asarray(im).astype(np.float64)
        with Image.open(_path(self.root, source['instance_mask_path'])) as im:
            instance_mask = np.asarray(im)
        scale = float(source['depth_scale_m'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('target depth_scale_m must be positive metres per stored unit')
        obs = prepare_instance(rgb, depth * scale, instance_mask == record['instance_id'],
            source['intrinsics'], source['g_cam'], depth_min_m=source.get('depth_min_m', .2),
            depth_max_m=source.get('depth_max_m', 1.2))
        obs.update(sample_uid=record['sample_id'], instance_uid=record['sample_id'])
        attempt_rows = [r['attempt'] for r in record['rows']]
        points = np.asarray([r['point_cam_m'] for r in attempt_rows], np.float64)
        normals = np.asarray([r['normal_out_cam'] for r in attempt_rows], np.float64)
        if points.shape != (len(attempt_rows), 3) or normals.shape != points.shape:
            raise ValueError('recorded poses must be length-3 vectors')
        lengths = np.linalg.norm(normals, axis=1)
        finite = np.isfinite(points).all(1) & np.isfinite(normals).all(1) & (lengths > 1e-12)
        # Unit normalization is explicit; never camera-flip or substitute observed axes.
        normalized = np.zeros_like(normals)
        normalized[finite] = normals[finite] / lengths[finite, None]
        support_valid = np.zeros(len(points), bool)
        support_distance = np.full(len(points), np.nan)
        for i in np.flatnonzero(finite):
            if len(obs['support_points']):
                distance = np.linalg.norm(obs['support_points'] - points[i], axis=1)
                support_distance[i] = distance.min()
                support_valid[i] = support_distance[i] <= .005
        attempts = dict(point_cam_m=points, normal_out_cam=normalized,
            success=np.asarray([r['success'] for r in attempt_rows], np.float32),
            label_valid=np.asarray([r['label_valid'] for r in attempt_rows], bool),
            pose_support_valid=support_valid, pose_valid=finite, support_distance_m=support_distance,
            attempt_ids=[r['sample_id'] for r in record['rows']],
            pose_source='planned_pipeline_contact',
            actual_execution_verified=all(bool(r.get('actual_execution_verified', False)) for r in record['rows']),
            actual_execution_verified_by_attempt=np.asarray([
                bool(r.get('actual_execution_verified', False)) for r in record['rows']], bool),
            original_normal_out_cam=normals.copy())
        result = dict(observation=obs, attempts=attempts, record=record)
        if self.cache_size > 0:
            self._cache[index] = copy.deepcopy(result)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result
