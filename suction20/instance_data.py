"""Manifest-backed RGB-D instances, exact label caches, and recorded attempts.

Source training uses all scenes 0..99. Source test scenes 100..189 can be used
for checkpoint selection only with explicit test-selected result metadata.
No data or geometry is synthesized when an asset is missing.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
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
                                BOUNDARY_ROUNDOFF_M, observation_descriptor,
                                query_descriptor, prepare_instance)
from .instance_teacher import InstanceTeacher, LABEL_SCHEMA, QUALITY_CONTRACT, validate_labels
from .metadata import without_legacy_checksums

OBSERVATION_SCHEMA = 'instance_observation_v2'
MANIFEST_VERSION = 'suction20.instance_manifest.v4'
READABLE_MANIFEST_VERSIONS = (MANIFEST_VERSION, 'suction20.instance_manifest.v3')
# These full-camera arrays have no consumer after source observation validation
# and before either route's compact training payload. Retain the organized XYZ
# and normal maps used to check canonical query/pixel identity.
_COMPACT_SOURCE_UNUSED_ARRAYS = frozenset(('depth_m', 'depth_valid', 'eligible_mask', 'component_map'))
QUERY_FIELDS = ('query_points', 'query_normals', 'query_uv', 'query_ids',
                'query_source_ids', 'query_components', 'query_valid',
                'query_cell_ids', 'query_rounds')
LABEL_ALIASES = {'S': 'seal', 'W': 'wrench', 'Q_inst': 'q_inst', 'valid': 'label_valid'}
LABEL_IDENTITY_FIELDS = ('quality_contract', 'schema_version', 'teacher_spec', 'physics',
    'scene_id', 'frame_id', 'instance_row', 'obj_id', 'dense_path', 'seal_candidates_path',
    'T_world_cam', 'T_world_obj')
MANIFEST_FIELDS = ('schema_version', 'manifest_version', 'sample_id', 'instance_uid', 'group_id',
    'capture_uid', 'split', 'test_subset', 'scene_id', 'frame_id', 'instance_row', 'obj_id', 'camera',
    'source_kind', 'observation_npz', 'observation_identity', 'query_cache', 'query_identity',
    'label_cache', 'label_identity', 'labelled_queries', 'normal_spec', 'sampler_spec',
    'precision_spec', 'quality_contract', 'source_paths', 'calibration', 'support_count',
    'query_count', 'trainable', 'status', 'min_support_points')


def _label_identity(identity):
    """Read declared semantics; discard obsolete, unused integrity metadata."""
    if not isinstance(identity, dict):
        raise ValueError('supervised manifest requires a declared teacher identity')
    if identity.get('quality_contract') != QUALITY_CONTRACT or identity.get('schema_version') != LABEL_SCHEMA:
        raise ValueError('frozen teacher schema/quality mismatch')
    result = without_legacy_checksums({key: identity[key]
                                     for key in LABEL_IDENTITY_FIELDS if key in identity})
    result.setdefault('teacher_spec', None)
    result.setdefault('physics', None)
    if result['physics'] is not None and not isinstance(result['physics'], dict):
        raise ValueError('declared teacher physics must be a mapping or unknown')
    return result


def _teacher_identity(identity):
    """Small explicit physics/version fields; unknown legacy provenance is None."""
    identity = _label_identity(identity)
    return {key: identity.get(key) for key in ('quality_contract', 'schema_version', 'teacher_spec', 'physics')}


def _compatible_teacher(left, right):
    for key in ('quality_contract', 'schema_version', 'teacher_spec', 'physics'):
        if left.get(key) is not None and right.get(key) is not None and left[key] != right[key]:
            raise ValueError(f'mixed declared teacher contracts: {key}')


def _same_label_identity(left, right):
    left, right = _label_identity(left), _label_identity(right)
    _compatible_teacher(left, right)
    for key in ('scene_id', 'frame_id', 'instance_row', 'obj_id', 'T_world_cam', 'T_world_obj'):
        if key in left and key in right and left[key] != right[key]:
            raise ValueError(f'cached label identity differs from manifest: {key}')


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
    value = without_legacy_checksums(value)
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
    return path


def _read_npz(path, *, exclude_arrays=()):
    exclude_arrays = frozenset(exclude_arrays)
    if 'metadata_json' in exclude_arrays:
        raise ValueError('NPZ metadata cannot be excluded')
    arrays = None
    cache_root = os.environ.get('SUCTION_INSTANCE_SELECTION_CACHE')
    if cache_root:
        from .instance_selection_cache import read_cached_arrays
        cached = read_cached_arrays(path, cache_root, exclude_arrays=exclude_arrays)
        if cached is not None:
            arrays, members = cached
    if arrays is None:
        with np.load(path, allow_pickle=False) as file:
            members = tuple(file.files)
            arrays = {key: file[key] for key in members if key not in exclude_arrays}
    encoded = arrays.pop('metadata_json', None)
    result = without_legacy_checksums(arrays)
    if encoded is not None:
        meta = without_legacy_checksums(json.loads(str(encoded)))
        # A skipped array must not conceal an ambiguous metadata declaration.
        array_names = without_legacy_checksums(dict.fromkeys(
            key for key in members if key != 'metadata_json'))
        if set(meta) & set(array_names):
            raise ValueError('array/metadata key collision')
        result.update(meta)
    return without_legacy_checksums(result)


def save_observation(path, observation):
    """Save a complete prepared instance, including its full query sequence."""
    if observation['normal_spec'] != NORMAL_SPEC or observation['sampler_spec'] != SAMPLER_SPEC:
        raise ValueError('only current v3 observations can be stored')
    return _save_npz(path, observation)


def load_observation(path):
    """Load once, then validate geometry in memory; never scan archive bytes."""
    return _load_observation(path)


def _load_observation(path, *, exclude_arrays=()):
    """Keep identical canonical checks for full and compact-source readers."""
    obs = _read_npz(path, exclude_arrays=exclude_arrays)
    if obs.get('normal_spec') != NORMAL_SPEC or obs.get('sampler_spec') != SAMPLER_SPEC:
        raise ValueError('observation normal/sampler version mismatch')
    if obs.get('precision_spec') != PRECISION_SPEC:
        raise ValueError('observation precision contract mismatch')
    k = len(obs['query_points'])
    if k > 256 or obs['query_points'].shape != (k, 3) or obs['query_normals'].shape != (k, 3):
        raise ValueError('malformed cached queries')
    if any(len(obs[key]) != k for key in QUERY_FIELDS):
        raise ValueError('unaligned query cache fields')
    if not np.isfinite(obs['query_points']).all() or not np.isfinite(obs['query_normals']).all():
        raise ValueError('cached canonical queries must be finite')
    source_ids = np.asarray(obs['query_source_ids'])
    image = np.asarray(obs['xyz_map'])
    if (image.ndim != 3 or image.shape[-1] != 3 or source_ids.shape != (k,)
            or source_ids.dtype.kind not in 'iu' or np.any(source_ids < 0)
            or np.any(source_ids >= image.shape[0] * image.shape[1])
            or len(np.unique(source_ids)) != k):
        raise ValueError('malformed cached source pixel IDs')
    if not np.array_equal(obs['query_points'], image.reshape(-1, 3)[source_ids]):
        raise ValueError('query points differ from cached raw source pixels')
    normal_map = np.asarray(obs['normal_map'])
    if normal_map.shape != image.shape or not np.array_equal(obs['query_normals'], normal_map.reshape(-1, 3)[source_ids]):
        raise ValueError('query normals differ from cached organized normals')
    if np.any(abs(np.linalg.norm(obs['query_normals'], axis=1) - 1) > 1e-4):
        raise ValueError('cached canonical normals must be unit vectors')
    if obs['query_valid'].dtype != np.bool_ or not np.all(obs['query_valid']):
        raise ValueError('cached canonical queries must be eligible')
    if not np.array_equal(obs['query_ids'], np.arange(k)):
        raise ValueError('canonical query IDs must preserve their original prefix order')
    obs.setdefault('min_support_points', int(obs.get('diagnostics', {}).get('min_support', 100)))
    obs['query_identity'] = query_descriptor(source_ids)
    obs['observation_identity'] = observation_descriptor(obs)
    return obs


def load_labels(path, *, points=None, normals=None):
    labels = _read_npz(path)
    validate_labels(labels, expected_points=points, expected_normals=normals)
    if 'identity' in labels:
        labels['identity'] = _label_identity(labels['identity'])
    for alias, source in LABEL_ALIASES.items():
        labels[alias] = labels[source]
    return labels


class InstanceDataset:
    """One visible GT instance per item; failures remain in evaluation counts."""
    def __init__(self, manifest_path, split=None, training_queries=None,
                 require_labels=True, *, cache_size=2):
        self.manifest = Path(manifest_path)
        self.root = self.manifest.parent
        if split not in (None, 'train', 'test'):
            raise ValueError('source instance split must be train/test')
        if training_queries is not None and (isinstance(training_queries, bool) or
            not isinstance(training_queries, int) or not 1 <= training_queries <= 256):
            raise ValueError('training_queries must be an integer in [1,256]')
        self.training_queries, self.require_labels = training_queries, bool(require_labels)
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError('cache_size cannot be negative')
        self.records, identifiers, identities = [], set(), set()
        teacher_identity, object_assets, known_teacher = None, {}, {}
        for number, line in enumerate(self.manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            decoded = without_legacy_checksums(json.loads(line))
            row = {key: decoded[key] for key in MANIFEST_FIELDS if key in decoded}
            if row.get('schema_version') != OBSERVATION_SCHEMA or row.get('manifest_version') not in READABLE_MANIFEST_VERSIONS:
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
                    row['label_identity'] = _label_identity(row.get('label_identity'))
                    current_teacher = _teacher_identity(row['label_identity'])
                    _compatible_teacher(known_teacher, current_teacher)
                    for key, value in current_teacher.items():
                        if known_teacher.get(key) is None and value is not None:
                            known_teacher[key] = copy.deepcopy(value)
                    if teacher_identity is not None:
                        _compatible_teacher(teacher_identity, current_teacher)
                        # One unknown legacy row means the shared provenance is
                        # not completely known; do not infer missing physics.
                        for key in ('teacher_spec', 'physics'):
                            if teacher_identity.get(key) is None or current_teacher.get(key) is None:
                                current_teacher[key] = None
                    teacher_identity = current_teacher
                    label_identity = row['label_identity']
                    for key in ('obj_id', 'scene_id', 'frame_id', 'instance_row'):
                        if label_identity.get(key) != row[key]:
                            raise ValueError(f'teacher identity differs from source instance: {key}')
                    object_key = str(int(row['obj_id']))
                    assets = {key: label_identity.get(key) for key in ('dense_path', 'seal_candidates_path')}
                    object_assets.setdefault(object_key, assets)
                self.records.append(row)
        if len(identities) > 1:
            raise ValueError('mixed label/observation contracts in one manifest')
        self.identity = dict(manifest_path=str(self.manifest.resolve()), normal_spec=NORMAL_SPEC,
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
        result = self.load_uncached(index)
        if self.cache_size:
            self._cache[index] = copy.deepcopy(result)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def load_uncached(self, index, *, _compact_source=False):
        record = copy.deepcopy(self.records[index])
        excluded = _COMPACT_SOURCE_UNUSED_ARRAYS if _compact_source and record['split'] == 'train' else ()
        obs = _load_observation(_path(self.root, record['observation_npz']), exclude_arrays=excluded)
        obs['sample_uid'], obs['instance_uid'] = record['sample_id'], record['instance_uid']
        obs['observation_identity']['instance_uid'] = record['instance_uid']
        full_count = len(obs['query_points'])
        count = min(full_count, self.training_queries) if self.training_queries is not None else full_count
        labels = None
        if self.require_labels and record.get('label_cache'):
            labels = load_labels(_path(self.root, record['label_cache']))
            _same_label_identity(labels.get('identity'), record.get('label_identity'))
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
        obs['query_identity'] = query_descriptor(obs['query_source_ids'])
        record['has_supervision'] = bool(labels is not None and labels['label_valid'].any())
        result = {'observation': obs, 'labels': labels, 'record': record}
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
    source_paths = {key: str(path.resolve()) for key, path in paths.items()}
    source_paths.update(camera_poses=str(pose_path.resolve()), cam0_wrt_table=str(table_path.resolve()))
    if instance_mask_root is not None:
        source_paths['instance_mask'] = str((Path(instance_mask_root) / f'scene_{scene_id:04d}' / camera / f'{frame_id:04d}.png').resolve())
    return dict(rgb=rgb, depth_m=depth.astype(np.float64) / scale, instance_ids=instance_ids,
                intrinsics=K, gravity=g, records=records, source_paths=source_paths,
                calibration={'intrinsics': K.tolist(), 'T_world_cam': T_world_cam.tolist(),
                             'depth_scale_m': 1. / scale})


def _source_selection(split, scenes, frames):
    allowed = source_scene_ids(split)
    selected_scenes = tuple(allowed if scenes is None else sorted(set(map(int, scenes))))
    selected_frames = tuple(range(256) if frames is None else sorted(set(map(int, frames))))
    if (not selected_scenes or not selected_frames or not set(selected_scenes) <= set(allowed)
            or not set(selected_frames) <= set(range(256))):
        raise ValueError('requested scenes/frames must be nonempty and inside the declared split')
    return allowed, selected_scenes, selected_frames


def _selection_components(name, values):
    """Encode actual IDs as readable ranges, splitting long path components."""
    runs = []
    first = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append(f'{first:04d}' if first == previous else f'{first:04d}-{previous:04d}')
            first = value
        previous = value
    runs.append(f'{first:04d}' if first == previous else f'{first:04d}-{previous:04d}')
    components, current = [], name
    for run in runs:
        if len(current) + len(run) + 1 > 180:
            components.append(current)
            current = name + '_more'
        current += '_' + run
    components.append(current)
    return components


def source_build_output(output, *, split='all', camera='realsense', scenes=None, frames=None):
    """Give each explicit scene/frame request its own deterministic namespace.

    A complete default build keeps its documented root manifest. Explicit
    selections never publish that root manifest, even if they list all IDs.
    Path names encode IDs directly and do not identify file contents.
    """
    if camera not in ('realsense', 'kinect'):
        raise ValueError('source camera must be realsense or kinect')
    _, selected_scenes, selected_frames = _source_selection(split, scenes, frames)
    output = Path(output)
    if scenes is None and frames is None:
        return output
    return output.joinpath('shards', camera, split,
        *_selection_components('scenes', selected_scenes),
        *_selection_components('frames', selected_frames))


@contextmanager
def source_output_lock(output):
    """One writer per publication directory; OS releases the lock on exit/crash."""
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Keep the inode after unlocking: unlinking permits two different lock
    # inodes to be held for the same directory during a writer race.
    with (output / '.instance-build.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'instance cache output is already being written: {output}') from exc
        try:
            stream.seek(0); stream.truncate()
            stream.write(json.dumps({'pid': os.getpid()})); stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_text(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish_source_manifest(manifest, rows, summary):
    """Publish only after complete construction, without losing old coverage."""
    summary_path = manifest.with_suffix('.summary.json')
    if manifest.exists():
        if not summary_path.is_file():
            raise ValueError('existing manifest has no coverage summary; use a new output directory')
        previous = json.loads(summary_path.read_text())
        if any(previous.get(key) != summary.get(key) for key in ('camera', 'split')):
            raise ValueError('existing manifest belongs to another camera/split; use a new output directory')
        def frame_set(value):
            pairs = value.get('frame_pairs')
            return ({(int(s), int(f)) for s in value['scene_ids'] for f in value['frame_ids']}
                    if pairs is None else {tuple(pair) for pair in pairs})
        if not frame_set(previous) <= frame_set(summary):
            raise ValueError('refusing to replace an existing manifest with reduced/different frame coverage; '
                             'merge all shards or use a new output directory')
    # The manifest is published last; an interrupted writer never announces a
    # new manifest before its coverage summary and completed NPZ assets exist.
    _atomic_text(summary_path, json.dumps(summary, indent=2, allow_nan=False) + '\n')
    _atomic_text(manifest, ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows))


def build_source_instances(dataset_root, label_root, output, *, split='all', camera='realsense',
                           scenes=None, frames=None, with_labels=True, train_queries=128,
                           instance_mask_root=None, depth_min_m=.2, depth_max_m=1.2, min_support_points=100):
    """Build into a uniquely scoped, exclusively owned publication directory."""
    _, selected_scenes, selected_frames = _source_selection(split, scenes, frames)
    destination = source_build_output(output, split=split, camera=camera,
        scenes=selected_scenes if scenes is not None else None,
        frames=selected_frames if frames is not None else None)
    with source_output_lock(destination):
        return _build_source_instances(dataset_root, label_root, destination, split=split, camera=camera,
            scenes=selected_scenes, frames=selected_frames, with_labels=with_labels, train_queries=train_queries,
            instance_mask_root=instance_mask_root, depth_min_m=depth_min_m,
            depth_max_m=depth_max_m, min_support_points=min_support_points)


def _build_source_instances(dataset_root, label_root, output, *, split='all', camera='realsense',
                           scenes=None, frames=None, with_labels=True, train_queries=128,
                           instance_mask_root=None, depth_min_m=.2, depth_max_m=1.2, min_support_points=100):
    """Build a deterministic shard; no missing frames/objects silently disappear.

    Full defaults cover 100 training and 90 test scenes, 256 views each. A
    caller-supplied shard is labeled partial in the summary, never as full test.
    """
    allowed, selected_scenes, selected_frames = _source_selection(split, scenes, frames)
    if not 1 <= int(train_queries) <= 256:
        raise ValueError('train_queries must be in [1,256]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    existing_summary = (output / f'manifest_{split}.summary.json')
    if existing_summary.is_file():
        previous = json.loads(existing_summary.read_text())
        if previous.get('camera') != camera or previous.get('split') != split:
            raise ValueError('existing manifest belongs to another camera/split; use a new output directory')
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
                    frame['intrinsics'], frame['gravity'], depth_min_m=depth_min_m, depth_max_m=depth_max_m,
                    min_support=min_support_points)
                obs.update(sample_uid=uid, instance_uid=uid)
                obs['observation_identity']['instance_uid'] = uid
                status = obs['diagnostics']['status']
                status_counts[status] = status_counts.get(status, 0) + 1
                obs_rel = Path('observations') / f'{uid}.npz'
                obs_path = output / obs_rel
                if obs_path.exists():
                    old = load_observation(obs_path)
                    for key in ('rgb', 'rgb_valid', 'depth_m', 'instance_mask', 'intrinsics', 'gravity',
                                'query_points', 'query_normals', 'query_source_ids'):
                        if not np.array_equal(old[key], obs[key], equal_nan=True):
                            raise ValueError(f'existing observation data changed: {key}; use a new output directory')
                    if old['min_support_points'] != int(min_support_points):
                        raise ValueError('existing observation support policy changed; use a new output directory')
                else:
                    save_observation(obs_path, obs)
                label_rel, label_identity, labelled_queries = None, None, 0
                if teacher is not None:
                    count = min(len(obs['query_points']), int(train_queries) if row_split == 'train' else 256)
                    label_rel = Path('labels') / f'{uid}.npz'
                    label_path = output / label_rel
                    if label_path.exists():
                        labels = load_labels(label_path, points=obs['query_points'][:count], normals=obs['query_normals'][:count])
                        _compatible_teacher(_teacher_identity(labels['identity']), teacher.identity)
                        for key, expected in [('scene_id', scene_id), ('frame_id', frame_id),
                                              ('instance_row', instance_row), ('obj_id', frame['records'][instance_row]['obj_id'])]:
                            if labels['identity'].get(key) != expected:
                                raise ValueError('existing teacher cache refers to another instance')
                        # Legacy cache provenance stays explicitly unknown. Reuse
                        # exact poses and values without rescanning geometry or
                        # rewriting a checksum into another identity format.
                    else:
                        labels = teacher.evaluate(scene_id, frame_id, instance_row,
                            obs['query_points'][:count], obs['query_normals'][:count], obs['query_ids'][:count])
                        _save_npz(label_path, labels)
                    label_identity, labelled_queries = labels['identity'], count
                row = dict(schema_version=OBSERVATION_SCHEMA, manifest_version=MANIFEST_VERSION,
                    sample_id=uid, instance_uid=uid, group_id=f'scene_{scene_id:04d}',
                    capture_uid=f'{camera}_s{scene_id:04d}_f{frame_id:04d}',
                    split=row_split, test_subset=subset, scene_id=scene_id, frame_id=frame_id,
                    instance_row=instance_row, obj_id=frame['records'][instance_row]['obj_id'], camera=camera,
                    source_kind='suctionnet_rgbd', observation_npz=obs_rel.as_posix(),
                    observation_identity=obs['observation_identity'],
                    query_cache=obs_rel.as_posix(), query_identity=obs['query_identity'],
                    label_cache=None if label_rel is None else label_rel.as_posix(),
                    label_identity=label_identity, labelled_queries=labelled_queries,
                    normal_spec=NORMAL_SPEC, sampler_spec=SAMPLER_SPEC, precision_spec=PRECISION_SPEC,
                    quality_contract=QUALITY_CONTRACT, source_paths=frame['source_paths'],
                    calibration=frame['calibration'], min_support_points=int(min_support_points),
                    support_count=len(obs['support_points']),
                    query_count=len(obs['query_points']), trainable=obs['trainable'], status=status)
                rows.append(row)
    manifest = output / f'manifest_{split}.jsonl'
    summary = dict(manifest_version=MANIFEST_VERSION, manifest=manifest.name,
        camera=camera, split=split,
        scene_ids=list(selected_scenes), frame_ids=list(selected_frames),
        complete_requested_split=selected_scenes == allowed and selected_frames == tuple(range(256)),
        frame_count=len(selected_scenes) * len(selected_frames), visible_instance_count=len(rows),
        empty_gt_frames=empty_frames, status_counts=status_counts, train_queries=int(train_queries),
        test_queries=256, labels_computed=bool(with_labels), quality_contract=QUALITY_CONTRACT,
        min_support_points=int(min_support_points),
        normal_spec=NORMAL_SPEC, sampler_spec=SAMPLER_SPEC, selection_policy='test_selected_if_used_for_best')
    _publish_source_manifest(manifest, rows, summary)
    return manifest, summary


def merge_source_manifests(manifest_paths, output, *, split, allow_partial=False):
    """Merge complete-frame shards while holding the destination writer lock."""
    with source_output_lock(output):
        return _merge_source_manifests(manifest_paths, output, split=split, allow_partial=allow_partial)


def _merge_source_manifests(manifest_paths, output, *, split, allow_partial=False):
    """Merge independently built scene/frame shards without reopening NPZ files.

    Shards own complete frames. Their explicit empty-frame lists are retained,
    overlap is rejected, and relative cache paths are relocated to the new
    manifest. A formal result requires every scene and view of the split.
    """
    if split not in ('train', 'test'):
        raise ValueError('merge requires one explicit train or test split')
    paths = [Path(p) for p in manifest_paths]
    if not paths or len({str(p.resolve()) for p in paths}) != len(paths):
        raise ValueError('provide nonempty distinct shard manifests')
    output = Path(output)
    rows, declared_frames, empty_frames, seen_ids = [], set(), set(), set()
    shared, teacher_contract = None, None
    for path in paths:
        summary = json.loads(path.with_suffix('.summary.json').read_text())
        fields = ('camera', 'split', 'train_queries', 'test_queries', 'labels_computed',
                  'quality_contract', 'normal_spec', 'sampler_spec')
        contract = {key: summary.get(key) for key in fields}
        contract['min_support_points'] = summary.get('min_support_points', 100)
        if contract['split'] != split:
            raise ValueError('shard summary crosses requested split')
        if shared is None:
            shared = contract
        elif shared != contract:
            raise ValueError('shards have different cache/query/physics policies')
        scenes = list(summary['scene_ids'])
        frames = list(summary['frame_ids'])
        if (len(set(scenes)) != len(scenes) or len(set(frames)) != len(frames)
                or not set(scenes) <= set(source_scene_ids(split))
                or not set(frames) <= set(range(256))):
            raise ValueError('invalid shard scene/frame declaration')
        pairs = summary.get('frame_pairs')
        claimed = ({(int(s), int(f)) for s in scenes for f in frames} if pairs is None
                   else {tuple(pair) for pair in pairs})
        if pairs is not None and (len(pairs) != len(claimed) or
                not claimed <= {(s, f) for s in scenes for f in frames}):
            raise ValueError('invalid explicit shard frame pairs')
        if len(claimed) != summary['frame_count'] or not claimed:
            raise ValueError('shard frame count does not match its declaration')
        if declared_frames & claimed:
            raise ValueError('overlapping shard frames are not allowed')
        dataset = InstanceDataset(path, require_labels=bool(summary['labels_computed']))
        observed = set()
        for raw in dataset.records:
            row = copy.deepcopy(raw)
            frame = (row['scene_id'], row['frame_id'])
            if row['split'] != split or row['camera'] != shared['camera'] or frame not in claimed:
                raise ValueError('shard record crosses declared split/camera/frame')
            if row['sample_id'] in seen_ids:
                raise ValueError('duplicate instance ID across shards')
            seen_ids.add(row['sample_id']); observed.add(frame)
            if row.get('label_identity') is not None:
                row['label_identity'] = _label_identity(row['label_identity'])
                current = _teacher_identity(row['label_identity'])
                if teacher_contract is not None:
                    _compatible_teacher(teacher_contract, current)
                # Keep known physics available for checking all later shards.
                if teacher_contract is None or teacher_contract['physics'] is None:
                    teacher_contract = current
            for field in ('observation_npz', 'query_cache', 'label_cache'):
                if row.get(field) is not None:
                    cache = _path(path.parent, row[field]).resolve()
                    if not cache.is_file():
                        raise ValueError(f'shard cache is missing: {cache}')
                    row[field] = os.path.relpath(cache, output.resolve())
            row['manifest_version'] = MANIFEST_VERSION
            rows.append(row)
        empty_list = [tuple(v) for v in summary.get('empty_gt_frames', [])]
        empty = set(empty_list)
        if len(empty) != len(empty_list) or not empty <= claimed or empty & observed:
            raise ValueError('invalid or contradictory empty-frame declaration')
        if observed | empty != claimed or len(dataset.records) != summary['visible_instance_count']:
            raise ValueError('shard has missing frames or an incorrect visible-instance count')
        declared_frames |= claimed; empty_frames |= empty
    expected = {(s, f) for s in source_scene_ids(split) for f in range(256)}
    complete = declared_frames == expected
    if not complete and not allow_partial:
        raise ValueError(f'merged {split} coverage is incomplete: {len(expected-declared_frames)} missing frames')
    rows.sort(key=lambda row: (row['scene_id'], row['frame_id'], row['instance_row']))
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f'manifest_{split}.jsonl'
    if manifest.resolve() in {p.resolve() for p in paths}:
        raise ValueError('merged output cannot replace an input shard manifest')
    status_counts = {}
    for row in rows:
        status_counts[row['status']] = status_counts.get(row['status'], 0) + 1
    summary = dict(shared, manifest_version=MANIFEST_VERSION, manifest=manifest.name,
        scene_ids=sorted({s for s, _ in declared_frames}), frame_ids=sorted({f for _, f in declared_frames}),
        # Explicit pairs prevent a nonrectangular partial merge from claiming
        # the Cartesian product of its scene and frame lists.
        frame_pairs=[list(pair) for pair in sorted(declared_frames)],
        frame_count=len(declared_frames), visible_instance_count=len(rows),
        complete_requested_split=complete, empty_gt_frames=[list(pair) for pair in sorted(empty_frames)],
        status_counts=status_counts, source_manifests=[str(p.resolve()) for p in paths],
        selection_policy='test_selected_if_used_for_best')
    _publish_source_manifest(manifest, rows, summary)
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
        groups, ids, observations, observation_owners = {}, set(), OrderedDict(), {}
        definitions = set()
        for line in self.manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = without_legacy_checksums(json.loads(line))
            if row.get('schema_version') != 'suction20.target.v1' or row.get('label_kind') != 'attempt_success':
                raise ValueError('target records must use suction20.target.v1 / attempt_success')
            sample_id, group = row['sample_id'], row['group_id']
            if row.get('split') not in ('train', 'val', 'test'):
                raise ValueError('target record split must be train/val/test')
            if sample_id in ids:
                raise ValueError('duplicate attempt ID')
            ids.add(sample_id)
            if group in groups and groups[group] != row['split']:
                raise ValueError('target group crosses splits')
            groups[group] = row['split']
            obs, attempt = row['observation'], row['attempt']
            if obs.get('format') != 'rgbd' or not obs.get('rgb_path'):
                raise ValueError('instance routes require organized RGB-D including rgb_path; pointcloud-only target needs original assets')
            # Shared captures must keep one owner across *all* target instances.
            # Paths and explicit capture IDs establish recorded provenance only;
            # no file is opened here and different paths are not content-deduped.
            for field in ('rgb_path', 'depth_path', 'instance_mask_path'):
                obs[field] = str(_path(self.root, obs[field]).resolve())
            owner = (row['split'], group)
            provenance_keys = [(field, obs[field]) for field in ('depth_path', 'rgb_path')]
            for declared in (row.get('capture_uid'), obs.get('capture_uid')):
                if declared is not None:
                    if not isinstance(declared, str) or not declared.strip():
                        raise ValueError('target capture_uid must be a nonempty string when provided')
                    provenance_keys.append(('capture_uid', declared))
            for provenance_key in provenance_keys:
                if observation_owners.setdefault(provenance_key, owner) != owner:
                    raise ValueError('shared target observation/capture crosses split or group')
            if not isinstance(attempt['instance_id'], int) or attempt['instance_id'] <= 0:
                raise ValueError('planned attempt requires a positive predicted instance_id')
            if not isinstance(attempt['label_valid'], bool) or (attempt['label_valid'] and attempt['success'] not in (0, 1)):
                raise ValueError('invalid attempt success/label validity')
            if row.get('success_definition'):
                definitions.add(row['success_definition'])
            key = (obs['depth_path'], attempt['instance_id'])
            if key not in observations:
                observations[key] = dict(observation=obs, rows=[], split=row['split'], group_id=group,
                    instance_id=attempt['instance_id'], sample_id=f"{group}:{obs['depth_path']}:instance_{attempt['instance_id']}")
            item = observations[key]
            if item['observation'] != obs or item['split'] != row['split'] or item['group_id'] != group:
                raise ValueError('repeated target observation has conflicting calibration/split/group')
            item['rows'].append(row)
        if len(definitions) != 1:
            raise ValueError('attempt dataset requires exactly one success_definition')
        self.success_definition = next(iter(definitions))
        self.observation_owners = observation_owners
        self.records = [r for r in observations.values() if r['split'] == split]
        self.identity = dict(manifest_path=str(self.manifest.resolve()), split=split, normal_spec=NORMAL_SPEC,
            sampler_spec=SAMPLER_SPEC, precision_spec=PRECISION_SPEC, success_definition=self.success_definition,
            observation_ids=[record['sample_id'] for record in self.records],
            observation_provenance=[dict(kind=key[0], value=key[1], split=owner[0], group_id=owner[1])
                for key, owner in sorted(observation_owners.items())],
            pose_source='planned_pipeline_contact')
        self.cache_size, self._cache = int(cache_size), OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return copy.deepcopy(self._cache[index])
        result = self.load_uncached(index)
        if self.cache_size > 0:
            self._cache[index] = copy.deepcopy(result)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def load_uncached(self, index):
        record = copy.deepcopy(self.records[index])
        source = record['observation']
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
            depth_max_m=source.get('depth_max_m', 1.2),
            min_support=source.get('min_support_points', 100))
        obs.update(sample_uid=record['sample_id'], instance_uid=record['sample_id'])
        obs['observation_identity']['instance_uid'] = record['sample_id']
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
                support_valid[i] = support_distance[i] <= .005 + BOUNDARY_ROUNDOFF_M
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
        return result
