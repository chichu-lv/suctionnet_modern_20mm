"""Read-only SuctionNet R20 adapter; frozen physics and model inputs stay separate.

Importing this module does not require torch. ``collate_public`` imports torch
only when packing a training batch; all source reads/derivations use NumPy.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .batching import collate_public
from .full_data import load_seal, scene_frame0
from .observation import AugmentConfig, apply_rigid, depth_geometry, rigid_transform, unit_vectors


FrameKey = tuple[int, str, int]


def frame_key(key, camera='realsense') -> FrameKey:
    if len(key) == 2:
        scene, frame = key
    elif len(key) == 3:
        scene, camera, frame = key
    else:
        raise ValueError('Expected (scene,frame) or (scene,camera,frame)')
    scene, frame = int(scene), int(frame)
    if not 0 <= scene < 190 or not 0 <= frame < 256:
        raise ValueError('Frame key outside public SuctionNet inventory')
    if camera not in ('realsense', 'kinect'):
        raise ValueError('Unknown public camera')
    return scene, camera, frame


def split_scene_ids(split):
    if split == 'train':
        return [s for s in range(100) if s % 10 != 9]
    if split in ('val', 'validation', 'dev'):
        return list(range(9, 100, 10))
    ranges = {'test_seen': (100, 130), 'test_similar': (130, 160), 'test_novel': (160, 190)}
    if split not in ranges:
        raise ValueError(f'Unknown public split: {split}')
    return list(range(*ranges[split]))


def public_frame_keys(split, camera='realsense', epoch=0, seed=42):
    """Train: 64 views/scene without replacement; validation: fixed 0:256:8.

    Official test keys cover all 256 views; constructing keys does not evaluate them.

    The caller owns epoch/consumption state; __getitem__ never changes a split.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    keys = []
    for scene in split_scene_ids(split):
        if split == 'train':
            views = rng.choice(256, 64, replace=False)
        else:
            views = range(256) if split.startswith('test_') else range(0, 256, 8)
        keys.extend((scene, camera, int(view)) for view in views)
    if split == 'train':
        rng.shuffle(keys)
    return keys


def resolve_source_format(frame, fallback='legacy_dc4ad6b', compatibility='auto'):
    if fallback not in ('legacy_dc4ad6b', 'r20_v2_1'):
        raise ValueError(f'Unsupported source format: {fallback}')
    if compatibility not in ('auto', 'strict'):
        raise ValueError('compatibility must be auto or strict')
    schema = str(np.asarray(frame.get('schema_version', '')).item())
    if schema and schema != 'r20-v2.1':
        raise ValueError(f'Unsupported explicit source schema: {schema}')
    if compatibility == 'auto' and schema == 'r20-v2.1':
        return 'r20_v2_1'
    if compatibility == 'strict' and schema and fallback != 'r20_v2_1':
        raise ValueError('Source schema contradicts explicitly requested format')
    return fallback


def read_view_payload(path, source_format, compatibility, corrupt_depth):
    """Decompress only consumed members; preserve source identity and trace fields."""
    common = {'xyz', 'source_indices', 'instance_id', 'gt_valid', 'gt_A', 'gt_B',
              'gt_seal_norm', 'gt_wrench_eval', 'collision_flag'}
    trace = {'candidate_row', 'candidate_id', 'class_id', 'match_distance', 'normal_valid'}
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        header = {'schema_version': archive['schema_version']} if 'schema_version' in names else {}
        effective = resolve_source_format(header, source_format, compatibility)
        needed = common | (trace & names)
        needed.add('candidate_row' if effective == 'legacy_dc4ad6b' else 'gt_normal')
        derive = corrupt_depth or not {'normals_raw', 'normals_raw_valid'} <= names
        if not derive:
            needed.update(('normals_raw', 'normals_raw_valid'))
        frame = dict(header)
        frame.update((key, archive[key]) for key in sorted(needed))
    return frame, effective, derive


def recover_candidate_axes(frame, records, seal_loader, T_world_cam):
    """Recover directed original candidate axes, never PCA or mesh-index lookup."""
    ids = np.asarray(frame['instance_id'], np.int64)
    valid = np.asarray(frame['gt_valid'], bool)
    rows = np.asarray(frame['candidate_row'], np.int64)
    normals = np.zeros((len(ids), 3), np.float32)
    if np.any(valid & (ids <= 0)):
        raise ValueError('Matched candidate has no foreground GT instance')
    available = {int(rec['instance_row']) + 1 for rec in records}
    if set(ids[valid]) - available:
        raise ValueError('Matched GT instance absent from scene records')
    T_cam_world = np.linalg.inv(T_world_cam)
    for rec in records:
        pix = np.flatnonzero(valid & (ids == rec['instance_row'] + 1))
        if not len(pix):
            continue
        seal = seal_loader(rec['obj_id'])
        row = rows[pix]
        if np.any((row < 0) | (row >= len(seal['normals']))):
            raise ValueError('candidate_row outside original candidate array')
        if 'candidate_id' in frame and not np.array_equal(np.asarray(frame['candidate_id'])[pix], seal['indices'][row]):
            raise ValueError('candidate_id does not identify the original candidate_row')
        R = (T_cam_world @ rec['T_world_obj'])[:3, :3]
        normals[pix], _ = unit_vectors(seal['normals'][row] @ R.T)
    return normals


def derive_masks(inputs, targets):
    inputs['normal_obs'], inputs['normal_obs_valid'] = unit_vectors(inputs['normal_obs'], inputs['normal_obs_valid'])
    inputs['foreground_input'] = inputs['instance_id_input'] > 0
    inputs['candidate_eligible'] = inputs['foreground_input'] & inputs['normal_obs_valid']
    if targets:
        targets['normal_label'], label_ok = unit_vectors(targets['normal_label'])
        targets['quality_valid'] = targets['gt_valid'] & (targets['instance_id_gt'] > 0)
        targets['axis_valid'] = targets['quality_valid'] & label_ok & inputs['normal_obs_valid']


def select_aligned(sample, selection):
    """Select all point arrays together, preserving original pixel trace IDs."""
    n = len(sample['inputs']['xyz_cam_m'])
    for section in ('inputs', 'targets'):
        for key, value in sample[section].items():
            if section == 'inputs' and key == 'g_cam':
                continue
            if not isinstance(value, np.ndarray) or len(value) != n:
                raise ValueError(f'Unaligned {section} field: {key}')
            sample[section][key] = value[selection]
    for key in ('source_indices', 'candidate_row', 'candidate_id', 'class_id', 'match_distance', 'normal_valid'):
        if key in sample['meta']:
            sample['meta'][key] = sample['meta'][key][selection]
    return sample


def read_input_instances(instance_source, key, pixels, shape, ids_gt=None):
    """Read observation instance IDs by original pixels; never overwrite GT IDs."""
    if isinstance(instance_source, str) and instance_source == 'gt':
        if ids_gt is None:
            raise ValueError('GT instance input requires explicit GT instance IDs')
        return np.asarray(ids_gt, np.int64).copy()
    if callable(instance_source):
        mask = np.asarray(instance_source(key))
    else:
        scene, camera, frame = key
        path = Path(instance_source) / f'scene_{scene:04d}' / camera / f'{frame:04d}.png'
        with Image.open(path) as image:
            mask = np.array(image)
    if mask.shape != tuple(shape) or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError('Input instance mask must be a nonnegative integer HxW image')
    return mask.reshape(-1)[pixels].astype(np.int64)



class PublicSuction20Dataset:
    """Map-style source dataset with bounded process-local scene/object caches.

    ``instance_source`` is ``'gt'``, a mask root (scene_XXXX/camera/FFFF.png),
    or a callable(key)->HxW instance image. Predicted IDs never replace GT IDs.
    For replay-safe worker prefetch, call get_sample(index, augmentation_seed)
    using a seed derived from the consumed epoch/sequence, not worker RNG state.
    """
    def __init__(self, dataset_root, label_root, frame_keys, camera='realsense',
                 augment=None, instance_source='gt', source_format='legacy_dc4ad6b',
                 compatibility='auto', adapter_cache=None, seed=42,
                 scene_cache_size=2, object_cache_size=16):
        if adapter_cache is not None:
            raise ValueError('Derived sidecars are not enabled; use the read-only adapter')
        self.dataset_root, self.label_root = Path(dataset_root), Path(label_root)
        self.keys = [frame_key(key, camera) for key in frame_keys]
        config_path = self.label_root / 'config_frozen.json'
        config_bytes = config_path.read_bytes()
        self.data_cfg = json.loads(config_bytes)
        self.source_label_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        if not np.isclose(self.data_cfg['radius_m'], .02):
            raise ValueError('This adapter requires the R20 (20mm radius) release')
        if any(key[1] != self.data_cfg['camera'] for key in self.keys):
            raise ValueError('Frame camera differs from frozen label camera')
        self.augment = AugmentConfig(**augment) if isinstance(augment, dict) else augment
        if self.augment is not None and not isinstance(self.augment, AugmentConfig):
            raise TypeError('augment must be None, AugmentConfig, or its field dictionary')
        self.instance_source = instance_source
        self.source_format, self.compatibility = source_format, compatibility
        self.seed, self.epoch = int(seed), 0
        self.scene_cache_size, self.object_cache_size = scene_cache_size, object_cache_size
        if min(scene_cache_size, object_cache_size) < 1:
            raise ValueError('Cache sizes must be positive')
        self._pid = None
        self._scenes, self._objects = OrderedDict(), OrderedDict()
        self._cache_counts = {'hits': 0, 'misses': 0, 'evictions': 0}

    def __getstate__(self):
        # Spawn workers start with their own small metadata caches, not copies of
        # an already used parent's arrays. No frame/depth/voxel cache is added.
        state = self.__dict__.copy()
        state.update(_pid=None, _scenes=OrderedDict(), _objects=OrderedDict(),
                     _cache_counts={'hits': 0, 'misses': 0, 'evictions': 0})
        return state

    def __len__(self):
        return len(self.keys)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        # Explicit per-sample RNG avoids hidden global or DataLoader worker state.
        seed = np.random.SeedSequence([self.seed, self.epoch, int(index)]).generate_state(1)[0]
        return self.get_sample(index, int(seed))

    def _cached(self, cache, key, loader, limit):
        if self._pid != os.getpid():
            self._scenes.clear()
            self._objects.clear()
            self._cache_counts = {'hits': 0, 'misses': 0, 'evictions': 0}
            self._pid = os.getpid()
        if key not in cache:
            self._cache_counts['misses'] += 1
            cache[key] = loader()
            if len(cache) > limit:
                cache.popitem(last=False)
                self._cache_counts['evictions'] += 1
        else:
            self._cache_counts['hits'] += 1
        cache.move_to_end(key)
        return cache[key]

    def _scene(self, scene, camera, need_records=False):
        base = self.dataset_root / 'scenes' / f'scene_{scene:04d}' / camera
        def load():
            return dict(table=np.load(base / 'cam0_wrt_table.npy', allow_pickle=False),
                        poses=np.load(base / 'camera_poses.npy', allow_pickle=False), records=None)
        cached = self._cached(self._scenes, (scene, camera), load, self.scene_cache_size)
        if need_records and cached['records'] is None:
            cached['records'] = scene_frame0(self.dataset_root, scene, camera,
                                             table=cached['table'], camera_poses=cached['poses'])
        return cached['table'], cached['poses'], cached['records']

    def _seal(self, object_id):
        def load_axes():
            full = load_seal(self.label_root / 'seal_r20' / 'poisson' / f'{object_id:03d}_labels.npz')
            # load_seal retains the original FP64 normalization and validation.
            return {key: full[key] for key in ('normals', 'indices')}
        return self._cached(self._objects, object_id,
                            load_axes,
                            self.object_cache_size)

    def cache_info(self):
        def nbytes(value):
            if isinstance(value, np.ndarray):
                return value.nbytes
            if isinstance(value, dict):
                return sum(nbytes(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(nbytes(v) for v in value)
            return 0
        return dict(self._cache_counts, scene_entries=len(self._scenes), object_entries=len(self._objects),
                    scene_array_bytes=nbytes(self._scenes), object_array_bytes=nbytes(self._objects))

    def get_sample(self, index, augmentation_seed):
        started = perf_counter()
        timing = {}
        key = self.keys[index]
        scene, camera, frame_id = key
        path = self.label_root / 'scenes' / f'scene_{scene:04d}' / camera / 'graspness_r20_v2' / f'{frame_id:04d}.npz'
        corrupt = self.augment is not None and (self.augment.depth_noise_std_m > 0 or self.augment.dropout_max > 0)
        frame, effective, derive = read_view_payload(path, self.source_format, self.compatibility, corrupt)
        timing['read_npz_s'] = perf_counter() - started
        xyz = np.asarray(frame['xyz'], np.float32)
        pixels = np.asarray(frame['source_indices'], np.int64)
        ids = np.asarray(frame['instance_id'], np.int64)
        n = len(xyz)
        if xyz.shape != (n, 3) or not np.isfinite(xyz).all() or pixels.shape != (n,) or ids.shape != (n,):
            raise ValueError('Malformed source geometry or point indices')
        base = self.dataset_root / 'scenes' / f'scene_{scene:04d}' / camera
        reading = perf_counter()
        with Image.open(base / 'depth' / f'{frame_id:04d}.png') as image:
            shape = (image.height, image.width)
            # Only read depth values when normals must be derived or corrupted.
            depth = np.array(image) if derive else None
        timing['read_depth_meta_s'] = perf_counter() - reading
        if np.any((pixels < 0) | (pixels >= np.prod(shape))) or len(np.unique(pixels)) != n:
            raise ValueError('source_indices must be unique pixels within the original image')
        reading = perf_counter()
        table, poses, records = self._scene(scene, camera, effective == 'legacy_dc4ad6b')
        T_world_cam = table @ poses[frame_id]
        timing['read_depth_meta_s'] += perf_counter() - reading
        axes_started = perf_counter()
        normal_label = (recover_candidate_axes(frame, records, self._seal, T_world_cam)
                        if effective == 'legacy_dc4ad6b' else np.asarray(frame['gt_normal'], np.float32))
        timing['axis_recovery_s'] = perf_counter() - axes_started
        geometry_started = perf_counter()
        keep_rows = np.ones(n, bool)
        rng = np.random.default_rng(augmentation_seed)
        if derive:
            reading = perf_counter()
            mat = loadmat(base / 'meta' / f'{frame_id:04d}.mat')
            metadata_seconds = perf_counter() - reading
            timing['read_depth_meta_s'] += metadata_seconds
            # Keep geometry timing separate from this metadata read.
            geometry_started += metadata_seconds
            scale = float(np.asarray(mat['factor_depth']).reshape(-1)[0])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid source depth scale')
            z = depth.astype(np.float64) / scale
            if corrupt:
                cfg = self.augment
                noise = np.clip(rng.normal(0, cfg.depth_noise_std_m, z.shape), -cfg.depth_noise_clip_m, cfg.depth_noise_clip_m)
                z = np.where(z > 0, z + noise, z)
                keep_rows = rng.random(n) >= rng.uniform(0, cfg.dropout_max)
                # Deleted points must also be absent from neighbours' normals.
                z.reshape(-1)[pixels[~keep_rows]] = 0
            pmap, normals, valid, keep = depth_geometry(z, mat['intrinsic_matrix'], self.data_cfg['depth_min_m'], self.data_cfg['depth_max_m'])
            if not corrupt and not np.all(keep.reshape(-1)[pixels]):
                raise ValueError('Frozen source pixels no longer match the original depth')
            keep_rows &= keep.reshape(-1)[pixels]
            normal_obs = normals.reshape(-1, 3)[pixels]
            normal_ok = valid.reshape(-1)[pixels]
            if corrupt:
                xyz = pmap.reshape(-1, 3)[pixels]
            elif not np.allclose(xyz, pmap.reshape(-1, 3)[pixels], atol=1e-6, rtol=0):
                raise ValueError('Source depth geometry differs from the frozen label observation')
        else:
            normal_obs = np.asarray(frame['normals_raw'], np.float32)
            normal_ok = np.asarray(frame['normals_raw_valid'], bool)
        targets = {name: np.asarray(frame[field], np.float32) for name, field in
                   [('gt_A', 'gt_A'), ('gt_B', 'gt_B'), ('seal_target', 'gt_seal_norm'), ('wrench_target', 'gt_wrench_eval')]}
        targets.update(collision_free_target=(~np.asarray(frame['collision_flag'], bool)).astype(np.float32),
                       normal_label=normal_label, instance_id_gt=ids,
                       gt_valid=np.asarray(frame['gt_valid'], bool))
        inputs = dict(xyz_cam_m=xyz, normal_obs=normal_obs, normal_obs_valid=normal_ok,
                      instance_id_input=read_input_instances(self.instance_source, key, pixels, shape, ids),
                      g_cam=(T_world_cam[:3, :3].T @ np.array([0., 0., -1.])).astype(np.float32),
                      source_indices=pixels)
        meta = dict(sample_uid=key, source_indices=pixels.copy(), image_shape=shape,
                    T_world_cam=T_world_cam, T_aug_from_cam=np.eye(4, dtype=np.float32),
                    source_format=effective, source_schema_version=str(np.asarray(frame.get('schema_version', '')).item()),
                    source_label_config_sha256=self.source_label_config_sha256,
                    augmentation_seed=int(augmentation_seed), label_path=str(path))
        for field in ('candidate_row', 'candidate_id', 'class_id', 'match_distance', 'normal_valid'):
            if field in frame:
                dtype = np.int64 if field in ('candidate_row', 'candidate_id', 'class_id') else None
                meta[field] = np.asarray(frame[field], dtype=dtype)
        for section in (inputs, targets):
            for field, value in section.items():
                if field != 'g_cam' and len(value) != n:
                    raise ValueError(f'Unaligned source field: {field}')
        sample = dict(inputs=inputs, targets=targets, meta=meta)
        if not keep_rows.all():
            select_aligned(sample, keep_rows)
        derive_masks(sample['inputs'], sample['targets'])
        if self.augment is not None:
            apply_rigid(sample, rigid_transform(sample['inputs']['g_cam'], rng, self.augment))
        timing['augment_geometry_s'] = perf_counter() - geometry_started
        timing['prepare_total_s'] = perf_counter() - started
        sample['production_timing'] = timing
        sample['cache_info'] = self.cache_info()
        return sample


class PublicObservationDataset:
    """Label-root-free inference on original public RGB-D and input instances.

    Pass the checkpoint's source ``data_cfg`` for its depth range. ``gt`` is an
    explicit oracle-segmentation baseline; predicted masks require no GT labels,
    candidate files, object poses, or scene metadata.
    """
    def __init__(self, dataset_root, frame_keys, *, data_cfg, instance_source='gt', camera='realsense'):
        self.dataset_root = Path(dataset_root)
        self.keys = [frame_key(key, camera) for key in frame_keys]
        self.data_cfg = dict(data_cfg)
        self.instance_source = instance_source

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        from .full_data import resolve_instance_ids

        key = self.keys[index]
        scene, camera, frame_id = key
        base = self.dataset_root / 'scenes' / f'scene_{scene:04d}' / camera
        with Image.open(base / 'depth' / f'{frame_id:04d}.png') as image:
            depth = np.array(image)
        mat = loadmat(base / 'meta' / f'{frame_id:04d}.mat')
        scale = float(np.asarray(mat['factor_depth']).reshape(-1)[0])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid source depth scale')
        pmap, normals, valid, keep = depth_geometry(depth.astype(np.float64)/scale, mat['intrinsic_matrix'],
                                                   self.data_cfg['depth_min_m'], self.data_cfg['depth_max_m'])
        pixels = np.flatnonzero(keep.reshape(-1)).astype(np.int64)
        shape = depth.shape
        if isinstance(self.instance_source, str) and self.instance_source == 'gt':
            with Image.open(base / 'label' / f'{frame_id:04d}.png') as image:
                classes = np.array(image)
            records = scene_frame0(self.dataset_root, scene, camera)
            cfg = dict(self.data_cfg, camera=camera)
            ids = resolve_instance_ids(classes.reshape(-1)[pixels], records, self.dataset_root,
                                       scene, frame_id, cfg, pixels, shape).astype(np.int64)
        else:
            # Reuse precisely the same image/pixel contract as supervised reads.
            ids = read_input_instances(self.instance_source, key, pixels, shape)
        T_world_cam = (np.load(base / 'cam0_wrt_table.npy', allow_pickle=False)
                       @ np.load(base / 'camera_poses.npy', allow_pickle=False)[frame_id])
        inputs = dict(xyz_cam_m=pmap.reshape(-1, 3)[pixels], normal_obs=normals.reshape(-1, 3)[pixels],
                      normal_obs_valid=valid.reshape(-1)[pixels], instance_id_input=ids,
                      source_indices=pixels, g_cam=(T_world_cam[:3, :3].T @ np.array([0., 0., -1.])).astype(np.float32))
        derive_masks(inputs, {})
        return dict(inputs=inputs, targets={}, meta=dict(sample_uid=key, source_indices=pixels.copy(),
                    image_shape=shape, T_world_cam=T_world_cam, T_aug_from_cam=np.eye(4, dtype=np.float32),
                    source_format='observation_only'))
