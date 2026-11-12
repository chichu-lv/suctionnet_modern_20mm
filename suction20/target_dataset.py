"""Target observations and supervision, isolated from source physical proxies.

A dataset item is one observation. Multiple manifest attempt rows referring to
that observation are grouped, so loss reduction and validation retain frame
identity. All relative file paths are resolved against the manifest directory.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .batching import collate_public
from .datasets import derive_masks
from .observation import AugmentConfig, apply_rigid, depth_geometry, rigid_transform, unit_vectors
from .training import content_digest


_ATTEMPT_DTYPES = {'point_cam_m': np.float32, 'normal_out_cam': np.float32,
                   'instance_id': np.int64, 'source_index': np.int64,
                   'success': np.float32, 'label_valid': bool}


def empty_attempts():
    return {key: np.empty((0, 3) if key in ('point_cam_m', 'normal_out_cam') else (0,), dtype)
            for key, dtype in _ATTEMPT_DTYPES.items()}


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _integer_vector(value, name, size=None):
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer) or (size is not None and len(array) != size):
        raise ValueError(f'{name} must be an aligned integer vector')
    return array.astype(np.int64)


def _observed_pca_normals(xyz, config):
    """Bounded PCA of the observed cloud only; labels never enter this function."""
    count = len(xyz)
    normal = np.zeros((count, 3), np.float32)
    valid = np.zeros(count, bool)
    if count < config['min_neighbors']:
        return normal, valid
    tree = cKDTree(xyz)
    k = min(config['max_neighbors'], count)
    for start in range(0, count, 4096):
        points = xyz[start:start+4096]
        distance, index = tree.query(points, k=k, distance_upper_bound=config['radius_m'], workers=1)
        neighbor_ok = np.isfinite(distance)
        neighbor_count = neighbor_ok.sum(axis=1)
        neighbors = xyz[np.minimum(index, count-1)].astype(np.float64)
        center = (neighbors * neighbor_ok[..., None]).sum(axis=1) / np.maximum(neighbor_count, 1)[:, None]
        centered = (neighbors-center[:, None]) * neighbor_ok[..., None]
        covariance = np.einsum('nki,nkj->nij', centered, centered) / np.maximum(neighbor_count, 1)[:, None, None]
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        normals = eigenvector[:, :, 0]
        ok = (neighbor_count >= config['min_neighbors']) & (eigenvalue[:, 1] > 1e-12)
        normals[np.sum(normals * -points, axis=1) < 0] *= -1
        normals[~ok] = 0
        normal[start:start+len(points)] = normals
        valid[start:start+len(points)] = ok
    return normal, valid


def _normal_settings(normal_estimation):
    settings = {'radius_m': .02, 'max_neighbors': 32, 'min_neighbors': 6}
    if normal_estimation:
        unknown = set(normal_estimation) - set(settings)
        if unknown:
            raise ValueError(f'Unsupported observation normal settings: {unknown}')
        settings.update(normal_estimation)
    normal_cfg = settings
    if any(not isinstance(normal_cfg[key], int) for key in ('min_neighbors', 'max_neighbors')):
        raise ValueError('PCA neighbor counts must be integers')
    if not np.isfinite(normal_cfg['radius_m']) or normal_cfg['radius_m'] <= 0 or not 3 <= normal_cfg['min_neighbors'] <= normal_cfg['max_neighbors']:
        raise ValueError('Invalid observed PCA neighbourhood')
    return settings


def _load_observation_geometry(observation, manifest_dir, normal_estimation, source_label_config=None):
    obs = observation
    if obs.get('format') not in ('rgbd', 'pointcloud_npz'):
        raise ValueError('Observation format must be rgbd or pointcloud_npz')
    root = Path(manifest_dir)
    source_label_config = source_label_config or {}
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else root / value
    gravity, gvalid = unit_vectors(np.asarray(obs['g_cam'], np.float32).reshape(1, 3))
    if not gvalid[0]:
        raise ValueError('Observation gravity must be finite and nonzero')
    if obs['format'] == 'rgbd':
        with Image.open(path(obs['depth_path'])) as image:
            depth = np.array(image)
        scale = float(obs['depth_scale_m'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('depth_scale_m must be finite and positive')
        pmap, normal, normal_ok, keep = depth_geometry(depth.astype(np.float64)*scale, obs['intrinsics'],
                obs.get('depth_min_m', source_label_config.get('depth_min_m', .2)),
                obs.get('depth_max_m', source_label_config.get('depth_max_m', 1.2)))
        with Image.open(path(obs['instance_mask_path'])) as image:
            mask = np.array(image)
        if mask.shape != depth.shape or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError('Target instance mask must match depth shape with nonnegative integer IDs')
        pixels = np.flatnonzero(keep.reshape(-1)).astype(np.int64)
        xyz, normals = pmap.reshape(-1, 3)[pixels], normal.reshape(-1, 3)[pixels]
        valid, ids = normal_ok.reshape(-1)[pixels], mask.reshape(-1)[pixels].astype(np.int64)
        metadata = dict(image_shape=depth.shape, source_index_kind='pixel', normal_preprocessing='organized_depth')
    else:
        arrays = _read_npz(path(obs['path']))
        xyz = np.asarray(arrays['xyz'], np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Target pointcloud xyz must be finite [N,3] metres')
        ids = _integer_vector(arrays['instance_id'], 'instance_id', len(xyz))
        if np.any(ids < 0):
            raise ValueError('Input instance IDs must be nonnegative')
        pixels = (_integer_vector(arrays['source_indices'], 'source_indices', len(xyz))
                  if 'source_indices' in arrays else np.arange(len(xyz), dtype=np.int64))
        if np.any(pixels < 0) or len(np.unique(pixels)) != len(pixels):
            raise ValueError('Observation source_indices must be unique nonnegative IDs')
        if 'normals_raw' in arrays:
            if np.asarray(arrays['normals_raw']).shape != xyz.shape:
                raise ValueError('Observed normal shape differs from xyz')
            mask = arrays.get('normals_raw_valid', np.ones(len(xyz), bool))
            if np.asarray(mask).shape != (len(xyz),):
                raise ValueError('Observed normal validity must align with xyz')
            normals, valid = unit_vectors(arrays['normals_raw'], mask)
            normals[np.sum(normals * -xyz, axis=1) < 0] *= -1
            method = 'stored_observed'
        else:
            normals, valid = _observed_pca_normals(xyz, normal_estimation)
            method = 'observed_pca'
        metadata = dict(image_shape=None, source_index_kind='pixel' if 'source_indices' in arrays else 'row',
                        normal_preprocessing=method, normal_estimation=copy.deepcopy(normal_estimation))
    inputs = dict(xyz_cam_m=xyz, normal_obs=normals, normal_obs_valid=valid,
                  instance_id_input=ids, source_indices=pixels, g_cam=gravity[0])
    derive_masks(inputs, {})
    return inputs, metadata


def load_target_observation(observation, manifest_dir, normal_estimation=None, source_label_config=None):
    """Read unlabeled RGB-D/pointcloud with the same observed geometry pipeline.

    No label files are needed. Optional saved source configuration supplies
    default depth limits (otherwise 0.2–1.2m); explicit observation limits take
    precedence. Physical labels are never inferred from an unlabeled observation.
    """
    inputs, provenance = _load_observation_geometry(observation, manifest_dir,
                                                   _normal_settings(normal_estimation), source_label_config)
    uid = content_digest(observation)
    meta = dict(sample_uid=uid, observation_id=uid, label_kind='unlabeled',
                source_indices=inputs['source_indices'].copy(),
                T_aug_from_cam=np.eye(4, dtype=np.float32), **provenance)
    return dict(inputs=inputs, targets={}, attempts=empty_attempts(), meta=meta)


class TargetSuction20Dataset:
    """Manifest-backed target data with deterministic, rigid-only augmentation.

    Dense labels must carry ``instance_id_gt`` or their own ``instance_id``.
    An explicit labels.instance_mapping='input_instance_ids' declaration permits
    a shared mapping when no independent GT instance annotation exists.
    ``normal_estimation`` configures stored-or-observed-PCA preprocessing for
    point clouds without normals; RGB-D always uses organized depth normals.
    """
    def __init__(self, manifest, *, mode, split, source_label_config, augment=None,
                 success_definition=None, seed=42, observation_cache_size=2,
                 normal_estimation=None):
        if mode not in ('dense_proxy', 'attempt_success') or split not in ('train', 'val', 'test'):
            raise ValueError('Unknown target mode or split')
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        raw = self.manifest.read_bytes()
        self.manifest_digest = hashlib.sha256(raw).hexdigest()
        self.source_label_config = copy.deepcopy(source_label_config)
        self.source_label_config_sha256 = content_digest(self.source_label_config)
        if not np.isclose(self.source_label_config['radius_m'], .02):
            raise ValueError('Target adapter requires the source R20 definition')
        self.mode, self.split = mode, split
        self.seed, self.epoch = int(seed), 0
        self.augment = AugmentConfig(**augment) if isinstance(augment, dict) else augment
        if self.augment is not None:
            if not isinstance(self.augment, AugmentConfig):
                raise TypeError('augment must be None, AugmentConfig, or a field dictionary')
            if split != 'train':
                raise ValueError('Target validation/test cannot use random augmentation')
            if self.augment.depth_noise_std_m or self.augment.dropout_max:
                raise ValueError('Target first-stage augmentation must preserve depth and points')
        self.normal_estimation = _normal_settings(normal_estimation)
        self.observation_cache_size = int(observation_cache_size)
        if self.observation_cache_size < 1:
            raise ValueError('Observation cache size must be positive')
        self._pid, self._cache = None, OrderedDict()
        self._checked_definitions = set()
        groups, observations, sample_ids = {}, OrderedDict(), set()
        definitions = set()
        if success_definition is not None:
            definitions.add(str(success_definition))
        for line_number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('schema_version') != 'suction20.target.v1' or row.get('label_kind') != mode:
                raise ValueError(f'Manifest line {line_number}: schema or label_kind mismatch')
            for field in ('sample_id', 'group_id'):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f'Manifest line {line_number}: missing {field}')
            if row['sample_id'] in sample_ids:
                raise ValueError(f'Duplicate sample_id: {row["sample_id"]}')
            sample_ids.add(row['sample_id'])
            row_split, group = row.get('split'), row['group_id']
            if row_split not in ('train', 'val', 'test'):
                raise ValueError('Target split must be train, val, or test')
            if group in groups and groups[group] != row_split:
                raise ValueError(f'Group crosses splits: {group}')
            groups[group] = row_split
            obs = row['observation']
            fmt = obs.get('format')
            path_key = {'pointcloud_npz': 'path', 'rgbd': 'depth_path'}.get(fmt)
            if path_key is None:
                raise ValueError(f'Unknown observation format: {fmt}')
            physical_key = (fmt, str(self._path(obs[path_key]).resolve()))
            observation_id = content_digest({'format': fmt, 'path': obs[path_key]})
            if physical_key not in observations:
                observations[physical_key] = dict(observation_id=observation_id, observation=obs,
                                                  group_id=group, split=row_split, rows=[])
            item = observations[physical_key]
            if item['split'] != row_split or item['group_id'] != group:
                raise ValueError('Same physical observation crosses groups or splits')
            if item['observation'] != obs:
                raise ValueError('Repeated observation has inconsistent calibration or masks')
            if mode == 'dense_proxy' and item['rows']:
                raise ValueError('Dense observation has duplicate label records')
            item['rows'].append(row)
            if mode == 'dense_proxy':
                self._check_definition(row['labels']['source_label_config_path'])
            elif row.get('success_definition'):
                definitions.add(str(row['success_definition']))
        if not sample_ids:
            raise ValueError('Target manifest contains no records')
        self.success_definition = None
        if mode == 'attempt_success':
            if len(definitions) != 1 or not next(iter(definitions), '').strip():
                raise ValueError('Attempt data require one consistent success_definition')
            self.success_definition = next(iter(definitions))
        self.split_groups = {name: sorted(group for group, group_split in groups.items() if group_split == name)
                             for name in ('train', 'val', 'test')}
        self.items = [item for item in observations.values() if item['split'] == split]
        self.sample_ids = [item['observation_id'] for item in self.items]
        self.group_to_indices = {group: [] for group in self.split_groups[split]}
        for index, item in enumerate(self.items):
            self.group_to_indices[item['group_id']].append(index)

    def _path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _check_definition(self, path):
        path = self._path(path)
        if path not in self._checked_definitions:
            definition = json.loads(path.read_text())
            if content_digest(definition) != self.source_label_config_sha256:
                raise ValueError(f'Target source physical definition differs from checkpoint: {path}')
            self._checked_definitions.add(path)

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        seed = int(np.random.SeedSequence([self.seed, self.epoch, index]).generate_state(1)[0])
        return self.get_sample(index, seed)

    def _read_observation(self, item):
        if self._pid != os.getpid():
            self._cache.clear()
            self._pid = os.getpid()
        key = item['observation_id']
        if key in self._cache:
            self._cache.move_to_end(key)
            return copy.deepcopy(self._cache[key])
        inputs, metadata = _load_observation_geometry(item['observation'], self.root,
                                                       self.normal_estimation, self.source_label_config)
        self._cache[key] = (inputs, metadata)
        if len(self._cache) > self.observation_cache_size:
            self._cache.popitem(last=False)
        return copy.deepcopy((inputs, metadata))

    def _dense_targets(self, item, inputs, meta):
        description = item['rows'][0]['labels']
        arrays = _read_npz(self._path(description['path']))
        label_valid = np.asarray(arrays['gt_valid'], bool)
        if label_valid.ndim != 1:
            raise ValueError('gt_valid must be a one-dimensional label mask')
        m, n = len(label_valid), len(inputs['xyz_cam_m'])
        if 'source_indices' in arrays:
            if meta['source_index_kind'] != 'pixel':
                raise ValueError('Pixel-aligned labels require observation source_indices')
            source = _integer_vector(arrays['source_indices'], 'label source_indices', m)
            if len(np.unique(source)) != m or np.any(source < 0):
                raise ValueError('Label source_indices must be unique nonnegative IDs')
            order = np.argsort(source)
            sorted_source = source[order]
            locations = np.searchsorted(sorted_source, inputs['source_indices'])
            exists = locations < m
            if m:
                exists &= sorted_source[np.minimum(locations, m-1)] == inputs['source_indices']
            mapping = np.zeros(n, np.int64)
            mapping[exists] = order[locations[exists]]
        else:
            if m != n:
                raise ValueError('Sparse labels need explicit source_indices for alignment')
            exists, mapping = np.ones(n, bool), np.arange(n)

        def aligned(name, dtype=np.float32, tail=(), default=0):
            out = np.full((n,) + tail, default, dtype=dtype)
            if name not in arrays:
                return out
            value = np.asarray(arrays[name], dtype=dtype)
            if value.shape != (m,) + tail:
                raise ValueError(f'Label {name} does not align with gt_valid')
            out[exists] = value[mapping[exists]]
            return out

        gt_valid = aligned('gt_valid', bool)
        if 'instance_id_gt' in arrays or 'instance_id' in arrays:
            id_key = 'instance_id_gt' if 'instance_id_gt' in arrays else 'instance_id'
            _integer_vector(arrays[id_key], id_key, m)
            ids = aligned(id_key, np.int64)
            id_source = f'labels.{id_key}'
        elif description.get('instance_mapping') == 'input_instance_ids':
            ids = inputs['instance_id_input'].copy()
            id_source = 'explicit_shared_input_mapping'
        else:
            raise ValueError('Dense labels need instance_id_gt/instance_id or explicit input_instance_ids mapping')
        if np.any(ids < 0):
            raise ValueError('GT instance IDs must be nonnegative')
        foreground = ids > 0
        targets = dict(gt_valid=gt_valid, instance_id_gt=ids)
        component_names = [('seal_target', 'seal_valid', 'gt_seal_norm'),
                           ('wrench_target', 'wrench_valid', 'gt_wrench_eval'),
                           ('collision_free_target', 'collision_valid', 'collision_flag')]
        for target_key, mask_key, field in component_names:
            present = field in arrays or (field == 'gt_seal_norm' and 'gt_seal_raw' in arrays)
            if field == 'gt_seal_norm' and field not in arrays and 'gt_seal_raw' in arrays:
                lo, hi = self.source_label_config['seal_norm_min'], self.source_label_config['seal_norm_max']
                if not np.isfinite([lo, hi]).all() or hi <= lo:
                    raise ValueError('Invalid frozen Seal normalization bounds')
                raw_seal = aligned('gt_seal_raw')
                value = np.clip((raw_seal - lo)/(hi-lo), 0, 1)
                value[~np.isfinite(raw_seal)] = np.nan
            elif field == 'collision_flag':
                value = 1 - aligned(field)
            else:
                value = aligned(field)
            valid = ((aligned(mask_key, bool) if mask_key in arrays else gt_valid.copy())
                     & exists & foreground & np.isfinite(value) & present)
            if np.any(valid & ((value < 0) | (value > 1))):
                raise ValueError(f'{field} is outside the source [0,1] definition')
            if field == 'collision_flag' and np.any(valid & (value != 0) & (value != 1)):
                raise ValueError('collision_flag must be boolean or binary')
            targets[target_key] = np.where(valid, value, 0).astype(np.float32)
            targets[mask_key] = valid
        if 'gt_B' in arrays:
            quality = aligned('gt_B')
            quality_valid = gt_valid & exists & foreground & np.isfinite(quality)
        elif all(field in arrays for field in ('gt_wrench_eval', 'collision_flag')) and ('gt_seal_norm' in arrays or 'gt_seal_raw' in arrays):
            quality = targets['seal_target'] * targets['wrench_target'] * targets['collision_free_target']
            quality_valid = gt_valid & targets['seal_valid'] & targets['wrench_valid'] & targets['collision_valid']
        else:
            raise ValueError('gt_B or all actual S/W/C components are required; missing is not a negative label')
        if np.any(quality_valid & ((quality < 0) | (quality > 1))):
            raise ValueError('gt_B must use the source [0,1] definition')
        targets['gt_B'] = np.where(quality_valid, quality, 0).astype(np.float32)
        targets['quality_valid'] = quality_valid
        if 'gt_normal' in arrays:
            if description.get('normal_target_semantics') != 'original_candidate_axis':
                raise ValueError('Target axis labels must explicitly declare original_candidate_axis semantics')
            label, valid = unit_vectors(aligned('gt_normal', tail=(3,)))
            valid &= aligned('axis_target_valid', bool) if 'axis_target_valid' in arrays else gt_valid
            valid &= exists & foreground & inputs['normal_obs_valid']
        else:
            label, valid = np.zeros((n, 3), np.float32), np.zeros(n, bool)
        targets['normal_label'], targets['axis_valid'] = label, valid
        meta['instance_gt_source'] = id_source
        return targets

    def _attempts(self, item):
        rows = [row['attempt'] for row in item['rows']]
        result = empty_attempts()
        for key, dtype in _ATTEMPT_DTYPES.items():
            values = [row.get(key, -1) if key == 'source_index' else row[key] for row in rows]
            result[key] = np.asarray(values, dtype=dtype)
        n = len(rows)
        if result['point_cam_m'].shape != (n, 3) or result['normal_out_cam'].shape != (n, 3):
            raise ValueError('Attempt poses must have three coordinates')
        for row in rows:
            if not isinstance(row['instance_id'], int) or row['instance_id'] <= 0:
                raise ValueError('Attempt instance_id must be a positive integer')
            if 'source_index' in row and (not isinstance(row['source_index'], int) or row['source_index'] < 0):
                raise ValueError('Attempt source_index must be a nonnegative integer')
            if not isinstance(row['label_valid'], bool):
                raise ValueError('Attempt label_valid must be boolean')
            if row['label_valid'] and row['success'] not in (0, 1):
                raise ValueError('Labeled attempt success must be 0 or 1')
        # Do not camera-flip executed axes. Invalid records retain a zero axis and
        # are excluded by finite/nonzero pose checks in scoring/loss.
        result['normal_out_cam'], _ = unit_vectors(result['normal_out_cam'])
        return result

    def get_sample(self, index, augmentation_seed):
        item = self.items[index]
        inputs, provenance = self._read_observation(item)
        meta = dict(sample_uid=item['observation_id'], observation_id=item['observation_id'],
                    sample_ids=[row['sample_id'] for row in item['rows']], group_id=item['group_id'],
                    split=self.split, label_kind=self.mode, source_indices=inputs['source_indices'].copy(),
                    T_aug_from_cam=np.eye(4, dtype=np.float32), augmentation_seed=int(augmentation_seed),
                    **provenance)
        if 'scene_id' in item['rows'][0]:
            meta['scene_id'] = item['rows'][0]['scene_id']
        targets = self._dense_targets(item, inputs, meta) if self.mode == 'dense_proxy' else {}
        attempts = self._attempts(item) if self.mode == 'attempt_success' else empty_attempts()
        sample = dict(inputs=inputs, targets=targets, attempts=attempts, meta=meta)
        if self.augment is not None:
            T = rigid_transform(inputs['g_cam'], np.random.default_rng(augmentation_seed), self.augment)
            apply_rigid(sample, T)
            attempts['point_cam_m'] = attempts['point_cam_m'] @ T[:3, :3].T + T[:3, 3]
            attempts['normal_out_cam'] = attempts['normal_out_cam'] @ T[:3, :3].T
        return sample


class TargetGroupSchedule:
    """Uniform group→observation draws, isolated RNG and exact resume position.

    Epochs are nominal blocks of len(dataset) draws, not without-replacement
    traversals. Prefetched samples receive explicit augmentation seeds.
    """
    def __init__(self, dataset, seed=42):
        self.seed = int(seed)
        self.groups = sorted(dataset.group_to_indices)
        self.members = [list(dataset.group_to_indices[group]) for group in self.groups]
        if not self.groups or any(not member for member in self.members):
            raise ValueError('Target training schedule requires nonempty groups')
        self.identity = dict(manifest_digest=dataset.manifest_digest, split=dataset.split, mode=dataset.mode,
                             groups={group: [dataset.sample_ids[i] for i in dataset.group_to_indices[group]]
                                     for group in self.groups})
        self.epoch_size = len(dataset)
        self.epoch, self.position = 0, 0
        self.rng = np.random.default_rng(self.seed)

    def next(self):
        if self.position == self.epoch_size:
            self.epoch += 1
            self.position = 0
        group = int(self.rng.integers(len(self.groups)))
        members = self.members[group]
        index = members[int(self.rng.integers(len(members)))]
        augmentation_seed = int(self.rng.integers(0, 2**32, dtype=np.uint64))
        self.position += 1
        return index, augmentation_seed

    def next_frames(self, count):
        return [self.next() for _ in range(count)]

    def state_dict(self):
        return dict(identity=copy.deepcopy(self.identity), seed=self.seed,
                    epoch=self.epoch, position=self.position, rng_state=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        if state['identity'] != self.identity or state['seed'] != self.seed:
            raise ValueError('Target resume manifest/split/groups/seed differs')
        if not 0 <= state['position'] <= self.epoch_size or state['epoch'] < 0:
            raise ValueError('Invalid target schedule resume position')
        self.epoch, self.position = int(state['epoch']), int(state['position'])
        self.rng.bit_generator.state = copy.deepcopy(state['rng_state'])


def collate_target(samples, voxel_size_m=.002):
    """Pack observations and separate real attempted poses, never fake negatives."""
    import torch

    modes = {sample['meta']['label_kind'] for sample in samples}
    if len(modes) != 1:
        raise ValueError('Collate target modes separately')
    batch = collate_public(samples, voxel_size_m)
    attempts = {key: torch.from_numpy(np.concatenate([sample['attempts'][key] for sample in samples]))
                for key in _ATTEMPT_DTYPES}
    batch_ids, anchors = [], []
    for b, sample in enumerate(samples):
        points, attempt = sample['inputs'], sample['attempts']
        count = len(attempt['success'])
        local = np.full(count, -1, np.int64)
        if sample['meta']['source_index_kind'] == 'pixel' and count:
            order = np.argsort(points['source_indices'])
            pixels = points['source_indices'][order]
            for a, pixel in enumerate(attempt['source_index']):
                location = np.searchsorted(pixels, pixel)
                if pixel >= 0 and location < len(pixels) and pixels[location] == pixel:
                    row = order[location]
                    if points['instance_id_input'][row] == attempt['instance_id'][a]:
                        local[a] = int(batch['inputs']['point_offsets'][b]) + row
        anchors.append(local)
        batch_ids.append(np.full(count, b, np.int64))
    attempts['batch_index'] = torch.from_numpy(np.concatenate(batch_ids))
    attempts['feature_point_index'] = torch.from_numpy(np.concatenate(anchors))
    batch['attempts'] = attempts
    return batch
