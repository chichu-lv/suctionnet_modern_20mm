"""Packed point/voxel contracts. Targets never contribute to voxel features."""
import numpy as np
from time import perf_counter


def concatenate_arrays(values):
    """Reuse single-frame storage; callers keep the packed batch alive and read-only."""
    return values[0] if len(values) == 1 else np.concatenate(values)


def unique_voxels_xyz(grid):
    """Mixed-radix keys preserve np.unique(axis=0)'s exact xyz lexicographic order."""
    if not len(grid):
        return np.zeros(3, np.int64), np.empty((0, 3), np.int64), np.empty(0, np.int64)
    origin = grid.min(axis=0)
    sizes = [int(hi) - int(lo) + 1 for hi, lo in zip(grid.max(axis=0), origin)]
    if max(sizes) - 1 > np.iinfo(np.int32).max:
        raise ValueError('Voxel grid exceeds int32 coordinates')
    nx, ny, nz = sizes
    if nx * ny * nz > np.iinfo(np.int64).max:
        raise ValueError('Voxel key exceeds int64')
    shifted = grid - origin
    keys = (shifted[:, 0] * ny + shifted[:, 1]) * nz + shifted[:, 2]
    unique, inverse = np.unique(keys, return_inverse=True)
    x, rest = np.divmod(unique, ny * nz)
    y, z = np.divmod(rest, nz)
    return origin, np.column_stack((x, y, z)), inverse


def collate_public(samples, voxel_size_m=.002):
    import torch

    started = perf_counter()
    timing = {'voxel_unique_s': 0., 'voxel_reduce_s': 0.}
    if not samples or not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError('Need samples and a positive voxel size')
    point_keys = ('xyz_cam_m', 'normal_obs', 'normal_obs_valid',
                  'instance_id_input', 'source_indices', 'candidate_eligible',
                  'foreground_input')
    packed = {key: [] for key in point_keys}
    coords, features, inverse, origins, batch_ids = [], [], [], [], []
    offsets, voxel_count = [0], 0
    for batch_id, sample in enumerate(samples):
        inputs = sample['inputs']
        xyz = np.asarray(inputs['xyz_cam_m'], np.float32)
        n = len(xyz)
        if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
            raise ValueError('Point geometry must be finite [N,3]')
        valid = np.asarray(inputs['normal_obs_valid'], bool)
        ids = np.asarray(inputs['instance_id_input'], np.int64)
        normals = np.where(valid[:, None], inputs['normal_obs'], 0).astype(np.float32)
        foreground = ids > 0
        local = dict(inputs, normal_obs=normals, foreground_input=foreground,
                     candidate_eligible=foreground & valid)
        for key in point_keys:
            value = np.asarray(local[key])
            if len(value) != n:
                raise ValueError(f'Unaligned point field: {key}')
            packed[key].append(value)
        grid = np.floor(xyz / voxel_size_m).astype(np.int64)
        unique_started = perf_counter()
        origin, unique, inv = unique_voxels_xyz(grid)
        timing['voxel_unique_s'] += perf_counter() - unique_started
        reduce_started = perf_counter()
        center = xyz.mean(axis=0, dtype=np.float64).astype(np.float32) if n else np.zeros(3, np.float32)
        point_features = np.column_stack(((xyz-center)/.1, normals, foreground, valid)).astype(np.float32)
        count = np.bincount(inv, minlength=len(unique))
        # bincount avoids a large torch scatter buffer in DataLoader workers.
        voxel_features = np.zeros((len(unique), 8), np.float32)
        for channel in range(8):
            voxel_features[:, channel] = np.bincount(inv, weights=point_features[:, channel], minlength=len(unique)) / count
        timing['voxel_reduce_s'] += perf_counter() - reduce_started
        coords.append(np.column_stack((np.full(len(unique), batch_id), unique[:, [2, 1, 0]])).astype(np.int32))
        features.append(voxel_features)
        inverse.append(inv.astype(np.int64) + voxel_count)
        origins.append(origin)
        batch_ids.append(np.full(n, batch_id, np.int64))
        offsets.append(offsets[-1]+n)
        voxel_count += len(unique)
    inputs = {key: torch.from_numpy(concatenate_arrays(values)) for key, values in packed.items()}
    for key in ('instance_id_input', 'source_indices'):
        inputs[key] = inputs[key].long()
    inputs.update(
        g_cam=torch.from_numpy(np.stack([s['inputs']['g_cam'] for s in samples]).astype(np.float32)),
        batch_index=torch.from_numpy(concatenate_arrays(batch_ids)),
        point_offsets=torch.tensor(offsets, dtype=torch.int64),
        voxel_coords=torch.from_numpy(np.ascontiguousarray(concatenate_arrays(coords))),
        voxel_features=torch.from_numpy(np.ascontiguousarray(concatenate_arrays(features))),
        inverse_map=torch.from_numpy(concatenate_arrays(inverse)),
        voxel_origins=torch.from_numpy(np.stack(origins)),
    )
    target_keys = set(samples[0]['targets'])
    if any(set(s['targets']) != target_keys for s in samples):
        raise ValueError('Batch samples require identical target fields; collate domains separately')
    targets = {}
    for key in sorted(target_keys):
        values = [np.asarray(s['targets'][key]) for s in samples]
        if any(len(value) != offsets[i+1]-offsets[i] for i, value in enumerate(values)):
            raise ValueError(f'Unaligned target: {key}')
        targets[key] = torch.from_numpy(concatenate_arrays(values))
    timing['collate_total_s'] = perf_counter() - started
    for sample in samples:
        for key, seconds in sample.get('production_timing', {}).items():
            timing[key] = timing.get(key, 0.) + seconds
    return {'inputs': inputs, 'targets': targets, 'meta': [s['meta'] for s in samples],
            'production_timing': timing}
