"""Instance-balanced, unique seed sampling with deterministic metric FPS."""
from __future__ import annotations

import numpy as np
import torch


def fps_indices(xyz, indices, count, first=None):
    """CPU geometry selection; equal distances resolve to the raw packed index."""
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    count = min(count, len(indices))
    if count <= 0:
        return indices[:0]
    points = np.asarray(xyz)[indices].astype(np.float64)
    chosen = np.zeros(len(indices), bool)
    distance = np.full(len(indices), np.inf)
    current = int(np.searchsorted(indices, indices[0] if first is None else first))
    output = []
    for _ in range(count):
        output.append(indices[current])
        chosen[current] = True
        distance = np.minimum(distance, np.sum((points - points[current]) ** 2, axis=1))
        distance[chosen] = -1
        current = int(np.argmax(distance))
    return np.asarray(output, np.int64)


def inference_pool(xyz, indices, quality, quota):
    ordered = indices[np.lexsort((indices, -quality[indices]))]
    pool = ordered[:min(len(ordered), 4 * quota)]
    return fps_indices(xyz, pool, quota, first=ordered[0])


class CandidateSampler:
    def __init__(self, num_seeds=512):
        self.num_seeds = num_seeds

    @torch.no_grad()
    def __call__(self, inputs, q0, *, mode='inference', targets=None, epoch=0):
        if mode not in {'train', 'inference', 'dense_proxy'}:
            raise ValueError(f'Unknown sampler mode {mode}')
        if mode != 'inference' and targets is None:
            raise ValueError('Training sampling requires separate targets')
        xyz = inputs['xyz_cam_m'].detach().float().cpu().numpy()
        inst = inputs['instance_id_input'].cpu().numpy()
        batch = inputs['batch_index'].cpu().numpy()
        eligible = inputs['normal_obs_valid'].cpu().numpy() & (inst > 0)
        quality = q0.detach().float().cpu().numpy()
        selected = np.zeros((len(inputs['g_cam']), self.num_seeds), np.int64)
        valid = np.zeros_like(selected, bool)
        if targets is not None and mode != 'inference':
            label_valid = targets['quality_valid'].cpu().numpy()
            label_a = targets['gt_A'].cpu().numpy() if mode == 'train' else None
        for b in range(len(selected)):
            ids = np.unique(inst[eligible & (batch == b)])
            cursor = 0
            for rank, instance in enumerate(ids):
                quota = self.num_seeds // len(ids) + (rank < self.num_seeds % len(ids))
                if quota == 0:
                    continue
                pool = np.flatnonzero(eligible & (batch == b) & (inst == instance))
                proposal = inference_pool(xyz, pool, quality, quota)
                if mode == 'inference':
                    result = proposal
                else:
                    def random_order(rows):
                        return rows[torch.randperm(len(rows)).numpy()]
                    if mode == 'dense_proxy':
                        high = random_order(pool[label_valid[pool]])
                        low = np.empty(0, np.int64)
                        budgets = (quota - quota // 2, quota // 2, 0)
                    else:
                        high = random_order(pool[label_valid[pool] & (label_a[pool] >= .2)])
                        low = random_order(pool[label_valid[pool] & (label_a[pool] < .2)])
                        budgets = ((0, quota - quota // 2, quota // 2) if epoch < 2 else
                                   (quota - 2 * (quota // 4), quota // 4, quota // 4))
                    # Independent group draws precede union and deterministic fill.
                    draws = [proposal[:budgets[0]], high[:budgets[1]], low[:budgets[2]]]
                    used = set()
                    result = []
                    for rows in draws + [high, low, proposal, pool]:
                        for row in rows:
                            if int(row) not in used:
                                used.add(int(row))
                                result.append(int(row))
                            if len(result) == min(quota, len(pool)):
                                break
                        if len(result) == min(quota, len(pool)):
                            break
                    result = np.asarray(result, np.int64)
                selected[b, cursor:cursor + len(result)] = result
                valid[b, cursor:cursor + len(result)] = True
                cursor += len(result)
        return (torch.as_tensor(selected, device=q0.device),
                torch.as_tensor(valid, device=q0.device))
