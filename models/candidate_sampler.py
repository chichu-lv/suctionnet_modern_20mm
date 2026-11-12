"""Instance-balanced, unique seed sampling with deterministic metric FPS."""
from __future__ import annotations

import time

import numpy as np
import torch


from .geometry_ops import fps_indices


def inference_pool(xyz, indices, quality, quota, *, fps_fn=fps_indices):
    ordered = indices[np.lexsort((indices, -quality[indices]))]
    pool = ordered[:min(len(ordered), 4 * quota)]
    return fps_fn(xyz, pool, quota, first=ordered[0])


class CandidateSampler:
    def __init__(self, num_seeds=512, *, fps_fn=fps_indices):
        self.num_seeds = num_seeds
        self.fps_fn = fps_fn
        self.last_timings = {}

    @torch.no_grad()
    def __call__(self, inputs, q0, *, mode='inference', targets=None, epoch=0,
                 cpu_geometry=None, sampler_targets_cpu=None):
        if mode not in {'train', 'inference', 'dense_proxy'}:
            raise ValueError(f'Unknown sampler mode {mode}')
        if mode != 'inference' and targets is None and sampler_targets_cpu is None:
            raise ValueError('Training sampling requires separate targets')
        started = time.perf_counter()
        geometry = cpu_geometry if cpu_geometry is not None else inputs.get('cpu_geometry')
        if geometry is None:
            xyz = inputs['xyz_cam_m'].detach().float().cpu().numpy()
            inst = inputs['instance_id_input'].cpu().numpy()
            batch = inputs['batch_index'].cpu().numpy()
            observed_valid = inputs['normal_obs_valid'].cpu().numpy()
        else:
            xyz, inst, batch = geometry.xyz_cam_m, geometry.instance_id_input, geometry.batch_index
            observed_valid = geometry.normal_obs_valid
        eligible = observed_valid & (inst > 0)
        setup_done = time.perf_counter()
        quality = q0.detach().float().cpu().numpy()
        transferred = time.perf_counter()
        selected = np.zeros((len(inputs['g_cam']), self.num_seeds), np.int64)
        valid = np.zeros_like(selected, bool)
        if mode != 'inference':
            if sampler_targets_cpu is None:
                label_valid = targets['quality_valid'].cpu().numpy()
                label_a = targets['gt_A'].cpu().numpy() if mode == 'train' else None
            else:
                label_valid = sampler_targets_cpu.quality_valid
                label_a = sampler_targets_cpu.gt_A if mode == 'train' else None
                if mode == 'train' and label_a is None:
                    raise ValueError('Training sampler requires CPU gt_A')
        for b in range(len(selected)):
            ids = np.unique(inst[eligible & (batch == b)])
            cursor = 0
            for rank, instance in enumerate(ids):
                quota = self.num_seeds // len(ids) + (rank < self.num_seeds % len(ids))
                if quota == 0:
                    continue
                pool = np.flatnonzero(eligible & (batch == b) & (inst == instance))
                proposal = inference_pool(xyz, pool, quality, quota, fps_fn=self.fps_fn)
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
        self.last_timings = {'sampler_cpu_s': time.perf_counter() - transferred,
                             'q0_d2h_host_s': transferred - setup_done,
                             'sampler_geometry_cpu_setup_s': setup_done - started}
        return (torch.as_tensor(selected, device=q0.device),
                torch.as_tensor(valid, device=q0.device))
