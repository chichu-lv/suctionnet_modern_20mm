import numpy as np
import torch

from models.candidate_sampler import CandidateSampler, fps_indices


def test_fps_ties_unique_and_padding():
    xyz = np.array([[0., 0, 0], [1., 0, 0], [-1., 0, 0], [0., 0, 0]])
    assert fps_indices(xyz, [3, 2, 1, 0], 4).tolist() == [0, 1, 2, 3]
    inputs = {'xyz_cam_m': torch.tensor(xyz), 'normal_obs_valid': torch.ones(4, dtype=torch.bool),
              'instance_id_input': torch.tensor([2, 2, 1, 0]), 'batch_index': torch.zeros(4, dtype=torch.long),
              'g_cam': torch.tensor([[0., 0., -1.]])}
    sampler = CandidateSampler(8)
    idx, valid = sampler(inputs, torch.tensor([1., 0., 0., 100.]))
    assert valid.sum() == 3
    assert idx[valid].tolist() == [2, 0, 1]
    # Inference ignores even deliberately malformed GT; unknown still eligible.
    other, _ = sampler(inputs, torch.tensor([1., 0., 0., 100.]), targets={})
    assert torch.equal(other, idx)


def test_training_a_pools_unknown_fallback_and_reproducibility():
    inputs = {'xyz_cam_m': torch.randn(10, 3), 'normal_obs_valid': torch.ones(10, dtype=torch.bool),
              'instance_id_input': torch.ones(10, dtype=torch.long), 'batch_index': torch.zeros(10, dtype=torch.long),
              'g_cam': torch.tensor([[0., 0., -1.]])}
    targets = {'quality_valid': torch.tensor([True] * 6 + [False] * 4),
               'gt_A': torch.tensor([1.] * 3 + [0.] * 7)}
    sampler = CandidateSampler(10)
    torch.manual_seed(42)
    idx, valid = sampler(inputs, torch.arange(10).float(), mode='train', targets=targets, epoch=0)
    torch.manual_seed(42)
    again, _ = sampler(inputs, torch.arange(10).float(), mode='train', targets=targets, epoch=0)
    assert torch.equal(idx, again)
    assert valid.all() and len(idx.unique()) == 10
    assert set(idx[0, :6].tolist()) == set(range(6))
