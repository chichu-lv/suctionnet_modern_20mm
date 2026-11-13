"""The fixed 840,000-point / 512-candidate input used by the initial P2 probes."""
from __future__ import annotations

import numpy as np
import torch

POINT_COUNT = 840_000
CANDIDATE_COUNT = 512
FROZEN = {
    'radius_m': .020,
    'self_height_threshold_m': .004,
    'envelope_segments_m': [[-.002, .025, .022], [.025, .070, .028], [.070, .150, .028]],
}


def make_fixture():
    """Return observation inputs and the five positional arguments to query()."""
    xx, yy = np.meshgrid(np.linspace(-.30, .30, 1200), np.linspace(-.175, .175, 700))
    col = np.minimum(((xx + .30) / .12).astype(int), 4)
    row = np.minimum(((yy + .175) / (.35 / 3)).astype(int), 2)
    zz = .65 + .04 * np.sin(14 * xx) + .018 * np.cos(16 * yy) + .050 * ((col + row) % 3)
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1).astype(np.float32)
    ids = (1 + col + 5 * row).ravel().astype(np.int64)
    seams = (np.abs(np.remainder(xx + .30, .12) - .06) > .057) | (
        np.abs(np.remainder(yy + .175, .35 / 3) - .35 / 6) > .055)
    ids[seams.ravel()] = 0
    normals = np.stack([.56 * np.cos(14 * xx), -.288 * np.sin(16 * yy), -np.ones_like(xx)], -1).reshape(-1, 3)
    normals = (normals / np.linalg.norm(normals, axis=1, keepdims=True)).astype(np.float32)
    rng = np.random.default_rng(109)
    candidates = rng.choice(np.flatnonzero(ids > 0), CANDIDATE_COUNT, replace=False)
    inputs = {
        'xyz_cam_m': torch.from_numpy(xyz),
        'normal_obs': torch.from_numpy(normals),
        'normal_obs_valid': torch.ones(len(xyz), dtype=torch.bool),
        'instance_id_input': torch.from_numpy(ids),
        'batch_index': torch.zeros(len(xyz), dtype=torch.long),
        'g_cam': torch.tensor([[0., 0., -1.]]),
    }
    args = (
        torch.from_numpy(xyz[candidates]),
        torch.from_numpy(normals[candidates]),
        torch.from_numpy(ids[candidates]),
        torch.zeros(CANDIDATE_COUNT, dtype=torch.long),
        torch.ones(CANDIDATE_COUNT, dtype=torch.bool),
    )
    return inputs, args
