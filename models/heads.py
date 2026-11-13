"""Feature MLPs and directed, bounded observation-axis residuals."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def mlp(widths, *, output=False):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if not (output and i == len(widths) - 2):
            layers.extend((nn.LayerNorm(b), nn.ReLU()))
    return nn.Sequential(*layers)


def compile_dense_modules(module, names):
    """Compile only named dense MLPs in place, retaining state-dict names.

    Default Inductor mode allows variable tail chunks without forcing CUDA
    graph capture across the surrounding geometry and per-instance loops.
    Compilation is lazy and remains an explicit from-scratch performance opt-in.
    """
    for name in names:
        module.get_submodule(name).compile(dynamic=True, mode='default')


def chunk_apply(fn, *tensors, chunk_size=16384, checkpointed=False, input_mode='slice'):
    """Bound transient dense activations without dropping observation points."""
    if len(tensors[0]) == 0:
        return fn(*tensors)
    if input_mode == 'split':
        # A shared multi-output view joins gradients once per input, instead of
        # constructing a full input-sized gradient for each independent slice.
        chunks = zip(*(torch.split(t, chunk_size, dim=0) for t in tensors))
    elif input_mode == 'slice':
        chunks = (tuple(t[start:start + chunk_size] for t in tensors)
                  for start in range(0, len(tensors[0]), chunk_size))
    else:
        raise ValueError(f'Unknown chunk input_mode: {input_mode}')
    result = []
    for args in chunks:
        if checkpointed and torch.is_grad_enabled():
            result.append(checkpoint(fn, *args, use_reentrant=False))
        else:
            result.append(fn(*args))
    return torch.cat(result)


def tangent_basis(normal):
    """Deterministic x/y/z tie order; caller supplies valid unit normals."""
    reference = F.one_hot(normal.abs().argmin(-1), 3).to(normal)
    u = F.normalize(torch.cross(reference, normal, dim=-1), dim=-1)
    return u, torch.cross(normal, u, dim=-1)


class AxisHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp((259, 128, 2), output=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, observed, valid):
        # Geometry remains FP32 under mixed precision, including the 30 degree cap.
        ab = self.net(torch.cat((features, observed.to(features)), -1)).float()
        with torch.autocast(device_type=features.device.type, enabled=False):
            n0 = observed.float()
            valid = valid & torch.isfinite(n0).all(-1) & (n0.norm(dim=-1) > 1e-6)
            safe = torch.where(valid[:, None], n0, n0.new_tensor([0., 0., 1.]))
            safe = F.normalize(safe, dim=-1)
            u, v = tangent_basis(safe)
            scale = (math.tan(math.pi / 6) / ab.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1.)
            ab = ab * scale
            n = F.normalize(safe + ab[:, :1] * u + ab[:, 1:] * v, dim=-1)
            return torch.where(valid[:, None], n, torch.zeros_like(n))
