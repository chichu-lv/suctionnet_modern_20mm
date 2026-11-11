"""Verified external SPconvUNet14D, without vendoring upstream source."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
import torch
from torch import nn

from ..heads import mlp

_LOAD_LOCK = threading.Lock()
_BLOBS = {
    'resnet.py': 'aa10c0f7810470c49a98572650f429d36a25b41f',
    'backbone_resunet14.py': 'ad1a0eef6b2ab7d8f03d8d7a4a981dcc1e301c96',
}


def coordinate_rows(input_coords, output_coords):
    """Return output row for every input coordinate, checking one-to-one coverage."""
    if input_coords.shape != output_coords.shape:
        raise ValueError('Sparse output must preserve the complete input active set')
    if len(input_coords) == 0:
        return torch.empty(0, dtype=torch.long, device=input_coords.device)
    all_coords = torch.cat((input_coords.long(), output_coords.long()))
    _, inverse = torch.unique(all_coords, dim=0, return_inverse=True)
    n = len(input_coords)
    left, right = inverse[:n], inverse[n:]
    if left.unique().numel() != n or right.unique().numel() != n:
        raise ValueError('Duplicate sparse coordinates')
    lookup = torch.full((int(inverse.max()) + 1,), -1, dtype=torch.long, device=inverse.device)
    lookup[right] = torch.arange(n, device=inverse.device)
    rows = lookup[left]
    if (rows < 0).any():
        raise ValueError('Sparse output missing an input coordinate')
    return rows


def _load_external(root):
    root = Path(root) / 'models'
    for name, expected in _BLOBS.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'{path}; run scripts/setup_backbone.py --output BACKBONE_ROOT')
        content = path.read_bytes()
        digest = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if digest != expected:
            raise ValueError(f'Pinned backbone source mismatch: {path}')
    # The pinned file imports models.resnet. Bind that single dependency only
    # during execution, preserving our own models package and any prior module.
    with _LOAD_LOCK:
        prior = sys.modules.get('models.resnet')
        try:
            for filename, module_name in [('resnet.py', 'models.resnet'),
                                          ('backbone_resunet14.py', '_suction20_pinned_backbone')]:
                spec = importlib.util.spec_from_file_location(module_name, root / filename)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            return module.SPconvUNet14D
        finally:
            if prior is None:
                sys.modules.pop('models.resnet', None)
            else:
                sys.modules['models.resnet'] = prior


class SpconvResUNet14(nn.Module):
    def __init__(self, backbone_root):
        super().__init__()
        self.network = _load_external(backbone_root)(8, 512)
        self.projection = mlp((512, 256))

    def forward(self, inputs):
        # spconv indice kernels interpret [N,4] as a row-major memory buffer.
        # NumPy column stacking can produce Fortran-contiguous coordinates;
        # dtype/device conversion alone preserves those strides in PyTorch.
        coords = inputs['voxel_coords'].to(dtype=torch.int32).contiguous()
        if len(coords) == 0:
            return self.projection(inputs['voxel_features'].new_empty((0, 512)))
        import spconv.pytorch as spconv
        # Four stride-2 stages require at least 16 cells along each axis.
        shape = (coords[:, 1:].max(0).values + 1).clamp_min(16).tolist()
        sparse = spconv.SparseConvTensor(inputs['voxel_features'].contiguous(), coords, shape,
                                        len(inputs['g_cam']))
        output = self.network(sparse)
        rows = coordinate_rows(coords, output.indices)
        return self.projection(output.features[rows])

    def parameter_groups(self):
        encoder, decoder = [], []
        for name, param in self.network.named_parameters():
            first = name.split('.')[0]
            is_decoder = first.startswith(('convtr', 'bntr')) or first in {
                'block5', 'block6', 'block7', 'block8', 'final'}
            (decoder if is_decoder else encoder).append(param)
        decoder.extend(self.projection.parameters())
        return {'encoder': encoder, 'decoder': decoder}
