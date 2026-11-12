"""Input-instance pooling without merging identities across packed frames."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .heads import mlp


class InstanceFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_projection = mlp((512, 256, 128))

    def forward(self, local, xyz, batch_index, instance_id, *, checkpointed=False):
        point_instance = torch.zeros(len(xyz), dtype=torch.long, device=xyz.device)
        foreground = instance_id > 0
        keys = torch.stack((batch_index[foreground], instance_id[foreground]), -1)
        unique, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        point_instance[foreground] = inverse + 1
        pooled, centers, extents = [], [], []

        def pool(features, rows):
            selected = features[rows].float()
            return torch.cat((selected.mean(0), selected.max(0).values))

        for slot in range(1, len(unique) + 1):
            rows = torch.where(point_instance == slot)[0]
            if checkpointed and torch.is_grad_enabled():
                pooled.append(checkpoint(pool, local, rows, use_reentrant=False))
            else:
                pooled.append(pool(local, rows))
            geometry = xyz[rows].float()
            centers.append(geometry.mean(0))
            extents.append(geometry.max(0).values - geometry.min(0).values)
        if pooled:
            features = self.pool_projection(torch.stack(pooled))
            centers, extents = torch.stack(centers), torch.stack(extents)
        else:
            features = self.pool_projection(local.new_empty((0, 512)))
            centers = xyz.new_empty((0, 3))
            extents = xyz.new_empty((0, 3))
        # Slot zero is background, which has no instance statistics.
        return {
            'point_instance_index': point_instance,
            'instance_features': torch.cat((features.new_zeros((1, 128)), features)),
            'instance_centers': torch.cat((xyz.new_zeros((1, 3)), centers)),
            'instance_extents': torch.cat((xyz.new_zeros((1, 3)), extents)),
            'instance_keys': unique,
        }


def instance_blocks(stats, xyz, slots):
    """Gather instance128/relative3/extent3 for points or explicit poses."""
    feature = stats['instance_features'][slots]
    relative = (xyz.float() - stats['instance_centers'][slots]) / .1
    relative = relative * (slots > 0)[:, None]
    extent = stats['instance_extents'][slots] / .1
    return feature, relative, extent
