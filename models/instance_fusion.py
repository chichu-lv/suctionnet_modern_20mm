"""Input-instance pooling without merging identities across packed frames."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .heads import mlp


@dataclass(frozen=True)
class InstanceLayout:
    """One ordered point layout reused by pooling and broadcast backward."""
    order: torch.Tensor
    tile_offsets: torch.Tensor
    tiles_per_instance: torch.Tensor
    row_bounds: tuple[int, ...]


@torch.no_grad()
def instance_layout(slots, slot_count, *, tile_size=1024):
    """Bound each first-stage segment, including for a large foreground object.

    The small count vector is read once, replacing the per-instance nonzero
    shape synchronizations in the indexed pooling path. Original point order
    inside an instance is retained for the existing mean/max pooling.
    """
    order = torch.argsort(slots, stable=True)
    counts = torch.bincount(slots, minlength=slot_count).cpu().tolist()
    bounds, offsets, tiles = [0], [0], []
    for count in counts:
        begin, end = bounds[-1], bounds[-1] + count
        bounds.append(end)
        tiles.append((count + tile_size - 1) // tile_size)
        offsets.extend(range(begin + tile_size, end, tile_size))
        if count:
            offsets.append(end)
    return InstanceLayout(order,
        torch.tensor(offsets, dtype=torch.long, device=slots.device),
        torch.tensor(tiles, dtype=torch.long, device=slots.device), tuple(bounds))


class _SegmentBroadcast(torch.autograd.Function):
    """Index forward once; reduce the repeated rows without scatter atomics."""
    @staticmethod
    def forward(ctx, features, slots, order, offsets, tile_counts):
        ctx.save_for_backward(order, offsets, tile_counts)
        ctx.feature_shape, ctx.feature_dtype = features.shape, features.dtype
        return features.index_select(0, slots)

    @staticmethod
    def backward(ctx, grad_points):
        order, offsets, tile_counts = ctx.saved_tensors
        if not len(grad_points):
            return grad_points.new_zeros(ctx.feature_shape), None, None, None, None
        # FP16/BF16 contributions accumulate in FP32. FP64 remains available for
        # numerical diagnostics; new training need not reproduce old AMP sums.
        dtype = torch.float64 if ctx.feature_dtype == torch.float64 else torch.float32
        ordered = grad_points.index_select(0, order).to(dtype)
        # Tiles avoid a single enormous segment for large instances. Offsets
        # and lengths are constructed here from the complete internal slot map;
        # unsafe=True skips redundant device-to-host validation of that layout.
        partial = torch.segment_reduce(ordered, 'sum', offsets=offsets, axis=0, unsafe=True)
        gradient = torch.segment_reduce(partial, 'sum', lengths=tile_counts, axis=0, unsafe=True)
        return gradient.to(ctx.feature_dtype), None, None, None, None


def segment_broadcast(features, slots, layout):
    """Broadcast a complete frame's instance features with a grouped adjoint."""
    return _SegmentBroadcast.apply(features, slots, layout.order,
                                   layout.tile_offsets, layout.tiles_per_instance)


class InstanceFusion(nn.Module):
    def __init__(self, *, group_points=False):
        super().__init__()
        self.group_points = group_points
        self.pool_projection = mlp((512, 256, 128))

    def forward(self, local, xyz, batch_index, instance_id, *, checkpointed=False):
        point_instance = torch.zeros(len(xyz), dtype=torch.long, device=xyz.device)
        foreground = instance_id > 0
        keys = torch.stack((batch_index[foreground], instance_id[foreground]), -1)
        unique, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        point_instance[foreground] = inverse + 1
        layout = instance_layout(point_instance, len(unique) + 1) if self.group_points else None
        pooled, centers, extents = [], [], []

        def pool(features, rows):
            selected = features[rows].float()
            return torch.cat((selected.mean(0), selected.max(0).values))

        for slot in range(1, len(unique) + 1):
            rows = (torch.where(point_instance == slot)[0] if layout is None else
                    layout.order[layout.row_bounds[slot]:layout.row_bounds[slot + 1]])
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
        result = {
            'point_instance_index': point_instance,
            'instance_features': torch.cat((features.new_zeros((1, 128)), features)),
            'instance_centers': torch.cat((xyz.new_zeros((1, 3)), centers)),
            'instance_extents': torch.cat((xyz.new_zeros((1, 3)), extents)),
            'instance_keys': unique,
        }
        if layout is not None:
            result['_instance_layout'] = layout
        return result


def instance_blocks(stats, xyz, slots, *, point_features=None):
    """Gather instance128/relative3/extent3 for points or explicit poses."""
    feature = stats['instance_features'][slots] if point_features is None else point_features
    relative = (xyz.float() - stats['instance_centers'][slots]) / .1
    relative = relative * (slots > 0)[:, None]
    extent = stats['instance_extents'][slots] / .1
    return feature, relative, extent
