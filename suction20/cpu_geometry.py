"""Batch-local CPU views; observations and sampler supervision stay separate.

Build these in the main process after DataLoader pinning. The referenced packed
batch must remain alive and must not be mutated until its consumers finish.
Neither context belongs in a state_dict, checkpoint or prediction export.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np


def _view(value, dtype):
    if hasattr(value, 'device'):
        if value.device.type != 'cpu':
            raise ValueError('CPU contexts must be built from the original CPU packed batch')
        value = value.detach().numpy()
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype):
        raise ValueError(f'CPU context expected {np.dtype(dtype)}, received {result.dtype}')
    return result


@dataclass(frozen=True)
class CpuGeometry:
    xyz_cam_m: np.ndarray
    normal_obs: np.ndarray
    normal_obs_valid: np.ndarray
    instance_id_input: np.ndarray
    batch_index: np.ndarray
    point_offsets: np.ndarray

    @classmethod
    def from_inputs(cls, inputs):
        return cls(**{key: _view(inputs[key], dtype) for key, dtype in (
            ('xyz_cam_m', np.float32), ('normal_obs', np.float32),
            ('normal_obs_valid', bool), ('instance_id_input', np.int64),
            ('batch_index', np.int64), ('point_offsets', np.int64))})

    @cached_property
    def xyz64(self):
        return np.ascontiguousarray(self.xyz_cam_m, dtype=np.float64)

    @cached_property
    def frame_rows(self):
        return tuple(np.arange(start, stop, dtype=np.int64)
                     for start, stop in zip(self.point_offsets[:-1], self.point_offsets[1:]))


@dataclass(frozen=True)
class SamplerTargetsCPU:
    quality_valid: np.ndarray | None
    gt_A: np.ndarray | None

    @classmethod
    def from_targets(cls, targets):
        return cls(quality_valid=_view(targets['quality_valid'], bool) if 'quality_valid' in targets else None,
                   gt_A=_view(targets['gt_A'], np.float32) if 'gt_A' in targets else None)
