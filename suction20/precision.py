"""Explicit autocast policy for new training; legacy ``amp`` stays compatible.

spconv 2.3.6 has no BF16 tensorview mapping and its custom AMP entry points
request FP16. The backbone wrapper therefore runs sparse layers in FP32 when
the surrounding dense model uses BF16. This is a mixed policy, not all-BF16.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    requested: str
    effective: str
    device_type: str

    @property
    def enabled(self):
        return self.effective != "fp32"

    @property
    def dtype(self):
        return {"fp32": torch.float32, "fp16": torch.float16,
                "bf16": torch.bfloat16}[self.effective]

    @property
    def scaler_enabled(self):
        return self.device_type == "cuda" and self.effective == "fp16"

    def autocast(self):
        return torch.autocast(self.device_type, dtype=self.dtype, enabled=self.enabled)

    def make_scaler(self):
        # A disabled scaler retains the existing checkpoint interface but does
        # not scale, unscale, check gradients, or update a dynamic loss scale.
        return torch.amp.GradScaler("cuda", enabled=self.scaler_enabled)

    def as_dict(self):
        return {"requested": self.requested, "dense": self.effective,
                "sparse_backbone": "fp32" if self.effective == "bf16" else self.effective,
                "loss": "fp32", "parameter_storage": "fp32",
                "grad_scaling": self.scaler_enabled,
                "bf16_sparse_policy": "fp32_spconv" if self.effective == "bf16" else None}


def resolve_precision(config, device, *, inference=False):
    """Explicit precision applies to training and inference on the same config.

    Old configs retain CUDA FP16 training and FP32 inference. Explicit BF16 can
    also run the small CPU reference tests; legacy amp never enables CPU AMP.
    """
    device = torch.device(device)
    train = config.get("training", {})
    explicit = "precision" in train
    requested = train.get("precision", "fp16" if train.get("amp", True) else "fp32")
    if requested not in ("fp32", "fp16", "bf16"):
        raise ValueError("training.precision must be fp32, fp16 or bf16")
    if inference and not explicit:
        effective = "fp32"
    elif device.type == "cuda" or (device.type == "cpu" and explicit and requested == "bf16"):
        effective = requested
    else:
        effective = "fp32"
    if device.type == "cuda" and effective == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 training requires a BF16-capable CUDA device")
    return PrecisionPolicy(requested, effective, device.type)


class UpdateLossAccumulator:
    """Retain detached scalars only; read their vector once per update."""
    def __init__(self, accumulation):
        self.accumulation = accumulation
        self.names = None
        self.values = None

    def add(self, losses):
        names = tuple(losses)
        if self.names is None:
            self.names = names
        elif names != self.names:
            raise ValueError("Loss terms changed within an accumulation update")
        values = torch.stack([losses[name].detach().float() for name in names]) / self.accumulation
        self.values = values if self.values is None else self.values + values

    def result(self):
        if self.values is None:
            return {}
        return dict(zip(self.names, self.values.cpu().tolist()))
