import sys
import types

import pytest
import torch

from suction20.precision import UpdateLossAccumulator, resolve_precision
from suction20.training import make_optimizer


def test_precision_preserves_legacy_cpu_and_explicit_bf16_policy():
    legacy = resolve_precision({"training": {"amp": True}}, "cpu")
    assert legacy.effective == "fp32" and not legacy.scaler_enabled
    bf16 = resolve_precision({"training": {"precision": "bf16"}}, "cpu")
    assert bf16.effective == "bf16" and not bf16.make_scaler().is_enabled()
    assert bf16.as_dict()["sparse_backbone"] == "fp32"
    assert resolve_precision({"training": {"amp": True}}, "cpu", inference=True).effective == "fp32"
    assert resolve_precision({"training": {"precision": "bf16"}}, "cpu", inference=True) == bf16
    with pytest.raises(ValueError, match="training.precision"):
        resolve_precision({"training": {"precision": "int8"}}, "cpu")


def test_update_loss_readback_detaches_and_averages_terms():
    value = torch.tensor(2., requires_grad=True)
    accumulator = UpdateLossAccumulator(2)
    accumulator.add({"total": value.square(), "q": value})
    accumulator.add({"total": (value + 1).square(), "q": value + 1})
    assert not accumulator.values.requires_grad
    assert accumulator.result() == {"total": 6.5, "q": 2.5}
    assert value.grad is None


def test_fused_request_uses_plain_cpu_adamw_and_original_decay_groups():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    optimizer = make_optimizer(model, {"fused": True, "weight_decay": .02,
                                      "lr_peak": .003, "eps": 1e-7}, "cpu")
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["fused"] is False
    assert optimizer.defaults["lr"] == .003 and optimizer.defaults["eps"] == 1e-7
    assert optimizer.param_groups[0]["weight_decay"] == .02
    assert optimizer.param_groups[1]["weight_decay"] == 0
    assert len(optimizer.param_groups[0]["params"]) == 1
    model(torch.ones(4, 3)).square().mean().backward()
    optimizer.step()
    assert optimizer.state


def test_bf16_dense_keeps_sparse_network_in_fp32(monkeypatch):
    from models.backbones import spconv_resunet14 as wrapper

    class SparseTensor:
        def __init__(self, features, indices, *args):
            self.features, self.indices = features, indices

    class Network(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.linear = torch.nn.Linear(n_in, n_out)

        def forward(self, sparse):
            assert not torch.is_autocast_enabled("cpu")
            assert sparse.features.dtype == torch.float32
            return SparseTensor(self.linear(sparse.features), sparse.indices)

    root, sub = types.ModuleType("spconv"), types.ModuleType("spconv.pytorch")
    root.pytorch, sub.SparseConvTensor = sub, SparseTensor
    monkeypatch.setitem(sys.modules, "spconv", root)
    monkeypatch.setitem(sys.modules, "spconv.pytorch", sub)
    monkeypatch.setattr(wrapper, "_load_external", lambda _: Network)
    backbone = wrapper.SpconvResUNet14("unused", backbone_coordinate_fastpath=True)
    inputs = {"voxel_features": torch.randn(4, 8), "g_cam": torch.zeros(1, 3),
              "voxel_coords": torch.tensor([[0, 0, 0, i] for i in range(4)])}
    with resolve_precision({"training": {"precision": "bf16"}}, "cpu").autocast():
        output = backbone(inputs)
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in backbone.parameters())


def test_inference_inherits_saved_explicit_precision():
    from suction20.training import content_digest
    from suction20.validation import checkpoint_inference_config
    source = {"radius": .02}
    state = {"format_version": "suction20.checkpoint.v1", "source_label_config": source,
             "source_label_config_sha256": content_digest(source), "experiment": "E1",
             "model_config": {"experiment": "E1"},
             "training_config": {"training": {"precision": "bf16"}}}
    resolved = checkpoint_inference_config({"experiment": "E1", "model": {}}, state)
    assert resolved["training"]["precision"] == "bf16"
