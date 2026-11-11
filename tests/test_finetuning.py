from copy import deepcopy

import pytest
import torch
from torch import nn

from suction20.finetuning import (apply_phase, make_finetune_optimizer, resolve_finetune_config,
                                 target_loss, target_selection)
from suction20.training import load_config


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.decoder = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        self.head = nn.Linear(3, 1)

    def parameter_groups(self):
        return {"encoder": list(self.encoder.parameters()), "decoder": list(self.decoder.parameters()),
                "heads": list(self.head.parameters())}

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))


def test_phase_transition_preserves_heads_optimizer_and_freezes_bn():
    model = TinyModel()
    optimizer = make_finetune_optimizer(model, {})
    before = deepcopy(model.state_dict())
    for step in [199, 200]:
        model.train()
        phase = apply_phase(model, optimizer, step)
        assert not model.decoder[1].training
        assert not model.decoder[1].weight.requires_grad
        model(torch.ones(3, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in model.encoder.parameters())
        assert int(optimizer.state[model.head.weight]["step"]) == (1 if step == 199 else 2)
        if step == 199:
            torch.testing.assert_close(model.decoder[0].weight, before["decoder.0.weight"])
            assert phase == "heads_only"
        else:
            assert not torch.equal(model.decoder[0].weight, before["decoder.0.weight"])
            assert phase == "decoder_and_heads"
    torch.testing.assert_close(model.encoder.weight, before["encoder.weight"])
    torch.testing.assert_close(model.decoder[1].running_mean, before["decoder.1.running_mean"])


def test_attempt_loss_scores_executed_pose_without_proposal_or_axis_head():
    class ExecutedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
        def encode_scene(self, inputs):
            return inputs
        def score_at_pose(self, encoded, p, n, instance_id, **kwargs):
            assert torch.equal(p, attempts["point_cam_m"])
            assert torch.equal(n, attempts["normal_out_cam"])
            return {"task_success_logit": self.weight.expand(len(p)),
                    "pose_feature_valid": torch.tensor([True, False])}
        def forward(self, *args, **kwargs):
            raise AssertionError("Attempts must not sample model proposals")
    attempts = {"point_cam_m": torch.tensor([[.01, 0., .4], [.02, 0., .4]]),
                "normal_out_cam": torch.tensor([[0., 0., -1.]] * 2),
                "instance_id": torch.ones(2, dtype=torch.long), "batch_index": torch.zeros(2, dtype=torch.long),
                "feature_point_index": torch.full((2,), -1), "label_valid": torch.ones(2, dtype=torch.bool),
                "success": torch.tensor([1., 0.])}
    model = ExecutedModel()
    losses, diagnostics = target_loss(model, {"inputs": {}, "targets": {}, "attempts": attempts}, "attempt_success")
    losses["total"].backward()
    assert model.weight.grad == -.5
    assert diagnostics["unmapped_attempt_count"] == 1


@pytest.mark.parametrize("mode", ["dense_proxy", "attempt_success"])
def test_shipped_finetune_configs_resolve_architecture_and_source_losses(mode):
    config = load_config(f"docs/implementation/examples/finetune_{mode}.yaml")
    checkpoint = {"quality_target": "gt_B", "experiment": "E5", "model_config": {"quality_target": "gt_B"},
                  "training_config": {"loss": {"weights": {"normal": 2.}}, "sampling": {}}}
    result = resolve_finetune_config(config, checkpoint)
    assert result["model"]["use_task_success_head"] == (mode == "attempt_success")
    assert result["source_loss"]["weights"]["normal"] == 2.
    invalid = deepcopy(config)
    invalid["training"]["phases"][0]["lr"]["heads"] = .1
    with pytest.raises(ValueError, match="phase"):
        resolve_finetune_config(invalid, checkpoint)


def test_undefined_attempt_auc_uses_bce_without_filling_zero():
    assert target_selection({"attempt_AUROC": None, "attempt_BCE": .6}, "attempt_success") == ("attempt_BCE", -.6)
    assert target_selection({"attempt_AUROC": None}, "attempt_success") is None
