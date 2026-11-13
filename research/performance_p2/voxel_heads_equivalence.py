"""Reproduce one synthetic CPU float64 algebra/gradient comparison.

The comparison jointly covers voxel-level fusion/q0, selected seed features,
and the optional first-affine split of the normal-conditioned axis MLP.
Its objective is synthetic squared error, not the production source_losses
objective or the full directed-axis geometry. It is not a speed benchmark and
does not exercise the sparse backbone, CUDA, or AMP.

Run from any directory:
    python research/performance_p2/voxel_heads_equivalence.py --output-file result.json

The checked-in result records the original completed run; repackaging this
script did not rerun that experiment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
from models.heads import mlp


def run_comparison() -> dict:
    """Use the original seed, shapes, operations and synthetic objective."""
    torch.set_num_threads(1)
    torch.manual_seed(72)
    point_count, voxel_count, frame_count = 2117, 113, 2
    voxel_frame = torch.arange(voxel_count).remainder(frame_count)
    inverse = torch.randint(voxel_count, (point_count,))
    frame = voxel_frame[inverse]
    gravity = F.normalize(torch.randn(frame_count, 3, dtype=torch.double), dim=-1)
    observed = F.normalize(torch.randn(point_count, 3, dtype=torch.double), dim=-1)
    seed = torch.randperm(point_count)[:71]
    target = torch.rand(point_count, dtype=torch.double)
    # Raw-point target heterogeneity stays in the objective after q0 gathering.
    valid = torch.rand(point_count) > .15
    raw_weights = torch.rand(point_count, dtype=torch.double)

    class Dense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion = mlp((393, 256, 256)).double()
            self.q0_head = mlp((256, 128, 1), output=True).double()
            self.axis_net = mlp((259, 128, 2), output=True).double()

    base = Dense()
    new = copy.deepcopy(base)
    features_old = torch.randn(voxel_count, 256, dtype=torch.double, requires_grad=True)
    features_new = features_old.detach().clone().requires_grad_()

    def baseline(net, features):
        local = features[inverse]
        fused = net.fusion(torch.cat((
            local, local.new_zeros(point_count, 134), gravity[frame]
        ), -1))
        q0 = net.q0_head(fused).sigmoid().squeeze(-1)
        ab = net.axis_net(torch.cat((local, observed), -1))
        return q0, fused[seed], ab

    def optimized(net, features):
        fused_voxel = net.fusion(torch.cat((
            features, features.new_zeros(voxel_count, 134), gravity[voxel_frame]
        ), -1))
        q0 = net.q0_head(fused_voxel).sigmoid().squeeze(-1)[inverse]
        # Optional follow-on optimization, not part of the initial P2 scope.
        first = net.axis_net[0]
        h_voxel = F.linear(features, first.weight[:, :256], first.bias)
        h_normal = F.linear(observed, first.weight[:, 256:], None)
        ab = net.axis_net[1:](h_voxel[inverse] + h_normal)
        return q0, fused_voxel[inverse[seed]], ab

    def synthetic_objective(outputs):
        q0, seed_features, ab = outputs
        return (
            ((q0[valid] - target[valid]).square() * raw_weights[valid]).mean()
            + seed_features.square().mean()
            + ab.square().mean()
        )

    out_old = baseline(base, features_old)
    out_new = optimized(new, features_new)
    loss_old = synthetic_objective(out_old)
    loss_new = synthetic_objective(out_new)
    loss_old.backward()
    loss_new.backward()
    new_parameters = dict(new.named_parameters())
    parameter_errors = {
        name: float((parameter.grad - new_parameters[name].grad).abs().max())
        for name, parameter in base.named_parameters()
    }
    return {
        "scope": "one synthetic CPU float64 algebra/gradient comparison; no CUDA speed or AMP claim",
        "checked_components": [
            "no-instance voxel fusion and q0",
            "selected seed fused features",
            "optional axis first-affine split with pointwise observed normals, before axis geometry"
        ],
        "objective": "synthetic weighted q0 squared error plus squared seed features and pregeometry axis ab; not production source_losses",
        "not_checked": [
            "sparse backbone",
            "production source_losses and instance-frame-batch reduction",
            "directed-axis geometric normalization",
            "CUDA/AMP execution",
            "runtime speed"
        ],
        "state_dict_comparison_scope": "the prototype Dense modules only",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "raw_points": point_count,
        "voxels": voxel_count,
        "frames": frame_count,
        "max_abs_q0": float((out_old[0] - out_new[0]).abs().max().detach()),
        "max_abs_seed_fused": float((out_old[1] - out_new[1]).abs().max().detach()),
        "max_abs_axis_pregeometry_ab": float((out_old[2] - out_new[2]).abs().max().detach()),
        "abs_loss": float((loss_old - loss_new).abs().detach()),
        "max_abs_voxel_feature_grad": float((features_old.grad - features_new.grad).abs().max()),
        "max_abs_parameter_grad": max(parameter_errors.values()),
        "state_dict_keys_unchanged": list(base.state_dict()) == list(new.state_dict()),
        "state_dict_shapes_unchanged": all(
            base.state_dict()[name].shape == new.state_dict()[name].shape
            for name in base.state_dict()
        ),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_comparison()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2) + "\n"
    args.output_file.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

