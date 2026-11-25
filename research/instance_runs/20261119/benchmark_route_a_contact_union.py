#!/usr/bin/env python3
"""Compare only the opt-in A contact-union projection on bounded CPU fixtures.

Both runs use the current implementation, identical prepared geometry, weights,
and input order. The injected Linear/BatchNorm backbone is explicitly synthetic;
this measures neither SpConv nor GPU speed. Allocation totals are cumulative
positive CPU operator allocations, not peak RAM/VRAM. No optimizer is stepped.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.instance_route_a import InstanceSuction3D, compact_prepared_source_item


class InjectedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(12, 256)
        self.norm = nn.BatchNorm1d(256)

    def forward(self, inputs):
        return self.norm(self.projection(inputs["voxel_features"]))


def observation(index, side=256, queries=128):
    y, x = np.mgrid[:side, :side]
    points = np.stack(((x.ravel() - side // 2) * .001,
                       (y.ravel() - side // 2) * .001, np.full(side*side, .6)), -1)
    normals = np.tile([0., 0., -1.], (len(points), 1))
    source_ids = np.arange(len(points), dtype=np.int64)
    chosen = np.linspace(0, len(points)-1, queries).astype(np.int64)
    rgb = np.stack((x / side, y / side, np.full((side, side), .3 + index * .2)), -1).astype(np.float32)
    return dict(rgb=rgb, support_points=points, support_rgb=rgb.reshape(-1, 3).astype(np.float64),
                support_normals=normals, support_normal_valid=np.ones(len(points), bool),
                support_rgb_valid=np.ones(len(points), bool), support_components=np.ones(len(points), np.int64),
                support_source_ids=source_ids, support_uv=np.column_stack((x.ravel(), y.ravel())),
                centroid=points.mean(0), gravity=np.array([0., 1., 0.]),
                query_points=points[chosen].copy(), query_normals=normals[chosen].copy(),
                query_ids=np.arange(queries, dtype=np.int64), sample_uid=f"synthetic-union/{index}")


def run(model, rows, enabled):
    os.environ["SUCTION_A_CONTACT_UNION"] = "1" if enabled else "0"
    model.zero_grad(set_to_none=True)
    output = model.forward_instances(rows)
    loss = sum(part[key].square().mean() for part in output
               for key in ("q_inst", "seal", "wrench", "ell", "task_logit")) / len(output)
    loss.backward()
    gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters()
                 if value.grad is not None}
    return output, gradients


def errors(before, after, old_model, new_model):
    maximum_output = maximum_gradient = 0.
    learned = {"q_inst", "seal", "wrench", "ell", "log_scale", "log_variance", "task_logit", "task_probability"}
    for expected, actual in zip(before[0], after[0]):
        assert expected.keys() == actual.keys()
        for name, value in expected.items():
            if torch.is_tensor(value):
                if name in learned:
                    torch.testing.assert_close(actual[name], value, rtol=2e-5, atol=1e-7, msg=name)
                    if value.numel():
                        maximum_output = max(maximum_output, float((actual[name] - value).abs().max()))
                else:
                    torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True, msg=name)
            else:
                assert actual[name] == value
    assert before[1].keys() == after[1].keys()
    for name, expected in before[1].items():
        actual = after[1][name]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7, msg=name)
        maximum_gradient = max(maximum_gradient, float((actual - expected).abs().max()))
    for (name, expected), (other, actual) in zip(old_model.named_buffers(), new_model.named_buffers()):
        assert name == other
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    return dict(max_prediction_abs_error=maximum_output, max_parameter_gradient_abs_error=maximum_gradient,
                parameters_with_gradients=len(before[1]), geometry_exact=True, bn_buffers_exact=True,
                acceptance_rtol=2e-5, acceptance_atol=1e-7)


def allocation_profile(model, rows, enabled):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        run(model, rows, enabled)
    operators = {}
    for event in profile.key_averages():
        if event.key.startswith("aten::") and event.self_cpu_memory_usage > 0:
            operators[event.key] = dict(calls=event.count, positive_self_cpu_bytes=event.self_cpu_memory_usage)
    return dict(cumulative_positive_self_cpu_bytes=sum(row["positive_self_cpu_bytes"] for row in operators.values()),
                operators=operators)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if not 3 <= args.rounds <= 20:
        parser.error("--rounds must be between 3 and 20")
    torch.set_num_threads(1)
    torch.manual_seed(20261119)
    before = InstanceSuction3D(backbone=InjectedBackbone(), backbone_points=2048, query_chunk_size=32).train()
    after = copy.deepcopy(before)
    observations = [observation(index) for index in range(2)]
    previous_preparation_setting = os.environ.get("SUCTION_A_CONTACT_UNION")
    try:
        os.environ["SUCTION_A_CONTACT_UNION"] = "1"
        rows = [compact_prepared_source_item({"observation": before.prepare_source_geometry(obs)})["observation"]
                for obs in observations]
    finally:
        if previous_preparation_setting is None:
            os.environ.pop("SUCTION_A_CONTACT_UNION", None)
        else:
            os.environ["SUCTION_A_CONTACT_UNION"] = previous_preparation_setting
    counts = {"reference": [], "union": []}
    hooks = [before.local_projection.register_forward_pre_hook(lambda _m, a: counts["reference"].append(len(a[0]))),
             after.local_projection.register_forward_pre_hook(lambda _m, a: counts["union"].append(len(a[0])))]
    old_setting = os.environ.get("SUCTION_A_CONTACT_UNION")
    try:
        first = run(before, rows, False)
        second = run(after, rows, True)
        comparison = errors(first, second, before, after)
        times = {"reference": [], "union": []}
        for index in range(args.rounds):
            jobs = [("reference", before, False), ("union", after, True)]
            if index % 2:
                jobs.reverse()
            results = {}
            for name, model, enabled in jobs:
                started = time.perf_counter()
                results[name] = run(model, rows, enabled)
                times[name].append((time.perf_counter() - started) * 1000.)
            this_comparison = errors(results["reference"], results["union"], before, after)
            for key in ("max_prediction_abs_error", "max_parameter_gradient_abs_error"):
                comparison[key] = max(comparison[key], this_comparison[key])
        allocation = {"reference": allocation_profile(before, rows, False),
                      "union": allocation_profile(after, rows, True)}
        # Verify the actual dense precision profile once on this same support
        # workload; this is a numerical comparison, not a second tuning sweep.
        before.configure_execution(precision="bf16_dense")
        after.configure_execution(precision="bf16_dense")
        bf16_comparison = errors(run(before, rows, False), run(after, rows, True), before, after)
        assert len(set(counts["reference"])) == len(set(counts["union"])) == 1
        result = dict(device="CPU only", torch=torch.__version__, numpy=np.__version__, threads=1,
            implementation="current modified worktree; only SUCTION_A_CONTACT_UNION differs",
            reference_flag="0 (default)", union_flag="1 (opt-in training compact inputs only)",
            scope="2 synthetic 65536-point instances, 128 queries each, injected Linear/BatchNorm backbone; no optimizer, file I/O, native geometry, IPC or GPU timing",
            dense_precision="fp32", comparison=comparison, bf16_comparison=bf16_comparison,
            work=dict(raw_support_rows=counts["reference"][0], contact_union_rows=counts["union"][0],
                      rows_per_instance=[len(row["_route_a_contact_feature_rows"]) for row in rows],
                      full_interpolated_feature_bytes=counts["reference"][0]*256*4,
                      union_interpolated_feature_bytes=counts["union"][0]*256*4,
                      full_projected_feature_bytes=counts["reference"][0]*64*4,
                      union_projected_feature_bytes=counts["union"][0]*64*4),
            timing={name: dict(median_ms=statistics.median(values), rounds_ms=values) for name, values in times.items()},
            allocation=allocation,
            limitation="Cumulative CPU operator allocations are not peak memory; synthetic CPU timing does not establish a GPU throughput gain. BF16 numerical comparison uses the same fixed inputs once.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("work", "comparison", "timing")}, ensure_ascii=False))
    finally:
        for hook in hooks:
            hook.remove()
        if old_setting is None:
            os.environ.pop("SUCTION_A_CONTACT_UNION", None)
        else:
            os.environ["SUCTION_A_CONTACT_UNION"] = old_setting


if __name__ == "__main__":
    main()
