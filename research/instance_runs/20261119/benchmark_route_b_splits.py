#!/usr/bin/env python3
"""Compare route-B batch splitting with a historical local Git revision.

This is a bounded, synthetic, single-thread CPU backward-allocation probe.
It does not measure GPU throughput, peak memory, or real-data training speed.
No checkpoint or tensor dump is written; --output receives a JSON summary.

Example, from the repository root::

    python research/instance_runs/20261119/benchmark_route_b_splits.py \
        --output /tmp/route-b-splits.json

The baseline is loaded from ``git show COMMIT:models/instance_route_b.py`` as
``models._baseline_route_b`` without changing the checkout. Other imports use
the current checkout, so this probe targets the route-B model changes only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = "e6d9578f5fb39174cf9a81c769672800c06a9306"
MODEL_PATH = "models/instance_route_b.py"
SELECTED_OPERATORS = ("aten::slice_backward", "aten::cat")


def _git(*arguments):
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments], check=True, capture_output=True,
        text=True, timeout=30,
    ).stdout.strip()


def load_baseline(ref):
    """Resolve a commit before passing its file path to git show; never a shell."""
    if not ref or any(ord(character) < 32 for character in ref):
        raise ValueError("--baseline-ref must be a nonempty Git revision without control characters")
    try:
        # --end-of-options prevents a user-supplied revision from becoming an
        # option. The second command receives only Git's resolved commit ID.
        commit = _git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
        source = _git("show", f"{commit}:{MODEL_PATH}")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise ValueError(
            f"Cannot read route-B baseline {ref!r} from local Git history. "
            "Fetch the required branch/revision or provide an available --baseline-ref. "
            f"Git detail: {detail.strip()}"
        ) from error
    name = "models._baseline_route_b"
    module = types.ModuleType(name)
    module.__file__ = f"git:{commit}:{MODEL_PATH}"
    module.__package__ = "models"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, commit


def plane_observation(height, width, uid):
    """A synthetic RGB-D plane with deterministic, supported canonical queries."""
    y, x = np.mgrid[:height, :width]
    intrinsics = np.array(((600., 0., (width - 1) / 2),
                           (0., 610., (height - 1) / 2), (0., 0., 1.)))
    z = np.full((height, width), .6, dtype=np.float64)
    xyz = np.stack(((x - intrinsics[0, 2]) * z / intrinsics[0, 0],
                    (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z), -1)
    normals = np.zeros_like(xyz)
    normals[..., 2] = -1
    mask = np.ones((height, width), dtype=bool)
    rgb = np.stack((x / width, y / height, np.full_like(z, .4)), -1).astype(np.float32)
    uv = np.array(((width // 2, height // 2),
                   (width // 2 - 8, height // 2 + 3),
                   (width // 2 + 9, height // 2 - 4)))
    return dict(rgb=rgb, xyz_map=xyz, mask=mask, instance_mask=mask.copy(),
                normal_map=normals, normal_valid=mask.copy(),
                gravity=np.array((0., 1., 0.)), intrinsics=intrinsics,
                query_points=xyz[uv[:, 1], uv[:, 0]],
                query_normals=normals[uv[:, 1], uv[:, 0]], sample_uid=uid)


def make_observations(instances):
    rows = []
    for index in range(instances):
        height, width = ((48, 64), (52, 60), (76, 100))[index % 3]
        row = plane_observation(height, width, f"cpu-audit/{index}")
        if index % 3 == 1:
            row["query_points"] = row["query_points"][:2]
            row["query_normals"] = row["query_normals"][:2]
        rows.append(row)
    return rows


def prediction_tensors(value, path=""):
    """Include prediction diagnostics as well as the public score tensors."""
    if torch.is_tensor(value):
        yield path, value.detach().clone()
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from prediction_tensors(child, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from prediction_tensors(child, f"{path}/{index}")


def measure(module, rows, *, seed, query_chunk, dense_chunk):
    torch.manual_seed(seed)
    model = module.InstanceSuction2DFastV3(query_chunk=query_chunk).train()
    model.configure_execution(dense_query_chunk=dense_chunk)
    model.set_training_context(13)
    prepared = [module.prepare_source_inputs_cpu(row, sample_image_channels=True)
                for row in rows]
    outputs = model.forward_instances(prepared)
    loss = torch.stack([
        sum(output[key].square().mean()
            for key in ("q_inst", "seal", "wrench", "ell", "task_logit"))
        for output in outputs
    ]).mean()
    # Profile backward only. Input preparation and forward activations are not
    # included in these operator-allocation totals.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
    ) as profile:
        loss.backward()
    operators = {name: {"calls": 0, "allocated_output_bytes": 0}
                 for name in SELECTED_OPERATORS}
    for event in profile.key_averages():
        if event.key in operators:
            operators[event.key] = {
                "calls": event.count,
                "allocated_output_bytes": event.cpu_memory_usage,
            }
    # These two operation families are disjoint. Do not also add nested
    # SliceBackward/SplitBackward, zeros, or empty events to these allocations.
    summary = {
        "backward_operators": operators,
        "selected_operator_allocation_bytes": sum(
            item["allocated_output_bytes"] for item in operators.values()),
    }
    predictions = dict(prediction_tensors(outputs))
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters() if parameter.grad is not None}
    return summary, predictions, gradients


def compare_tensors(baseline, candidate):
    base_predictions, base_gradients = baseline
    predictions, gradients = candidate
    prediction_keys_match = base_predictions.keys() == predictions.keys()
    mismatches = [key for key in base_predictions.keys() & predictions.keys()
                  if not torch.equal(base_predictions[key], predictions[key])]
    gradient_keys_match = base_gradients.keys() == gradients.keys()
    max_error, max_error_parameter = 0., None
    gradients_exact = gradient_keys_match
    gradients_close = gradient_keys_match
    for name in base_gradients.keys() & gradients.keys():
        before, after = base_gradients[name], gradients[name]
        if before.shape != after.shape:
            gradients_exact = gradients_close = False
            continue
        difference = (before - after).abs().max().item()
        if difference > max_error:
            max_error, max_error_parameter = difference, name
        gradients_exact &= torch.equal(before, after)
        gradients_close &= torch.allclose(before, after, rtol=1e-3, atol=2e-5)
    return {
        "all_prediction_tensors_exact": prediction_keys_match and not mismatches,
        "prediction_tensor_count": len(base_predictions),
        "prediction_keys_match": prediction_keys_match,
        "prediction_mismatches": sorted(mismatches),
        "parameter_gradient_count": len(base_gradients),
        "gradient_parameter_keys_match": gradient_keys_match,
        "all_parameter_gradients_exact": gradients_exact,
        "all_parameter_gradients_close": gradients_close,
        "gradient_tolerance": {"rtol": 1e-3, "atol": 2e-5},
        "max_gradient_absolute_error": max_error,
        "max_gradient_error_parameter": max_error_parameter,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=2)
    parser.add_argument("--dense-chunk", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=672)
    args = parser.parse_args(argv)
    if min(args.instances, args.query_chunk, args.dense_chunk) < 1:
        parser.error("--instances, --query-chunk, and --dense-chunk must be positive")
    sys.path.insert(0, str(ROOT))
    import models.instance_route_b as current
    try:
        historical, commit = load_baseline(args.baseline_ref)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    rows = make_observations(args.instances)
    settings = dict(seed=args.seed, query_chunk=args.query_chunk, dense_chunk=args.dense_chunk)
    before, before_predictions, before_gradients = measure(historical, rows, **settings)
    after, after_predictions, after_gradients = measure(current, rows, **settings)
    comparison = compare_tensors((before_predictions, before_gradients),
                                 (after_predictions, after_gradients))
    old_bytes = before["selected_operator_allocation_bytes"]
    new_bytes = after["selected_operator_allocation_bytes"]
    result = {
        "scope": "synthetic CPU backward allocations; not GPU throughput or peak memory",
        "torch_version": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
        "baseline_ref": args.baseline_ref, "baseline_commit": commit,
        "candidate": f"current working tree {MODEL_PATH}",
        "instances": args.instances, "query_lengths": [len(row["query_points"]) for row in rows],
        **settings,
        "allocation_accounting": (
            "Sum only aten::slice_backward and aten::cat CPU output allocations; "
            "exclude their nested autograd, zeros, and empty events. "
            "Cumulative operator allocations are not peak memory."
        ),
        "baseline": before, "candidate_result": after, "equivalence": comparison,
        "selected_allocation_reduction_fraction": (1 - new_bytes / old_bytes) if old_bytes else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if (comparison["all_prediction_tensors_exact"]
                 and comparison["all_parameter_gradients_close"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
