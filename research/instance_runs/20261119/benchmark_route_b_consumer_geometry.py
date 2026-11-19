#!/usr/bin/env python3
"""Paired CPU projected-ring consumer work, without a GPU throughput claim.

The historical model AND its preparation module are loaded from local Git.
Each side receives its own valid private wire format and the same immutable
synthetic observation, texture values, poses and per-update phases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import models.instance_route_b as current
from benchmark_route_b_splits import (_git, compare_tensors, make_observations,
                                      measure, prediction_tensors)
from benchmark_route_b_static_roi import byte_count, observation, prepared_item

DEFAULT_BASELINE = "d1c600b1b8ee81f49c9b988c51e217a311699e3b"
SELECTED = ("aten::index", "aten::clamp", "aten::where", "aten::sin", "aten::cos",
            "aten::linalg_cross", "aten::grid_sampler_2d", "aten::sort")


def historical_modules(ref):
    if not ref or any(ord(c) < 32 for c in ref):
        raise ValueError("baseline-ref must be a nonempty Git revision without control characters")
    commit = _git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    for filename, name in (("instance_route_b_preparation.py", "models._consumer_baseline_preparation"),
                           ("instance_route_b.py", "models._consumer_baseline_model")):
        source = _git("show", f"{commit}:models/{filename}")
        if filename == "instance_route_b.py":
            source = source.replace("from .instance_route_b_preparation import",
                                    "from ._consumer_baseline_preparation import")
        module = types.ModuleType(name)
        module.__file__, module.__package__ = f"git:{commit}:models/{filename}", "models"
        sys.modules[name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, commit


def consumer(module, obs, texture):
    item = prepared_item(module, obs)
    encoded = module._restore_training_encoded(item["observation"]["_route_b_training_inputs"], "cpu")
    encoded["texture_feature"] = texture.clone().requires_grad_()
    model = module.InstanceSuction2DFastV3(query_chunk=128)
    points = torch.as_tensor(obs["query_points"])
    normals = torch.as_tensor(obs["query_normals"])
    phase = torch.linspace(0., 6.2, len(points), dtype=torch.float64)
    radial = module.ring_directions(normals, phase)[0]
    ring = points[:, None, None] + points.new_tensor(module.RADII_M[1:])[None, :, None, None] * radial[:, None]
    unique = torch.cat((points[:, None], ring.flatten(1, 2)), 1)
    return dict(model=model, encoded=encoded, wire=item, points=points, normals=normals, phase=phase,
                slots=lambda: model._read_slots(encoded, unique),
                contacts=lambda: model._contact_inputs(encoded, points, normals, phase))


def profile_calls(function):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        output = function()
    events = profile.key_averages()
    result = dict(operator_calls_total=sum(e.count for e in events),
                  cumulative_positive_self_cpu_allocated_bytes=sum(max(e.self_cpu_memory_usage, 0) for e in events),
                  operators={e.key: dict(calls=e.count, cpu_output_bytes=e.cpu_memory_usage)
                             for e in events if e.key in SELECTED})
    del output
    return result


def exact_contact_pair(before, after):
    old, new = before["contacts"](), after["contacts"]()
    old_values, new_values = dict(prediction_tensors(old)), dict(prediction_tensors(new))
    assert old_values.keys() == new_values.keys()
    errors = [name for name in old_values if not torch.equal(old_values[name], new_values[name])]
    old[0].square().mean().backward()
    new[0].square().mean().backward()
    old_gradient = before["encoded"]["texture_feature"].grad
    new_gradient = after["encoded"]["texture_feature"].grad
    return dict(tensor_count=len(old_values), all_contact_tensors_exact=not errors, mismatches=errors,
                texture_gradient_exact=torch.equal(old_gradient, new_gradient),
                texture_gradient_max_absolute_error=(old_gradient - new_gradient).abs().max().item())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rounds < 3:
        parser.error("--rounds must be at least three")
    torch.set_num_threads(1)
    historical, commit = historical_modules(args.baseline_ref)
    obs = observation()
    prepared = prepared_item(current, obs)
    hw = prepared["observation"]["_route_b_training_inputs"]["encoded"]["texture_padded_hw"]
    texture = torch.randn((1, 16, *hw), generator=torch.Generator().manual_seed(21))
    pairs = dict(baseline=consumer(historical, obs, texture), candidate=consumer(current, obs, texture))
    comparison = exact_contact_pair(pairs["baseline"], pairs["candidate"])
    for pair in pairs.values():
        for _ in range(2):
            pair["contacts"]()
    timing = {name: {scope: [] for scope in ("slots", "contacts")} for name in pairs}
    for round_index in range(args.rounds):
        for name in (("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")):
            for scope in ("slots", "contacts"):
                start = time.perf_counter()
                output = pairs[name][scope]()
                timing[name][scope].append(1000 * (time.perf_counter() - start))
                del output
    profiles = {name: {scope: profile_calls(pair[scope]) for scope in ("slots", "contacts")}
                for name, pair in pairs.items()}
    xyz = pairs["candidate"]["encoded"]["xyz_map"]
    mask = pairs["candidate"]["encoded"]["mask"]
    precompute = []
    for _ in range(args.rounds):
        start = time.perf_counter()
        current.bilinear_validity_map(xyz, mask)
        precompute.append(1000 * (time.perf_counter() - start))
    # A separate small full-network pair verifies all public predictions,
    # diagnostics and trainable gradients. It is not the stage timing above.
    rows = make_observations(8)
    settings = dict(seed=672, query_chunk=2, dense_chunk=4096)
    _, old_predictions, old_gradients = measure(historical, rows, **settings)
    _, new_predictions, new_gradients = measure(current, rows, **settings)
    network = compare_tensors((old_predictions, old_gradients), (new_predictions, new_gradients))
    result = dict(
        scope="synthetic CPU projected-ring consumer only; no I/O, IPC, CUDA or end-to-end training throughput",
        torch_version=torch.__version__, numpy_version=np.__version__, device="cpu", torch_threads=1,
        baseline_commit=commit, candidate="current working-tree route-B model and preparation module",
        image_hw=list(obs["mask"].shape), support_points=int(obs["mask"].sum()),
        queries=len(obs["query_points"]), slots_per_query=1 + 8 * current.ANGLES,
        rounds=args.rounds, order="alternating baseline/candidate, each after two contact warmups",
        timings_ms=timing,
        median_ms={name: {scope: statistics.median(samples) for scope, samples in scopes.items()}
                   for name, scopes in timing.items()},
        profile=profiles,
        allocation_scope="sum of positive self CPU operator allocations, not peak RSS or GPU memory",
        static_quad_map=dict(shape=list(mask.shape), bytes=int(mask.numel()),
                             precompute_median_ms=statistics.median(precompute)),
        wire_numeric_bytes={name: byte_count(pair["wire"]) for name, pair in pairs.items()},
        contact_equivalence=comparison, full_network_equivalence=network,
        remaining="CUDA nonzero in arbitrary/canonical support checking and real GPU timing are not changed or measured",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if (comparison["all_contact_tensors_exact"] and comparison["texture_gradient_exact"]
                 and network["all_prediction_tensors_exact"] and network["all_parameter_gradients_exact"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
