#!/usr/bin/env python3
"""Bounded CPU pairing of route-B preparation and retained geometry ROI.

The baseline is read from local Git, with no checkout change. Synthetic RGB-D
inputs are fixed before timing. Cold preparation and ideal warm-cache reuse are
reported separately; neither is a measured training-cache hit rate or GPU gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark_route_b_splits import load_baseline
import models.instance_route_b as current


def observation():
    height, width = 720, 1280
    y, x = np.mgrid[:height, :width]
    intrinsics = np.array(((1000., 0., 639.5), (0., 1010., 359.5), (0., 0., 1.)), np.float64)
    z = np.full((height, width), .65, np.float64)
    xyz = np.stack(((x - 639.5) * z / 1000., (y - 359.5) * z / 1010., z), -1)
    mask = np.zeros((height, width), bool)
    mask[280:440, 540:740] = True
    xyz[~mask] = 0
    normals = np.zeros_like(xyz)
    normals[..., 2][mask] = -1
    rng = np.random.default_rng(20261119)
    rgb = rng.random((height, width, 3), dtype=np.float32)
    rgb[~mask] = 0
    indices = np.flatnonzero(mask.ravel())[np.linspace(0, mask.sum()-1, 128).astype(int)]
    return dict(xyz_map=xyz, mask=mask, instance_mask=mask.copy(), normal_map=normals,
                normal_valid=mask.copy(), rgb=rgb, rgb_valid=mask.copy(),
                gravity=np.array((0., 1., 0.)), intrinsics=intrinsics,
                support_points=xyz[mask].copy(), query_points=xyz.reshape(-1, 3)[indices],
                query_normals=normals.reshape(-1, 3)[indices], sample_uid="synthetic/720p/32000")


def byte_count(value):
    """Count owned numeric objects, deduplicating aliases by object identity."""
    seen = set()
    def visit(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            return item.nbytes
        if torch.is_tensor(item):
            return item.numel() * item.element_size()
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(v) for v in item)
        return 0
    return visit(value)


def prepared_item(module, obs, static=None):
    options = {} if static is None else {"static_inputs": static}
    prepared = module.prepare_source_inputs_cpu(obs, sample_image_channels=True, **options)
    return module.compact_prepared_source_item({"observation": prepared, "record": {}, "labels": {}})


def check_pair(baseline, obs, static):
    before = prepared_item(baseline, obs)
    after = prepared_item(current, obs, static)
    full = baseline._restore_training_encoded(before["observation"]["_route_b_training_inputs"], "cpu")
    roi = current._restore_training_encoded(after["observation"]["_route_b_training_inputs"], "cpu")
    y0, x0 = roi["geometry_top_left"]
    rh, rw = roi["mask"].shape
    for key, expected in full.items():
        actual = roi[key]
        if key in current._GEOMETRY_MAPS:
            expected = expected[y0:y0+rh, x0:x0+rw]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=key)
        else:
            assert actual == expected, key
    return before, after, full, roi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="f7a3ffc83f4993e3543dcc8ece7080b3dc905596")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.rounds <= 100:
        parser.error("--rounds must be between 3 and 100")
    torch.set_num_threads(1)
    baseline, commit = load_baseline(args.baseline_ref)
    obs = observation()
    static = current.prepare_source_static_inputs(obs)
    before, after, full, roi = check_pair(baseline, obs, static)
    # Pre-create exposure variants; color transform cost is outside this probe.
    exposures = [dict(obs, rgb=obs["rgb"] * np.float32(.45 + .5 * index / args.rounds))
                 for index in range(args.rounds)]
    durations = {name: [] for name in ("baseline_prepare_pack", "current_cold_prepare_pack",
                 "current_warm_prepare_pack", "baseline_restore", "current_restore")}
    def measure(name, function):
        started = time.perf_counter()
        value = function()
        durations[name].append((time.perf_counter() - started) * 1000.)
        return value
    # Warm the import/allocator path separately from the measured rounds.
    prepared_item(baseline, obs)
    prepared_item(current, obs)
    for index, exposure in enumerate(exposures):
        jobs = [("baseline_prepare_pack", lambda: prepared_item(baseline, exposure)),
                ("current_cold_prepare_pack", lambda: prepared_item(current, exposure)),
                ("current_warm_prepare_pack", lambda: prepared_item(current, exposure, static))]
        if index % 2:
            jobs.reverse()
        for name, job in jobs:
            measure(name, job)
        jobs = [("baseline_restore", lambda: baseline._restore_training_encoded(
                    before["observation"]["_route_b_training_inputs"], "cpu")),
                ("current_restore", lambda: current._restore_training_encoded(
                    after["observation"]["_route_b_training_inputs"], "cpu"))]
        if index % 2:
            jobs.reverse()
        for name, job in jobs:
            measure(name, job)
        check_pair(baseline, exposure, static)
    summarize = lambda values: dict(median_ms=statistics.median(values), mean_ms=statistics.mean(values), rounds_ms=values)
    result = dict(
        baseline_commit=commit, implementation="current modified worktree", device="CPU only", threads=1,
        scope="Synthetic 720x1280 camera, 32000 visible pixels, 128 queries; no file I/O, augmentation, IPC or GPU timing",
        static_reuse_scope="Warm case assumes a cache hit. Actual training hit rate is not measured.",
        python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, rounds=args.rounds,
        shape=dict(camera=[720, 1280], geometry_roi=list(roi["mask"].shape),
                   context=list(roi["context"].shape), texture=list(roi["texture"].shape)),
        numerical_comparison="All encoded values exactly equal at every exposure, geometry compared at matching original ROI pixels",
        timing={name: summarize(values) for name, values in durations.items()},
        numeric_bytes=dict(baseline_wire=byte_count(before), current_wire=byte_count(after),
                           baseline_consumer_geometry=byte_count({key: full[key] for key in current._GEOMETRY_MAPS}),
                           current_consumer_geometry=byte_count({key: roi[key] for key in current._GEOMETRY_MAPS}),
                           static_cache_entry=byte_count(static)),
        work_counts=dict(full_camera_pixels=720*1280, geometry_roi_pixels=roi["mask"].numel(),
                         baseline_rgb_normalization_pixels=720*1280,
                         current_rgb_normalization_pixels=128*128 + roi["texture"].shape[-1]*roi["texture"].shape[-2]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "timing": {k: v["median_ms"] for k, v in result["timing"].items()},
                      "numeric_bytes": result["numeric_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
