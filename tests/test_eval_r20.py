"""Synthetic contracts plus an explicitly enabled bounded frozen-release check."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pytest
from scipy.spatial import cKDTree

from suction20.eval_r20 import SceneEvalContext, evaluate_poses, load_scene_eval_context, seal_at_pose
from suction20.full_data import DEFAULT_CONFIG, load_seal, transform_points


def ring_points(height=0.):
    theta = np.arange(72) * 2 * np.pi / 72 - np.pi + np.pi / 72
    return np.column_stack([.02 * np.cos(theta), .02 * np.sin(theta), np.full(72, height)])


def test_seal_full_cylinder_closed_ring_and_self_collision():
    ring = ring_points()
    assert seal_at_pose(ring, [0, 0, 0], [0, 0, 1]) == (1., False)
    assert seal_at_pose(ring[:-1], [0, 0, 0], [0, 0, 1])[0] == 0
    # Far axial ring must not be pruned by a local sphere or observation band.
    assert seal_at_pose(ring_points(.4), [0, 0, 0], [0, 0, 1])[0] == pytest.approx(1)
    assert seal_at_pose(np.vstack([ring, [0, 0, 1.]]), [0, 0, 0], [0, 0, 1]) == (1., True)
    raised = ring.copy()
    raised[-1, 2] = .003
    theta = 2 * np.pi / 72
    chord = 2 * .02 * np.sin(theta / 2)
    expected = chord / np.hypot(chord, .003) * np.exp(-1e5 * np.var(raised[:, 2]))
    assert seal_at_pose(raised, [0, 0, 0], [0, 0, 1], chunk_points=7)[0] == pytest.approx(expected)


def test_rotated_pose_recomputes_and_invalid_pose_rejected():
    ring = ring_points()
    n = np.array([np.sin(np.deg2rad(10)), 0, np.cos(np.deg2rad(10))])
    score, _ = seal_at_pose(ring, [0, 0, 0], n)
    assert score < .99
    with pytest.raises(ValueError, match="normal"):
        seal_at_pose(ring, [0, 0, 0], [0, 0, 0])


def test_scene_evaluation_self_collision_and_world_camera_transform():
    cfg = dict(DEFAULT_CONFIG)
    world_object = np.eye(4)
    world_object[2, 3] = .2
    world_camera = np.eye(4)
    world_camera[:3, 3] = [1., 2., 3.]
    obstacles = np.array([[0., 0., 0.]])
    dense = np.vstack([ring_points(), [0., 0., .3]])
    context = SceneEvalContext(0, "realsense", cfg,
                               [{"instance_row": 0, "obj_id": 0, "T_world_obj": world_object}],
                               world_camera[None], {0: dense}, {0: np.array([0., 0., .2])},
                               obstacles, np.array([-1]), cKDTree(obstacles))
    points = np.tile([-1., -2., -2.8], (2, 1))
    result = evaluate_poses(context, 0, points, [[0., 0, 1], [0., 0, -1]], np.array([0, 0]))
    # The only protrusion is beyond the scene envelope. Original Seal still
    # flags self collision; reversing the executed axis removes that collision.
    np.testing.assert_allclose(result["S"], [1., 1.], atol=1e-12)
    np.testing.assert_array_equal(result["collision"], [True, False])
    np.testing.assert_allclose(result["Q20"], [0., 1.], atol=1e-12)
    # A contact 10 cm above the table, aimed toward it, hits the table envelope.
    table_hit = evaluate_poses(context, 0, [[-1., -2., -2.9]], [[0., 0., -1.]], np.array([0]))
    assert table_hit["collision"][0]


@pytest.mark.skipif(not os.environ.get("R20_RELEASE_ROOT"), reason="requires explicit frozen release root")
def test_frozen_32_candidate_consistency():
    """16 unique candidates/scene; high, low nonzero, zero, collision buckets."""
    release = Path(os.environ["R20_RELEASE_ROOT"])
    cfg = json.loads((release / "output/config_frozen.json").read_text())
    evidence = {"scene_ids": [0, 1], "tolerance": 1e-5, "candidates": [], "rotations": [],
                "release_manifest_sha256": hashlib.sha256((release / "output/manifest.json").read_bytes()).hexdigest()}
    begin = time.monotonic()
    for scene_id in (0, 1):
        context = load_scene_eval_context(release / "input_raw", release / "output", scene_id, "realsense", cfg)
        candidates = []
        with np.load(release / f"output/collision_label_r20/{scene_id:04d}_realsense_collision.npz") as cf, \
             np.load(release / f"output/wrench_label_r20/{scene_id:04d}_realsense_wrench_scores.npz") as wf:
            for rec in context.records:
                row, oid = rec["instance_row"], rec["obj_id"]
                source = release / "input_raw/dense_point_clouds" / f"{oid:03d}.npz"
                # Assert same official dense input referenced by producer command.
                assert source.samefile(release / "input/dense_point_clouds" / f"{oid:03d}.npz")
                seal = load_seal(release / "output/seal_r20/poisson" / f"{oid:03d}_labels.npz")
                for i in range(len(seal["points"])):
                    candidates.append((row, oid, i, float(seal["scores"][i]), bool(cf[f"arr_{row}"][i]),
                                       float(wf[f"wrench_eval_{row}"][i]), seal["points"][i], seal["normals"][i],
                                       bool(seal["collision"][i])))
        groups = {
            "high": sorted([c for c in candidates if c[3] > .5 and not c[4]], key=lambda c: -c[3]),
            "low": sorted([c for c in candidates if 0 < c[3] < .5], key=lambda c: abs(c[3] - .1)),
            "zero": [c for c in candidates if c[3] == 0],
            "collision": sorted([c for c in candidates if c[4]], key=lambda c: -c[3]),
        }
        chosen, used = [], set()
        for bucket, pool in groups.items():
            selected = []
            for c in pool:
                if (c[0], c[2]) not in used:
                    selected.append((bucket, c))
                    used.add((c[0], c[2]))
                if len(selected) == 4:
                    break
            assert len(selected) == 4, f"insufficient {bucket} candidates"
            chosen.extend(selected)
        p_cam, n_cam, gt_rows = [], [], []
        for bucket, c in chosen:
            T = np.linalg.inv(context.world_camera[8]) @ context.records[c[0]]["T_world_obj"]
            p_cam.append(transform_points(c[6], T))
            n_cam.append(c[7] @ T[:3, :3].T)
            gt_rows.append(c[0])
        p_cam, n_cam = np.asarray(p_cam), np.asarray(n_cam)
        actual = evaluate_poses(context, 8, p_cam, n_cam, np.asarray(gt_rows))
        for j, (bucket, c) in enumerate(chosen):
            raw, coll, wrench = c[3:6]
            normalized = float(np.clip((raw - cfg["seal_norm_min"]) / (cfg["seal_norm_max"] - cfg["seal_norm_min"]), 0, 1))
            expected = {"raw_seal": raw, "S": normalized, "W": wrench, "Q20": normalized * wrench * (not coll)}
            errors = {key: float(abs(actual[key][j] - value)) for key, value in expected.items()}
            evidence["candidates"].append({"scene_id": scene_id, "frame_id": 8, "instance_row": c[0], "object_id": c[1],
                                            "candidate_row": c[2], "bucket": bucket, "expected": expected,
                                            "collision": coll, "seal_self_collision": c[8], "absolute_error": errors})
            assert all(value <= 1e-5 for value in errors.values()), evidence["candidates"][-1]
            assert actual["collision"][j] == coll
        # Rotate four high-quality axes 10 degrees about a perpendicular axis.
        normals = n_cam[:4] / np.linalg.norm(n_cam[:4], axis=1, keepdims=True)
        tangent = np.cross(normals, [1., 0, 0])
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        rotated = normals * np.cos(np.deg2rad(10)) + tangent * np.sin(np.deg2rad(10))
        changed = evaluate_poses(context, 8, p_cam[:4], rotated, np.asarray(gt_rows[:4]))
        assert np.any(np.abs(changed["raw_seal"] - actual["raw_seal"][:4]) > 1e-5)
        for j in range(4):
            evidence["rotations"].append({"scene_id": scene_id, "candidate_row": chosen[j][1][2], "rotation_deg": 10,
                                           "original_raw_seal": float(actual["raw_seal"][j]),
                                           "rotated_raw_seal": float(changed["raw_seal"][j])})
        background = evaluate_poses(context, 8, p_cam[:1], n_cam[:1], np.array([-1]))
        assert background["Q20"][0] == 0
    evidence.update(status="passed", elapsed_seconds=time.monotonic() - begin,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    if os.environ.get("R20_EVIDENCE_PATH"):
        Path(os.environ["R20_EVIDENCE_PATH"]).write_text(json.dumps(evidence, indent=2) + "\n")
