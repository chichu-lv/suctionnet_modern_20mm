"""Regression tests for the finalized dual Wrench labels and per-view GT selection."""
import json

import numpy as np
import pytest

from suction20 import full_data as fd


def test_wrench_thresholds_share_torque_but_differ_by_sqrt2():
    cfg = dict(fd.DEFAULT_CONFIG)
    p = np.array([[0., 0., 0.], [0.02, 0., 0.]])
    n = np.array([[0., 0., 1.], [0., 0., 1.]])
    center = np.array([0.01, 0., 0.])
    values = fd.wrench_scores_dual(p, n, center, 1.0, cfg)
    tau = 0.01 * cfg["gravity_m_s2"]
    threshold_eval = cfg["wrench_k"] * cfg["radius_m"] * np.pi
    expected_eval = 1. - min(1., tau / threshold_eval)
    expected_annot = 1. - min(1., tau / (threshold_eval * np.sqrt(2.)))
    np.testing.assert_allclose(values["wrench_eval"], expected_eval, atol=1e-6)
    np.testing.assert_allclose(values["wrench_annot"], expected_annot, atol=1e-6)
    assert np.all(values["wrench_annot"] >= values["wrench_eval"])
    np.testing.assert_array_equal(
        fd.wrench_scores(p, n, center, 1.0, cfg), values["wrench_eval"]
    )


def test_scene_and_view_persist_both_wrench_fields_and_gtb_uses_eval(tmp_path, monkeypatch):
    cfg = dict(fd.DEFAULT_CONFIG)
    dataset = tmp_path / "dataset"
    seal_root = tmp_path / "seal"
    out = tmp_path / "output"
    seal_root.mkdir()
    cam = dataset / "scenes" / "scene_0000" / "realsense"
    cam.mkdir(parents=True)
    np.save(cam / "camera_poses.npy", np.eye(4)[None])
    np.save(cam / "cam0_wrt_table.npy", np.eye(4))

    points = np.array([[0., 0., 0.], [0.02, 0., 0.]], np.float32)
    normals = np.repeat(np.array([[0., 0., 1.]], np.float32), 2, axis=0)
    scores = np.array([0.8, 0.4], np.float32)
    np.savez(
        seal_root / "003_labels.npz", points=points, normals=normals,
        scores=scores, collision=np.zeros(2, bool),
        indices=np.array([11, 22], np.int32)
    )
    rec = {"instance_row": 0, "obj_id": 3, "T_world_obj": np.eye(4)}
    monkeypatch.setattr(fd, "scene_frame0", lambda ds, scene, camera: [rec])
    monkeypatch.setattr(fd, "load_dense_points", lambda ds, obj, voxel: points)
    monkeypatch.setattr(fd, "scene_table_cloud", lambda cfg: np.array([[10., 10., 10.]]))

    fd.compute_scene(dataset, seal_root, out, 0, cfg, resume=False)
    wfile = out / "wrench_label_r20" / "0000_realsense_wrench_scores.npz"
    with np.load(wfile, allow_pickle=False) as f:
        assert "wrench_annot_0" in f.files and "wrench_eval_0" in f.files
        np.testing.assert_array_equal(f["arr_0"], f["wrench_eval_0"])
        assert np.all(f["wrench_annot_0"] >= f["wrench_eval_0"])
        evals = f["wrench_eval_0"].copy()
        annots = f["wrench_annot_0"].copy()

    manifest = out / "scene_meta_r20" / "0000_realsense.json"
    assert json.loads(manifest.read_text())["wrench_schema"] == "annot_eval_dual_v1"

    extra_dir = out / "seal_extras_r20"
    extra_dir.mkdir(parents=True)
    np.savez_compressed(
        extra_dir / "003.npz",
        pca_normals=np.repeat(np.array([[0., 1., 0.]], np.float32), 2, axis=0),
        normal_valid=np.ones(2, dtype=bool),
        seal_custom=np.array([0.6, 0.7], dtype=np.float32),
        indices=np.array([11, 22], dtype=np.int32),
    )

    monkeypatch.setattr(
        fd, "read_frame",
        lambda ds, scene, frame, config: (
            points.copy(), normals.copy(), np.ones(2, bool),
            np.array([4, 4], np.int32), np.array([0, 1], np.int32), (1, 2)
        ),
    )
    fd.generate_view(dataset, seal_root, out, 0, 0, cfg, resume=False)
    gtfile = out / "scenes" / "scene_0000" / "realsense" / "graspness_r20_v2" / "0000.npz"
    with np.load(gtfile, allow_pickle=False) as f:
        assert f["gt_valid"].tolist() == [True, True]
        assert f["instance_id"].tolist() == [1, 1]
        np.testing.assert_allclose(f["gt_wrench"], evals)
        np.testing.assert_allclose(f["gt_wrench_eval"], evals)
        np.testing.assert_allclose(f["gt_wrench_annot"], annots)
        np.testing.assert_allclose(f["gt_B"], scores * evals)
        np.testing.assert_allclose(f["gt_A"], scores)
        np.testing.assert_allclose(f["gt_normal"], normals)
        np.testing.assert_allclose(f["gt_normal_pca"], [[0., 1., 0.], [0., 1., 0.]])
        np.testing.assert_allclose(f["normals_raw"], normals)
        assert f["normal_pca_valid"].tolist() == [True, True]

    audit = fd.dataset_stats(out, cfg, [0], [0])
    assert audit["frame_count"] == 1
    assert audit["g2_errors"] == []


def test_old_scene_wrench_schema_must_be_regenerated(tmp_path, monkeypatch):
    cfg = dict(fd.DEFAULT_CONFIG)
    out = tmp_path / "output"
    wd = out / "wrench_label_r20"
    cd = out / "collision_label_r20"
    md = out / "scene_meta_r20"
    for d in (wd, cd, md):
        d.mkdir(parents=True)
    np.savez(wd / "0000_realsense_wrench_scores.npz", arr_0=np.array([0.5]))
    np.savez(cd / "0000_realsense_collision.npz", arr_0=np.array([False]))
    (md / "0000_realsense.json").write_text(json.dumps({
        "configuration_hash": fd.stable_hash(cfg),
        "objects": [{"obj_id": 3, "row": 0}]
    }))
    with pytest.raises(ValueError, match="Old scene Wrench schema"):
        fd.compute_scene(tmp_path, tmp_path, out, 0, cfg, resume=True)
