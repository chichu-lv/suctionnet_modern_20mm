"""CPU-only validation orchestration, checkpoint and label-free inference checks."""
import copy
import json
import sys

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from PIL import Image

from models import Suction20Net
from suction20 import eval_r20, validation
from suction20.training import FrameSchedule, load_checkpoint, save_checkpoint


class FixtureBackbone(nn.Module):
    """Test-only dense stand-in; never a production backbone."""
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(8, 256)

    def forward(self, inputs):
        return self.project(inputs["voxel_features"])


def sample_fixture(count=12):
    points = np.column_stack([np.arange(count) * .02, np.zeros(count), np.ones(count)]).astype(np.float32)
    return {"inputs": {"xyz_cam_m": points, "normal_obs": np.tile([0., 0., -1.], (count, 1)).astype(np.float32),
                       "normal_obs_valid": np.ones(count, bool), "instance_id_input": np.ones(count, np.int64),
                       "source_indices": np.arange(count, dtype=np.int64) * 2, "g_cam": np.array([0., 0., -1.], np.float32)},
            "meta": {"sample_uid": (9, "realsense", 0)}, "targets": {}}


def model_config():
    return {"experiment": "E1", "model": {"seed_count": 8, "dense_chunk_size": 4},
            "data": {"source_root": "unused", "label_root": "must-not-read"}}


def make_model(config):
    return Suction20Net(dict(config["model"], experiment=config["experiment"]), {}, FixtureBackbone())


def test_prediction_never_reads_targets_and_restores_camera_coordinates():
    class NoTargets(dict):
        def __getitem__(self, key):
            if key == "targets":
                raise AssertionError("inference read supervision")
            return super().__getitem__(key)

    config = model_config()
    model = make_model(config)
    sample = NoTargets(sample_fixture())
    before = validation.predict_sample(model, sample, config, torch.device("cpu"))
    repeated = validation.predict_sample(model, sample, config, torch.device("cpu"))
    for key in before:
        np.testing.assert_array_equal(before[key], repeated[key])
    transform = np.eye(4)
    transform[:3, 3] = [.1, .2, .3]
    sample["meta"]["T_aug_from_cam"] = transform
    moved = validation.predict_sample(model, sample, config, torch.device("cpu"))
    np.testing.assert_allclose(moved["point_cam_m"], before["point_cam_m"] - transform[:3, 3])
    np.testing.assert_array_equal(moved["normal_out_cam"], before["normal_out_cam"])


def test_empty_prediction_and_nms():
    config = model_config()
    prediction = validation.predict_sample(make_model(config), sample_fixture(0), config, torch.device("cpu"))
    assert prediction["point_cam_m"].shape == (0, 3)
    assert prediction["point_index"].shape == (0,)
    assert validation.nms_prediction(prediction, config)["scene_indices"].size == 0


def test_checkpoint_inference_round_trip_and_config_mismatch(tmp_path):
    config = model_config()
    model = make_model(config)
    sample = sample_fixture()
    before = validation.predict_sample(model, sample, config, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    schedule = FrameSchedule([0], frame_keys=[(0, 0)])
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scaler, schedule, global_step=0, config=config, source_label_config={})
    state = torch.load(path, weights_only=False)
    relocated = copy.deepcopy(config)
    relocated["model"].update(backbone_root="relocated", dense_chunk_size=5)
    resolved = validation.checkpoint_inference_config(relocated, state)
    restored = make_model(resolved)
    load_checkpoint(path, restored)
    after = validation.predict_sample(restored, sample, resolved, torch.device("cpu"))
    for key in before:
        np.testing.assert_allclose(after[key], before[key], atol=1e-7)
    bad = copy.deepcopy(config)
    bad["experiment"] = "E0"
    with pytest.raises(ValueError, match="experiment"):
        validation.checkpoint_inference_config(bad, state)
    bad = copy.deepcopy(config)
    bad["model"]["seed_count"] = 9
    with pytest.raises(ValueError, match="seed_count"):
        validation.checkpoint_inference_config(bad, state)
    state["source_label_config"]["radius_m"] = .04
    with pytest.raises(ValueError, match="source label definitions"):
        validation.checkpoint_inference_config(config, state)


def test_success_head_presence_does_not_imply_trained_ranking():
    source = {"stage": "public_pretrain", "global_step": 100}
    assert not validation.use_task_success_ranking(source, {})
    explicit = {"ranking": {"final_score": "task_success_score"}}
    with pytest.raises(ValueError, match="trained"):
        validation.use_task_success_ranking(source, explicit)
    target = {"stage": "target_finetune", "target_mode": "attempt_success", "global_step": 1}
    assert validation.use_task_success_ranking(target, explicit)
    assert not validation.use_task_success_ranking(target, {"ranking": {"final_score": "q"}})
    target["global_step"] = 0
    with pytest.raises(ValueError, match="trained"):
        validation.use_task_success_ranking(target, explicit)


@pytest.mark.parametrize("empty", [False, True])
def test_evaluation_uses_raw_point_ownership_before_global_cap(monkeypatch, empty):
    count = 62
    sample = sample_fixture(count)
    target = dict(instance_id_gt=np.arange(count) + 1, gt_valid=np.zeros(count, bool),
                  quality_valid=np.zeros(count, bool), gt_B=np.ones(count),
                  normal_label=sample["inputs"]["normal_obs"], axis_valid=np.zeros(count, bool))
    sample["targets"] = target
    indices = np.arange(count - 2, -1, -1) if not empty else np.empty(0, np.int64)
    n = len(indices)
    prediction = dict(point_index=indices, point_cam_m=sample["inputs"]["xyz_cam_m"][indices],
                      normal_out_cam=sample["inputs"]["normal_obs"][indices], q=np.linspace(1., .1, n),
                      q0=np.ones(n), instance_id=np.arange(n) + 100,
                      source_indices=sample["inputs"]["source_indices"][indices],
                      normal_out_all=sample["inputs"]["normal_obs"], q0_all=np.ones(count))

    class Dataset(list):
        data_cfg = {}

    monkeypatch.setattr(validation, "make_source_dataset", lambda *a, **kw: Dataset([sample]))
    monkeypatch.setattr(validation, "predict_sample", lambda *a, **kw: prediction)
    monkeypatch.setattr(eval_r20, "load_scene_eval_context", lambda *a: object())
    calls = []

    def evaluate(context, frame, points, normals, rows):
        calls.append(rows)
        np.testing.assert_array_equal(rows, target["instance_id_gt"][indices] - 1)
        return {"Q20": np.full(n, .8), "collision": np.zeros(n, bool)}

    monkeypatch.setattr(eval_r20, "evaluate_poses", evaluate)
    result = validation.evaluate_model(object(), model_config(), torch.device("cpu"))
    assert len(calls) == 1
    assert result["instance_Top1_Q20_at_0_4"] == pytest.approx(n / count)
    assert result["no_candidate_rate"] == pytest.approx(1 - n / count)
    assert result["AP_R20"] == (1. if n else 0.)
    assert result["frames"][0]["scene_candidate_count"] == min(50, n)
    assert result["frames"][0]["masked_seed_q_mae"] is None
    assert result["frames"][0]["proposal_coverage_at_512"] is None
    json.dumps(result, allow_nan=False)


def test_empty_evaluation_reports_no_measurement(monkeypatch):
    monkeypatch.setattr(validation, "make_source_dataset", lambda *a, **kw: [])
    result = validation.evaluate_model(object(), model_config(), torch.device("cpu"))
    assert result["view_count"] == 0
    assert result["AP_R20"] is None
    assert result["frames"] == []


def test_inference_cli_observation_only_checkpoint_export(tmp_path, monkeypatch):
    import scripts.infer as command

    config = model_config()
    model = make_model(config)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()),
                     torch.amp.GradScaler("cpu", enabled=False), FrameSchedule([0], frame_keys=[(0, 0)]),
                     global_step=0, config=config, source_label_config={})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    sample = sample_fixture()
    del sample["targets"]
    observed = []

    def observation_dataset(source_root, keys, **kwargs):
        observed.append((source_root, keys, kwargs))
        assert "label_root" not in kwargs
        return [sample]

    monkeypatch.setattr(command, "PublicObservationDataset", observation_dataset)
    monkeypatch.setattr(command, "make_model", lambda cfg, frozen, device: make_model(cfg))
    output = tmp_path / "nested" / "predictions.npz"
    monkeypatch.setattr(sys, "argv", ["infer.py", "--config", str(config_path), "--checkpoint", str(checkpoint),
                                     "--frames", "9:0", "--output", str(output), "--device", "cpu"])
    command.main()
    assert len(observed) == 1
    with np.load(output, allow_pickle=False) as data:
        prefix = "scene_0009_realsense_0000/"
        api = data[prefix + "suction_api"]
        np.testing.assert_array_equal(api[:, 0], data[prefix + "q"])
        np.testing.assert_array_equal(api[:, 1:4], data[prefix + "normal_out_cam"])
        np.testing.assert_array_equal(api[:, 4:7], data[prefix + "point_cam_m"])
        assert len(api) <= 10


def test_r20_rejects_changed_protocol_limits():
    config = model_config()
    config["ranking"] = {"max_per_instance": 3}
    with pytest.raises(ValueError, match="AP-R20 protocol"):
        validation.evaluate_model(object(), config, torch.device("cpu"))


def test_target_observation_cli_without_dataset_roots_and_stable_prefixes(tmp_path, monkeypatch):
    import scripts.infer as command

    config = model_config()
    config["model"]["use_task_success_head"] = True
    config["ranking"] = {"final_score": "task_success_score"}
    model = make_model(config)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()),
                     torch.amp.GradScaler("cpu", enabled=False), FrameSchedule([0], frame_keys=[(0, 0)]),
                     global_step=1, config=config, source_label_config={"depth_min_m": .6, "depth_max_m": 1.8},
                     stage="target_finetune", target_mode="attempt_success")
    # This is a synthetic checkpoint state, not an actual optimizer update.
    inference_config = copy.deepcopy(config)
    del inference_config["data"]
    config_path = tmp_path / "resolved_config.json"
    config_path.write_text(json.dumps(inference_config))
    assets = tmp_path / "observations"
    assets.mkdir()
    sample = sample_fixture()
    np.savez(assets / "cloud.npz", xyz=sample["inputs"]["xyz_cam_m"],
             instance_id=sample["inputs"]["instance_id_input"], normals_raw=sample["inputs"]["normal_obs"])
    np.savez(assets / "empty.npz", xyz=np.empty((0, 3), np.float32), instance_id=np.empty(0, np.int64))
    Image.fromarray(np.full((7, 7), 500, np.uint16)).save(assets / "near.png")
    Image.fromarray(np.full((7, 7), 1500, np.uint16)).save(assets / "far.png")
    Image.fromarray(np.ones((7, 7), np.uint8)).save(assets / "mask.png")
    manifest = assets / "observations.jsonl"
    rows = [{"sample_id": "capture/01", "observation": {"format": "pointcloud_npz", "path": "cloud.npz", "g_cam": [0, 0, -1]}},
            {"sample_id": "capture%2F01", "observation": {"format": "pointcloud_npz", "path": "empty.npz", "g_cam": [0, 0, -1]}}]
    rgbd = {"format": "rgbd", "instance_mask_path": "mask.png", "intrinsics": [[100, 0, 3], [0, 100, 3], [0, 0, 1]],
            "depth_scale_m": .001, "g_cam": [0, 0, -1]}
    rows.extend([
        {"sample_id": "near_inherited", "observation": dict(rgbd, depth_path="near.png")},
        {"sample_id": "far_inherited", "observation": dict(rgbd, depth_path="far.png")},
        {"sample_id": "near_override", "observation": dict(rgbd, depth_path="near.png", depth_min_m=.4)},
    ])
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def no_public_data(*args, **kwargs):
        raise AssertionError("Target inference must not access public data or label roots")

    monkeypatch.setattr(command, "PublicObservationDataset", no_public_data)
    monkeypatch.setattr(command, "make_model", lambda cfg, frozen, device: make_model(cfg))
    output = tmp_path / "target_predictions.npz"
    monkeypatch.setattr(sys, "argv", ["infer.py", "--config", str(config_path), "--checkpoint", str(checkpoint),
                                     "--observation-manifest", str(manifest), "--output", str(output), "--device", "cpu"])
    command.main()
    with np.load(output, allow_pickle=False) as data:
        full, empty = "sample_capture%2F01/", "sample_capture%252F01/"
        assert data[full + "source_index_kind"].item() == "row"
        np.testing.assert_array_equal(data[full + "source_indices"], data[full + "point_index"])
        np.testing.assert_array_equal(data[full + "suction_api"][:, 0], data[full + "task_success_score"])
        assert data[empty + "suction_api"].shape == (0, 7)
        # Training and inference inherit the same saved .6–1.8 m interval.
        assert data["sample_near_inherited/suction_api"].shape == (0, 7)
        assert len(data["sample_far_inherited/suction_api"]) > 0
        # Explicit per-observation bounds continue to take precedence.
        assert len(data["sample_near_override/suction_api"]) > 0
        assert data["sample_near_override/source_index_kind"].item() == "pixel"
        previous = {key: data[key] for key in data.files}
    command.main()
    with np.load(output, allow_pickle=False) as data:
        for key, value in previous.items():
            np.testing.assert_array_equal(data[key], value)


def test_observation_manifest_rejects_duplicate_ids_and_cli_modes_are_exclusive(tmp_path, monkeypatch):
    import scripts.infer as command

    path = tmp_path / "duplicate.jsonl"
    row = json.dumps({"sample_id": "same", "observation": {}})
    path.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        command.read_observation_manifest(path)
    monkeypatch.setattr(sys, "argv", ["infer.py", "--config", "unused", "--checkpoint", "unused",
                                     "--frames", "0:0", "--observation-manifest", str(path), "--output", "unused"])
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2
