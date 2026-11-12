"""CPU-only orchestration regression; no real data or sparse GPU training."""
import copy
import json

import numpy as np
import pytest
import torch

from models import Suction20Net
from scripts import finetune as entry
from suction20.full_data import DEFAULT_CONFIG
from suction20.target_dataset import load_target_observation
from suction20.training import FrameSchedule, load_config, save_checkpoint
from test_model import TinyTestBackbone


@pytest.mark.parametrize("mode,min_neighbors", [("dense_proxy", 6), ("attempt_success", 6),
                                                ("dense_proxy", 20)])
def test_target_loop_resume_matches_uninterrupted_updates(tmp_path, monkeypatch, mode, min_neighbors):
    frozen = dict(DEFAULT_CONFIG)
    (tmp_path / "source.json").write_text(json.dumps(frozen))
    xyz = np.array([[i * .003, j * .003, .5] for i in range(4) for j in range(4)], np.float32)
    arrays = dict(xyz=xyz, instance_id=np.ones(16, np.int64), source_indices=np.arange(16),
                  normals_raw=np.tile([0., 0., -1.], (16, 1)), normals_raw_valid=np.ones(16, bool))
    if min_neighbors > len(xyz):
        # Default observed PCA can fit this plane, but the configured minimum
        # cannot. Validation must honor the same custom preprocessing as train.
        arrays.pop("normals_raw")
        arrays.pop("normals_raw_valid")
    records = []
    for index, split in enumerate(("train", "train", "val")):
        np.savez(tmp_path / f"cloud{index}.npz", **arrays)
        observation = dict(format="pointcloud_npz", path=f"cloud{index}.npz", g_cam=[0., 0., -1.])
        record = dict(schema_version="suction20.target.v1", sample_id=str(index),
                      group_id=f"group{index}", split=split, observation=observation, label_kind=mode)
        if mode == "dense_proxy":
            np.savez(tmp_path / f"labels{index}.npz", gt_B=np.linspace(0, 1, 16),
                     gt_valid=np.ones(16, bool), instance_id_gt=np.ones(16, np.int64))
            record["labels"] = dict(path=f"labels{index}.npz", source_label_config_path="source.json")
        else:
            record["attempt"] = dict(point_cam_m=xyz[5].tolist(), normal_out_cam=[0., 0., -1.],
                                     instance_id=1, source_index=5, success=index % 2, label_valid=True)
        records.append(record)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    source_sample = load_target_observation(records[0]["observation"], tmp_path)
    source_sample["targets"] = dict(instance_id_gt=np.ones(16, np.int64),
                                    gt_A=np.linspace(0, 1, 16, dtype=np.float32),
                                    gt_B=np.linspace(0, .8, 16, dtype=np.float32),
                                    quality_valid=np.ones(16, bool), axis_valid=np.ones(16, bool),
                                    normal_label=np.tile([0., 0., -1.], (16, 1)).astype(np.float32))

    class SyntheticSource:
        data_cfg = frozen
        keys = [(scene, "realsense", frame) for scene in range(100) if scene % 10 != 9
                for frame in range(256)]

        def get_sample(self, index, seed):
            sample = copy.deepcopy(source_sample)
            # Explicit augmentation seeds affect input, detecting schedule drift.
            sample["inputs"]["xyz_cam_m"] += np.random.default_rng(seed).uniform(-.001, .001, 3)
            return sample

    def make_model(config, physics, device):
        return Suction20Net(dict(config["model"], experiment="E1"), physics, TinyTestBackbone()).to(device)

    monkeypatch.setattr(entry, "make_source_dataset", lambda *args, **kwargs: SyntheticSource())
    monkeypatch.setattr(entry, "make_model", make_model)
    domain_losses = {"target": [], "source": []}
    real_target_loss, real_source_loss = entry.target_loss, entry.source_losses

    def target_loss(*args, **kwargs):
        losses, diagnostics = real_target_loss(*args, **kwargs)
        domain_losses["target"].append(float(losses["total"].detach()))
        return losses, diagnostics

    def source_loss(*args, **kwargs):
        losses = real_source_loss(*args, **kwargs)
        domain_losses["source"].append(float(losses["total"].detach()))
        return losses

    monkeypatch.setattr(entry, "target_loss", target_loss)
    monkeypatch.setattr(entry, "source_losses", source_loss)
    public_config = dict(experiment="E1", model=dict(quality_target="gt_B", seed_count=8), data={})
    torch.manual_seed(17)
    public = make_model(public_config, frozen, "cpu")
    public_optimizer = torch.optim.AdamW(public.parameters())
    # Give the synthetic source checkpoint nonempty Adam moments. Target init
    # must still create fresh state, not merely reset the visible step counter.
    sum(parameter.square().mean() for parameter in public.parameters()).backward()
    public_optimizer.step()
    public_optimizer.zero_grad(set_to_none=True)
    source_checkpoint = tmp_path / "public.pt"
    save_checkpoint(source_checkpoint, public, public_optimizer, torch.amp.GradScaler("cuda", enabled=False),
                    FrameSchedule([0]), global_step=777, config=public_config, source_label_config=frozen)
    config = load_config(f"docs/implementation/examples/finetune_{mode}.yaml")
    config["target_data"]["manifest"] = str(manifest)
    config["target_data"]["normal_estimation"] = {"min_neighbors": min_neighbors}
    config["runtime"] = {"cpu_threads": 1}
    config["training"].update(max_optimizer_steps=2, amp=False)
    config["validation"]["every_optimizer_steps"] = 1
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    assert entry.finetune(config, full, torch.device("cpu"), init_checkpoint=source_checkpoint) == 2
    assert entry.finetune(config, resumed, torch.device("cpu"), init_checkpoint=source_checkpoint, stop_at_step=1) == 1
    assert entry.finetune(config, resumed, torch.device("cpu"), resume=resumed / "last.pt") == 2
    expected = torch.load(full / "last.pt", weights_only=False)
    actual = torch.load(resumed / "last.pt", weights_only=False)
    assert actual["global_step"] == 2  # Public optimizer step 777 is not inherited.
    assert actual["sampler_state"] == expected["sampler_state"]
    assert actual["target_sampler_state"] == expected["target_sampler_state"]
    assert actual["source_checkpoint_ref"] == str(source_checkpoint)
    assert actual["parameter_phase"] == "heads_only"
    if mode == "dense_proxy":
        validations = [json.loads(line) for line in (full / "validation.jsonl").read_text().splitlines()]
        expected_queries = 0 if min_neighbors > len(xyz) else len(xyz)
        assert all(row["fixed_query_count"] == expected_queries for row in validations)
    updates = [json.loads(line) for directory in (full, resumed)
               for line in (directory / "updates.jsonl").read_text().splitlines()]
    for index, update in enumerate(updates):
        for domain, weight in (("target", .5), ("source", .25)):
            raw = domain_losses[domain][2 * index:2 * index + 2]
            assert len(raw) == 2
            assert update["loss"][domain] == pytest.approx(weight * sum(raw))
    for key, value in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][key], value, rtol=0, atol=0)
    for key, value in public.backbone.state_dict().items():
        torch.testing.assert_close(actual["model_state_dict"][f"backbone.{key}"], value, rtol=0, atol=0)
    assert all(int(state["step"]) == 2 for state in actual["optimizer_state_dict"]["state"].values())
