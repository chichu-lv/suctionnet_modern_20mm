"""Scheduling checks for the bounded CPU geometry/GPU preparation overlap."""
import threading
import json

import pytest
import torch

from suction20.geometry_pipeline import GeometryPipeline


def test_two_slots_overlap_and_finish_before_the_next_optimizer_window():
    main_thread = threading.get_ident()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    events = []
    prepared_second = threading.Event()
    finished_first = threading.Event()

    def prepare(index):
        assert threading.get_ident() == main_thread
        events.append(("prepare", index))
        graph = model(torch.tensor([[float(index + 1), 1.]]))
        if index == 1:
            prepared_second.set()

        def geometry():
            assert threading.get_ident() != main_thread
            # These waits prove useful main-thread work can run while the
            # geometry job is unfinished; no timing/speed assertion is needed.
            if index == 0:
                assert prepared_second.wait(3)
            elif index == 1:
                assert finished_first.wait(3)
            return index + 1

        return (index, graph), geometry

    before = model.weight.detach().clone()
    with GeometryPipeline() as pipeline:
        for completed in pipeline.iter_update(4, prepare):
            index, graph = completed.payload
            assert completed.geometry_result == index + 1
            assert completed.timing["geometry_job_wall_s"] >= 0
            assert completed.timing["geometry_wait_host_s"] >= 0
            (graph.square().mean() / 4).backward()
            events.append(("finish", index))
            if index == 0:
                finished_first.set()
            del completed, graph
        # Iterating one update cannot prepare work from the following update.
        assert events == [("prepare", 0), ("prepare", 1), ("finish", 0),
                          ("prepare", 2), ("finish", 1), ("prepare", 3),
                          ("finish", 2), ("finish", 3)]
        optimizer.step()
    assert not torch.equal(before, model.weight)


def test_cpu_geometry_exception_reaches_the_trainer():
    def prepare(index):
        def geometry():
            if index == 0:
                raise RuntimeError("native query failed")
            return index
        return index, geometry

    with GeometryPipeline() as pipeline:
        with pytest.raises(RuntimeError, match="native query failed"):
            list(pipeline.iter_update(2, prepare))


def test_production_trainer_completes_two_pipelined_updates(tmp_path, monkeypatch):
    """Real loss/backward/AdamW, with the small CPU backbone and local loader."""
    pytest.importorskip("models._fps_cpu")
    pytest.importorskip("models._spatial_index_cpu")
    from models import Suction20Net
    from scripts import train_pretrain as entry
    from suction20.from_scratch import from_scratch_config
    from suction20.performance import attach_performance
    from test_contact_grouping import FROZEN
    from test_model import TinyTestBackbone
    from test_prefetch import ToySource

    class Source(ToySource):
        data_cfg = FROZEN

        def get_sample(self, index, augmentation_seed):
            sample = super().get_sample(index, augmentation_seed)
            sample["targets"]["gt_B"] = sample["targets"]["gt_A"].copy()
            return sample

    model_holder = []

    def make_model(config, frozen, device):
        model = Suction20Net(config["model"], frozen, TinyTestBackbone(),
                             performance=config["runtime"]["performance"]).to(device)
        model_holder.append(model)
        return model

    monkeypatch.setattr(entry, "make_source_dataset", lambda *args, **kwargs: Source())
    monkeypatch.setattr(entry, "make_model", make_model)
    monkeypatch.setattr(entry, "worktree_state", lambda: {"revision": "synthetic-fixed-source", "tracked_changes": []})
    base = {"experiment": "E3", "seed": 42,
            "model": {"experiment": "E3", "quality_target": "gt_B", "seed_count": 4,
                      "pose_chunk_size": 4},
            "data": {"smoke_frame_keys": [list(key) for key in Source.keys]},
            "training": {"max_optimizer_steps": 3, "microbatch_size": 1,
                         "gradient_accumulation_steps": 2, "effective_batch_size": 2,
                         "validate_every_optimizer_steps": 3},
            "optimizer": {}, "runtime": {"cpu_threads": 1}}
    config, performance = from_scratch_config(base, workers=0, cpu_threads=1,
        geometry_threads=2, dense_chunk_size=32, geometry_pipeline_depth=2)
    performance["loader"]["pin_memory"] = False
    sidecar = tmp_path / "performance.json"
    sidecar.write_text(json.dumps(performance))
    run = tmp_path / "run"
    assert entry.train(attach_performance(config, sidecar), run, torch.device("cpu"),
                       pause_at_step=2, attempt_id="pipeline", profile_update=1) == 2
    rows = [json.loads(line) for line in (run / "attempts/pipeline/updates.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in rows] == [1, 2]
    assert [row["consumed_samples"] for row in rows] == [2, 4]
    assert all(row["gradients_finite"] and row["updated"] for row in rows)
    assert all(row["geometry_pipeline"]["depth"] == 2 for row in rows)
    assert all(row["geometry_pipeline"]["geometry_job_wall_s"] > 0 for row in rows)
    assert (run / rows[0]["profile_trace"]).is_file()
    assert rows[1]["profile_trace"] is None
    assert [frame["microbatch_index"] for frame in rows[-1]["frame_stats"]] == [0, 1]
    assert [frame["sequence_id"] for row in rows for frame in row["frame_details"]] == [0, 1, 2, 3]
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert saved["global_step"] == 2
    assert all(value["step"].item() == 2 for value in saved["optimizer_state_dict"]["state"].values())
