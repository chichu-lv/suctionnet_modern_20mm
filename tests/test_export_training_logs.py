"""Commit/rollback/tail analysis without checkpoint deserialization."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from suction20.training_telemetry import export_training_logs


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def update(step, attempt="a", ordinal=None, *, updated=True, micro=1, accumulation=2):
    frames = [[1, "realsense", i] for i in range(micro * accumulation)]
    return dict(global_step=step, updated=updated, attempt_id=attempt,
        optimizer_attempt=ordinal if ordinal is not None else step,
        frames=frames, microbatch_size=micro, gradient_accumulation_steps=accumulation,
        effective_batch_size=micro * accumulation,
        frame_details=[dict(sample_uid=frame, augmentation_seed=i + 123, microbatch_index=i // micro,
                            point_count=10, voxel_count=5) for i, frame in enumerate(frames)],
        frame_stats=[dict(microbatch_index=i, point_count=10 * micro, voxel_count=5 * micro)
                     for i in range(accumulation)],
        elapsed_seconds=2., lr=.001, loss={"total": .25})


def write_log(root, attempt, rows, tail=b""):
    path = root / "attempts" / attempt / "updates.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join((json.dumps(r) + "\n").encode() for r in rows)
    path.write_bytes(raw + tail)
    return dict(path=str(path.relative_to(root)), attempt_id=attempt, end_offset=len(raw), sha256=digest(raw))


def checkpoint(root, name, step, segments, *, skipped=0, frames=None, attempt="a", validations=None):
    path = root / "checkpoints" / (name + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"Deliberately not a pickle checkpoint: " + name.encode()
    path.write_bytes(raw)
    return dict(path=str(path.relative_to(root)), size_bytes=len(raw), sha256=digest(raw),
        generation=name, global_step=step, attempt_id=attempt,
        analysis=dict(schema_version=1, global_step=step, attempt_id=attempt,
            identity={"budget_contract": "test", "runtime_profile": "tiny"}, budget={"target_samples": 20},
            ledger=dict(lineage=segments, attempts=step + skipped, skipped=skipped,
                        frames=frames if frames is not None else (step + skipped) * 2),
            validations=validations or {}, best=None))


def manifest(root, current, predecessor=None):
    write_json(root / "checkpoint-manifest.json", dict(schema_version=1, current=current, predecessor=predecessor))


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_committed_and_live_tail_separate_and_outputs_hashed(tmp_path):
    root, out = tmp_path / "run", tmp_path / "export"
    segment = write_log(root, "a", [update(1)], (json.dumps(update(2)) + "\n").encode())
    manifest(root, checkpoint(root, "current", 1, [segment]))
    summary = export_training_logs(root, out, live_attempt_id="a")
    assert summary["categories"]["committed"]["successful_updates"] == 1
    assert summary["categories"]["live_uncommitted"]["successful_updates"] == 1
    assert summary["categories"]["committed"]["frames_per_training_second"] == 1
    assert len(read_csv(out / "frames.csv")) == 4
    exported = json.loads((out / "export-manifest.json").read_text())
    for entry in exported["outputs"]:
        assert digest((out / entry["path"]).read_bytes()) == entry["sha256"]
    assert any(s["role"] == "committed_update_prefix" for s in exported["sources"])
    with pytest.raises(ValueError, match="already exists"):
        export_training_logs(root, out)


def test_replayed_steps_never_double_count_abandoned_tail(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)], (json.dumps(update(2)) + "\n").encode())
    old_raw = (root / segment["path"]).read_bytes()
    resumed = write_log(root, "b", [update(2, "b", 2)])
    write_json(root / "attempts/b/identity.json", dict(abandoned_tails=[dict(path=segment["path"],
        start_offset=segment["end_offset"], end_offset=len(old_raw), sha256=digest(old_raw))]))
    manifest(root, checkpoint(root, "current", 2, [segment, resumed], attempt="b"))
    summary = export_training_logs(root, tmp_path / "export")
    assert summary["categories"]["committed"]["successful_updates"] == 2
    assert summary["categories"]["abandoned"]["successful_updates"] == 1
    assert summary["categories"]["uncommitted"]["attempts"] == 0


def test_predecessor_requires_explicit_selection_and_labels_rollback(tmp_path):
    root = tmp_path / "run"
    first = write_log(root, "a", [update(1)])
    predecessor = checkpoint(root, "previous", 1, [first])
    second = write_log(root, "a", [update(1), update(2)])
    current = checkpoint(root, "current", 2, [second])
    manifest(root, current, predecessor)
    (root / current["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checkpoint checksum mismatch"):
        export_training_logs(root, tmp_path / "refused")
    summary = export_training_logs(root, tmp_path / "export", checkpoint_role="predecessor")
    assert summary["checkpoint_role"] == "predecessor"
    assert summary["categories"]["committed"]["successful_updates"] == 1
    assert summary["categories"]["uncommitted"]["successful_updates"] == 1


def test_amp_skip_is_attempt_and_consumption_but_not_credit(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1), update(1, ordinal=2, updated=False), update(2, ordinal=3)])
    manifest(root, checkpoint(root, "current", 2, [segment], skipped=1))
    row = export_training_logs(root, tmp_path / "export")["categories"]["committed"]
    assert row["attempts"] == 3 and row["successful_updates"] == 2
    assert row["consumed_sample_presentations"] == 6 and row["credited_sample_presentations"] == 4


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(effective_batch_size=9), "Frame cardinality"),
    (lambda r: r["frame_details"].pop(), "Frame details cardinality"),
    (lambda r: r["frame_details"][0].update(microbatch_index=1), "membership"),
    (lambda r: r["frame_stats"][0].update(point_count=99), "point_count"),
    (lambda r: r.update(global_step=3), "not continuous"),
])
def test_rejects_incoherent_committed_batch_or_sequence(tmp_path, mutation, match):
    root = tmp_path / "run"
    row = update(1, micro=2, accumulation=2)
    mutation(row)
    segment = write_log(root, "a", [row])
    manifest(root, checkpoint(root, "current", 1, [segment], frames=4))
    with pytest.raises(ValueError, match=match):
        export_training_logs(root, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_microbatch_two_by_two_exports_four_frame_details(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1, micro=2, accumulation=2)])
    manifest(root, checkpoint(root, "current", 1, [segment], frames=4))
    export_training_logs(root, tmp_path / "export")
    rows = read_csv(tmp_path / "export/frames.csv")
    assert [r["microbatch_index"] for r in rows] == ["0", "0", "1", "1"]
    assert [r["augmentation_seed"] for r in rows] == ["123", "124", "125", "126"]


@pytest.mark.parametrize("bad", [b'{"updated": NaN}\n', b'{"updated":1e999}\n', b'{malformed}\n', b'{"truncated":'])
def test_malformed_committed_refused_but_tail_is_reported_gap(tmp_path, bad):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)], bad)
    current = checkpoint(root, "current", 1, [segment])
    manifest(root, current)
    summary = export_training_logs(root, tmp_path / "export")
    assert len(summary["gaps"]) == 1
    assert summary["categories"]["committed"]["successful_updates"] == 1
    raw = (root / segment["path"]).read_bytes()
    segment.update(end_offset=len(raw), sha256=digest(raw))
    manifest(root, current)
    with pytest.raises(ValueError, match="Invalid committed update"):
        export_training_logs(root, tmp_path / "refused")


def test_committed_hash_mismatch_refused_even_if_json_valid(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)])
    manifest(root, checkpoint(root, "current", 1, [segment]))
    path = root / segment["path"]
    path.write_bytes(path.read_bytes().replace(b'0.25', b'0.75'))
    with pytest.raises(ValueError, match="checksum mismatch"):
        export_training_logs(root, tmp_path / "export")


def test_validation_and_resource_observations_remain_separate(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)])
    val = root / "validations/one.json"
    write_json(val, dict(global_step=1, metrics=dict(split="val", AP_R20=.2,
        frames=[dict(scene_id=9, frame_id=0, AP_R20=.2)])))
    refs = {"1": dict(path="validations/one.json", sha256=digest(val.read_bytes()))}
    manifest(root, checkpoint(root, "current", 1, [segment], validations=refs))
    monitor = tmp_path / "resources.jsonl"
    monitor.write_text(json.dumps(dict(status=dict(global_step=999), telemetry=dict(gpu=dict(power_watts=None)))) + "\n")
    summary = export_training_logs(root, tmp_path / "export", resource_logs=[monitor])
    assert summary["categories"]["committed"]["successful_updates"] == 1
    assert summary["resource_sample_count"] == 1
    assert summary["validations"][0]["metrics"]["AP_R20"] == .2
    assert len(read_csv(tmp_path / "export/validation_frames.csv")) == 1
    assert read_csv(tmp_path / "export/resources.csv")[0]["classification"] == "observation_not_committed_progress"


def test_legacy_missing_analysis_and_path_escape_fail_closed(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)])
    entry = checkpoint(root, "current", 1, [segment])
    analysis = entry.pop("analysis")
    manifest(root, entry)
    with pytest.raises(ValueError, match="analysis metadata"):
        export_training_logs(root, tmp_path / "export")
    entry["analysis"] = analysis
    segment["path"] = "../outside.jsonl"
    manifest(root, entry)
    with pytest.raises(ValueError, match="outside run"):
        export_training_logs(root, tmp_path / "export")


def test_event_prefix_is_verified_and_tail_remains_uncommitted(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)])
    current = checkpoint(root, "current", 1, [segment])
    event = root / "attempts/a/events.jsonl"
    first = (json.dumps(dict(event_id="one", attempt_id="a", kind="checkpoint_started")) + "\n").encode()
    second = (json.dumps(dict(event_id="two", attempt_id="a", kind="checkpoint_committed")) + "\n").encode()
    event.write_bytes(first + second)
    current["analysis"]["ledger"]["events_lineage"] = [dict(path="attempts/a/events.jsonl",
        attempt_id="a", end_offset=len(first), sha256=digest(first))]
    manifest(root, current)
    export_training_logs(root, tmp_path / "export")
    assert [r["classification"] for r in read_csv(tmp_path / "export/events.csv")] == ["committed_event", "uncommitted_event"]
    event.write_bytes(first.replace(b"one", b"bad") + second)
    with pytest.raises(ValueError, match="checksum mismatch"):
        export_training_logs(root, tmp_path / "refused")


def test_credited_counters_cannot_disagree_with_logged_frames(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)])
    current = checkpoint(root, "current", 1, [segment])
    current["analysis"]["counters"] = {"effective_samples_completed": 20}
    manifest(root, current)
    with pytest.raises(ValueError, match="log-derived counter"):
        export_training_logs(root, tmp_path / "refused")


def test_validation_hash_mismatch_does_not_publish_partial_export(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)])
    val = root / "validations/one.json"
    write_json(val, dict(global_step=1, metrics=dict(split="val", frames=[dict(scene_id=9, frame_id=0)])))
    refs = {"1": dict(path="validations/one.json", sha256="wrong")}
    manifest(root, checkpoint(root, "current", 1, [segment], validations=refs))
    with pytest.raises(ValueError, match="committed_validation checksum mismatch"):
        export_training_logs(root, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_public_checkpoint_alias_cannot_disagree_with_analysis(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)])
    current = checkpoint(root, "current", 1, [segment])
    current["budget"] = {"target_samples": 999}
    manifest(root, current)
    with pytest.raises(ValueError, match="public budget disagrees"):
        export_training_logs(root, tmp_path / "export")


def test_formal_validation_cannot_claim_completion_with_short_result(tmp_path):
    root = tmp_path / "run"
    segment = write_log(root, "a", [update(1)])
    val = root / "validations/one.json"
    write_json(val, dict(global_step=1, metrics=dict(split="val", frames=[dict(scene_id=9, frame_id=0)])))
    refs = {"1": dict(path="validations/one.json", sha256=digest(val.read_bytes()))}
    current = checkpoint(root, "current", 1, [segment], validations=refs)
    current["analysis"]["budget"]["formal"] = True
    current["budget"] = current["analysis"]["budget"]
    current["runtime_profile"] = "tiny"
    manifest(root, current)
    with pytest.raises(ValueError, match="exactly 320"):
        export_training_logs(root, tmp_path / "export")
