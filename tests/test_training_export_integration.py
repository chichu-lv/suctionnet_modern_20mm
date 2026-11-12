"""Real CPU trainer publication -> no-pickle exporter integration."""
import torch

from scripts import train_pretrain as entry
from suction20 import validation
from suction20.training_telemetry import export_training_logs
from test_pretrain_runtime import harness  # shared tiny stochastic trainer fixture


def test_pause_and_resume_exports_exact_canonical_prefix(tmp_path, harness):
    config, calls, _ = harness
    run = tmp_path / "run"
    assert entry.train(config, run, torch.device("cpu"), pause_at_step=1) == 1
    first = export_training_logs(run, tmp_path / "first")
    assert first["counters"]["effective_samples_completed"] == 2
    assert first["event_count"] > 0
    # Pause again before the synthetic validation node, avoiding test-only
    # validation mocks whose result intentionally omits real protocol metadata.
    assert entry.train(config, run, torch.device("cpu"), resume=run / "last.pt", pause_at_step=2) == 2
    second = export_training_logs(run, tmp_path / "second")
    assert second["counters"]["effective_samples_completed"] == 4
    assert second["categories"]["committed"]["attempts"] == 2
    assert second["categories"]["uncommitted"]["attempts"] == 0


def test_completed_cpu_validation_exports_manifest_references(tmp_path, harness, monkeypatch):
    config, _, evaluate = harness
    def with_protocol(*args, **kwargs):
        return dict(evaluate(*args, **kwargs), split="smoke_val")
    monkeypatch.setattr(validation, "evaluate_model", with_protocol)
    run = tmp_path / "run"
    assert entry.train(config, run, torch.device("cpu")) == 4
    result = export_training_logs(run, tmp_path / "export")
    assert [record["global_step"] for record in result["validations"]] == [2, 4]
    assert result["best"]["global_step"] == 2  # tied final metric preserves the earlier best
    assert result["sample_progress"]["fraction"] == 1
