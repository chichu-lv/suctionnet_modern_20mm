"""No torch, GPU, SSH, or real training required."""
import copy
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys

import pytest
from suction20 import run_supervisor as s


@pytest.fixture
def config(tmp_path):
    root, backup = tmp_path / "runtime", tmp_path / "backup"
    root.mkdir()
    backup.mkdir()
    return {"root": str(root), "backup_root": str(backup), "python": sys.executable,
            "source_root": str(tmp_path / "dataset/input_raw"),
            "label_root": str(tmp_path / "dataset/output"), "backbone_root": str(root / "external"),
            "manifest_sha256": "manifest", "hosts": {
                name: {"machine_id": "test", "hostname": name, "ssh": name, "gpus": {
                    str(i): {"uuid": f"GPU-{name}-{i}", "cpus": list(range(i * 8, i * 8 + 8))}
                    for i in range(count)}} for name, count in (("H20", 2), ("L20", 4))}}


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.view = {"state": "running", "locks_free": False}

    def call(self, host, request, timeout=30):
        self.calls.append((host, request))
        if self.fail:
            raise subprocess.TimeoutExpired("ssh", 30)
        return dict(self.view) if request["op"] == "inspect" else {"ok": True}


@pytest.fixture
def coordinator(config, tmp_path):
    return s.Coordinator(config, tmp_path / "ledger.json", FakeTransport())


def test_ssh_ambiguity_reserves_before_launch_and_restart(coordinator):
    coordinator.transport.fail = True
    assignment = coordinator.assign("E5", "H20", 0)
    assert assignment["state"] == "launch_unknown"
    restarted = s.Coordinator(coordinator.config, coordinator.path, coordinator.transport)
    with pytest.raises(s.Refused, match="already reserved"):
        restarted.assign("E5", "L20", 0)
    with pytest.raises(s.Refused, match="device already reserved"):
        restarted.assign("E2", "H20", 0)
    restarted.reconcile()
    assert restarted.state["assignments"]["E5"]["state"] == "unknown"
    assert len(coordinator.transport.calls) == 2


def test_all_six_independent_assignments_allowed(coordinator):
    for experiment, host, gpu in s.INITIAL:
        coordinator.assign(experiment, host, gpu)
    coordinator.assign("E2", "H20", 1)
    assert len(coordinator.state["assignments"]) == 6
    assert len({a["token"] for a in coordinator.state["assignments"].values()}) == 6


def test_release_requires_positive_exit_and_checkpoint(coordinator):
    coordinator.assign("E5", "H20", 0)
    with pytest.raises(s.Refused, match="prove"):
        coordinator.release("E5", {})
    coordinator.transport.view = {"state": "exited", "locks_free": True}
    with pytest.raises(s.Refused, match="checkpoint"):
        coordinator.release("E5", {})
    coordinator.transport.view["checkpoint_verified"] = True
    coordinator.release("E5", {"reviewed": True})
    assert "E5" not in coordinator.state["assignments"]
    assert coordinator.state["history"][0]["release_evidence"] == {"reviewed": True}


def test_validation_serialized_and_network_loss_keeps_owner(coordinator):
    for experiment, host, gpu in s.INITIAL[:2]:
        assignment = coordinator.assign(experiment, host, gpu)
        assignment.update(state="running", view={"validation_request": {
            "request_id": experiment, "attempt_id": assignment["token"], "step": 5760}})
    coordinator.grants()
    assert coordinator.state["validation"]["request_id"] == "E5"
    coordinator.state["assignments"]["E5"]["state"] = "unknown"
    coordinator.grants()
    assert coordinator.state["validation"]["request_id"] == "E5"
    coordinator.state["assignments"]["E5"].update(state="running", view={
        "validation_done": {"request_id": "E5"}})
    coordinator.grants()
    assert coordinator.state["validation"]["request_id"] == "E3"


def test_reserved_grant_replayed_after_delivery_timeout(coordinator):
    assignment = coordinator.assign("E5", "H20", 0)
    request = {"request_id": "val1", "attempt_id": assignment["token"], "step": 5760}
    assignment.update(state="running", view={"validation_request": request})
    coordinator.transport.fail = True
    coordinator.grants()
    assert s.read_json(coordinator.path)["validation"]["request_id"] == "val1"
    coordinator.transport.fail = False
    coordinator.grants()
    assert coordinator.transport.calls[-1][1]["grant"]["request_id"] == "val1"


def test_exited_owner_releases_validation_after_crash(coordinator):
    assignment = coordinator.assign("E5", "H20", 0)
    coordinator.state["validation"] = {"token": assignment["token"], "request_id": "old"}
    assignment.update(state="exited", view={"state": "exited", "locks_free": True})
    coordinator.grants()
    assert coordinator.state["validation"] is None


def test_staged_launch_uses_evidence_not_two_job_cap(coordinator):
    coordinator.state["evidence"].update({"host:H20": True, "host:L20": True})
    assert coordinator.next_launch() == ("E5", "H20", 0)
    coordinator.assign("E5", "H20", 0)
    assert coordinator.next_launch() == ("E3", "L20", 0)
    coordinator.assign("E3", "L20", 0)
    assert coordinator.next_launch() is None
    coordinator.state["evidence"]["stable:E3"] = True
    assert coordinator.next_launch() == ("E0", "L20", 1)
    coordinator.assign("E0", "L20", 1)
    coordinator.state["evidence"]["stable:E0"] = True
    assert coordinator.next_launch() == ("E1", "L20", 2)


def test_busy_gpu_invisible_pid_and_uuid_change_rejected():
    for device in ({"uuid": "GPU-a", "memory_mib": 1024, "pids": []},
                   {"uuid": "GPU-a", "memory_mib": 0, "pids": [1]},
                   {"uuid": "GPU-b", "memory_mib": 0, "pids": []}):
        with pytest.raises(s.Refused):
            s.assert_idle({1: device}, 1, "GPU-a")
    s.assert_idle({1: {"uuid": "GPU-a", "memory_mib": 66, "pids": []}}, 1, "GPU-a")


def test_resource_budget_and_validation_wait():
    sample = {"available_bytes": 512 * s.GIB, "local_free": 300 * s.GIB,
              "backup_free": 300 * s.GIB, "local_bytes": 0, "backup_bytes": 0, "rss_bytes": 0}
    s.launch_resources(sample, 4)
    with pytest.raises(s.Refused, match="RAM"):
        s.launch_resources(sample, 5)
    assert s.pause_reasons(sample, {"phase": "waiting_validation", "progress_unix": 1}, 5000, 4999, 1) == []
    assert "no compute progress for 45 minutes" in s.pause_reasons(
        sample, {"phase": "training", "progress_unix": 1}, 5000, 4999, 1)
    sample["rss_bytes"] = 64 * s.GIB
    assert any("RSS" in r for r in s.pause_reasons(sample, {}, 5000, 4999, 4999))


def test_config_rejects_unauthorized_gpu_and_overlapping_affinity(config):
    s.validate_config(config)
    config["hosts"]["L20"]["gpus"]["4"] = config["hosts"]["L20"]["gpus"]["0"]
    with pytest.raises(s.Refused, match="authorized"):
        s.validate_config(config)
    del config["hosts"]["L20"]["gpus"]["4"]
    config["hosts"]["L20"]["gpus"]["1"]["cpus"] = list(range(8))
    with pytest.raises(s.Refused, match="overlaps"):
        s.validate_config(config)


def test_data_roots_not_writable_destination(config):
    config["backup_root"] = config["label_root"] + "/backups"
    with pytest.raises(s.Refused, match="read-only"):
        s.validate_config(config)


def test_locks_inherited_by_child(tmp_path):
    path = tmp_path / "gpu.lock"
    with s.lock(path) as descriptor:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=(descriptor,))
    try:
        with pytest.raises(s.Refused, match="held"):
            with s.lock(path):
                pass
    finally:
        child.terminate()
        child.wait()
    with s.lock(path):
        pass


def test_wrong_process_identity_cannot_be_killed():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = s.process_identity(child.pid)
        wrong = dict(identity, start_ticks=identity["start_ticks"] + 1)
        assert s.signal_owned(wrong, signal.SIGTERM) is False
        assert child.poll() is None
        assert s.signal_owned(identity, signal.SIGTERM) is True
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_trainer_cli_resume_and_runtime_caps(coordinator):
    assignment = coordinator.assign("E3", "L20", 0, resume="checkpoints/a.pt", prefix_passed=True)
    command = s.trainer_command(coordinator.config, assignment)
    assert "--resume" in command and "--pause-at-step" not in command
    assert command[command.index("--gpu-memory-limit-gib") + 1] == "40"
    assert "--init-checkpoint" not in command
    assert assignment["token"] == command[command.index("--attempt-id") + 1]


def test_backup_checks_hash_deduplicates_and_keeps_two_hourlies(coordinator, monkeypatch):
    assignment = coordinator.assign("E5", "H20", 0)
    config = coordinator.config
    run, _ = s.assignment_paths(config, assignment)
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/a.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    entry = {"path": "checkpoints/a.pt", "sha256": s.digest(checkpoint), "size_bytes": checkpoint.stat().st_size}
    s.atomic_json(run / "checkpoint-manifest.json", {"current": entry, "predecessor": None, "best": entry, "nodes": {}})
    (run / "attempts").mkdir()
    (run / "attempts/attempt.jsonl").write_text('{"step":1}\n')
    original_usage = s.shutil.disk_usage
    monkeypatch.setattr(s.shutil, "disk_usage", lambda path: original_usage(path)._replace(free=300 * s.GIB))
    destinations = []
    for number in range(3):
        request = {"request_id": str(number), "attempt_id": assignment["token"]}
        destinations.append(Path(s.backup_snapshot(config, assignment, request)))
    assert not destinations[0].exists()
    assert destinations[-1].joinpath("backup-complete.json").exists()
    assert destinations[-1].joinpath("checkpoints/a.pt").read_bytes() == b"immutable-checkpoint"
    assert len(list((Path(config["backup_root"]) / "objects").iterdir())) == 1
    checkpoint.write_bytes(b"corruption")
    backup = Path(s.backup_snapshot(config, assignment, {"request_id": "cached"}))
    assert (backup / "checkpoints/a.pt").read_bytes() == b"immutable-checkpoint"
    (Path(config["backup_root"]) / "objects" / entry["sha256"]).unlink()
    with pytest.raises(s.Refused, match="truncated|checksum"):
        s.backup_snapshot(config, assignment, {"request_id": "bad"})


def test_transport_passes_json_without_shell_payload(config, tmp_path, monkeypatch):
    captured = {}
    config["hosts"]["L20"]["ssh_options"] = ["-i", "/private/key", "-p", "20007"]
    monkeypatch.setattr(s, "machine_id", lambda: "different")
    def run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, stdout='{"ok":true}')
    monkeypatch.setattr(s.subprocess, "run", run)
    transport = s.Transport(config, tmp_path / "config with spaces.json")
    assert transport.call("L20", {"op": "probe"}) == {"ok": True}
    assert json.loads(captured["kwargs"]["input"]) == {"op": "probe"}
    assert "-i" in captured["command"]
    assert " rpc --host L20" in captured["command"][-1]


def test_cuda_ready_maps_container_pid_and_does_not_yield_after_ownership():
    assignment = {"token": "token", "gpu_uuid": "GPU-one"}
    ready = {"attempt_id": "token", "gpu_uuid": "GPU-one", "trainer_pid": 10, "cuda_context_ready": True}
    assert s.gpu_owner({"uuid": "GPU-one", "pids": [999]}, {}, assignment, trainer_pid=10) == (None, None)
    assert s.gpu_owner({"uuid": "GPU-one", "pids": [999]}, ready, assignment, trainer_pid=10) == (999, None)
    assert s.gpu_owner({"uuid": "GPU-one", "pids": [999, 1000]}, ready, assignment, 999, trainer_pid=10) == (999, None)
    assert s.gpu_owner({"uuid": "GPU-one", "pids": [999, 1000]}, ready, assignment, trainer_pid=10)[1]
    assert s.gpu_owner({"uuid": "GPU-one", "pids": []}, ready, assignment, trainer_pid=10) == (None, None)
    assert s.gpu_owner({"uuid": "GPU-one", "pids": [999]}, dict(ready, trainer_pid=99), assignment, trainer_pid=10)[1]
    assert s.gpu_owner({"uuid": "GPU-one", "pids": [999]}, ready, assignment, trainer_alive=False, trainer_pid=10) == (None, None)


def test_data_pressure_same_run_baseline_and_ten_minute_threshold():
    monitor = s.DataPressure()
    def status(marker, data):
        return {"phase": "training", "progress_unix": marker,
                "last_update": {"data_seconds": data, "elapsed_seconds": 10}}
    for i in range(20):
        assert monitor.observe(status(i, 1), i) is False
    assert monitor.observe(status(20, 5), 100) is False
    assert monitor.observe(status(21, 5), 699) is False
    assert monitor.observe(status(22, 5), 701) is True
    for i in range(30):
        monitor.observe(status(30 + i, 1), 702 + i)
    assert monitor.since is None


def test_unknown_owner_freezes_expansion(coordinator):
    coordinator.state["evidence"].update({"host:H20": True, "host:L20": True})
    a = coordinator.assign("E5", "H20", 0)
    a["state"] = "unknown"
    assert coordinator.next_launch() is None


def test_cloned_machine_id_still_requires_hostname(config, monkeypatch):
    monkeypatch.setattr(s, "machine_id", lambda: "test")
    monkeypatch.setattr(s.socket, "gethostname", lambda: "H20")
    assert s.is_local_host(config, "H20")
    assert not s.is_local_host(config, "L20")
    with pytest.raises(s.Refused, match="hostname"):
        s.host_guard(config, "L20")


def test_committed_manifest_recovery_ignores_stale_status_and_reports_fallback(tmp_path):
    run = tmp_path
    (run / "checkpoints").mkdir()
    current, predecessor = run / "checkpoints/current.pt", run / "checkpoints/predecessor.pt"
    current.write_bytes(b"current")
    predecessor.write_bytes(b"previous")
    entries = {role: {"path": str(path.relative_to(run)), "sha256": s.digest(path), "global_step": step}
               for role, path, step in (("current", current, 200), ("predecessor", predecessor, 100))}
    s.atomic_json(run / "checkpoint-manifest.json", entries)
    assert s.checkpoint_verified(run, {"durable_checkpoint": {"path": "pruned.pt", "sha256": "bad"}})
    assert s.committed_checkpoint(run)["global_step"] == 200
    current.write_bytes(b"corrupt")
    assert s.committed_checkpoint(run)["recovery_role"] == "predecessor"
    assert s.committed_checkpoint(run)["global_step"] == 100


def test_backup_pins_survive_pruning_and_include_only_selected_control(coordinator, monkeypatch):
    assignment = coordinator.assign("E5", "H20", 0)
    run, control = s.assignment_paths(coordinator.config, assignment)
    (run / "checkpoints").mkdir(parents=True)
    control.mkdir(parents=True)
    original = run / "checkpoints/current.pt"
    original.write_bytes(b"pinned")
    entry = {"path": "checkpoints/current.pt", "sha256": s.digest(original), "size_bytes": 6}
    s.atomic_json(run / "checkpoint-manifest.json", {"current": entry})
    (control / "resources.jsonl").write_text('{"rss":42}\n')
    (control / "transport_ed25519").write_text('private key must never be backed up')
    original_copy = s._backup_snapshot
    def prune_then_copy(*args):
        original.unlink()
        return original_copy(*args)
    monkeypatch.setattr(s, "_backup_snapshot", prune_then_copy)
    usage = s.shutil.disk_usage
    monkeypatch.setattr(s.shutil, "disk_usage", lambda p: usage(p)._replace(free=300 * s.GIB))
    result = Path(s.backup_snapshot(coordinator.config, assignment, {"request_id": "pin-test"}))
    assert (result / "checkpoints/current.pt").read_bytes() == b"pinned"
    assert (result / "supervisor/resources.jsonl").read_text() == '{"rss":42}\n'
    assert not (result / "supervisor/transport_ed25519").exists()
    assert not list((control / "backup-pins").iterdir())


def test_atomic_control_replacement_does_not_truncate_backup_json(coordinator, monkeypatch):
    assignment = coordinator.assign("E5", "H20", 0)
    run, control = s.assignment_paths(coordinator.config, assignment)
    (run / "checkpoints").mkdir(parents=True)
    control.mkdir(parents=True)
    checkpoint = run / "checkpoints/current.pt"
    checkpoint.write_bytes(b"pinned")
    entry = {"path": "checkpoints/current.pt", "sha256": s.digest(checkpoint), "size_bytes": 6}
    s.atomic_json(run / "checkpoint-manifest.json", {"current": entry})
    s.atomic_json(control / "status.json", {"original": True})
    original_pin = s.pin_file
    def replace_after_pin(path, root):
        result = original_pin(path, root)
        if path == control / "status.json":
            s.atomic_json(path, {"replacement": "a much longer value than the original"})
        return result
    monkeypatch.setattr(s, "pin_file", replace_after_pin)
    usage = s.shutil.disk_usage
    monkeypatch.setattr(s.shutil, "disk_usage", lambda p: usage(p)._replace(free=300 * s.GIB))
    result = Path(s.backup_snapshot(coordinator.config, assignment, {"request_id": "atomic-json"}))
    assert s.read_json(result / "supervisor/status.json") == {"original": True}


@pytest.mark.parametrize("copy_failure", [False, True])
@pytest.mark.parametrize("control_override", [False, True])
def test_worker_final_backup_grant_and_bounded_copy_failure(coordinator, monkeypatch, copy_failure, control_override):
    assignment = coordinator.assign("E5", "H20", 0)
    config = coordinator.config
    if control_override:
        config["control_repo"] = str(Path(config["root"]) / "control-repo")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/imports")
    run, control = s.assignment_paths(config, assignment)
    label = Path(config["label_root"])
    label.mkdir(parents=True)
    (label / "manifest.json").write_text('{}')
    config["manifest_sha256"] = s.digest(label / "manifest.json")
    monkeypatch.setattr(s, "host_guard", lambda *a: None)
    monkeypatch.setattr(s.os, "sched_getaffinity", lambda pid: set(range(8)))
    monkeypatch.setattr(s.time, "sleep", lambda seconds: None)
    sample = {"available_bytes": 512 * s.GIB, "local_free": 300 * s.GIB,
              "backup_free": 300 * s.GIB, "local_bytes": 0, "backup_bytes": 0, "rss_bytes": 0, "psi": {}}
    monkeypatch.setattr(s, "resources", lambda *a: dict(sample))
    monkeypatch.setattr(s, "gpu_snapshot", lambda: {0: {"uuid": assignment["gpu_uuid"], "memory_mib": 0, "pids": []}})
    class FinishedTrainer:
        pid = 999999999
        returncode = 75
        def poll(self):
            return 75
    def spawn(*args, **kwargs):
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "PYTHONPATH" not in kwargs["env"]
        assert kwargs["cwd"] == Path(config["root"]) / "repo"
        assert str(Path(config["root"]) / "repo/scripts/train_pretrain.py") in args[0]
        assert len(kwargs["pass_fds"]) == 2
        (run / "checkpoints").mkdir()
        checkpoint = run / "checkpoints/current.pt"
        checkpoint.write_bytes(b"checkpoint")
        entry = {"path": "checkpoints/current.pt", "sha256": s.digest(checkpoint), "global_step": 100}
        s.atomic_json(run / "checkpoint-manifest.json", {"current": entry})
        s.atomic_json(control / "status.json", {"phase": "paused", "global_step": 100, "durable_checkpoint": entry})
        return FinishedTrainer()
    monkeypatch.setattr(s.subprocess, "Popen", spawn)
    read = s.read_json
    def grants(path, default=None):
        if Path(path).name == "backup.grant.json":
            return read(control / "backup.request.json", {})
        return read(path, default)
    monkeypatch.setattr(s, "read_json", grants)
    calls = []
    def backup(*args):
        calls.append(args)
        if copy_failure:
            raise s.Refused("corrupt current blob")
        return "verified-snapshot"
    monkeypatch.setattr(s, "backup_snapshot", backup)
    if copy_failure:
        with pytest.raises(s.Refused, match="repeated final backup"):
            s.run_worker(config, "H20", assignment)
        assert len(calls) == 2
        assert read(control / "worker.json")["exit_code"] == 1
    else:
        s.run_worker(config, "H20", assignment)
        assert len(calls) == 1
        assert read(control / "worker.json")["exit_code"] == 75
        assert read(control / "backup.done.json")["destination"] == "verified-snapshot"
    for path in s.lock_paths(config, assignment):
        with s.lock(path):
            pass


def test_missing_nvml_snapshot_does_not_disable_readiness_timeout():
    ready = {"cuda_context_ready": True, "attempt_id": "token", "trainer_pid": 10, "gpu_uuid": "GPU-one"}
    assignment = {"token": "token", "gpu_uuid": "GPU-one"}
    assert s.gpu_owner(None, ready, assignment, trainer_pid=10) == (None, None)
    since, expired = s.gpu_readiness_timeout(ready, None, None, 100)
    assert since == 100 and not expired
    assert s.gpu_readiness_timeout(ready, None, since, 400) == (100, True)
    assert s.gpu_owner({"uuid": "GPU-other", "pids": [999]}, ready, assignment, trainer_pid=10)[1]


@pytest.mark.parametrize('host', ['H20', 'L20'])
def test_only_h20_snapshots_include_allocator_and_sanitized_config(coordinator, monkeypatch, host):
    assignment = coordinator.assign('E5', host, 0)
    config = coordinator.config
    run, control = s.assignment_paths(config, assignment)
    (run / 'checkpoints').mkdir(parents=True)
    control.mkdir(parents=True)
    checkpoint = run / 'checkpoints/current.pt'
    checkpoint.write_bytes(b'checkpoint')
    entry = {'path': 'checkpoints/current.pt', 'sha256': s.digest(checkpoint), 'size_bytes': 10}
    s.atomic_json(run / 'checkpoint-manifest.json', {'current': entry})
    s.atomic_json(Path(config['root']) / 'control/coordinator.json', coordinator.state)
    config['private_key'] = 'excluded-root-secret'
    config['hosts']['H20']['ssh_options'] = ['excluded-transport-secret']
    config['control_repo'] = str(Path(config['root']) / 'control-repo')
    authorize_gpu4(config)
    usage = s.shutil.disk_usage
    monkeypatch.setattr(s.shutil, 'disk_usage', lambda p: usage(p)._replace(free=300 * s.GIB))
    result = Path(s.backup_snapshot(config, assignment, {'request_id': 'allocator'}))
    ledger = result / 'supervisor/coordinator.json'
    safe_config = result / 'supervisor/coordinator-config.json'
    assert ledger.exists() == safe_config.exists() == (host == 'H20')
    if host == 'H20':
        assert s.read_json(ledger)['assignments']['E5']['token'] == assignment['token']
        saved = s.read_json(safe_config)
        assert saved['manifest_sha256'] == config['manifest_sha256']
        assert saved['hosts']['H20']['gpus'] == config['hosts']['H20']['gpus']
        assert saved['extra_authorized_gpus'] == config['extra_authorized_gpus']
        assert saved['control_repo'] == config['control_repo']
        assert 'excluded-' not in safe_config.read_text()


def pre_initialization_failure(coordinator):
    assignment = coordinator.assign("E5", "H20", 0)
    run, control = s.assignment_paths(coordinator.config, assignment)
    (run / "checkpoints").mkdir(parents=True)
    control.mkdir(parents=True)
    (run / ".trainer.lock").touch()
    s.atomic_json(control / "worker.json", {"token": assignment["token"], "exit_code": 1,
        "worker_identity": {"pid": 999999999, "start_ticks": 1, "boot_id": "old"}})
    s.atomic_json(control / "status.json", {"attempt_id": assignment["token"], "phase": "failed",
        "global_step": 0, "durable_checkpoint": None, "cuda_context_ready": False})
    proof = {"passed": True, "health_check": True, "data_check": True, "state_check": True,
             "diagnosis": "indexed CUDA allocator initialization failed before optimizer updates"}
    return assignment, run, control, proof


def test_trainer_command_uses_explicit_cuda_index(coordinator):
    assignment = coordinator.assign("E5", "H20", 0)
    command = s.trainer_command(coordinator.config, assignment)
    assert command[command.index("--device") + 1] == "cuda:0"


def test_pre_initialization_retirement_preserves_and_replays_receipt(coordinator):
    assignment, run, control, proof = pre_initialization_failure(coordinator)
    receipt = s.retire_pre_initialization(coordinator.config, assignment, proof)
    assert not run.exists()
    assert Path(receipt["preserved_run"]).joinpath(".trainer.lock").exists()
    assert receipt["successful_updates"] == 0
    assert s.retire_pre_initialization(coordinator.config, assignment, proof) == receipt


@pytest.mark.parametrize("artifact", ["checkpoint-manifest.json", "last.pt", "checkpoints/step0.pt",
                                      "attempts/token/updates.jsonl", "attempts/token/identity.json",
                                      "updates.jsonl"])
def test_pre_initialization_rejects_any_training_artifact(coordinator, artifact):
    assignment, run, control, proof = pre_initialization_failure(coordinator)
    path = run / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    with pytest.raises(s.Refused, match="artifacts"):
        s.retire_pre_initialization(coordinator.config, assignment, proof)
    assert run.exists()
    assert not (control / "pre-initialization-retired.json").exists()


@pytest.mark.parametrize("change", [{"global_step": 1}, {"phase": "training"},
                                     {"durable_checkpoint": {"path": "missing.pt"}},
                                     {"cuda_context_ready": True}, {"attempt_id": "wrong"}])
def test_pre_initialization_rejects_unproven_zero_state(coordinator, change):
    assignment, run, control, proof = pre_initialization_failure(coordinator)
    status = s.read_json(control / "status.json")
    s.atomic_json(control / "status.json", dict(status, **change))
    with pytest.raises(s.Refused, match="before initialization"):
        s.retire_pre_initialization(coordinator.config, assignment, proof)


def test_pre_initialization_rejects_live_owner_and_held_lock(coordinator):
    assignment, run, control, proof = pre_initialization_failure(coordinator)
    worker = s.read_json(control / "worker.json")
    import os
    s.atomic_json(control / "worker.json", dict(worker, worker_identity=s.process_identity(os.getpid())))
    with pytest.raises(s.Refused, match="exited"):
        s.retire_pre_initialization(coordinator.config, assignment, proof)
    s.atomic_json(control / "worker.json", worker)
    with s.lock(s.lock_paths(coordinator.config, assignment)[0]):
        with pytest.raises(s.Refused, match="held"):
            s.retire_pre_initialization(coordinator.config, assignment, proof)


def test_pre_initialization_coordinator_retains_owner_on_timeout_then_new_token(coordinator):
    assignment, run, control, proof = pre_initialization_failure(coordinator)
    old_token = assignment["token"]
    original_call = coordinator.transport.call
    response_lost = [True]
    def call(host, request, timeout=30):
        if request["op"] == "retire_pre_initialization":
            receipt = s.retire_pre_initialization(coordinator.config, request["assignment"], request["evidence"])
            if response_lost[0]:
                response_lost[0] = False
                raise subprocess.TimeoutExpired("ssh", 30)
            return receipt
        return original_call(host, request, timeout)
    coordinator.transport.call = call
    with pytest.raises(subprocess.TimeoutExpired):
        coordinator.recover_pre_initialization("E5", proof)
    persisted = s.read_json(coordinator.path)
    assert persisted["assignments"]["E5"]["token"] == old_token
    assert persisted["assignments"]["E5"]["state"] == "pre_initialization_retiring"
    with pytest.raises(s.Refused, match="already reserved"):
        coordinator.assign("E5", "L20", 0)
    new = coordinator.recover_pre_initialization("E5", proof)
    assert new["token"] != old_token and new["resume"] is None
    assert coordinator.state["history"][0]["pre_initialization_retirement"]["retired"]
    with pytest.raises(s.Refused, match="once only"):
        coordinator.recover_pre_initialization("E5", proof)


def test_pre_initialization_replays_after_rename_before_receipt(coordinator, monkeypatch):
    assignment, run, control, proof = pre_initialization_failure(coordinator)
    publish = s.atomic_json
    def crash_before_receipt(path, value):
        if Path(path).name == "pre-initialization-retired.json":
            raise OSError("simulated crash before receipt publication")
        return publish(path, value)
    monkeypatch.setattr(s, "atomic_json", crash_before_receipt)
    with pytest.raises(OSError, match="before receipt"):
        s.retire_pre_initialization(coordinator.config, assignment, proof)
    assert not run.exists()
    assert (control / "failed-run-before-initialization/.trainer.lock").exists()
    monkeypatch.setattr(s, "atomic_json", publish)
    receipt = s.retire_pre_initialization(coordinator.config, assignment, proof)
    assert receipt["retired"] and receipt["token"] == assignment["token"]


def selected_v3(config, effective=32):
    path = Path(config['root']) / 'selected-H20-E5.json'
    raw = {'experiment': 'E5', 'training': {'budget_contract': s.CONTRACT_VERSION,
            'microbatch_size': effective, 'gradient_accumulation_steps': 1},
           'runtime_profile': {'profile_id': 'H20-E5-m32', 'training_version': 'v3',
            'selection_status': 'selected', 'host': 'H20', 'experiment': 'E5'},
           'runtime': {'cpu_threads': 4, 'gpu_memory_limit_gib': 110}}
    s.atomic_json(path, raw)
    config.update(contract_version=s.CONTRACT_VERSION, lock_root=str(Path(config['root']) / 'common-locks'),
                  profiles={'H20': {'E5': {'config_path': str(path), 'config_sha256': s.digest(path)}}})
    return path, raw


def test_selected_profile_binds_decreased_formal_budget_and_hardware(config, coordinator):
    selected_v3(config)
    assignment = coordinator.assign('E5', 'H20', 0)
    assert assignment['budget']['formal'] and assignment['budget']['target_updates'] == 7200
    assert assignment['budget']['prefix_step'] == 13
    assert assignment['cpus'] == [0, 1, 2, 3]
    command = s.trainer_command(config, assignment)
    assert command[command.index('--pause-at-step') + 1] == '13'
    assert command[command.index('--cpu-threads') + 1] == '4'
    assert command[command.index('--gpu-memory-limit-gib') + 1] == '110'
    with pytest.raises(s.Refused, match='missing selected'):
        coordinator.assign('E3', 'L20', 0)


def test_from_scratch_profile_passes_v4_sidecar_to_trainer(config, coordinator):
    from suction20.from_scratch import from_scratch_config
    path, _ = selected_v3(config, effective=4)
    source = Path(__file__).parents[1] / 'configs/runtime_profiles/cd009-v3/H20-E5.json'
    fresh, performance = from_scratch_config(json.loads(source.read_text()))
    s.atomic_json(path, fresh)
    sidecar = path.with_name('fresh-performance.json')
    s.atomic_json(sidecar, performance)
    config['profiles']['H20']['E5']['config_sha256'] = s.digest(path)
    config['performance_configs'] = {'H20': {'E5': {
        'config_path': str(sidecar), 'config_sha256': s.digest(sidecar)}}}
    assignment = coordinator.assign('E5', 'H20', 0)
    command = s.trainer_command(config, assignment)
    assert assignment['runtime_profile']['selection_status'] == 'from_scratch'
    assert assignment['budget']['target_updates'] == 57600
    assert command[command.index('--performance-config') + 1] == str(sidecar)
    assert '--init-checkpoint' not in command


def test_changed_profile_or_budget_rejected(config, coordinator):
    path, raw = selected_v3(config)
    assignment = coordinator.assign('E5', 'H20', 0)
    raw['runtime']['cpu_threads'] = 8
    s.atomic_json(path, raw)
    with pytest.raises(s.Refused, match='hash'):
        s.trainer_command(config, assignment)
    config['profiles']['H20']['E5']['config_sha256'] = s.digest(path)
    with pytest.raises(s.Refused, match='changed'):
        s.trainer_command(config, assignment)
    config.pop('contract_version')
    with pytest.raises(s.Refused, match='fall back'):
        s.trainer_command(config, assignment)


def test_selected_wrong_host_and_cpu_pool_rejected(config, coordinator):
    path, raw = selected_v3(config)
    raw['runtime_profile']['host'] = 'L20'
    s.atomic_json(path, raw)
    config['profiles']['H20']['E5']['config_sha256'] = s.digest(path)
    with pytest.raises(s.Refused, match='host/experiment'):
        coordinator.assign('E5', 'H20', 0)
    raw['runtime_profile']['host'] = 'H20'
    raw['runtime']['cpu_threads'] = 12
    s.atomic_json(path, raw)
    config['profiles']['H20']['E5']['config_sha256'] = s.digest(path)
    with pytest.raises(s.Refused, match='affinity pool'):
        coordinator.assign('E5', 'H20', 0)
    assert not coordinator.state['assignments']


def v3_checkpoint(assignment):
    budget = assignment['budget']
    nodes = [{'step': node, 'effective_samples': node * budget['effective_batch_size'],
              'frame_count': 320} for node in budget['validation_steps']]
    return {'global_step': budget['target_updates'], 'budget': budget, 'attempt_id': assignment['token'],
            'runtime_profile': assignment['runtime_profile'], 'contract_version': s.CONTRACT_VERSION,
            'effective_samples_completed': budget['target_samples'], 'completed_validation_nodes': nodes,
            'analysis': {'global_step': budget['target_updates'], 'attempt_id': assignment['token'], 'budget': budget,
                'identity': {'runtime_profile': assignment['runtime_profile'], 'budget_contract': s.CONTRACT_VERSION},
                'counters': {'effective_samples_completed': budget['target_samples']}, 'completed_validation_nodes': nodes}}


def migrated_release(coordinator):
    config = coordinator.config
    selected_v3(config, effective=4)
    sidecars = Path(__file__).parents[1] / 'configs/performance'
    p1, p2 = sidecars / 'cd009-p1.json', sidecars / 'cd009-p2.json'
    config['performance_configs'] = {'H20': {'E5': {
        'config_path': str(p1), 'config_sha256': s.digest(p1)}}}
    assignment = coordinator.assign('E5', 'H20', 0)
    checkpoint = v3_checkpoint(assignment)
    checkpoint.update(global_step=1030, effective_samples_completed=4120,
                      attempt_id='migration-p2', completed_validation_nodes=[])
    checkpoint['analysis'].update(global_step=1030, attempt_id='migration-p2',
        counters={'effective_samples_completed': 4120}, completed_validation_nodes=[])
    checkpoint['analysis']['identity']['performance'] = {'config_sha256': s.digest(p2)}
    config['performance_configs']['H20']['E5'] = {
        'config_path': str(p2), 'config_sha256': s.digest(p2)}
    view = {'state': 'exited', 'locks_free': True, 'checkpoint_verified': True,
            'worker': {'exit_code': 75}, 'committed_checkpoint': checkpoint,
            'performance_migration_verified': True, 'performance_source_backup_verified': True,
            'performance_migration_source_config_sha256': s.digest(p1)}
    assignment['view'] = copy.deepcopy(view)
    coordinator.transport.view = view
    return assignment, view


def test_resume_release_verifies_migrated_target_without_rewriting_old_assignment(coordinator):
    assignment, view = migrated_release(coordinator)
    original = copy.deepcopy(assignment)
    # Neither ordinary release nor diagnosed recovery acquires this exception.
    for command in ('release', 'recover'):
        with pytest.raises(s.Refused, match='performance sidecar identity mismatch'):
            coordinator.release('E5', {'command': command})
    released = coordinator.release('E5', {'command': 'resume'})
    assert 'E5' not in coordinator.state['assignments']
    assert released is coordinator.state['history'][-1]
    for key in ('performance_config_path', 'performance_config_sha256', 'performance_settings',
                'runtime_profile', 'budget', 'token'):
        assert released[key] == original[key]
    assert released['view'] == view
    assert released['view']['committed_checkpoint']['analysis']['identity']['performance']['config_sha256'] != released['performance_config_sha256']


@pytest.mark.parametrize('bad_proof', [
    {'performance_migration_verified': False},
    {'performance_source_backup_verified': False},
    {'performance_migration_source_config_sha256': 'wrong-source'},
    {'worker': {'exit_code': 1}},
])
def test_resume_release_rechecks_fresh_migration_proof(coordinator, bad_proof):
    assignment, view = migrated_release(coordinator)
    original = copy.deepcopy(assignment)
    # A previously reconciled valid view must not hide fresh contradictory proof.
    coordinator.transport.view = dict(view, **bad_proof)
    with pytest.raises(s.Refused, match='explicit migration|boundary pause'):
        coordinator.release('E5', {'command': 'resume'})
    assert coordinator.state['assignments']['E5'] == original
    assert coordinator.state['history'] == []


def test_completion_requires_exact_samples_and_full_nodes(config, coordinator):
    selected_v3(config)
    assignment = coordinator.assign('E5', 'H20', 0)
    checkpoint = v3_checkpoint(assignment)
    s.verify_checkpoint_contract(assignment, checkpoint, complete=True)
    checkpoint['effective_samples_completed'] -= 1
    with pytest.raises(s.Refused, match='sample count'):
        s.verify_checkpoint_contract(assignment, checkpoint, complete=True)
    checkpoint['effective_samples_completed'] += 1
    checkpoint['completed_validation_nodes'][-1]['frame_count'] = 319
    with pytest.raises(s.Refused, match='ten full'):
        s.verify_checkpoint_contract(assignment, checkpoint, complete=True)
    checkpoint['completed_validation_nodes'][-1]['frame_count'] = 320
    checkpoint['runtime_profile'] = dict(checkpoint['runtime_profile'], profile_id='wrong')
    with pytest.raises(s.Refused, match='identity'):
        s.verify_checkpoint_contract(assignment, checkpoint, complete=True)


def test_prefix_evidence_scoped_to_v3_assignment(config, coordinator):
    selected_v3(config)
    assignment = coordinator.assign('E5', 'H20', 0)
    evidence = {'attempt_id': assignment['token'], 'profile_sha256': assignment['profile_sha256'],
                'contract_version': s.CONTRACT_VERSION, 'paused_step': 13, 'resumed_step': 14,
                'identity_and_sampler_passed': True}
    s.check_evidence(assignment, 'prefix', evidence)
    with pytest.raises(s.Refused, match='boundary'):
        s.check_evidence(assignment, 'prefix', dict(evidence, paused_step=100))
    with pytest.raises(s.Refused, match='attempt'):
        s.check_evidence(assignment, 'prefix', dict(evidence, attempt_id='old'))


def test_retire_preserves_history_without_completion_or_relaunch(coordinator):
    coordinator.assign('E5', 'H20', 0)
    coordinator.transport.view = {'state': 'exited', 'locks_free': True, 'checkpoint_verified': True,
        'worker': {'exit_code': 75}, 'final_backup_verified': False}
    proof = {'passed': True, 'reason': 'approved sample-budget contract supersedes paused v2'}
    with pytest.raises(s.Refused, match='matching final backup'):
        coordinator.retire('E5', s.CONTRACT_VERSION, proof)
    assert 'E5' in coordinator.state['assignments']
    coordinator.transport.view['final_backup_verified'] = True
    retired = coordinator.retire('E5', s.CONTRACT_VERSION, proof)
    assert retired['state'] == 'superseded' and not retired['accepted_complete']
    assert retired is coordinator.state['history'][-1]
    with pytest.raises(s.Refused, match='superseded'):
        coordinator.assign('E5', 'H20', 0)


def test_shared_lock_root_prevents_cross_version_gpu_overlap(config, tmp_path):
    assignment = {'experiment': 'E5', 'gpu_uuid': 'GPU-test'}
    config['lock_root'] = str(tmp_path / 'common-locks')
    other = dict(config, root=str(tmp_path / 'v3'))
    assert s.lock_paths(config, assignment) == s.lock_paths(other, assignment)
    with s.lock(s.lock_paths(config, assignment)[0]):
        with pytest.raises(s.Refused):
            with s.lock(s.lock_paths(other, assignment)[0]):
                pass


def test_v3_cannot_reinterpret_existing_legacy_ledger(config, coordinator):
    coordinator.assign('E5', 'H20', 0)
    selected_v3(config)
    with pytest.raises(s.Refused, match='ledger contract differs'):
        s.Coordinator(config, coordinator.path, coordinator.transport)


def test_v3_complete_evidence_needs_committed_full_validation(config, coordinator):
    selected_v3(config)
    assignment = coordinator.assign('E5', 'H20', 0)
    evidence = {'attempt_id': assignment['token'], 'profile_sha256': assignment['profile_sha256'],
        'contract_version': s.CONTRACT_VERSION, 'global_step': 7200, 'validation_nodes': 10,
        'effective_samples_completed': 230400, 'hashes_passed': True, 'ledger_passed': True, 'backup_passed': True}
    with pytest.raises(s.Refused, match='checkpoint profile'):
        s.check_evidence(assignment, 'complete', evidence)
    assignment['view'] = {'committed_checkpoint': v3_checkpoint(assignment)}
    s.check_evidence(assignment, 'complete', evidence)
    with pytest.raises(s.Refused, match='sample count'):
        s.check_evidence(assignment, 'complete', dict(evidence, effective_samples_completed=230399))


def test_retirement_backup_checks_matching_bytes_and_attempt(config):
    assignment = {'token': 'old', 'host': 'H20', 'experiment': 'E5'}
    destination = Path(config['backup_root']) / 'old/final'
    destination.mkdir(parents=True)
    checkpoint_path = destination / 'weights.pt'
    checkpoint_path.write_bytes(b'committed model')
    checkpoint = {'path': 'weights.pt', 'sha256': s.digest(checkpoint_path)}
    request = {'attempt_id': 'old', 'final': True}
    s.atomic_json(destination / 'checkpoint-manifest.json', {'current': checkpoint})
    files = {'weights.pt': {'size_bytes': checkpoint_path.stat().st_size, 'sha256': checkpoint['sha256']}}
    for name in ('resolved_config.json', 'source_config_frozen.json', 'runtime.json', 'supervisor/assignment.json',
                 'supervisor/command.json', 'supervisor/status.json', 'supervisor/resources.jsonl',
                 'attempts/old/identity.json', 'attempts/old/updates.jsonl'):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
        files[name] = {'size_bytes': path.stat().st_size, 'sha256': s.digest(path)}
    s.atomic_json(destination / 'backup-complete.json', {'request': request, 'files': files})
    done = dict(request, destination=str(destination))
    assert s.final_backup_verified(config, assignment, checkpoint, done)
    assert not s.final_backup_verified(config, assignment, checkpoint, dict(done, attempt_id='other'))
    log = destination / 'attempts/old/updates.jsonl'
    log.write_text('bad')
    assert not s.final_backup_verified(config, assignment, checkpoint, done)
    log.unlink()
    assert not s.final_backup_verified(config, assignment, checkpoint, done)
    log.write_text('{}\n')
    assert s.final_backup_verified(config, assignment, checkpoint, done)
    checkpoint_path.write_bytes(b'corruption')
    assert not s.final_backup_verified(config, assignment, checkpoint, done)


def test_v3_requires_explicit_common_lock_root(config):
    selected_v3(config)
    s.validate_config(config)
    config.pop('lock_root')
    with pytest.raises(s.Refused, match='explicit common lock_root'):
        s.validate_config(config)


@pytest.mark.parametrize('field', ['global_step', 'budget', 'counters', 'identity', 'completed_validation_nodes'])
def test_v3_rejects_inline_analysis_disagreement(config, coordinator, field):
    selected_v3(config)
    assignment = coordinator.assign('E5', 'H20', 0)
    checkpoint = v3_checkpoint(assignment)
    checkpoint['analysis'][field] = None
    with pytest.raises(s.Refused, match='inline analysis'):
        s.verify_checkpoint_contract(assignment, checkpoint, complete=True)


def test_missing_profile_blocks_launch_without_stopping_coordinator(config, monkeypatch, tmp_path):
    selected_v3(config)
    config['profiles'].clear()
    path = tmp_path / 'supervisor.json'
    s.atomic_json(path, config)
    ledger = Path(config['root']) / 'control/coordinator.json'
    coordinator = s.Coordinator(config, ledger, FakeTransport())
    coordinator.state['evidence']['host:H20'] = {'passed': True}
    coordinator.save()
    class ProbeTransport:
        def call(self, host, request):
            assert request['op'] == 'probe'
            return {'gpus': {}, 'resources': {}}
    monkeypatch.setattr(s, 'Transport', lambda *args: ProbeTransport())
    monkeypatch.setattr(s, 'host_guard', lambda *args: None)
    monkeypatch.setattr(s, 'assert_idle', lambda *args: None)
    monkeypatch.setattr(s, 'launch_resources', lambda *args: None)
    s.main(['--config', str(path), 'tick'])
    state = s.read_json(ledger)
    assert 'missing selected profile' in state['launch_gate']
    assert not state['assignments']
    s.main(['--config', str(path), 'tick'])
    assert not s.read_json(ledger)['assignments']


def authorize_gpu4(config):
    config["hosts"]["L20"]["gpus"]["4"] = {"uuid": "GPU-L20-4", "cpus": list(range(32, 40))}
    config["extra_authorized_gpus"] = {"L20": {"4": "GPU-L20-4"}}


def test_gpu4_requires_explicit_uuid_binding_and_unique_resources(config, coordinator):
    authorize_gpu4(config)
    s.validate_config(config)
    assignment = coordinator.assign("E4", "L20", 4)
    assert assignment["gpu_index"] == 4 and assignment["gpu_uuid"] == "GPU-L20-4"
    assert assignment["cpus"] == list(range(32, 40))
    config["extra_authorized_gpus"]["L20"]["4"] = "GPU-wrong"
    with pytest.raises(s.Refused, match="UUID differs"):
        s.validate_config(config)
    config["extra_authorized_gpus"]["L20"]["4"] = "GPU-L20-4"
    config["hosts"]["L20"]["gpus"]["4"]["cpus"] = [0]
    with pytest.raises(s.Refused, match="overlaps"):
        s.validate_config(config)
    config["hosts"]["L20"]["gpus"]["4"]["cpus"] = list(range(32, 40))
    config["extra_authorized_gpus"]["L20"]["4"] = "GPU-H20-0"
    config["hosts"]["L20"]["gpus"]["4"]["uuid"] = "GPU-H20-0"
    with pytest.raises(s.Refused, match="more than once"):
        s.validate_config(config)


@pytest.mark.parametrize("extra", [None, [], {"H20": {"2": "GPU-extra"}},
                                   {"L20": []}, {"L20": {"5": "GPU-extra"}},
                                   {"L20": {"04": "GPU-extra"}}, {"L20": {4: "GPU-extra"}}])
def test_extra_gpu_schema_rejects_other_scopes(config, extra):
    config["extra_authorized_gpus"] = extra
    with pytest.raises(s.Refused, match="authorized"):
        s.validate_config(config)


def test_gpu4_authorization_requires_matching_registry_entry(config):
    config["extra_authorized_gpus"] = {"L20": {"4": "GPU-L20-4"}}
    with pytest.raises(s.Refused, match="authorized"):
        s.validate_config(config)


@pytest.mark.parametrize("path", [None, "relative/repo", "/absolute/../repo", 4])
def test_control_repository_requires_absolute_nontraversing_path(config, path):
    config["control_repo"] = path
    with pytest.raises(s.Refused, match="absolute path"):
        s.validate_config(config)


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("local", [False, True])
def test_control_repository_routes_rpc_only(config, tmp_path, monkeypatch, override, local):
    expected = Path(config["root"]) / ("control repo" if override else "repo")
    if override:
        config["control_repo"] = str(expected)
    s.validate_config(config)
    monkeypatch.setattr(s, "is_local_host", lambda *args: local)
    captured = {}
    def run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, stdout='{"ok":true}')
    monkeypatch.setattr(s.subprocess, "run", run)
    path = tmp_path / "config with spaces.json"
    assert s.Transport(config, path).call("L20", {"op": "probe"}) == {"ok": True}
    command = captured["command"] if local else shlex.split(captured["command"][-1])
    assert command == [config["python"], str(expected / "scripts/supervise_pretrain.py"),
                       "--config", str(path), "rpc", "--host", "L20"]
    assert json.loads(captured["kwargs"]["input"]) == {"op": "probe"}


@pytest.mark.parametrize("override", [False, True])
def test_control_repository_routes_worker_but_keeps_training_cwd(config, coordinator, tmp_path, monkeypatch, override):
    authorize_gpu4(config)
    expected = Path(config["root"]) / ("control-repo" if override else "repo")
    if override:
        config["control_repo"] = str(expected)
    s.validate_config(config)
    assignment = coordinator.assign("E4", "L20", 4)
    captured = {}
    def spawn(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return type("Worker", (), {"pid": 999999999})()
    monkeypatch.setattr(s.subprocess, "Popen", spawn)
    path = tmp_path / "config.json"
    result = s.launch_worker(config, path, "L20", assignment)
    assert result["state"] == "dispatched"
    assert captured["command"][:2] == [config["python"], str(expected / "scripts/supervise_pretrain.py")]
    assert captured["command"][4:7] == ["worker", "--host", "L20"]
    assert captured["kwargs"]["cwd"] == Path(config["root"]) / "repo"
    assert captured["kwargs"]["start_new_session"] is True
    assert s.launch_worker(config, path, "L20", assignment)["state"] == "already_dispatched"


@pytest.mark.parametrize("resume", [None, "last.pt"])
def test_control_override_preserves_exact_selected_trainer_command(config, coordinator, resume):
    selected_v3(config)
    assignment = coordinator.assign("E5", "H20", 0, resume=resume, prefix_passed=bool(resume))
    before = s.trainer_command(config, assignment)
    config["control_repo"] = str(Path(config["root"]) / "control-repo")
    authorize_gpu4(config)
    s.validate_config(config)
    assert s.trainer_command(config, assignment) == before
    assert before[1] == str(Path(config["root"]) / "repo/scripts/train_pretrain.py")
