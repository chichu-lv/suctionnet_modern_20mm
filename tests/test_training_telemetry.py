"""Synthetic /proc and NVML fixtures; no GPU, Torch, network or training."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from suction20.training_telemetry import ResourceSampler


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def proc_process(root, *, start=123, user=100, system=50, read_bytes=1000):
    fields = ["S"] + ["0"] * 49
    fields[11], fields[12], fields[19] = str(user), str(system), str(start)
    write(root, "42/stat", "42 (trainer (worker)) " + " ".join(fields))
    write(root, "42/status", "VmRSS:\t2048 kB\nCpus_allowed_list:\t0-7\n")
    write(root, "42/io", f"read_bytes: {read_bytes}\nwrite_bytes: 200\nrchar: 300\nwchar: 400\n")


@pytest.fixture
def proc(tmp_path):
    proc_process(tmp_path)
    write(tmp_path, "sys/kernel/random/boot_id", "test-boot\n")
    write(tmp_path, "meminfo", "MemAvailable: 1000 kB\n")
    write(tmp_path, "loadavg", "1.0 2.0 3.0 1/4 42\n")
    for kind in ("cpu", "memory", "io"):
        write(tmp_path, "pressure/" + kind, "some avg10=1.0 avg60=2.0 avg300=3.0 total=100\n")
    return tmp_path


def nvml(command, **kwargs):
    assert kwargs["timeout"] == 5
    assert command[1:3] == ["-i", "GPU-test"]
    if "--query-gpu=" in command[3]:
        return SimpleNamespace(stdout="GPU-test, 20000, 46068, 80, 42, 220.5, 65, 1500, 9000\n")
    return SimpleNamespace(stdout="GPU-test, 9001, 19000\nGPU-test, 9002, 1000\n")


def test_sample_has_explicit_gpu_scope_and_cpu_delta(proc):
    now = [100.0]
    sampler = ResourceSampler(proc_root=proc, command=nvml, clock=lambda: now[0], wall_clock=lambda: 1000.)
    first = sampler.sample(pid=42, gpu_uuid="GPU-test", nvml_pid=9001, phase="training", attempt_id="a")
    assert first["gpu"]["process_memory_mib"] == 19000
    assert first["gpu"]["memory_used_mib"] == 20000
    assert first["gpu"]["utilization_percent"] == 80
    assert first["gpu"]["power_watts"] == 220.5
    assert len(first["gpu"]["compute_processes"]) == 2
    assert first["process"]["rss_bytes"] == 2048 * 1024
    assert first["process"]["cpu_core_equivalents"] is None
    assert first["process"]["affinity"] == "0-7"
    assert first["utc"].endswith("+00:00") and first["boot_id"] == "test-boot"
    now[0] += 2
    proc_process(proc, user=100 + 3 * sampler.ticks, read_bytes=3000)
    second = sampler.sample(pid=42, gpu_uuid="GPU-test", nvml_pid=9001)
    assert second["process"]["cpu_core_equivalents"] == 1.5
    assert second["process"]["cpu_window_seconds"] == 2
    assert second["process"]["io_bytes_per_second"]["read_bytes"] == 1000
    json.dumps(second, allow_nan=False)


def test_pid_reuse_resets_cpu_and_io_deltas(proc):
    now = [10.]
    sampler = ResourceSampler(proc_root=proc, command=nvml, clock=lambda: now[0])
    sampler.sample(pid=42)
    proc_process(proc, start=999, user=10000)
    now[0] += 2
    result = sampler.sample(pid=42)
    assert result["process"]["cpu_core_equivalents"] is None
    assert result["process"]["io_bytes_per_second"] is None
    assert "changed process" in result["missing_reasons"]["process.cpu_core_equivalents"]


@pytest.mark.parametrize("failure", [FileNotFoundError("no smi"), subprocess.TimeoutExpired("smi", 5)])
def test_missing_telemetry_is_null_with_reasons(tmp_path, failure):
    def unavailable(*args, **kwargs):
        raise failure
    result = ResourceSampler(proc_root=tmp_path, command=unavailable).sample(pid=42, gpu_uuid="GPU-test")
    assert result["process"]["rss_bytes"] is None
    assert result["gpu"]["memory_used_mib"] is None
    assert result["gpu"]["process_memory_mib"] is None
    assert result["host_resources"]["available_bytes"] is None
    assert "process.rss_bytes" in result["missing_reasons"]
    assert "gpu.query" in result["missing_reasons"]
    json.dumps(result, allow_nan=False)


def test_unsupported_values_preserved_and_no_pid_guess(proc):
    def unsupported(command, **kwargs):
        if "--query-gpu=" in command[3]:
            return SimpleNamespace(stdout="GPU-test, 20, 40, [N/A], 0, N/A, nan, 1, 2\n")
        return SimpleNamespace(stdout="GPU-test, 42, 12\n")
    result = ResourceSampler(proc_root=proc, command=unsupported).sample(pid=42, gpu_uuid="GPU-test")
    assert result["gpu"]["memory_utilization_percent"] == 0
    for key in ("utilization_percent", "power_watts", "temperature_c", "process_memory_mib"):
        assert result["gpu"][key] is None
        assert "gpu." + key in result["missing_reasons"]
    json.dumps(result, allow_nan=False)


def test_wrong_gpu_identity_is_not_attributed(proc):
    def wrong(*args, **kwargs):
        return SimpleNamespace(stdout="GPU-other, 2, 4, 8, 8, 8, 8, 8, 8\n")
    result = ResourceSampler(proc_root=proc, command=wrong).sample(pid=42, gpu_uuid="GPU-test", nvml_pid=42)
    assert result["gpu"]["memory_used_mib"] is None
    assert "UUID/row mismatch" in result["missing_reasons"]["gpu.query"]


def test_malformed_proc_counters_do_not_break_sampling(proc):
    write(proc, "42/stat", "malformed")
    write(proc, "42/io", "read_bytes: NaN\n")
    write(proc, "pressure/io", "full avg10=nan\n")
    result = ResourceSampler(proc_root=proc, command=nvml).sample(pid=42)
    assert result["process"]["cpu_core_equivalents"] is None
    assert result["process"]["io"] is None
    assert result["host_resources"]["psi"]["io"] is None
    json.dumps(result, allow_nan=False)
