"""Bounded CD-009 supervision. Linux/stdlib only; no training or CUDA imports.

The H20 ledger is the authority. A timeout never releases an assignment. Host
workers keep local locks inherited by the trainer, including if the worker dies.
No lease expiry, automatic failover, preemption, or scientific configuration edits.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

from suction20.pretrain_budget import CONTRACT_VERSION, resolve_budget, resolve_config

GIB = 1024 ** 3
EXPERIMENTS = tuple(f"E{i}" for i in range(6))
INITIAL = (("E5", "H20", 0), ("E3", "L20", 0), ("E0", "L20", 1),
           ("E1", "L20", 2), ("E4", "L20", 3))


class Refused(RuntimeError):
    """A safety or operational gate needs evidence before proceeding."""


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def machine_id():
    return Path("/etc/machine-id").read_text().strip()


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused(f"lock held: {path}") from exc
        yield stream.fileno()


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_alive(identity):
    return bool(identity) and process_identity(identity["pid"]) == identity


def signal_owned(identity, sig):
    """pidfd closes PID reuse race; never signal a process group or unknown PID."""
    if not identity_alive(identity):
        return False
    fd = os.pidfd_open(identity["pid"])
    try:
        if not identity_alive(identity):
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    finally:
        os.close(fd)


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                             "--format=csv,noheader,nounits"], check=True,
                            capture_output=True, text=True, timeout=20)
    devices = {}
    for line in result.stdout.strip().splitlines():
        index, gpu, used = [v.strip() for v in line.split(",")]
        devices[int(index)] = {"uuid": gpu, "memory_mib": int(used), "pids": [], "process_memory_mib": {}}
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                             "--format=csv,noheader,nounits"], check=True,
                            capture_output=True, text=True, timeout=20)
    for line in result.stdout.strip().splitlines():
        gpu, pid, memory = [v.strip() for v in line.split(",")]
        for device in devices.values():
            if device["uuid"] == gpu:
                device["pids"].append(int(pid))
                device["process_memory_mib"][str(pid)] = int(memory) if memory.isdigit() else None
    return devices


def assert_idle(snapshot, index, expected_uuid):
    device = snapshot[index]
    if device["uuid"] != expected_uuid:
        raise Refused("GPU index/UUID identity changed")
    if device["pids"] or device["memory_mib"] >= 1024:
        raise Refused("GPU busy (including invisible PID memory usage)")


def tree_bytes(root):
    # Count hardlinked checkpoint aliases only once.
    total, seen = 0, set()
    for base, _, files in os.walk(root):
        for name in files:
            try:
                stat = (Path(base) / name).stat()
            except FileNotFoundError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                total += stat.st_size
                seen.add(identity)
    return total


def resources(root, backup_root, pid=None):
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key == "MemAvailable":
            info["available_bytes"] = int(value.split()[0]) * 1024
    info.update(local_free=shutil.disk_usage(root).free,
                backup_free=shutil.disk_usage(backup_root).free,
                local_bytes=tree_bytes(root),
                backup_bytes=tree_bytes(backup_root), rss_bytes=0)
    if pid:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    info["rss_bytes"] = int(line.split()[1]) * 1024
        except FileNotFoundError:
            pass
    info["psi"] = {}
    for kind in ("cpu", "memory", "io"):
        path = Path("/proc/pressure") / kind
        if path.exists():
            info["psi"][kind] = {line.split()[0]: {
                key: float(value) for key, value in
                (item.split("=") for item in line.split()[1:])}
                for line in path.read_text().splitlines()}
    return info


def launch_resources(sample, concurrency):
    if sample["available_bytes"] < 128 * GIB * concurrency:
        raise Refused("insufficient available RAM for target host concurrency")
    if min(sample["local_free"], sample["backup_free"]) < 200 * GIB:
        raise Refused("launch requires 200 GiB free local and shared space")
    if max(sample["local_bytes"], sample["backup_bytes"]) >= 100 * GIB:
        raise Refused("CD-009 disk budget exhausted")


def pause_reasons(sample, status, now, last_backup, started):
    reasons = []
    if sample["rss_bytes"] >= 64 * GIB:
        reasons.append("RSS stop limit 64 GiB")
    if min(sample["local_free"], sample["backup_free"]) < 100 * GIB:
        reasons.append("disk free below 100 GiB")
    if max(sample["local_bytes"], sample["backup_bytes"]) >= 100 * GIB:
        reasons.append("CD-009 disk usage reached 100 GiB")
    if now - (last_backup or started) > 7200:
        reasons.append("no successful backup for two hours")
    if status.get("phase") not in ("waiting_validation", "paused", "complete", "completed"):
        if now - status.get("progress_unix", started) > 2700:
            reasons.append("no compute progress for 45 minutes")
    return reasons


class DataPressure:
    """Compare a run with its own first 20 observed updates, never another GPU."""
    def __init__(self):
        self.baseline = []
        self.recent = []
        self.seen = None
        self.since = None

    def observe(self, status, now):
        row = status.get("last_update")
        marker = status.get("progress_unix")
        if not row or marker == self.seen or status.get("phase") != "training":
            return bool(self.since and now - self.since >= 600)
        self.seen = marker
        data, elapsed = row["data_seconds"], row["elapsed_seconds"]
        if len(self.baseline) < 20:
            self.baseline.append(data)
            return False
        self.recent.append((data, elapsed))
        self.recent = self.recent[-20:]
        mean_data = statistics.mean(v[0] for v in self.recent)
        mean_elapsed = statistics.mean(v[1] for v in self.recent)
        bad = mean_data > 2 * statistics.mean(self.baseline) and mean_data > .3 * mean_elapsed
        self.since = (self.since or now) if bad else None
        return bool(self.since and now - self.since >= 600)


def gpu_owner(device, status, assignment, established_pid=None, *, trainer_alive=True, trainer_pid=None):
    """Map host NVML PID only after authenticated local CUDA-context readiness.

    The trainer's synchronized live context on this UUID must appear in NVML.
    Exactly one PID then identifies it; multiple PIDs are ambiguous, never killed.
    Once established, other users arriving later do not trigger yielding.
    """
    if established_pid is not None:
        return established_pid, None
    if not trainer_alive or not status.get("cuda_context_ready"):
        return None, None
    if (status.get("attempt_id") != assignment["token"] or status.get("gpu_uuid") != assignment["gpu_uuid"]
            or status.get("trainer_pid") != trainer_pid):
        return None, "CUDA readiness identity does not match assignment"
    if device is None:
        return None, None
    if device.get("uuid") != assignment["gpu_uuid"]:
        return None, "NVML snapshot UUID differs from assignment"
    if len(device["pids"]) > 1:
        return None, "multiple NVML PIDs at CUDA readiness; launch ownership ambiguous"
    if len(device["pids"]) == 1:
        return device["pids"][0], None
    return None, None


def gpu_readiness_timeout(status, established_pid, ready_since, now):
    if status.get("cuda_context_ready"):
        ready_since = ready_since or now
    timed_out = ready_since is not None and established_pid is None and now - ready_since >= 300
    return ready_since, timed_out


def pin_file(source, pin_root):
    """Pin the inode before measuring size; atomic replacement cannot truncate JSON."""
    target = pin_root / uuid.uuid4().hex
    os.link(source, target)
    return target, target.stat().st_size


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_config(config):
    if set(config["hosts"]) != {"H20", "L20"}:
        raise Refused("only H20 and L20 are authorized")
    for host, maximum in (("H20", 2), ("L20", 4)):
        entry = config["hosts"][host]
        indexes = [int(index) for index in entry["gpus"]]
        if sorted(indexes) != list(range(maximum)):
            raise Refused(f"{host} requires exactly authorized indices 0..{maximum - 1}")
        cpus = []
        for gpu in entry["gpus"].values():
            if not gpu["uuid"].startswith("GPU-") or "REPLACE" in gpu["uuid"]:
                raise Refused("fill observed authorized GPU UUIDs before running")
            if not gpu["cpus"] or any(type(cpu) is not int or cpu < 0 for cpu in gpu["cpus"]):
                raise Refused("each device requires a nonempty CPU affinity pool")
            cpus.extend(gpu["cpus"])
        if len(cpus) != len(set(cpus)):
            raise Refused("CPU affinity overlaps on a host")
    if config.get("contract_version") not in (None, "suction20.pretrain.updates.legacy", CONTRACT_VERSION):
        raise Refused("unsupported supervisor contract")
    if config.get("contract_version") == CONTRACT_VERSION and not config.get("lock_root"):
        raise Refused("v3 requires explicit common lock_root shared with prior versions")
    root = Path(config["root"])
    backup = Path(config["backup_root"])
    for source in (config["source_root"], config["label_root"]):
        source = Path(source)
        lock_root = Path(config.get("lock_root", root / "control/locks"))
        if not lock_root.is_absolute():
            raise Refused("lock_root must be absolute")
        if any(path == source or source in path.parents for path in (root, backup, lock_root)):
            raise Refused("runtime/backup directory must be outside read-only data")


def is_local_host(config, host):
    identity = config["hosts"][host]
    return machine_id() == identity["machine_id"] and socket.gethostname() == identity["hostname"]


def host_guard(config, host):
    # Container images may clone /etc/machine-id; hostname is also mandatory.
    if not is_local_host(config, host):
        raise Refused(f"expected {host} machine identity and hostname")


def assignment_paths(config, assignment):
    root = Path(config["root"])
    run = root / "runs" / (assignment["experiment"] + "_seed42")
    control = root / "control" / "attempts" / assignment["token"]
    return run, control


def assignment_budget(assignment):
    return assignment.get("budget") or resolve_budget({"training": {"max_optimizer_steps": 57600}})


def selected_profile(config, experiment, host):
    """Selected configs are JSON for stdlib-only host verification; JSON is valid YAML."""
    if config.get("contract_version") != CONTRACT_VERSION:
        return {}
    selection = config.get("profiles", {}).get(host, {}).get(experiment)
    if not selection:
        raise Refused(f"missing selected profile for {host}/{experiment}")
    path = Path(selection["config_path"])
    payload = path.read_bytes()
    if not path.is_absolute() or hashlib.sha256(payload).hexdigest() != selection.get("config_sha256"):
        raise Refused("selected profile config hash/path mismatch")
    raw = json.loads(payload)
    if raw.get("experiment") != experiment or raw.get("seed", 42) != 42:
        raise Refused("selected config experiment/seed mismatch")
    resolved = resolve_config(raw)
    budget = resolve_budget(resolved)
    profile = resolved.get("runtime_profile", {})
    if budget["contract_version"] != CONTRACT_VERSION or not budget["formal"]:
        raise Refused("selected profile requires formal v3 sample budget")
    if (profile.get("selection_status") != "selected" or profile.get("training_version") != "v3" or
            profile.get("profile_id") in (None, "", "pending") or
            profile.get("host") != host or profile.get("experiment") != experiment):
        raise Refused("selected profile host/experiment/version mismatch")
    runtime = resolved.get("runtime", {})
    threads, memory = runtime.get("cpu_threads"), runtime.get("gpu_memory_limit_gib")
    if type(threads) is not int or threads < 1 or isinstance(memory, bool) or not isinstance(memory, (int, float)) or not 0 < memory < 1000:
        raise Refused("selected profile must specify positive CPU threads and GPU memory limit")
    return {"config_path": str(path), "config_sha256": selection["config_sha256"],
            "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "runtime_profile": profile, "budget": budget,
            "runtime": {"cpu_threads": threads, "gpu_memory_limit_gib": memory}}


def verify_assignment_profile(config, assignment):
    selected = selected_profile(config, assignment["experiment"], assignment["host"])
    if any(assignment.get(key) != value for key, value in selected.items()):
        raise Refused("assignment profile/budget changed; fresh version required")
    if assignment.get("budget", {}).get("contract_version") == CONTRACT_VERSION and not selected:
        raise Refused("v3 assignment cannot fall back to legacy configuration")
    return selected


def verify_checkpoint_contract(assignment, checkpoint, *, complete=False):
    budget = assignment_budget(assignment)
    if budget["contract_version"] != CONTRACT_VERSION:
        return
    if (checkpoint.get("budget") != budget or checkpoint.get("runtime_profile") != assignment["runtime_profile"] or
            checkpoint.get("contract_version") != CONTRACT_VERSION):
        raise Refused("checkpoint profile/budget identity mismatch")
    step = checkpoint.get("global_step")
    if type(step) is not int or step < 0 or step > budget["target_updates"] or checkpoint.get("effective_samples_completed") != step * budget["effective_batch_size"]:
        raise Refused("checkpoint effective sample count mismatch")
    analysis = checkpoint.get("analysis") or {}
    if (analysis.get("global_step") != step or analysis.get("attempt_id") != checkpoint.get("attempt_id") or
            analysis.get("budget") != budget or (analysis.get("identity") or {}).get("runtime_profile") != assignment["runtime_profile"] or
            (analysis.get("identity") or {}).get("budget_contract") != CONTRACT_VERSION or
            (analysis.get("counters") or {}).get("effective_samples_completed") != checkpoint["effective_samples_completed"] or
            analysis.get("completed_validation_nodes") != checkpoint.get("completed_validation_nodes") or
            analysis.get("last_completed_validation") != checkpoint.get("last_completed_validation")):
        raise Refused("checkpoint inline analysis disagrees with committed identity/counters")
    if complete:
        expected = [{"step": node, "effective_samples": node * budget["effective_batch_size"], "frame_count": 320}
                    for node in budget["validation_steps"]]
        if step != budget["target_updates"] or checkpoint["effective_samples_completed"] != budget["target_samples"] or checkpoint.get("completed_validation_nodes") != expected:
            raise Refused("completion requires exact sample budget and ten full validation nodes")


def trainer_command(config, assignment):
    verify_assignment_profile(config, assignment)
    root = Path(config["root"])
    run, control = assignment_paths(config, assignment)
    runtime = assignment.get("runtime") or {"cpu_threads": 8, "gpu_memory_limit_gib": 64 if assignment["host"] == "H20" else 40}
    config_path = assignment.get("config_path", str(root / "repo/configs" / ("pretrain_" + assignment["experiment"].lower() + ".yaml")))
    command = [config["python"], str(root / "repo/scripts/train_pretrain.py"),
               "--config", config_path,
               "--run-dir", str(run), "--device", "cuda:0", "--source-root", config["source_root"],
               "--label-root", config["label_root"], "--backbone-root", config["backbone_root"],
               "--control-dir", str(control), "--attempt-id", assignment["token"],
               "--validation-gate", "--cpu-threads", str(runtime["cpu_threads"]), "--gpu-memory-limit-gib",
               str(runtime["gpu_memory_limit_gib"])]
    if assignment.get("resume"):
        command.extend(["--resume", str(run / assignment["resume"])])
    if not assignment.get("prefix_passed"):
        command.extend(["--pause-at-step", str(assignment_budget(assignment)["prefix_step"])])
    return command


class Transport:
    """One JSON request per SSH command; timeout means unknown, never dead."""
    def __init__(self, config, config_path):
        self.config, self.config_path = config, str(Path(config_path).resolve())

    def call(self, host, request, timeout=30):
        config = self.config
        command = [config["python"], str(Path(config["root"]) / "repo/scripts/supervise_pretrain.py"),
                   "--config", self.config_path, "rpc", "--host", host]
        if not is_local_host(config, host):
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                       *config["hosts"][host].get("ssh_options", []),
                       config["hosts"][host]["ssh"], shlex.join(command)]
        result = subprocess.run(command, input=json.dumps(request), capture_output=True,
                                text=True, check=True, timeout=timeout)
        return json.loads(result.stdout)


class Coordinator:
    def __init__(self, config, ledger_path, transport):
        self.config, self.path, self.transport = config, Path(ledger_path), transport
        version = config.get("contract_version", "suction20.pretrain.updates.legacy")
        self.state = read_json(self.path, {"version": 1, "contract_version": version, "assignments": {}, "history": [],
                                          "validation": None, "backup": None, "evidence": {}})
        if self.state.get("contract_version", "suction20.pretrain.updates.legacy") != version:
            raise Refused("coordinator ledger contract differs; use a new version namespace")

    def save(self):
        atomic_json(self.path, self.state)

    def assign(self, experiment, host, index, *, resume=None, prefix_passed=False):
        if experiment not in EXPERIMENTS or experiment in self.state["assignments"]:
            raise Refused("experiment already reserved or invalid")
        if any(a["experiment"] == experiment and a.get("state") == "superseded" for a in self.state["history"]):
            raise Refused("superseded experiment requires a separate version namespace")
        selected = selected_profile(self.config, experiment, host)
        gpu = self.config["hosts"][host]["gpus"][str(index)]
        threads = selected.get("runtime", {}).get("cpu_threads", len(gpu["cpus"]))
        if threads > len(gpu["cpus"]):
            raise Refused("profile CPU threads exceed device affinity pool")
        for other in self.state["assignments"].values():
            if other["host"] == host and other["gpu_uuid"] == gpu["uuid"]:
                raise Refused("device already reserved (unknown owners also reserve it)")
        assignment = {"experiment": experiment, "host": host, "gpu_index": index,
                      "gpu_uuid": gpu["uuid"], "cpus": gpu["cpus"][:threads], "token": uuid.uuid4().hex,
                      "created_unix": time.time(), "state": "launch_unknown", "resume": resume,
                      "prefix_passed": prefix_passed}
        assignment.update(selected)
        self.state["assignments"][experiment] = assignment
        self.save()  # The only permissible ordering is reserve BEFORE transport.
        try:
            result = self.transport.call(host, {"op": "launch", "assignment": assignment})
            assignment["launch_result"] = result
            assignment["state"] = "launched"
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            assignment["transport_error"] = str(exc)
        self.save()
        return assignment

    def reconcile(self):
        for assignment in self.state["assignments"].values():
            try:
                view = self.transport.call(assignment["host"], {"op": "inspect", "assignment": assignment})
                assignment["view"] = view
                assignment["observed_unix"] = time.time()
                assignment["state"] = view["state"]
                assignment.pop("transport_error", None)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                assignment["state"] = "unknown"
                assignment["transport_error"] = str(exc)
        self.save()

    def release(self, experiment, evidence):
        assignment = self.state["assignments"][experiment]
        view = self.transport.call(assignment["host"], {"op": "inspect", "assignment": assignment})
        if view["state"] != "exited" or not view.get("locks_free"):
            raise Refused("old host must prove worker/trainer exit and local locks free")
        if not view.get("checkpoint_verified"):
            raise Refused("verify published checkpoint before releasing owner")
        verify_checkpoint_contract(assignment, view.get("committed_checkpoint") or {})
        if any(self.state[k] and self.state[k]["token"] == assignment["token"]
               for k in ("validation", "backup")):
            raise Refused("reconcile outstanding validation/backup before release")
        assignment.update(state="released", release_evidence=evidence, view=view)
        self.state["history"].append(assignment)
        del self.state["assignments"][experiment]
        self.save()
        return assignment

    def retire(self, experiment, superseded_by, evidence):
        if not superseded_by or superseded_by == self.config.get("contract_version") or not evidence.get("passed") or not evidence.get("reason"):
            raise Refused("retirement requires a new contract and explicit reviewed reason")
        assignment = self.state["assignments"][experiment]
        view = self.transport.call(assignment["host"], {"op": "inspect", "assignment": assignment})
        if view.get("state") != "exited" or not view.get("locks_free") or view.get("worker", {}).get("exit_code") != 75:
            raise Refused("retirement requires positively exited boundary pause and free locks")
        if not view.get("checkpoint_verified") or not view.get("final_backup_verified"):
            raise Refused("retirement requires verified checkpoint and matching final backup")
        verify_checkpoint_contract(assignment, view.get("committed_checkpoint") or {})
        if any(self.state[key] and self.state[key]["token"] == assignment["token"] for key in ("validation", "backup")):
            raise Refused("reconcile outstanding grants before retirement")
        assignment.update(state="superseded", superseded_by=superseded_by, accepted_complete=False, view=view,
                          release_evidence={"command": "retire", "unix": time.time(), "evidence": evidence})
        self.state["history"].append(assignment)
        del self.state["assignments"][experiment]
        self.save()  # One atomic state transition; never expose an unretired release.
        return assignment

    def recover_pre_initialization(self, experiment, proof):
        """Explicit one-time retry; remote proof/preservation precedes new ownership."""
        if not all(proof.get(key) for key in ("passed", "health_check", "data_check", "state_check", "diagnosis")):
            raise Refused("pre-initialization recovery requires diagnosed health/data/state evidence")
        old = self.state["assignments"][experiment]
        if old.get("resume") or old.get("prefix_passed"):
            raise Refused("pre-initialization retry cannot reset a resumed experiment")
        if any(a["experiment"] == experiment for a in self.state["history"]):
            raise Refused("pre-initialization retry is once only and excludes previously trained runs")
        if any(self.state[key] and self.state[key]["token"] == old["token"] for key in ("validation", "backup")):
            raise Refused("outstanding work grant prevents pre-initialization retirement")
        old["pre_initialization_recovery_evidence"] = proof
        old["state"] = "pre_initialization_retiring"
        self.save()  # A transport timeout retains this owner; replay is idempotent.
        receipt = self.transport.call(old["host"], {"op": "retire_pre_initialization", "assignment": old,
                                                   "evidence": proof})
        if receipt.get("token") != old["token"] or not receipt.get("retired"):
            raise Refused("host did not prove pre-initialization retirement")
        old.update(state="released", pre_initialization_retirement=receipt,
                   release_evidence={"command": "recover --pre-initialization", "unix": time.time()})
        self.state["history"].append(old)
        del self.state["assignments"][experiment]
        self.save()
        return self.assign(experiment, old["host"], old["gpu_index"])

    def grants(self):
        # Granted work remains reserved across coordinator restart/network loss.
        for kind in ("validation", "backup"):
            current = self.state[kind]
            if current:
                assignment = next((a for a in self.state["assignments"].values()
                                   if a["token"] == current["token"]), None)
                if assignment is None or assignment["state"] == "unknown":
                    continue
                view = assignment.get("view", {})
                done = view.get(kind + "_done") or {}
                if done.get("request_id") == current["request_id"]:
                    self.state[kind] = None
                    self.save()
                elif view.get("state") == "exited" and view.get("locks_free"):
                    self.state[kind] = None
                    self.save()
                else:
                    # Idempotent delivery repairs crash after durable reservation.
                    self.deliver(assignment, kind, current)
                    continue
            for assignment in self.state["assignments"].values():
                if assignment["state"] != "running":
                    continue
                request = assignment.get("view", {}).get(kind + "_request")
                done = assignment.get("view", {}).get(kind + "_done") or {}
                if not request or request["request_id"] == done.get("request_id"):
                    continue
                if request.get("attempt_id") != assignment["token"]:
                    raise Refused("request attempt does not match assigned owner")
                grant = dict(request, token=assignment["token"])
                self.state[kind] = grant
                self.save()
                self.deliver(assignment, kind, grant)
                break

    def deliver(self, assignment, kind, grant):
        try:
            self.transport.call(assignment["host"], {"op": "grant", "assignment": assignment,
                                                     "kind": kind, "grant": grant})
        except (OSError, subprocess.SubprocessError, ValueError):
            pass  # Persisted grant remains owned; never grant another on timeout.

    def next_launch(self):
        evidence = self.state["evidence"]
        assigned = self.state["assignments"]
        for owner in assigned.values():
            if owner.get("state") in ("unknown", "launch_unknown"):
                return None  # No new assignments during unresolved network ownership.
            worker = owner.get("view", {}).get("worker", {})
            if worker.get("data_pressure") or worker.get("pause_reasons"):
                return None
            sample = worker.get("resources", {})
            if any(sample.get("psi", {}).get(kind, {}).get("full", {}).get("avg60", 0) > 10
                   for kind in ("memory", "io")):
                return None
        finished = {a["experiment"] for a in self.state["history"] if a.get("accepted_complete") or a.get("state") == "superseded"}
        for experiment, host, index in INITIAL:
            if experiment in assigned or experiment in finished:
                continue
            if not evidence.get("host:" + host):
                continue
            previous = {"E0": "E3", "E1": "E0", "E4": "E1"}.get(experiment)
            if previous and not evidence.get("stable:" + previous):
                continue
            return experiment, host, index
        if "E2" not in assigned and "E2" not in finished:
            occupied = {(a["host"], a["gpu_index"]) for a in assigned.values()}
            # Prefer L20 when simultaneously free. GPU1 can join independently.
            for host, index in (("L20", 0), ("L20", 1), ("L20", 2), ("L20", 3), ("H20", 0), ("H20", 1)):
                if (host, index) in occupied or not evidence.get("host:" + host):
                    continue
                freed = any(a["host"] == host and a["gpu_index"] == index and
                            a.get("accepted_complete") for a in self.state["history"])
                if freed or (host == "H20" and index == 1 and evidence.get("stable:E3")):
                    return "E2", host, index
        return None


def lock_paths(config, assignment):
    root = Path(config.get("lock_root", Path(config["root"]) / "control/locks"))
    return (root / (assignment["gpu_uuid"] + ".lock"),
            root / (assignment["experiment"] + ".lock"))


def committed_checkpoint(run):
    """Manifest is authoritative; status can lag its durable commit point."""
    manifest = read_json(run / "checkpoint-manifest.json", {})
    for role in ("current", "predecessor"):
        entry = manifest.get(role)
        if not entry:
            continue
        path = (run / entry["path"]).resolve()
        if run.resolve() not in path.parents:
            raise Refused("checkpoint path escapes run")
        if path.is_file() and digest(path) == entry["sha256"]:
            return dict(entry, recovery_role=role)
    return None


def checkpoint_verified(run, status=None):
    return committed_checkpoint(run) is not None


def final_backup_verified(config, assignment, checkpoint, done):
    if not checkpoint or not done or not done.get("final") or done.get("attempt_id") != assignment["token"]:
        return False
    backup_root = Path(config["backup_root"]).resolve()
    destination = Path(done.get("destination", "")).resolve()
    if backup_root not in destination.parents:
        return False
    marker = read_json(destination / "backup-complete.json", {})
    if marker.get("request", {}).get("attempt_id") != assignment["token"] or not marker.get("request", {}).get("final"):
        return False
    manifest = read_json(destination / "checkpoint-manifest.json", {})
    current = manifest.get("current") or {}
    files = marker.get("files") or {}
    required = {checkpoint["path"], "resolved_config.json", "source_config_frozen.json", "runtime.json",
                "supervisor/assignment.json", "supervisor/command.json", "supervisor/status.json", "supervisor/resources.jsonl"}
    # A zero-length log is legitimate only before updates; a retired prefix must
    # preserve its attempt identity and at least one actual update log.
    identities = [name for name in files if name.startswith("attempts/") and name.endswith("/identity.json")]
    logs = [name for name in files if name.startswith("attempts/") and name.endswith("/updates.jsonl")]
    if not required.issubset(files) or not identities or not logs or len(files) > 10000:
        return False
    if not any(files[name].get("size_bytes", 0) > 0 for name in logs):
        return False
    if current.get("sha256") != checkpoint["sha256"] or files[checkpoint["path"]].get("sha256") != checkpoint["sha256"]:
        return False
    total = 0
    for name, entry in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = (destination / relative).resolve()
        size = entry.get("size_bytes")
        if type(size) is not int or size < 0 or backup_root not in path.parents:
            return False
        total += size
        if total > 100 * GIB or not path.is_file() or path.stat().st_size != size or digest(path) != entry.get("sha256"):
            return False
    return True


def check_evidence(assignment, category, evidence):
    budget = assignment_budget(assignment)
    if budget["contract_version"] == CONTRACT_VERSION:
        if (evidence.get("profile_sha256") != assignment.get("profile_sha256") or
                evidence.get("contract_version") != CONTRACT_VERSION or evidence.get("attempt_id") != assignment["token"]):
            raise Refused("evidence contract/profile/attempt mismatch")
    if category == "prefix":
        if evidence.get("paused_step") != budget["prefix_step"] or evidence.get("resumed_step", 0) < budget["prefix_step"] + 1 or not evidence.get("identity_and_sampler_passed"):
            raise Refused("prefix requires budget boundary pause, next update resume and sampler/identity check")
    if category == "complete":
        if evidence.get("global_step") != budget["target_updates"] or evidence.get("validation_nodes") != 10 or not all(
                evidence.get(key) for key in ("hashes_passed", "ledger_passed", "backup_passed")):
            raise Refused("completion evidence incomplete")
        if budget["contract_version"] == CONTRACT_VERSION:
            if evidence.get("effective_samples_completed") != budget["target_samples"]:
                raise Refused("completion sample count mismatch")
            checkpoint = assignment.get("view", {}).get("committed_checkpoint") or {}
            verify_checkpoint_contract(assignment, checkpoint, complete=True)


def inspect_worker(config, assignment):
    run, control = assignment_paths(config, assignment)
    worker = read_json(control / "worker.json", {})
    status = read_json(control / "status.json", {})
    if worker and worker.get("token") != assignment["token"]:
        raise Refused("worker token mismatch")
    alive = identity_alive(worker.get("worker_identity")) or identity_alive(worker.get("trainer_identity"))
    locks_free = False
    try:
        with ExitStack() as stack:
            for path in lock_paths(config, assignment):
                stack.enter_context(lock(path))
            locks_free = True
    except Refused:
        pass
    # Missing worker.json is not exit proof: launch may still be in flight.
    state = "running" if alive else ("exited" if worker and locks_free and
             (worker.get("exit_code") is not None or worker.get("worker_identity")) else "unknown")
    committed = committed_checkpoint(run) if state == "exited" else None
    result = {"state": state, "locks_free": locks_free, "worker": worker, "status": status,
              "checkpoint_verified": committed is not None, "committed_checkpoint": committed}
    for kind in ("validation", "backup"):
        for suffix in ("request", "done"):
            result[kind + "_" + suffix] = read_json(control / f"{kind}.{suffix}.json")
    result["final_backup_verified"] = final_backup_verified(config, assignment, committed, result.get("backup_done")) if state == "exited" else False
    return result


def launch_worker(config, config_path, host, assignment):
    run, control = assignment_paths(config, assignment)
    control.mkdir(parents=True, exist_ok=True)
    # Per-token launch lock and immutable marker make repeated RPC delivery safe.
    with lock(control / "launch.lock"):
        previous = read_json(control / "assignment.json")
        if previous:
            if previous != assignment:
                raise Refused("assignment token reused with different payload")
            return {"state": "already_dispatched", "token": assignment["token"]}
        atomic_json(control / "assignment.json", assignment)
        command = [config["python"], str(Path(config["root"]) / "repo/scripts/supervise_pretrain.py"),
                   "--config", str(Path(config_path).resolve()), "worker", "--host", host,
                   "--assignment", str(control / "assignment.json")]
        # start_new_session is setsid; survival after logout is separately evidenced.
        with (control / "worker.stdout.log").open("ab") as output:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=Path(config["root"]) / "repo")
        atomic_json(control / "dispatch.json", {"pid": child.pid, "unix": time.time()})
        return {"state": "dispatched", "pid": child.pid, "token": assignment["token"]}


def backup_snapshot(config, assignment, request):
    """Pin all referenced blobs locally before async transfer can race pruning."""
    run, control = assignment_paths(config, assignment)
    pin_root = control / "backup-pins" / request["request_id"]
    pin_root.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(3):
            manifest = read_json(run / "checkpoint-manifest.json")
            if not manifest:
                raise Refused("no committed checkpoint manifest")
            entries = [manifest[k] for k in ("current", "predecessor", "best") if manifest.get(k)]
            entries.extend(manifest.get("nodes", {}).values())
            pins = {}
            try:
                for entry in entries:
                    relative = entry["path"]
                    source = (run / relative).resolve()
                    if run.resolve() not in source.parents:
                        raise Refused("checkpoint path escapes run")
                    target = pin_root / entry["sha256"]
                    if not target.exists():
                        os.link(source, target)
                    pins[relative] = target
                break
            except FileNotFoundError:
                if attempt == 2:
                    raise Refused("checkpoint publication/pruning raced backup pinning")
        return _backup_snapshot(config, assignment, request, manifest, pins)
    finally:
        shutil.rmtree(pin_root)


def _backup_snapshot(config, assignment, request, manifest, pins):
    """Copy pinned immutable blobs and bounded logs under one global grant."""
    run, control = assignment_paths(config, assignment)
    backup = Path(config["backup_root"])
    if shutil.disk_usage(backup).free < 100 * GIB or tree_bytes(backup) >= 100 * GIB:
        raise Refused("backup free-space/budget gate")
    entries = [manifest[k] for k in ("current", "predecessor", "best") if manifest.get(k)]
    entries.extend(manifest.get("nodes", {}).values())
    files = {entry["path"]: entry for entry in entries}
    pin_root = control / "backup-pins" / request["request_id"]
    # Logs may be ahead of checkpoint; canonical ledger cursor stays in checkpoint.
    for pattern in ("*.json", "*.jsonl", "attempts/**/*.json", "attempts/**/*.jsonl",
                    "validations/*.json"):
        for path in run.glob(pattern):
            if path.name != "checkpoint-manifest.json" and path.is_file():
                relative = str(path.relative_to(run))
                if relative not in files:
                    pins[relative], size = pin_file(path, pin_root)
                    files[relative] = {"size_bytes": size}
    control_sources = {}
    for name in ("assignment.json", "command.json", "worker.json", "status.json",
                 "resources.jsonl", "selected-config.json", "pause.request", "validation.request.json", "validation.done.json"):
        path = control / name
        if path.is_file():
            relative = "supervisor/" + name
            control_sources[relative], size = pin_file(path, pin_root)
            files[relative] = {"size_bytes": size}
    if assignment["host"] == "H20":
        # Back up the sole allocator's state with H20 snapshots. Transport
        # credentials/options are provisioned separately, never copied here.
        ledger = Path(config["root"]) / "control/coordinator.json"
        if ledger.is_file():
            relative = "supervisor/coordinator.json"
            control_sources[relative], size = pin_file(ledger, pin_root)
            files[relative] = {"size_bytes": size}
        safe_config = {key: config[key] for key in (
            "root", "python", "source_root", "label_root", "backup_root",
            "backbone_root", "manifest_sha256")}
        safe_config.update({key: config[key] for key in ("contract_version", "profiles", "lock_root") if key in config})
        safe_config["hosts"] = {host: {key: value[key] for key in (
            "machine_id", "hostname", "ssh", "gpus") if key in value}
            for host, value in config["hosts"].items()}
        snapshot = pin_root / "coordinator-config.json"
        atomic_json(snapshot, safe_config)
        relative = "supervisor/coordinator-config.json"
        control_sources[relative] = snapshot
        files[relative] = {"size_bytes": snapshot.stat().st_size}
    destination = backup / assignment["experiment"] / assignment["token"] / request["request_id"]
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    started, written = time.monotonic(), 0
    for relative, entry in sorted(files.items()):
        source = pins.get(relative) or control_sources.get(relative) or (run / relative).resolve()
        if not (run.resolve() in source.resolve().parents or control.resolve() in source.resolve().parents):
            raise Refused("backup source path escapes run/control")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".partial")
        # Reuse verified immutable objects across hourly snapshots. Links stay
        # inside the backup root; restore with cp -L or equivalent.
        object_path = (backup / "objects" / entry["sha256"]) if entry.get("sha256") else None
        if object_path and object_path.exists() and digest(object_path) == entry["sha256"]:
            target.unlink(missing_ok=True)
            target.symlink_to(os.path.relpath(object_path, target.parent))
            fsync_directory(target.parent)
            hashes[relative] = {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
            continue
        remaining = entry["size_bytes"]
        result = hashlib.sha256()
        with source.open("rb") as src, temp.open("wb") as out:
            while remaining:
                block = src.read(min(1024 * 1024, remaining))
                if not block:
                    raise Refused("source truncated during backup")
                out.write(block)
                result.update(block)
                remaining -= len(block)
                written += len(block)
                delay = written / (25 * 1024 * 1024) - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
            out.flush()
            os.fsync(out.fileno())
        sha = result.hexdigest()
        if entry.get("sha256") and entry["sha256"] != sha:
            raise Refused("checkpoint checksum changed during backup")
        if digest(temp) != sha:
            raise Refused("backup readback hash mismatch")
        if object_path:
            object_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp, object_path)
            fsync_directory(object_path.parent)
            target.unlink(missing_ok=True)
            target.symlink_to(os.path.relpath(object_path, target.parent))
        else:
            os.replace(temp, target)
        fsync_directory(target.parent)
        hashes[relative] = {"sha256": sha, "size_bytes": entry["size_bytes"]}
    atomic_json(destination / "checkpoint-manifest.json", manifest)
    atomic_json(destination / "backup-complete.json", {"files": hashes, "request": request,
                                                        "completed_unix": time.time()})
    # Keep two ordinary hourly snapshots per attempt plus all node/final ones.
    # The newest snapshot includes complete bounded logs; originals stay local.
    snapshots = []
    for marker in destination.parent.glob("*/backup-complete.json"):
        content = read_json(marker)
        req = content["request"]
        if not req.get("final") and not req.get("validation_node"):
            snapshots.append((content["completed_unix"], marker.parent))
    for _, old in sorted(snapshots)[:-2]:
        shutil.rmtree(old)
    # Objects are retained only while referenced by a completed snapshot. Never
    # collect while copying (this function runs under the global backup grant).
    referenced = set()
    for marker in backup.glob("E*/*/*/backup-complete.json"):
        referenced.update(item["sha256"] for item in read_json(marker)["files"].values())
    for path in (backup / "objects").glob("*"):
        if path.name not in referenced:
            path.unlink()
    return str(destination)


def run_worker(config, host, assignment):
    import concurrent.futures
    from suction20.training_telemetry import ResourceSampler
    sampler = ResourceSampler()
    host_guard(config, host)
    if assignment["host"] != host:
        raise Refused("assignment host mismatch")
    gpu = config["hosts"][host]["gpus"][str(assignment["gpu_index"])]
    if gpu["uuid"] != assignment["gpu_uuid"] or not set(assignment["cpus"]).issubset(gpu["cpus"]):
        raise Refused("assignment resource identity mismatch")
    verify_assignment_profile(config, assignment)
    budget = assignment_budget(assignment)
    memory_limit = assignment.get("runtime", {}).get("gpu_memory_limit_gib", 64 if host == "H20" else 40)
    run, control = assignment_paths(config, assignment)
    control.mkdir(parents=True, exist_ok=True)
    worker = {"token": assignment["token"], "worker_identity": process_identity(os.getpid()),
              "started_unix": time.time(), "host": host}
    atomic_json(control / "worker.json", worker)
    child = None
    try:
        with ExitStack() as stack:
            descriptors = [stack.enter_context(lock(path)) for path in lock_paths(config, assignment)]
            # Serialize launches per host, so target concurrency and CPU/RAM checks agree.
            with lock(lock_paths(config, assignment)[0].parent / "host-launch.lock"):
                concurrency = 1
                for path in (Path(config["root"]) / "control/attempts").glob("*/worker.json"):
                    other = read_json(path, {})
                    if other.get("token") != assignment["token"] and (
                        identity_alive(other.get("trainer_identity")) or identity_alive(other.get("worker_identity"))):
                        concurrency += 1
                launch_resources(resources(config["root"], config["backup_root"]), concurrency)
                if not set(assignment["cpus"]).issubset(os.sched_getaffinity(0)):
                    raise Refused("configured CPU affinity unavailable")
                for sample in range(3):
                    assert_idle(gpu_snapshot(), assignment["gpu_index"], assignment["gpu_uuid"])
                    if sample < 2:
                        time.sleep(30)
                manifest = Path(config["label_root"]) / "manifest.json"
                if digest(manifest) != config["manifest_sha256"]:
                    raise Refused("dataset manifest identity mismatch")
                if assignment.get("resume"):
                    path = run / assignment["resume"]
                    checkpoint = committed_checkpoint(run)
                    if not checkpoint:
                        raise Refused("resume checkpoint missing/corrupt; initialization forbidden")
                    verify_checkpoint_contract(assignment, checkpoint)
                elif run.exists() and any(run.iterdir()):
                    raise Refused("fresh initialization requires empty run directory")
                run.mkdir(parents=True, exist_ok=True)
                command = trainer_command(config, assignment)
                if assignment.get("config_path"):
                    payload = Path(assignment["config_path"]).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != assignment["config_sha256"]:
                        raise Refused("selected config changed before launch")
                    snapshot = control / "selected-config.json"
                    temporary = snapshot.with_name(snapshot.name + "." + uuid.uuid4().hex + ".tmp")
                    with temporary.open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, snapshot)
                    fsync_directory(snapshot.parent)
                    command[command.index("--config") + 1] = str(snapshot)
                atomic_json(control / "command.json", {"argv": command, "assignment": assignment})
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=assignment["gpu_uuid"],
                                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                   PYTHONDONTWRITEBYTECODE="1")
                environment.pop("PYTHONPATH", None)
                assert_idle(gpu_snapshot(), assignment["gpu_index"], assignment["gpu_uuid"])
                with (control / "trainer.stdout.log").open("ab") as output:
                    child = subprocess.Popen(["taskset", "--cpu-list", ",".join(map(str, assignment["cpus"]))] + command,
                                             stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                             cwd=Path(config["root"]) / "repo", env=environment,
                                             pass_fds=tuple(descriptors))
                worker["trainer_identity"] = process_identity(child.pid)
                atomic_json(control / "worker.json", worker)
            pause_since, pressure_since = None, None
            established_gpu_pid = None
            cuda_ready_since = None
            backup_errors = 0
            data_pressure = DataPressure()
            last_backup, last_request, last_step = None, 0, None
            future, pending_request = None, None
            # Threads are joined before worker exit: a backup grant cannot be freed
            # while this worker still copies after its trainer exits.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                while child.poll() is None:
                    now = time.time()
                    status = read_json(control / "status.json", {})
                    sample = resources(config["root"], config["backup_root"], child.pid)
                    try:
                        device = gpu_snapshot()[assignment["gpu_index"]]
                        sample["gpu"] = device
                    except (OSError, subprocess.SubprocessError, ValueError) as exc:
                        sample["gpu_error"] = str(exc)
                    reasons = pause_reasons(sample, status, now, last_backup, worker["started_unix"])
                    # Only launch-race detection. Once GPU ownership is established,
                    # another user's later arrival never triggers automatic yielding.
                    established_gpu_pid, conflict = gpu_owner(
                        sample.get("gpu"), status, assignment, established_gpu_pid,
                        trainer_alive=identity_alive(worker.get("trainer_identity")), trainer_pid=child.pid)
                    cuda_ready_since, ownership_timed_out = gpu_readiness_timeout(
                        status, established_gpu_pid, cuda_ready_since, now)
                    if conflict:
                        reasons.append(conflict)
                    elif ownership_timed_out:
                        reasons.append("CUDA context ownership unverified after five minutes")
                    pressure = any(sample["psi"].get(kind, {}).get("full", {}).get("avg60", 0) > 10
                                   for kind in ("memory", "io"))
                    pressure_since = (pressure_since or now) if pressure else None
                    data_bad = data_pressure.observe(status, now)
                    if (pressure_since and now - pressure_since >= 600) or data_bad:
                        newer = any(read_json(path, {}).get("started_unix", 0) > worker["started_unix"] and
                                    identity_alive(read_json(path, {}).get("worker_identity"))
                                    for path in (Path(config["root"]) / "control/attempts").glob("*/worker.json"))
                        if not newer:
                            reasons.append("sustained shared pressure; pause newest local job")
                    if (sample.get("gpu", {}).get("process_memory_mib", {}).get(str(established_gpu_pid)) or 0) > memory_limit * 1024:
                        reasons.append("GPU process memory budget exceeded")
                    if reasons:
                        if not (control / "pause.request").exists():
                            atomic_json(control / "pause.request", {"reasons": reasons, "unix": now})
                        pause_since = pause_since or now
                    if (control / "pause.request").exists():
                        pause_since = pause_since or now
                    if pause_since and now - pause_since >= 300:
                        signal_owned(worker.get("trainer_identity"), signal.SIGTERM)
                    if pause_since and now - pause_since >= 360:
                        signal_owned(worker.get("trainer_identity"), signal.SIGKILL)
                    if future and future.done():
                        try:
                            destination = future.result()
                            last_backup = now
                            backup_errors = 0
                            atomic_json(control / "backup.done.json", dict(pending_request,
                                        completed_unix=now, destination=destination))
                        except Exception as exc:
                            backup_errors += 1
                            atomic_json(control / "backup.error.json", {"error": str(exc), "unix": now, "failures": backup_errors})
                            if backup_errors >= 2:
                                atomic_json(control / "pause.request", {"reason": "repeated backup failure", "unix": now})
                        future = None
                    step = status.get("global_step", 0)
                    checkpoint = status.get("durable_checkpoint")
                    existing = read_json(control / "backup.request.json")
                    done = read_json(control / "backup.done.json") or {}
                    outstanding = existing and existing["request_id"] != done.get("request_id")
                    node = step in budget["validation_steps"] and step != last_step
                    if checkpoint and not outstanding and (now - last_request >= 3600 or node):
                        existing = {"request_id": uuid.uuid4().hex, "attempt_id": assignment["token"],
                                    "experiment": assignment["experiment"], "step": step, "validation_node": bool(node)}
                        atomic_json(control / "backup.request.json", existing)
                        last_request, last_step = now, step
                    grant = read_json(control / "backup.grant.json") or {}
                    if existing and not future and backup_errors < 2 and grant.get("request_id") == existing["request_id"] and \
                            done.get("request_id") != existing["request_id"]:
                        pending_request = existing
                        future = pool.submit(backup_snapshot, config, assignment, existing)
                    sample["telemetry"] = sampler.sample(pid=child.pid, gpu_uuid=assignment["gpu_uuid"],
                        nvml_pid=established_gpu_pid, phase=status.get("phase"), attempt_id=assignment["token"],
                        run_id=str(run), contract_revision=budget["contract_version"],
                        profile_id=assignment.get("runtime_profile", {}).get("profile_id"))
                    worker.update(heartbeat_unix=now, resources=sample,
                                  warning_rss=sample["rss_bytes"] >= 48 * GIB,
                                  progress_warning=(status.get("phase") != "waiting_validation" and
                                      now - status.get("progress_unix", worker["started_unix"]) > 900),
                                  pause_reasons=reasons, last_backup_unix=last_backup,
                                  data_pressure=data_bad, nvml_host_pid=established_gpu_pid,
                                  checkpoint_warning=(status.get("phase") == "training" and
                                      status.get("global_step", 0) - (status.get("durable_checkpoint") or {}).get("global_step", 0) > budget["checkpoint_interval_updates"]))
                    atomic_json(control / "worker.json", worker)
                    with (control / "resources.jsonl").open("a") as stream:
                        stream.write(json.dumps({"unix": now, "status": status, "resources": sample}) + "\n")
                    time.sleep(60)
                # Final/pause backup stays under the coordinator grant, also after
                # training exits. Publish request and wait without releasing locks.
                status = read_json(control / "status.json", {})
                if backup_errors >= 2:
                    raise Refused("repeated backup failures; local evidence retained for diagnosis")
                if committed_checkpoint(run):
                    final = {"request_id": uuid.uuid4().hex, "attempt_id": assignment["token"],
                             "experiment": assignment["experiment"], "step": status.get("global_step"),
                             "final": True}
                    if future:
                        try:
                            destination = future.result()
                            atomic_json(control / "backup.done.json", dict(pending_request, destination=destination))
                        except Exception as exc:
                            backup_errors += 1
                            atomic_json(control / "backup.error.json", {"error": str(exc), "failures": backup_errors})
                    existing = read_json(control / "backup.request.json")
                    done = read_json(control / "backup.done.json") or {}
                    if existing and existing["request_id"] != done.get("request_id"):
                        while True:
                            grant = read_json(control / "backup.grant.json") or {}
                            if grant.get("request_id") == existing["request_id"]:
                                try:
                                    destination = backup_snapshot(config, assignment, existing)
                                    atomic_json(control / "backup.done.json", dict(existing, destination=destination,
                                                                                 completed_unix=time.time()))
                                    break
                                except Exception as exc:
                                    backup_errors += 1
                                    atomic_json(control / "backup.error.json", {"error": str(exc), "failures": backup_errors})
                                    if backup_errors >= 2:
                                        raise Refused("repeated final backup failures") from exc
                            worker["heartbeat_unix"] = time.time()
                            atomic_json(control / "worker.json", worker)
                            time.sleep(60)
                    atomic_json(control / "backup.request.json", final)
                    worker["phase"] = "final_backup"
                    atomic_json(control / "worker.json", worker)
                    while True:
                        grant = read_json(control / "backup.grant.json") or {}
                        if grant.get("request_id") == final["request_id"]:
                            try:
                                destination = backup_snapshot(config, assignment, final)
                                atomic_json(control / "backup.done.json", dict(final, destination=destination,
                                                                             completed_unix=time.time()))
                                break
                            except Exception as exc:
                                backup_errors += 1
                                atomic_json(control / "backup.error.json", {"error": str(exc), "failures": backup_errors})
                                if backup_errors >= 2:
                                    raise Refused("repeated final backup failures") from exc
                        worker["heartbeat_unix"] = time.time()
                        atomic_json(control / "worker.json", worker)
                        time.sleep(60)
            worker["exit_code"] = child.returncode
    except BaseException as exc:
        worker["error"] = str(exc)
        if child and child.poll() is None:
            # Trainer inherits locks if supervision dies; owner remains unknown.
            atomic_json(control / "pause.request", {"reason": "worker failure", "error": str(exc)})
        else:
            worker["exit_code"] = child.returncode if child and child.returncode not in (0, 75) else 1
        raise
    finally:
        worker["ended_unix"] = time.time()
        atomic_json(control / "worker.json", worker)


def retire_pre_initialization(config, assignment, evidence):
    """Preserve only a positively exited, wholly unstarted run; never clear data."""
    if not all(evidence.get(key) for key in ("passed", "health_check", "data_check", "state_check", "diagnosis")):
        raise Refused("pre-initialization retirement requires explicit diagnostic evidence")
    run, control = assignment_paths(config, assignment)
    receipt_path = control / "pre-initialization-retired.json"
    archive = control / "failed-run-before-initialization"
    with ExitStack() as stack:
        for path in lock_paths(config, assignment):
            stack.enter_context(lock(path))
        stack.enter_context(lock(Path(config["root"]) / "control/locks/host-launch.lock"))
        worker = read_json(control / "worker.json", {})
        status = read_json(control / "status.json", {})
        if (worker.get("token") != assignment["token"] or not worker.get("worker_identity") or
                worker.get("exit_code") in (None, 0, 75) or
                identity_alive(worker.get("worker_identity")) or identity_alive(worker.get("trainer_identity"))):
            raise Refused("pre-initialization retry requires positively exited failed owner")
        if (status.get("attempt_id") != assignment["token"] or status.get("global_step") != 0 or
                status.get("phase") != "failed" or status.get("durable_checkpoint") is not None or
                status.get("cuda_context_ready")):
            raise Refused("trainer status does not prove failure before initialization")
        if assignment.get("resume") or assignment.get("prefix_passed"):
            raise Refused("a resumed run cannot use pre-initialization retry")
        if run.exists() and archive.exists():
            raise Refused("both active and preserved run directories exist")
        candidate = run if run.exists() else archive
        if not candidate.is_dir() or candidate.is_symlink():
            raise Refused("no original or preserved pre-initialization run directory")
        # The known allocator failure occurs before dataset/model/log construction.
        # A strict allowlist rejects even empty attempt logs and step-zero weights.
        for path in candidate.iterdir():
            if path.is_symlink():
                raise Refused("pre-initialization directory contains a symlink")
            if path.name == ".trainer.lock" and path.is_file():
                continue
            if path.name == "checkpoints" and path.is_dir() and not any(path.iterdir()):
                continue
            raise Refused("run contains artifacts beyond untouched lock/empty checkpoint directory")
        stack.enter_context(lock(candidate / ".trainer.lock"))
        previous = read_json(receipt_path)
        if previous:
            if run.exists() or previous.get("token") != assignment["token"]:
                raise Refused("pre-initialization receipt conflicts with active run")
            return previous
        if run.exists():
            run.rename(archive)
            fsync_directory(run.parent)
            fsync_directory(control)
        receipt = {"token": assignment["token"], "retired": True, "experiment": assignment["experiment"],
                   "preserved_run": str(archive), "retired_unix": time.time(), "evidence": evidence,
                   "worker": worker, "status": status, "successful_updates": 0}
        atomic_json(receipt_path, receipt)
        return receipt


def rpc(config, config_path, host, request):
    host_guard(config, host)
    op = request["op"]
    if op == "probe":
        return {"machine_id": machine_id(), "hostname": socket.gethostname(),
                "gpus": gpu_snapshot(), "resources": resources(config["root"], config["backup_root"]),
                "cpu_affinity": sorted(os.sched_getaffinity(0))}
    assignment = request["assignment"]
    if assignment["host"] != host or assignment["experiment"] not in EXPERIMENTS:
        raise Refused("invalid assignment identity")
    if len(assignment["token"]) != 32 or any(c not in "0123456789abcdef" for c in assignment["token"]):
        raise Refused("invalid assignment token")
    run, control = assignment_paths(config, assignment)
    if op == "launch":
        return launch_worker(config, config_path, host, assignment)
    if op == "inspect":
        return inspect_worker(config, assignment)
    if read_json(control / "assignment.json") != assignment:
        # Coordinator carries observations absent from the immutable host payload.
        original = read_json(control / "assignment.json", {})
        if any(original.get(key) != assignment.get(key) for key in
               ("token", "host", "gpu_uuid", "experiment", "created_unix")):
            raise Refused("host assignment identity differs")
    if op == "retire_pre_initialization":
        return retire_pre_initialization(config, assignment, request["evidence"])
    if op == "pause":
        atomic_json(control / "pause.request", {"reason": request["reason"], "unix": time.time()})
        return {"requested": True}
    if op == "grant":
        kind, grant = request["kind"], request["grant"]
        if kind not in ("backup", "validation") or grant["token"] != assignment["token"]:
            raise Refused("invalid grant")
        current = read_json(control / f"{kind}.request.json")
        if not current or any(grant.get(key) != value for key, value in current.items()):
            raise Refused("grant does not match live request")
        atomic_json(control / f"{kind}.grant.json", grant)
        return {"granted": grant["request_id"]}
    raise Refused("unknown host operation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "tick", "serve", "plan"):
        commands.add_parser(name)
    probe = commands.add_parser("probe")
    probe.add_argument("--host", choices=("H20", "L20"), required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--host", choices=("H20", "L20"), required=True)
    worker.add_argument("--assignment", type=Path, required=True)
    remote = commands.add_parser("rpc")
    remote.add_argument("--host", choices=("H20", "L20"), required=True)
    evidence = commands.add_parser("evidence", help="record reviewed host/stability/completion evidence")
    evidence.add_argument("key", help="host:H20, host:L20, stable:E3/E0/E1, prefix:E0..E5, complete:E0..E5")
    evidence.add_argument("--file", required=True, type=Path)
    pause = commands.add_parser("pause")
    pause.add_argument("experiment", choices=EXPERIMENTS)
    pause.add_argument("--reason", required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("experiment", choices=EXPERIMENTS)
    recover = commands.add_parser("recover", help="one diagnosed failure recovery from verified checkpoint")
    recover.add_argument("experiment", choices=EXPERIMENTS)
    recover.add_argument("--evidence", required=True, type=Path)
    recover.add_argument("--pre-initialization", action="store_true",
                         help="once-only audited retry before any checkpoint, attempt log or update")
    retire = commands.add_parser("retire", help="preserve a paused run as superseded; never mark complete")
    retire.add_argument("experiment", choices=EXPERIMENTS)
    retire.add_argument("--superseded-by", required=True)
    retire.add_argument("--evidence", required=True, type=Path)
    release = commands.add_parser("release")
    release.add_argument("experiment", choices=EXPERIMENTS)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    validate_config(config)
    if args.command == "worker":
        run_worker(config, args.host, read_json(args.assignment))
        return
    if args.command == "rpc":
        print(json.dumps(rpc(config, args.config, args.host, json.load(sys.stdin))))
        return
    transport = Transport(config, args.config)
    if args.command == "probe":
        print(json.dumps(transport.call(args.host, {"op": "probe"}), indent=2))
        return
    ledger = Path(config["root"]) / "control/coordinator.json"
    if args.command in ("status", "plan"):
        coordinator = Coordinator(config, ledger, transport)
        result = coordinator.state if args.command == "status" else {"next": coordinator.next_launch(),
                  "note": "read-only; live launch still checks GPU/RAM/data/stage gates"}
        print(json.dumps(result, indent=2))
        return
    host_guard(config, "H20")
    if args.command == "serve":
        with lock(ledger.with_name("coordinator-daemon.lock")):
            while True:
                main(["--config", str(args.config), "tick"])
                time.sleep(60)
    with lock(ledger.with_suffix(".lock")):
        coordinator = Coordinator(config, ledger, transport)
        if args.command == "evidence":
            category, experiment = args.key.split(":", 1)
            if category not in ("host", "stable", "prefix", "complete"):
                raise Refused("unknown evidence category")
            evidence = read_json(args.file)
            if evidence.get("passed") is not True:
                raise Refused("evidence must explicitly record passed=true")
            if category == "host":
                if experiment not in config["hosts"] or not all(evidence.get(k) for k in
                        ("persistence_survival", "environment_identity", "kernel_check", "storage_check")):
                    raise Refused("host evidence lacks persistence/environment/kernel/storage checks")
            elif category == "stable":
                check_evidence(coordinator.state["assignments"][experiment], category, evidence)
                if evidence.get("successful_updates", 0) < 20 or evidence.get("observed_seconds", 0) < 600:
                    raise Refused("stability requires 20 successful updates and 600 seconds")
                if not evidence.get("pressure_passed") or not coordinator.state["evidence"].get("prefix:" + experiment):
                    raise Refused("stability requires prefix/resume and pressure evidence")
            elif category in ("prefix", "complete"):
                if category == "complete":
                    coordinator.reconcile()
                check_evidence(coordinator.state["assignments"][experiment], category, evidence)
            coordinator.state["evidence"][args.key] = {"content": evidence, "sha256": digest(args.file),
                                                        "recorded_unix": time.time()}
            coordinator.save()
        elif args.command == "pause":
            assignment = coordinator.state["assignments"][args.experiment]
            transport.call(assignment["host"], {"op": "pause", "assignment": assignment, "reason": args.reason})
        elif args.command in ("resume", "release", "recover", "retire"):
            coordinator.reconcile()
            coordinator.grants()
            old = coordinator.state["assignments"][args.experiment]
            status = old.get("view", {}).get("status", {})
            if args.command == "retire":
                coordinator.retire(args.experiment, args.superseded_by, read_json(args.evidence))
                return
            if args.command in ("resume", "recover"):
                verify_assignment_profile(config, old)
            if args.command == "release" and not coordinator.state["evidence"].get("complete:" + args.experiment):
                raise Refused("completion acceptance evidence required")
            if args.command == "resume" and old.get("view", {}).get("worker", {}).get("exit_code") != 75:
                raise Refused("automatic retry disabled; only normal boundary pause may resume")
            if args.command == "recover" and args.pre_initialization:
                coordinator.recover_pre_initialization(args.experiment, read_json(args.evidence))
                return
            if args.command == "recover":
                proof = read_json(args.evidence)
                if not all(proof.get(k) for k in ("passed", "health_check", "data_check", "state_check", "diagnosis")):
                    raise Refused("recovery requires diagnosed health/data/state evidence")
                if any(a["experiment"] == args.experiment and a.get("recovery_evidence")
                       for a in coordinator.state["history"]):
                    raise Refused("repeated failure: blocked for diagnosis, no further automatic retry")
                old["recovery_evidence"] = proof
            if args.command == "release":
                check_evidence(old, "complete", coordinator.state["evidence"]["complete:" + args.experiment]["content"])
                old["accepted_complete"] = True
            released = coordinator.release(args.experiment, {"command": args.command, "unix": time.time()})
            if args.command in ("resume", "recover"):
                checkpoint = released["view"]["committed_checkpoint"]
                coordinator.assign(args.experiment, released["host"], released["gpu_index"],
                                   resume="last.pt", prefix_passed=checkpoint["global_step"] >= assignment_budget(released)["prefix_step"])
        else:
            while True:
                coordinator.reconcile()
                coordinator.grants()
                candidate = coordinator.next_launch()
                if candidate:
                    experiment, host, index = candidate
                    try:
                        probe = transport.call(host, {"op": "probe"})
                        assert_idle({int(k): v for k, v in probe["gpus"].items()}, index,
                                    config["hosts"][host]["gpus"][str(index)]["uuid"])
                        count = 1 + sum(a["host"] == host for a in coordinator.state["assignments"].values())
                        launch_resources(probe["resources"], count)
                    except (Refused, OSError, subprocess.SubprocessError, ValueError) as exc:
                        coordinator.state["launch_gate"] = str(exc)
                        coordinator.save()
                    else:
                        try:
                            coordinator.assign(*candidate)
                            coordinator.state.pop("launch_gate", None)
                        except (Refused, OSError, ValueError) as exc:
                            coordinator.state["launch_gate"] = str(exc)
                        coordinator.save()
                if args.command == "tick":
                    break
                time.sleep(60)


if __name__ == "__main__":
    main()
