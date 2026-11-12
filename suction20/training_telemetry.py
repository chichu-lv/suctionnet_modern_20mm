"""Read-only training telemetry and checkpoint-linked analysis exports.

Only the Python standard library is used. Sampling never imports Torch, touches
training RNG, or initializes CUDA. Missing measurements are null with reasons.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite measurement")
    return result


class ResourceSampler:
    """One sampler per worker; CPU rates are deltas for the same process identity.

    ``pid`` is the local /proc PID; ``nvml_pid`` is the supervisor's independently
    verified host PID. This collector never guesses their mapping or signals PIDs.
    GPU fields describe the assigned device; device power is not process power.
    """

    def __init__(self, *, proc_root="/proc", command=None, clock=None, wall_clock=None):
        self.proc = Path(proc_root)
        self.command = command or subprocess.run
        self.clock = clock or time.monotonic
        self.wall_clock = wall_clock or time.time
        self.previous = None
        self.ticks = os.sysconf("SC_CLK_TCK")

    def sample(self, *, pid=None, gpu_uuid=None, nvml_pid=None, phase=None,
               attempt_id=None, run_id=None, contract_revision=None, profile_id=None):
        started, unix = self.clock(), self.wall_clock()
        missing = {}
        result = dict(schema_version="suction20.resources.v1", event_type="resources",
                      unix=unix, utc=datetime.fromtimestamp(unix, timezone.utc).isoformat(),
                      monotonic_seconds=started, host=socket.gethostname(),
                      attempt_id=attempt_id, run_id=run_id, contract_revision=contract_revision,
                      profile_id=profile_id, phase=phase, missing_reasons=missing)

        def read(path, key):
            try:
                return (self.proc / path).read_text()
            except (OSError, UnicodeError) as error:
                missing[key] = f"{type(error).__name__}: {error}"
                return None

        result["boot_id"] = read("sys/kernel/random/boot_id", "boot_id")
        if result["boot_id"] is not None:
            result["boot_id"] = result["boot_id"].strip()
        process = dict(pid=pid, start_ticks=None, cpu_user_seconds=None, cpu_system_seconds=None,
                       cpu_core_equivalents=None, cpu_window_seconds=None, rss_bytes=None,
                       affinity=None, io=None, io_bytes_per_second=None)
        result["process"] = process
        if pid is not None:
            stat = read(f"{pid}/stat", "process.stat")
            status = read(f"{pid}/status", "process.status")
            raw_io = read(f"{pid}/io", "process.io")
            try:
                if status is not None:
                    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
                    process["rss_bytes"] = int(fields["VmRSS"].split()[0]) * 1024
                    process["affinity"] = fields.get("Cpus_allowed_list", "").strip() or None
            except (KeyError, ValueError, IndexError) as error:
                missing["process.rss_bytes"] = f"invalid /proc status: {error}"
            try:
                if raw_io is not None:
                    process["io"] = {k: int(v.strip()) for k, v in
                                     (line.split(":", 1) for line in raw_io.splitlines())}
            except (ValueError, TypeError) as error:
                missing["process.io"] = f"invalid /proc io: {error}"
            try:
                if stat is not None:
                    # comm can contain spaces and parentheses; fields follow its last ')'.
                    fields = stat[stat.rfind(")") + 2:].split()
                    user, system, start = int(fields[11]), int(fields[12]), int(fields[19])
                    process.update(start_ticks=start, cpu_user_seconds=user / self.ticks,
                                   cpu_system_seconds=system / self.ticks)
                    current = (pid, start, started, user + system, process["io"])
                    old = self.previous
                    if old and old[:2] == current[:2] and started > old[2] and user + system >= old[3]:
                        window = started - old[2]
                        process.update(cpu_window_seconds=window,
                                       cpu_core_equivalents=(user + system - old[3]) / self.ticks / window)
                        if old[4] is not None and process["io"] is not None:
                            process["io_bytes_per_second"] = {
                                key: (process["io"][key] - old[4][key]) / window
                                for key in ("read_bytes", "write_bytes", "rchar", "wchar")
                                if key in process["io"] and key in old[4] and process["io"][key] >= old[4][key]}
                    else:
                        missing["process.cpu_core_equivalents"] = "first sample or changed process identity/counters"
                    self.previous = current
            except (ValueError, IndexError) as error:
                missing["process.stat"] = f"invalid /proc stat: {error}"
        else:
            missing["process"] = "local trainer PID not available"
        for key, value in process.items():
            if value is None:
                missing.setdefault(f"process.{key}", "measurement unavailable; see process source errors")

        host = dict(available_bytes=None, load_average=None, psi={})
        result["host_resources"] = host
        info = read("meminfo", "host_resources.available_bytes")
        if info is not None:
            try:
                fields = dict(line.split(":", 1) for line in info.splitlines())
                host["available_bytes"] = int(fields["MemAvailable"].split()[0]) * 1024
            except (ValueError, KeyError, IndexError) as error:
                missing["host_resources.available_bytes"] = f"invalid meminfo: {error}"
        load = read("loadavg", "host_resources.load_average")
        if load is not None:
            try:
                host["load_average"] = [_number(v) for v in load.split()[:3]]
                if len(host["load_average"]) != 3:
                    raise ValueError("expected three load averages")
            except ValueError as error:
                host["load_average"] = None
                missing["host_resources.load_average"] = str(error)
        for kind in ("cpu", "memory", "io"):
            raw = read(f"pressure/{kind}", f"host_resources.psi.{kind}")
            host["psi"][kind] = None
            if raw is not None:
                try:
                    host["psi"][kind] = {line.split()[0]: {k: _number(v) for k, v in
                        (field.split("=", 1) for field in line.split()[1:])} for line in raw.splitlines()}
                except (ValueError, IndexError) as error:
                    missing[f"host_resources.psi.{kind}"] = f"invalid PSI: {error}"

        gpu_fields = ("memory_used_mib", "memory_total_mib", "utilization_percent",
                      "memory_utilization_percent", "power_watts", "temperature_c",
                      "sm_clock_mhz", "memory_clock_mhz")
        gpu = dict(uuid=gpu_uuid, nvml_pid=nvml_pid, process_memory_mib=None,
                   compute_processes=None, **dict.fromkeys(gpu_fields))
        result["gpu"] = gpu
        if gpu_uuid:
            try:
                query = "uuid,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,clocks.mem"
                completed = self.command(["nvidia-smi", "-i", gpu_uuid, "--query-gpu=" + query,
                                          "--format=csv,noheader,nounits"], check=True,
                                         capture_output=True, text=True, timeout=5)
                values = list(csv.reader(io.StringIO(completed.stdout)))
                if len(values) != 1 or len(values[0]) != len(gpu_fields) + 1 or values[0][0].strip() != gpu_uuid:
                    raise ValueError("GPU UUID/row mismatch")
                for name, value in zip(gpu_fields, values[0][1:]):
                    try:
                        gpu[name] = _number(value.strip())
                    except ValueError:
                        missing[f"gpu.{name}"] = f"unsupported or invalid NVML value: {value.strip()}"
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                missing["gpu.query"] = f"{type(error).__name__}: {error}"
            try:
                completed = self.command(["nvidia-smi", "-i", gpu_uuid,
                    "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5)
                processes = []
                for row in csv.reader(io.StringIO(completed.stdout)):
                    if not row:
                        continue
                    if len(row) != 3 or row[0].strip() != gpu_uuid:
                        raise ValueError("compute process GPU UUID/row mismatch")
                    host_pid = int(row[1].strip())
                    try:
                        memory = _number(row[2].strip())
                    except ValueError:
                        memory = None
                        missing[f"gpu.compute_processes.{host_pid}.memory_mib"] = "NVML memory unsupported"
                    processes.append(dict(pid=host_pid, memory_mib=memory))
                gpu["compute_processes"] = processes
                own = [p for p in processes if p["pid"] == nvml_pid]
                if len(own) == 1:
                    gpu["process_memory_mib"] = own[0]["memory_mib"]
                else:
                    missing["gpu.process_memory_mib"] = "verified NVML PID not supplied or not present"
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                missing["gpu.process_query"] = f"{type(error).__name__}: {error}"
        else:
            missing["gpu"] = "assigned GPU UUID not supplied"
        for key, value in gpu.items():
            if value is None:
                missing.setdefault(f"gpu.{key}", "measurement unavailable; see GPU query errors")
        result["sample_duration_seconds"] = self.clock() - started
        return result


def _json(raw):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(raw, parse_constant=reject, parse_float=_number)


def _safe_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Analysis reference outside run directory: {name}")
    return path


def _sha(path, length=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = length
        while remaining is None or remaining:
            block = stream.read(1024 * 1024 if remaining is None else min(remaining, 1024 * 1024))
            if not block:
                if remaining:
                    raise ValueError(f"Truncated source: {path}")
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, list):
            result[name] = json.dumps(item, sort_keys=True, allow_nan=False)
        else:
            result[name] = item
    return result


def _csv(path, rows):
    flat = [_flatten(row) for row in rows]
    columns = sorted({key for row in flat for key in row}) or ["classification"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)


def _validate_update(row):
    if not isinstance(row, dict) or type(row.get("updated")) is not bool:
        raise ValueError("Update must contain a boolean updated field")
    if not isinstance(row.get("attempt_id"), str) or not row["attempt_id"]:
        raise ValueError("Update must identify its attempt")
    for key in ("global_step", "optimizer_attempt"):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError(f"Invalid {key}")
    losses = row.get("loss", {})
    if not isinstance(losses, dict) or any(v is not None and type(v) not in (int, float) for v in losses.values()):
        raise ValueError("Loss measurements must be numeric or null")
    for key in ("elapsed_seconds", "lr"):
        if row.get(key) is not None and (type(row[key]) not in (int, float) or row[key] < 0):
            raise ValueError(f"Invalid {key} measurement")
    stats = row.get("frame_stats", [])
    if not isinstance(stats, list):
        raise ValueError("Microbatch statistics must be a list")
    for stat in stats:
        if not isinstance(stat, dict):
            raise ValueError("Microbatch statistics must be objects")
        for key in ("point_count", "voxel_count"):
            if key in stat and (type(stat[key]) is not int or stat[key] < 0):
                raise ValueError(f"Invalid {key}")
    frames = row.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Update must contain consumed frames")
    for frame in frames:
        if not isinstance(frame, list) or len(frame) != 3:
            raise ValueError("Invalid frame UID cardinality")
    if "effective_batch_size" in row and len(frames) != row["effective_batch_size"]:
        raise ValueError("Frame cardinality differs from effective batch")
    if "microbatch_size" in row:
        micro, accumulation = row["microbatch_size"], row.get("gradient_accumulation_steps")
        if type(micro) is not int or micro <= 0 or type(accumulation) is not int or accumulation <= 0:
            raise ValueError("Invalid microbatch/accumulation")
        if len(frames) != micro * accumulation:
            raise ValueError("Frame cardinality differs from microbatch times accumulation")
    details = row.get("frame_details")
    if details is not None:
        if not isinstance(details, list) or len(details) != len(frames):
            raise ValueError("Frame details cardinality mismatch")
        for i, (frame, detail) in enumerate(zip(frames, details)):
            if detail.get("sample_uid") != frame:
                raise ValueError("Frame details UID/order mismatch")
            if "microbatch_size" in row and detail.get("microbatch_index") != i // row["microbatch_size"]:
                raise ValueError("Frame microbatch membership mismatch")
        stats = row.get("frame_stats")
        if "gradient_accumulation_steps" in row and stats is not None:
            if len(stats) != row["gradient_accumulation_steps"]:
                raise ValueError("Microbatch statistics cardinality mismatch")
            for index, stat in enumerate(stats):
                for key in ("point_count", "voxel_count"):
                    counts = [d.get(key) for d in details if d.get("microbatch_index") == index]
                    if all(type(c) is int for c in counts) and key in stat and sum(counts) != stat[key]:
                        raise ValueError(f"Per-frame {key} differs from microbatch aggregate")


def export_training_logs(run_dir, output_dir, *, checkpoint_role="current", resource_logs=(),
                         live_attempt_id=None):
    """Export a verified manifest generation without importing Torch/pickle.

    No automatic predecessor fallback: an explicit predecessor selection records
    the rollback. A caller-designated live attempt is a label, not liveness proof.
    Existing output directories are refused; all inputs are read-only snapshots.
    Manifest analysis metadata is trusted as the run's publication authority;
    checkpoint bytes and all committed log/result references are hash-verified.
    """
    root, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if checkpoint_role not in ("current", "predecessor", "best"):
        raise ValueError("Checkpoint role must be current, predecessor or best")
    if output.exists():
        raise ValueError("Export output already exists; choose a new snapshot directory")
    if output == root or output in root.parents:
        raise ValueError("Output cannot replace the run directory")
    sources, gaps = [], []

    def snapshot(path, role, length=None, expected=None):
        path = Path(path)
        size = path.stat().st_size if length is None else length
        if type(size) is not int or size < 0:
            raise ValueError("Invalid source prefix length")
        with path.open("rb") as stream:
            raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f"Truncated {role}: {path}")
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"{role} checksum mismatch: {path}")
        sources.append(dict(path=str(path), role=role, size_bytes=size, sha256=digest))
        return raw

    manifest_raw = snapshot(root / "checkpoint-manifest.json", "checkpoint_manifest")
    manifest = _json(manifest_raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint manifest schema")
    entry = manifest.get(checkpoint_role)
    if not isinstance(entry, dict):
        raise ValueError(f"No {checkpoint_role} checkpoint")
    checkpoint = _safe_path(root, entry["path"])
    if checkpoint.stat().st_size != entry["size_bytes"] or _sha(checkpoint) != entry["sha256"]:
        raise ValueError("Checkpoint checksum mismatch; select a verified predecessor explicitly if needed")
    sources.append(dict(path=str(checkpoint), role="checkpoint", size_bytes=entry["size_bytes"], sha256=entry["sha256"]))
    analysis = entry.get("analysis")
    if not isinstance(analysis, dict) or analysis.get("schema_version") != 1:
        raise ValueError("Checkpoint lacks supported analysis metadata; raw logs cannot establish accepted progress")
    if analysis.get("global_step") != entry["global_step"] or analysis.get("attempt_id") != entry["attempt_id"]:
        raise ValueError("Analysis/checkpoint step or attempt identity mismatch")
    public_aliases = dict(budget=analysis.get("budget"),
        runtime_profile=(analysis.get("identity") or {}).get("runtime_profile"),
        contract_version=(analysis.get("budget") or {}).get("contract_version"),
        effective_samples_completed=(analysis.get("counters") or {}).get("effective_samples_completed"),
        completed_validation_nodes=analysis.get("completed_validation_nodes"),
        last_completed_validation=analysis.get("last_completed_validation"))
    formal = bool((analysis.get("budget") or {}).get("formal"))
    for key, value in public_aliases.items():
        if (key in entry or formal) and entry.get(key) != value:
            raise ValueError(f"Checkpoint public {key} disagrees with analysis metadata")
    ledger = analysis["ledger"]
    updates, ends = [], {}

    def parse_rows(raw, path, classification, start=0, strict=True):
        rows, offset = [], start
        for line in raw.splitlines(keepends=True):
            end = offset + len(line)
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("Incomplete JSONL record")
                row = _json(line)
                _validate_update(row)
                # Classification/source fields cannot be overridden by a raw record.
                row.update(classification=classification, source_path=str(path),
                           source_start_offset=offset, source_end_offset=end)
                rows.append(row)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                if strict:
                    raise ValueError(f"Invalid committed update at {path}:{offset}: {error}") from error
                gaps.append(dict(path=str(path), start_offset=offset, end_offset=end,
                                 classification=classification, reason=str(error)))
            offset = end
        return rows

    step, attempts, skipped, frames_count = 0, 0, 0, 0
    credited_frames = 0
    for segment in ledger["lineage"]:
        path = _safe_path(root, segment["path"])
        if path in ends:
            raise ValueError("Repeated lineage segment would count updates twice")
        end = segment["end_offset"]
        raw = snapshot(path, "committed_update_prefix", end, segment["sha256"])
        rows = parse_rows(raw, path, "committed")
        for row in rows:
            attempts += 1
            step += int(row["updated"])
            skipped += int(not row["updated"])
            frames_count += len(row["frames"])
            credited_frames += len(row["frames"]) if row["updated"] else 0
            if row.get("attempt_id") != segment["attempt_id"]:
                raise ValueError("Lineage/update attempt identity mismatch")
            if row.get("global_step") != step or row.get("optimizer_attempt") != attempts:
                raise ValueError("Committed update sequence is not continuous")
            budget = analysis.get("budget") or {}
            if budget.get("effective_batch_size") is not None and len(row["frames"]) != budget["effective_batch_size"]:
                raise ValueError("Committed batch differs from checkpoint budget")
            if row.get("contract_version") is not None and row["contract_version"] != budget.get("contract_version"):
                raise ValueError("Committed contract differs from checkpoint budget")
            profile = (analysis.get("identity") or {}).get("runtime_profile")
            if row.get("runtime_profile") is not None and row["runtime_profile"] != profile:
                raise ValueError("Committed runtime profile differs from checkpoint identity")
            for key, value in (("consumed_samples", frames_count), ("effective_samples_completed", credited_frames),
                               ("skipped_samples", frames_count - credited_frames)):
                if key in row and row[key] != value:
                    raise ValueError(f"Committed {key} differs from log-derived counter")
        updates.extend(rows)
        ends[path] = end
    expected = (entry["global_step"], ledger["attempts"], ledger["skipped"], ledger["frames"])
    if (step, attempts, skipped, frames_count) != expected:
        raise ValueError("Committed lineage counters differ from checkpoint ledger")
    derived_counters = dict(effective_samples_completed=credited_frames, consumed_samples=frames_count,
                            skipped_samples=frames_count - credited_frames, optimizer_attempts=attempts)
    for key, value in (analysis.get("counters") or {}).items():
        if key in derived_counters and value != derived_counters[key]:
            raise ValueError(f"Checkpoint {key} differs from log-derived counter")

    # Explicit abandoned tails survive resume. Their full-file prefix hashes bind
    # the old bytes; never infer acceptance merely from successful step numbers.
    abandoned = {}
    for path in sorted((root / "attempts").glob("*/identity.json")):
        path = _safe_path(root, path)
        identity = _json(snapshot(path, "attempt_identity"))
        for tail in identity.get("abandoned_tails", []) + identity.get("abandoned_event_tails", []):
            old = _safe_path(root, tail["path"])
            lo, hi = tail["start_offset"], tail["end_offset"]
            if type(lo) is not int or type(hi) is not int or lo < 0 or hi < lo:
                raise ValueError("Invalid abandoned tail interval")
            snapshot(old, "abandoned_prefix", hi, tail["sha256"])
            abandoned.setdefault(old, []).append((lo, hi))
    for path in sorted((root / "attempts").glob("*/updates.jsonl")):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Attempt log escapes run directory")
        raw = snapshot(path, "update_file_snapshot")
        start = ends.get(path, 0)
        if len(raw) < start:
            raise ValueError("Committed log truncated during export")
        # Recheck the committed bytes in the larger snapshot to detect in-place mutation.
        for segment in ledger["lineage"]:
            if _safe_path(root, segment["path"]) == path:
                if hashlib.sha256(raw[:start]).hexdigest() != segment["sha256"]:
                    raise ValueError("Committed prefix changed during export")
        offset = start
        for line in raw[start:].splitlines(keepends=True):
            end = offset + len(line)
            discarded = any(lo <= offset and end <= hi for lo, hi in abandoned.get(path, []))
            classification = "abandoned" if discarded else (
                "live_uncommitted" if path.parent.name == live_attempt_id else "uncommitted")
            updates.extend(parse_rows(line, path, classification, offset, strict=False))
            offset = end

    frame_rows = []
    for row in updates:
        details = row.get("frame_details")
        if details is None:
            details = [dict(sample_uid=frame, augmentation_seed=None, microbatch_index=None)
                       for frame in row["frames"]]
        for index, detail in enumerate(details):
            frame_rows.append(dict(detail, classification=row["classification"],
                attempt_id=row["attempt_id"], global_step=row["global_step"],
                optimizer_attempt=row["optimizer_attempt"], frame_index=index,
                source_path=row["source_path"], source_start_offset=row["source_start_offset"]))

    validation_rows, validation_summary = [], []
    for key, reference in analysis.get("validations", {}).items():
        path = _safe_path(root, reference["path"])
        result = _json(snapshot(path, "committed_validation", expected=reference["sha256"]))
        if result.get("global_step") != int(key) or int(key) > step:
            raise ValueError("Validation step differs from committed reference")
        metrics = result["metrics"]
        records = metrics["frames"]
        if not records or metrics.get("split") not in ("val", "smoke_val"):
            raise ValueError("Validation export requires complete development validation records")
        if formal and (len(records) != 320 or metrics.get("split") != "val"):
            raise ValueError("Formal validation requires exactly 320 development frames")
        seen = set()
        for record in records:
            uid = (record["scene_id"], record.get("camera", "realsense"), record["frame_id"])
            if uid in seen:
                raise ValueError("Duplicate frame in committed validation")
            seen.add(uid)
            validation_rows.append(dict(record, global_step=int(key), classification="committed",
                                        source_sha256=reference["sha256"], source_path=str(path)))
        validation_summary.append(dict(global_step=int(key), frame_count=len(records),
            source_sha256=reference["sha256"], metrics={k: v for k, v in metrics.items() if k != "frames"}))
    if "completed_validation_nodes" in analysis:
        effective = (analysis.get("budget") or {}).get("effective_batch_size")
        expected_nodes = [dict(step=r["global_step"], frame_count=r["frame_count"],
                               effective_samples=r["global_step"] * effective)
                          for r in sorted(validation_summary, key=lambda r: r["global_step"])]
        if analysis["completed_validation_nodes"] != expected_nodes:
            raise ValueError("Completed validation nodes differ from verified result records")

    resource_rows = []
    paths = [_safe_path(root, p) for p in (root / "attempts").glob("*/resources.jsonl")]
    paths += [Path(p).resolve() for p in resource_logs]
    for path in sorted(set(paths)):
        raw = snapshot(path, "resource_snapshot")
        for index, line in enumerate(raw.splitlines(keepends=True)):
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("Incomplete resource record")
                row = _json(line)
                if not isinstance(row, dict):
                    raise ValueError("Resource record must be an object")
                resource_rows.append(dict(row, source_path=str(path), source_line=index + 1,
                                          classification="observation_not_committed_progress"))
            except (ValueError, TypeError) as error:
                gaps.append(dict(path=str(path), line=index + 1, reason=str(error)))

    event_rows, event_ends, event_ids = [], {}, set()
    for segment in ledger.get("events_lineage", []):
        path = _safe_path(root, segment["path"])
        if path in event_ends or path in ends:
            raise ValueError("Repeated or overlapping event lineage segment")
        raw = snapshot(path, "committed_event_prefix", segment["end_offset"], segment["sha256"])
        event_ends[path] = segment
        for index, line in enumerate(raw.splitlines(keepends=True)):
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete committed event record")
            row = _json(line)
            if not isinstance(row, dict) or row.get("attempt_id") != segment["attempt_id"]:
                raise ValueError("Event lineage/attempt identity mismatch")
            if not isinstance(row.get("event_id"), str) or row["event_id"] in event_ids:
                raise ValueError("Missing or duplicate committed event ID")
            event_ids.add(row["event_id"])
            event_rows.append(dict(row, source_path=str(path), source_line=index + 1,
                                   classification="committed_event"))
    for path in sorted((root / "attempts").glob("*/events.jsonl")):
        path = _safe_path(root, path)
        raw = snapshot(path, "event_snapshot")
        segment = event_ends.get(path)
        offset = segment["end_offset"] if segment else 0
        if segment and hashlib.sha256(raw[:offset]).hexdigest() != segment["sha256"]:
            raise ValueError("Committed event prefix changed during export")
        line_offset = raw[:offset].count(b"\n")
        for index, line in enumerate(raw[offset:].splitlines(keepends=True), start=line_offset):
            end = offset + len(line)
            discarded = any(lo <= offset and end <= hi for lo, hi in abandoned.get(path, []))
            category = "abandoned_event" if discarded else (
                "live_uncommitted_event" if path.parent.name == live_attempt_id else "uncommitted_event")
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("Incomplete event record")
                row = _json(line)
                if not isinstance(row, dict):
                    raise ValueError("Event record must be an object")
                event_rows.append(dict(row, source_path=str(path), source_line=index + 1,
                                       classification=category))
            except (ValueError, TypeError) as error:
                gaps.append(dict(path=str(path), line=index + 1, reason=str(error)))
            offset = end

    summaries = {}
    for category in ("committed", "live_uncommitted", "uncommitted", "abandoned"):
        rows = [r for r in updates if r["classification"] == category]
        durations = [r["elapsed_seconds"] for r in rows
                     if isinstance(r.get("elapsed_seconds"), (int, float)) and r["elapsed_seconds"] > 0]
        measured = [r for r in rows if isinstance(r.get("elapsed_seconds"), (int, float))
                    and r["elapsed_seconds"] > 0]
        elapsed = sum(durations)
        samples = sum(len(r["frames"]) for r in rows)
        losses = {}
        for key in sorted({k for r in rows for k in r.get("loss", {})}):
            values = [r["loss"][key] for r in rows if isinstance(r.get("loss", {}).get(key), (int, float))]
            if values:
                losses[key] = dict(mean=sum(values) / len(values), last=values[-1], count=len(values),
                                   aggregation="unweighted mean of logged attempt reductions")
        rates = [r["lr"] for r in rows if isinstance(r.get("lr"), (int, float))]
        measured_points = sum(stat.get("point_count", 0) for r in measured for stat in r.get("frame_stats", []))
        measured_voxels = sum(stat.get("voxel_count", 0) for r in measured for stat in r.get("frame_stats", []))
        summaries[category] = dict(attempts=len(rows), successful_updates=sum(r["updated"] for r in rows),
            skipped_attempts=sum(not r["updated"] for r in rows), consumed_sample_presentations=samples,
            credited_sample_presentations=sum(len(r["frames"]) for r in rows if r["updated"]),
            measured_attempts=len(measured), training_block_seconds=elapsed,
            frames_per_training_second=sum(len(r["frames"]) for r in measured) / elapsed if elapsed else None,
            successful_updates_per_training_second=sum(r["updated"] for r in measured) / elapsed if elapsed else None,
            points_per_training_second=measured_points / elapsed if elapsed and measured_points else None,
            voxels_per_training_second=measured_voxels / elapsed if elapsed and measured_voxels else None,
            loss=losses, lr=dict(first=rates[0], last=rates[-1], minimum=min(rates), maximum=max(rates)) if rates else None)
    target_samples = (analysis.get("budget") or {}).get("target_samples")
    summary = dict(schema_version="suction20.analysis.v1", checkpoint_role=checkpoint_role,
        checkpoint=entry, identity=analysis.get("identity"), budget=analysis.get("budget"),
        counters=derived_counters, sample_progress=dict(completed=credited_frames, target=target_samples,
            fraction=credited_frames / target_samples if target_samples else None),
        categories=summaries, validations=validation_summary,
        best=analysis.get("best"), resource_sample_count=len(resource_rows), event_count=len(event_rows), gaps=gaps,
        limitations=["Manifest analysis metadata is publication authority; no checkpoint pickle was loaded.",
            "Training-block throughput excludes logging/checkpoint/validation/backup/pause overhead.",
            "Raw tails, event/resource observations and abandoned work never count as committed training progress.",
            "A live attempt label is supplied by the caller and does not prove process liveness.",
            "Hard crashes may leave unrecorded physical work; incomplete tail rows are listed as gaps.",
            "Device power/utilization and sampled memory are observations, not per-run exclusive cost.",
            "Validation aggregates are the original stored protocol values, not recomputed pooled metrics."])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, rows in (("updates", updates), ("frames", frame_rows), ("resources", resource_rows),
                           ("validation_frames", validation_rows), ("events", event_rows)):
            _csv(temporary / f"{name}.csv", rows)
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        outputs = [dict(path=p.name, size_bytes=p.stat().st_size, sha256=_sha(p)) for p in sorted(temporary.iterdir())]
        export_manifest = dict(schema_version="suction20.export.v1", run_dir=str(root),
            checkpoint_role=checkpoint_role, checkpoint_sha256=entry["sha256"],
            manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), live_attempt_id=live_attempt_id,
            created_utc=datetime.now(timezone.utc).isoformat(), sources=sources, outputs=outputs,
            exporter_sha256=_sha(Path(__file__)))
        (temporary / "export-manifest.json").write_text(json.dumps(export_manifest, indent=2, sort_keys=True) + "\n")
        # The destination must not already exist; rename publishes the complete snapshot.
        if output.exists():
            raise ValueError("Export output appeared during analysis")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary
