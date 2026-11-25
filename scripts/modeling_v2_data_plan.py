#!/usr/bin/env python3
"""Create a frozen P0 asset plan or a metadata-only file preflight.

This standard-library tool never converts training data. It only selects frames,
checks selected raw paths with stat(), and records explicit provenance metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import stat
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "plan_v1"
TOOL_VERSION = "1.0"
FRAME_RE = re.compile(r"^(\d{6})-depth\.png$")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/modeling_v2/data_protocol_v1.json"
ASSET_COLUMNS = [
    "dataset", "split", "scene_id", "camera", "frame_id", "role",
    "relative_path", "path", "status", "file_size_bytes", "error",
]


def load_protocol(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("schema") != SCHEMA or protocol.get("training_ready") is not False:
        raise ValueError("config must declare schema=plan_v1 and training_ready=false")
    sn = protocol["suctionnet"]
    cp = protocol["clearpose"]
    if [sn[name] for name in ("train_scenes", "dev_scenes", "reserved_test_scenes")] != [[0, 89], [90, 99], [100, 189]]:
        raise ValueError("plan_v1 freezes SuctionNet scenes to 0..89 / 90..99 / 100..189")
    if sn["cameras"] != ["kinect", "realsense"] or sn["view_ids"] != list(range(0, 256, 16)):
        raise ValueError("plan_v1 requires two cameras and views 0,16,...,240")
    expected_cp = {"set1": [1, 2, 3, 4], **{f"set{i}": [1, 2, 3, 4, 5] for i in range(4, 8)}}
    expected_dev = {"set1/scene4", *(f"set{i}/scene5" for i in range(4, 8))}
    if cp["official_train_scenes"] != expected_cp or set(cp["dev_scenes"]) != expected_dev:
        raise ValueError("plan_v1 requires the frozen 19-train/5-dev ClearPose scene split")
    if cp["max_camera_frames_per_scene"] != 96:
        raise ValueError("plan_v1 freezes the ClearPose per-scene cap to 96")
    if sn["audit_reserved_test"] or cp["audit_reserved_test"]:
        raise ValueError("reserved test data must not be selected or audited")
    return protocol


def frame_header(dataset: str, split: str, scene_id: str, camera: str, frame_id: str) -> dict:
    return {
        "schema": SCHEMA,
        "artifact_kind": "asset_plan_and_preflight",
        "training_ready": False,
        "dataset": dataset,
        "split": split,
        "scene_id": scene_id,
        "camera": camera,
        "frame_id": frame_id,
        "split_group": f"{dataset}:{scene_id}",
    }


def suctionnet_scenes(protocol: dict) -> list[dict]:
    rows = []
    sn = protocol["suctionnet"]
    for split, field in (("train", "train_scenes"), ("dev", "dev_scenes"), ("reserved_test", "reserved_test_scenes")):
        first, last = sn[field]
        for scene in range(first, last + 1):
            rows.append({"scene_id": f"scene_{scene:04d}", "split": split, "selected_camera_frame_budget": 0 if split == "reserved_test" else len(sn["cameras"]) * len(sn["view_ids"])})
    return rows


def clearpose_scenes(protocol: dict) -> list[dict]:
    cp = protocol["clearpose"]
    dev = set(cp["dev_scenes"])
    rows = []
    for set_id, scene_ids in cp["official_train_scenes"].items():
        for scene in scene_ids:
            scene_id = f"{set_id}/scene{scene}"
            rows.append({"scene_id": scene_id, "split": "dev" if scene_id in dev else "train", "selected_camera_frame_budget": cp["max_camera_frames_per_scene"]})
    return rows


def suctionnet_frame_plan(protocol: dict) -> list[dict]:
    frames = []
    for scene in suctionnet_scenes(protocol):
        if scene["split"] == "reserved_test":
            continue
        for camera in protocol["suctionnet"]["cameras"]:
            for view in protocol["suctionnet"]["view_ids"]:
                frame_id = f"{view:04d}"
                prefix = Path("scenes") / scene["scene_id"] / camera
                row = frame_header("suctionnet", scene["split"], scene["scene_id"], camera, frame_id)
                row["asset_relative_paths"] = {
                    "rgb": (prefix / "rgb" / f"{frame_id}.png").as_posix(),
                    "depth": (prefix / "depth" / f"{frame_id}.png").as_posix(),
                    "label": (prefix / "label" / f"{frame_id}.png").as_posix(),
                    "metadata": (prefix / "meta" / f"{frame_id}.mat").as_posix(),
                }
                frames.append(row)
    return frames


def center_quantile_indices(n: int, cap: int) -> list[int]:
    """Equivalent to floor((i + 0.5) * n / k), using exact integers."""
    k = min(n, cap)
    return [((2 * i + 1) * n) // (2 * k) for i in range(k)] if k else []


def new_output(path: Path) -> Path:
    path = path.expanduser().absolute()
    path.mkdir(parents=True, exist_ok=False)
    (path / "INCOMPLETE").write_text("Report generation has not completed.\n", encoding="utf-8")
    return path


def write_atomic(path: Path, writer) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer(handle)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    def writer(handle):
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_atomic(path, writer)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    def writer(handle):
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_atomic(path, writer)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    def writer(handle):
        output = csv.DictWriter(handle, fieldnames=columns)
        output.writeheader()
        output.writerows(rows)
    write_atomic(path, writer)


def write_plan_files(output: Path, protocol: dict, config_path: Path, git_revision: str | None) -> dict:
    plan = {
        "schema": SCHEMA,
        "tool_version": TOOL_VERSION,
        "tool_git_revision": git_revision,
        "config_path": str(config_path.absolute()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "asset_plan_and_preflight",
        "training_ready": False,
        "note": "This is an asset selection plan, not a training manifest. ClearPose frame IDs require source directory discovery.",
        "suctionnet": protocol["suctionnet"],
        "clearpose": protocol["clearpose"],
        "preflight_scope": protocol["preflight"],
        "provenance_note": "Release names are declarations, not verified contents. File sizes do not establish content integrity.",
    }
    write_json(output / "plan.json", plan)
    write_json(output / "config_snapshot.json", protocol)
    columns = ["scene_id", "split", "selected_camera_frame_budget"]
    write_csv(output / "scenes_suctionnet.csv", suctionnet_scenes(protocol), columns)
    write_csv(output / "scenes_clearpose.csv", clearpose_scenes(protocol), columns)
    return plan


class Preflight:
    def __init__(self):
        self.assets: dict[tuple[str, str], dict] = {}
        self.issues: list[dict] = []

    def asset(self, root: Path, relative: Path | str, context: dict, role: str) -> dict:
        path = root / relative
        key = (context["dataset"], str(path))
        if key in self.assets:
            return self.assets[key]
        row = {name: context.get(name, "") for name in ("dataset", "split", "scene_id", "camera", "frame_id")}
        row.update({"role": role, "relative_path": Path(relative).as_posix(), "path": str(path), "status": "ok", "file_size_bytes": None, "error": ""})
        try:
            info = path.stat()
            row["file_size_bytes"] = info.st_size
            if not stat.S_ISREG(info.st_mode):
                row["status"] = "not_regular_file"
            elif info.st_size == 0:
                row["status"] = "empty_file"
        except FileNotFoundError:
            row["status"] = "missing"
        except OSError as error:
            row["status"] = "stat_error"
            row["error"] = str(error)
        self.assets[key] = row
        return row

    def issue(self, dataset: str, scene_id: str, severity: str, code: str, **details) -> None:
        self.issues.append({"dataset": dataset, "scene_id": scene_id, "severity": severity, "code": code, **details})


def audit_suctionnet(root: Path, protocol: dict, audit: Preflight) -> list[dict]:
    rows = suctionnet_frame_plan(protocol)
    for row in rows:
        shared_context = {**row, "frame_id": ""}
        prefix = Path("scenes") / row["scene_id"] / row["camera"]
        shared = [audit.asset(root, prefix / name, shared_context, role) for name, role in (
            ("camK.npy", "intrinsics"), ("camera_poses.npy", "camera_poses"), ("cam0_wrt_table.npy", "table_alignment"))]
        scene_context = {**shared_context, "camera": ""}
        shared.append(audit.asset(root, Path("scenes") / row["scene_id"] / "object_id_list.txt", scene_context, "scene_object_ids"))
        selected = [audit.asset(root, relative, row, role) for role, relative in row["asset_relative_paths"].items()]
        row["asset_status"] = "ok" if all(item["status"] == "ok" for item in selected + shared) else "incomplete"
        row["asset_checks"] = selected
        row["shared_asset_paths"] = [item["relative_path"] for item in shared]
        row["source_root"] = str(root)
    return rows


def audit_clearpose(root: Path, protocol: dict, audit: Preflight) -> tuple[list[dict], list[dict]]:
    rows, selection_rows = [], []
    cap = protocol["clearpose"]["max_camera_frames_per_scene"]
    for scene in clearpose_scenes(protocol):
        scene_id = scene["scene_id"]
        folder = root / scene_id
        context = frame_header("clearpose", scene["split"], scene_id, "realsense_l515", "")
        metadata = audit.asset(root, Path(scene_id) / "metadata.mat", context, "scene_metadata")
        frame_ids = []
        ignored_names = 0
        try:
            if not folder.is_dir():
                audit.issue("clearpose", scene_id, "error", "scene_directory_missing", path=str(folder))
            else:
                for entry in folder.iterdir():
                    if not entry.name.endswith("-depth.png"):
                        continue
                    match = FRAME_RE.fullmatch(entry.name)
                    if match:
                        frame_ids.append(match.group(1))
                    else:
                        ignored_names += 1
        except OSError as error:
            audit.issue("clearpose", scene_id, "error", "scene_directory_read_error", path=str(folder), error=str(error))
        frame_ids.sort(key=int)
        selected_ids = [frame_ids[index] for index in center_quantile_indices(len(frame_ids), cap)]
        if not selected_ids:
            audit.issue("clearpose", scene_id, "error", "no_eligible_depth_frames", available_frames=0, budget=cap)
        elif len(selected_ids) < cap:
            audit.issue("clearpose", scene_id, "warning", "budget_reduced_without_duplication", available_frames=len(frame_ids), selected_frames=len(selected_ids), budget=cap)
        if ignored_names:
            audit.issue("clearpose", scene_id, "warning", "ignored_nonstandard_depth_names", count=ignored_names)
        selection_rows.append({**scene, "available_depth_frames": len(frame_ids), "selected_camera_frames": len(selected_ids)})
        for frame_id in selected_ids:
            row = frame_header("clearpose", scene["split"], scene_id, "realsense_l515", frame_id)
            row["asset_relative_paths"] = {role: (Path(scene_id) / f"{frame_id}-{suffix}.png").as_posix() for role, suffix in (
                ("rgb", "color"), ("depth", "depth"), ("gt_depth", "depth_true"), ("label", "label"))}
            selected = [audit.asset(root, relative, row, role) for role, relative in row["asset_relative_paths"].items()]
            row["asset_status"] = "ok" if all(item["status"] == "ok" for item in selected + [metadata]) else "incomplete"
            row["asset_checks"] = selected
            row["shared_asset_paths"] = [metadata["relative_path"]]
            row["source_root"] = str(root)
            rows.append(row)
    return rows, selection_rows


def run_audit(args, protocol: dict, output: Path) -> int:
    audit = Preflight()
    frame_sets = {}
    source_roots = {}
    if args.suctionnet_root is not None:
        root = args.suctionnet_root.expanduser().absolute()
        source_roots["suctionnet"] = str(root)
        frame_sets["suctionnet"] = audit_suctionnet(root, protocol, audit)
    if args.clearpose_root is not None:
        root = args.clearpose_root.expanduser().absolute()
        source_roots["clearpose"] = str(root)
        frames, selection = audit_clearpose(root, protocol, audit)
        frame_sets["clearpose"] = frames
        write_csv(output / "selection_clearpose.csv", selection, ["scene_id", "split", "selected_camera_frame_budget", "available_depth_frames", "selected_camera_frames"])
    for dataset, frames in frame_sets.items():
        write_jsonl(output / f"frames_{dataset}.jsonl", frames)
    assets = list(audit.assets.values())
    counts = Counter(item["status"] for item in assets)
    invalid_assets = sum(count for status, count in counts.items() if status != "ok")
    errors = sum(issue["severity"] == "error" for issue in audit.issues)
    incomplete = bool(invalid_assets or errors)
    summaries = {}
    for dataset, frames in frame_sets.items():
        selected_by_split = Counter(row["split"] for row in frames)
        complete_by_split = Counter(row["split"] for row in frames if row["asset_status"] == "ok")
        summaries[dataset] = {"selected_camera_frames": len(frames), "selected_by_split": dict(selected_by_split), "file_preflight_complete_camera_frames": sum(complete_by_split.values()), "file_preflight_complete_by_split": dict(complete_by_split)}
    report = {
        "schema": SCHEMA,
        "artifact_kind": "asset_plan_and_preflight",
        "training_ready": False,
        "status": "incomplete_assets" if incomplete else "file_preflight_passed",
        "exit_code": 2 if incomplete else 0,
        "source_roots": source_roots,
        "sources_not_audited": [dataset for dataset in ("suctionnet", "clearpose") if dataset not in source_roots],
        "reserved_test_audited": False,
        "datasets": summaries,
        "unique_asset_files": len(assets),
        "asset_status_counts": dict(counts),
        "missing_asset_files": counts["missing"],
        "invalid_asset_files": invalid_assets,
        "scene_errors": errors,
        "issues": audit.issues,
        "preflight_scope": protocol["preflight"],
        "next_gate": "Decode and semantic geometry/pose/CAD/teacher validation are still required before constructing a training manifest.",
    }
    write_csv(output / "audit.csv", assets, ASSET_COLUMNS)
    write_json(output / "audit.json", report)
    print(json.dumps({"schema": SCHEMA, "status": report["status"], "training_ready": False, "datasets": summaries, "missing_asset_files": counts["missing"], "invalid_asset_files": invalid_assets, "scene_errors": errors, "output": str(output)}, ensure_ascii=False))
    return report["exit_code"]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are never overwritten.")
        command.add_argument("--git-revision", default=None, help="Explicit code revision to record, if known.")
        if name == "audit":
            command.add_argument("--suctionnet-root", type=Path)
            command.add_argument("--clearpose-root", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "audit" and args.suctionnet_root is None and args.clearpose_root is None:
        cli.error("audit requires --suctionnet-root and/or --clearpose-root")
    try:
        protocol = load_protocol(args.config)
        output = new_output(args.output)
        write_plan_files(output, protocol, args.config, args.git_revision)
        if args.command == "plan":
            write_jsonl(output / "frames_suctionnet.jsonl", suctionnet_frame_plan(protocol))
            print(json.dumps({"schema": SCHEMA, "artifact_kind": "asset_plan_and_preflight", "training_ready": False, "suctionnet_selected_camera_frames": protocol["suctionnet"]["expected_selected_camera_frames"], "clearpose_camera_frame_budget": protocol["clearpose"]["expected_camera_frame_budget"], "output": str(output)}, ensure_ascii=False))
            code = 0
        else:
            code = run_audit(args, protocol, output)
        (output / "INCOMPLETE").unlink()
        return code
    except FileExistsError as error:
        print(f"ERROR: output already exists; choose a new --output directory: {error.filename}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
