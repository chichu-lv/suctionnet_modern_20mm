"""Behavioral regression checks for the metadata-only Model 2 data planner."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/modeling_v2_data_plan.py"


def run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def clearpose_root(tmp_path: Path) -> Path:
    """Nonempty placeholders intentionally test stat-only preflight, not decoding."""
    root = tmp_path / "clearpose"
    scenes = [f"set1/scene{i}" for i in range(1, 5)]
    scenes += [f"set{set_id}/scene{i}" for set_id in range(4, 8) for i in range(1, 6)]
    for scene in scenes:
        folder = root / scene
        folder.mkdir(parents=True)
        (folder / "metadata.mat").write_bytes(b"fixture")
        # Sparse IDs and reverse creation order prevent assumptions that IDs are
        # consecutive or directory iteration order is the sampling order.
        ids = [10 + 2 * i for i in range(100)] if scene == "set1/scene1" else [5, 17, 32]
        for frame_id in reversed(ids):
            for suffix in ("color", "depth", "depth_true", "label"):
                (folder / f"{frame_id:06d}-{suffix}.png").write_bytes(b"fixture")
    return root


def test_plan_cli_freezes_scene_split_and_double_camera_budget(tmp_path: Path) -> None:
    output = tmp_path / "plan"
    result = run_cli("plan", "--output", output)
    assert result.returncode == 0, result.stderr
    frames = read_jsonl(output / "frames_suctionnet.jsonl")
    assert Counter(frame["split"] for frame in frames) == {"train": 2880, "dev": 320}
    identity = {(frame["scene_id"], frame["camera"], frame["frame_id"]) for frame in frames}
    assert len(identity) == len(frames) == 3200
    assert {frame["scene_id"] for frame in frames} == {f"scene_{i:04d}" for i in range(100)}
    assert all(frame["schema"] == "plan_v1" and frame["training_ready"] is False for frame in frames)
    assert all((int(frame["scene_id"][-4:]) < 90) == (frame["split"] == "train") for frame in frames)
    with (output / "scenes_suctionnet.csv").open(newline="", encoding="utf-8") as handle:
        reserved = [row for row in csv.DictReader(handle) if row["split"] == "reserved_test"]
    assert len(reserved) == 90
    assert all(row["selected_camera_frame_budget"] == "0" for row in reserved)


def test_clearpose_quantiles_short_sequences_and_shared_metadata(clearpose_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "audit"
    result = run_cli("audit", "--clearpose-root", clearpose_root, "--output", output)
    assert result.returncode == 0, result.stderr
    frames = read_jsonl(output / "frames_clearpose.jsonl")
    full_scene_ids = [int(row["frame_id"]) for row in frames if row["scene_id"] == "set1/scene1"]
    # For 100 inputs and 96 center-quantile picks, these four source indices are
    # skipped; this also verifies sparse source IDs are preserved, not renumbered.
    assert full_scene_ids == [10 + 2 * i for i in range(100) if i not in {12, 37, 62, 87}]
    counts = Counter(row["scene_id"] for row in frames)
    assert len(counts) == 24
    assert counts["set1/scene1"] == 96
    assert all(count == 3 for scene, count in counts.items() if scene != "set1/scene1")
    assert len({(row["scene_id"], row["frame_id"]) for row in frames}) == len(frames) == 165
    with (output / "audit.csv").open(newline="", encoding="utf-8") as handle:
        metadata = [row for row in csv.DictReader(handle) if row["role"] == "scene_metadata"]
    assert len(metadata) == len({row["path"] for row in metadata}) == 24
    assert all(row["frame_id"] == "" for row in metadata)
    report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert report["status"] == "file_preflight_passed"
    assert report["training_ready"] is False
    assert report["reserved_test_audited"] is False
    assert len([issue for issue in report["issues"] if issue["code"] == "budget_reduced_without_duplication"]) == 23


def test_missing_selected_gt_fails_without_replacing_the_frame(clearpose_root: Path, tmp_path: Path) -> None:
    baseline = tmp_path / "before"
    assert run_cli("audit", "--clearpose-root", clearpose_root, "--output", baseline).returncode == 0
    original = read_jsonl(baseline / "frames_clearpose.jsonl")
    chosen = original[0]
    gt_asset = next(asset for asset in chosen["asset_checks"] if asset["role"] == "gt_depth")
    Path(gt_asset["path"]).unlink()

    output = tmp_path / "after"
    result = run_cli("audit", "--clearpose-root", clearpose_root, "--output", output)
    assert result.returncode == 2, result.stdout
    after = read_jsonl(output / "frames_clearpose.jsonl")
    assert [(row["scene_id"], row["frame_id"]) for row in after] == [(row["scene_id"], row["frame_id"]) for row in original]
    changed = next(row for row in after if (row["scene_id"], row["frame_id"]) == (chosen["scene_id"], chosen["frame_id"]))
    assert changed["asset_status"] == "incomplete"
    report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert report["status"] == "incomplete_assets"
    assert report["missing_asset_files"] == 1
    assert report["datasets"]["clearpose"]["selected_camera_frames"] == 165
    assert report["datasets"]["clearpose"]["file_preflight_complete_camera_frames"] == 164


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "owned_by_user.txt"
    sentinel.write_text("Keep this existing result.\n", encoding="utf-8")
    result = run_cli("plan", "--output", output)
    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "Keep this existing result.\n"
    assert list(output.iterdir()) == [sentinel]
