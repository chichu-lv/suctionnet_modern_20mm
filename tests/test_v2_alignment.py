"""Regression tests for V2 data requirements: Wrench and instance identity."""
import json

import numpy as np
import pytest
from PIL import Image

from suction20.full_data import (
    DEFAULT_CONFIG, config_from_file, wrench_scores, resolve_instance_ids,
)


def records(*object_ids):
    return [
        {"instance_row": row, "obj_id": int(obj)}
        for row, obj in enumerate(object_ids)
    ]


def resolve(class_ids, items, tmp_path, root=None, image_shape=(2, 3)):
    cfg = dict(DEFAULT_CONFIG)
    cfg["instance_mask_root"] = str(root) if root is not None else None
    source_indices = np.arange(len(class_ids), dtype=np.int32)
    return resolve_instance_ids(
        np.asarray(class_ids, dtype=np.int32), items, tmp_path,
        scene_id=0, frame_id=0, cfg=cfg,
        source_indices=source_indices, image_shape=image_shape,
    )


def test_v2_default_wrench_threshold_is_r_k_pi():
    assert "wrench_threshold_factor" not in DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    point = np.array([[0.0, 0.0, 0.0]], np.float64)
    normal = np.array([[0.0, 0.0, 1.0]], np.float64)
    com = np.array([0.01, 0.0, 0.0], np.float64)
    actual = wrench_scores(point, normal, com, 1.0, cfg)[0]
    torque = 0.01 * cfg["gravity_m_s2"]
    expected = 1.0 - min(1.0, torque / (cfg["wrench_k"] * cfg["radius_m"] * np.pi))
    assert actual == pytest.approx(expected, abs=1e-6)


def test_config_rejects_legacy_sqrt2_factor(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"wrench_threshold_factor": 2**0.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        config_from_file(path)


def test_unique_classes_map_to_scene_rows_not_class_ids(tmp_path):
    got = resolve([4, 8, 0, 4, 8, 0], records(3, 7), tmp_path)
    assert got.tolist() == [1, 2, 0, 1, 2, 0]


def test_repeated_classes_require_instance_mask(tmp_path):
    with pytest.raises(ValueError, match="instance_mask_root"):
        resolve([4, 4, 0, 4, 4, 0], records(3, 3), tmp_path)


def test_repeated_classes_resolve_from_external_row_order_mask(tmp_path):
    root = tmp_path / "instance_masks"
    folder = root / "scene_0000" / "realsense"
    folder.mkdir(parents=True)
    mask = np.array([[1, 2, 0], [1, 2, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(folder / "0000.png")

    got = resolve([4, 4, 0, 4, 4, 0], records(3, 3), tmp_path, root)
    assert got.tolist() == [1, 2, 0, 1, 2, 0]


def test_instance_mask_rejects_wrong_class_assignment(tmp_path):
    root = tmp_path / "instances"
    folder = root / "scene_0000" / "realsense"
    folder.mkdir(parents=True)
    Image.fromarray(np.array([[2, 1, 0], [2, 1, 0]], dtype=np.uint8)).save(folder / "0000.png")
    with pytest.raises(ValueError, match="does not match class"):
        resolve([4, 8, 0, 4, 8, 0], records(3, 7), tmp_path, root)


def test_instance_mask_rejects_missing_foreground(tmp_path):
    root = tmp_path / "instances"
    folder = root / "scene_0000" / "realsense"
    folder.mkdir(parents=True)
    Image.fromarray(np.array([[1, 0, 0], [1, 2, 0]], dtype=np.uint8)).save(folder / "0000.png")
    with pytest.raises(ValueError, match="foreground"):
        resolve([4, 8, 0, 4, 8, 0], records(3, 7), tmp_path, root)
