import json
from pathlib import Path
import numpy as np
import pytest

from suction20.core import compare_labels, validate_pair, read_label
from suction20.patch_official import patch_seal_source, create_variant, create_wrench_g0_variant
from suction20.data_utils import pack_dense


def write_label(path, points, scores, indices=None, collision=None):
    kwargs = dict(points=np.asarray(points, np.float32),
                  normals=np.tile(np.array([[0., 0., 1.]], np.float32), (len(points), 1)),
                  scores=np.asarray(scores, np.float32))
    if indices is not None:
        kwargs['indices'] = np.asarray(indices, np.int32)
    if collision is not None:
        kwargs['collision'] = np.asarray(collision, bool)
    np.savez(path, **kwargs)


def test_patch_strict_and_manifest(tmp_path):
    source = tmp_path / 'seal_annotate.py'
    source.write_text(
        'if __name__ == "__main__":\n'
        '    radius = 0.01\n'
        '    num_split = 72\n'
        '    idx = np.where((dist > radius - 0.001) & (dist < radius + 0.001))[0]\n'
        '    for i in range(88):\n'
        '        pass\n'
    )
    out = tmp_path / 'patch.py'
    info = create_variant(source, out, 20, [2, 0, 2])
    text = out.read_text()
    assert 'radius = 0.020000' in text
    assert 'radius - (0.1 * radius)' in text
    assert 'radius + (0.1 * radius)' in text
    assert 'for i in [0, 2]:' in text
    assert info['ring_half_width_m'] == pytest.approx(0.002)
    g0 = create_variant(source, tmp_path / 'seal_r10.py', 10, [2, 0])
    assert g0['ring_half_width_m'] == pytest.approx(0.001)
    assert '(dist > radius - 0.001) & (dist < radius + 0.001)' in (
        tmp_path / 'seal_r10.py'
    ).read_text()
    assert 'radius = 0.01' in source.read_text()
    assert info['radius_mm'] == 20
    with pytest.raises(RuntimeError):
        patch_seal_source('radius = 0.015', .020, [0])


def test_candidate_order_does_not_change_g0_diff(tmp_path):
    a, b = tmp_path / 'ref.npz', tmp_path / 'new.npz'
    points = np.array([[0,0,0], [0.01,0,0], [0.02,0,0]])
    write_label(a, points, [0, .8, .2], indices=[7, 8, 9], collision=[False, False, True])
    write_label(b, points[[2,0,1]], [.2, 0, .8], indices=[9,7,8], collision=[True, False, False])
    result = compare_labels(a, b)
    assert result['match_method'] == 'candidate_index'
    assert result['matched_count'] == 3
    assert result['score_abs_max'] == 0
    assert result['collision_disagreement_rate'] == 0
    assert read_label(a)['scores'][0] == 0  # keep negative sample


def test_candidate_missing_fails_same_set(tmp_path):
    a, b = tmp_path / 'ref.npz', tmp_path / 'new.npz'
    points = np.array([[0,0,0], [.01,0,0]])
    write_label(a, points, [0, 1], indices=[3, 5])
    write_label(b, points[:1], [.2], indices=[3])
    assert validate_pair(a, b)['same_candidate_set'] is False


def test_coordinate_matching_when_release_has_no_indices(tmp_path):
    a, b = tmp_path / 'ref.npz', tmp_path / 'new.npz'
    points = np.array([[0,0,0], [.01,0,0], [.02,0,0]])
    write_label(a, points, [0, .4, .7])
    write_label(b, points[[1,2,0]], [.4, .7, 0], indices=[20,30,10])
    val = compare_labels(a, b, max_distance_mm=.01)
    assert val['match_method'] == 'nearest_coordinate'
    assert val['matched_count'] == 3
    assert val['score_abs_mean'] == 0


def test_pack_dense(tmp_path):
    dense = tmp_path / '000'
    dense.mkdir()
    np.save(dense / 'points.npy', np.asarray([[0,0,0], [1,0,0]], np.float32))
    np.save(dense / 'normals.npy', np.asarray([[0,0,1], [0,0,1]], np.float32))
    dest = tmp_path / '000.npz'
    result = pack_dense(dense, dest)
    with np.load(dest) as f:
        assert f['points'].shape == (2, 3)
        assert f['normals'].shape == (2, 3)
    assert result['count'] == 2


def test_reject_invalid_lengths(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, points=np.zeros((2,3)), normals=np.zeros((2,3)), scores=np.array([.1]))
    with pytest.raises(ValueError):
        read_label(path)

def test_official_wrench_g0_replay_keeps_sqrt2_formula(tmp_path):
    upstream = tmp_path / "wrench_annotate.py"
    source = (
        "radius = 0.01\n"
        "wrench_thre = k * radius * np.pi * np.sqrt(2)\n"
        "if __name__ == '__main__':\n"
        "    for scene_idx in range(190):\n"
        "        pass\n"
    )
    upstream.write_text(source, encoding="utf-8")
    target = tmp_path / "wrench_r10_official.py"
    meta = create_wrench_g0_variant(upstream, target, [3, 0, 3])
    patched = target.read_text(encoding="utf-8")
    assert meta["scene_ids"] == [0, 3]
    assert "for scene_idx in [0, 3]:" in patched
    assert "wrench_thre = k * radius * np.pi * np.sqrt(2)" in patched
    assert "radius = 0.01" in patched
    assert upstream.read_text(encoding="utf-8") == source


def test_wrench_g0_source_without_historical_formula_is_rejected(tmp_path):
    upstream = tmp_path / "wrench_annotate.py"
    upstream.write_text("radius = 0.01\nfor scene_idx in range(190):\n    pass\n")
    with pytest.raises(RuntimeError, match=r"sqrt\(2\)"):
        create_wrench_g0_variant(upstream, tmp_path / "patched.py", [0])
