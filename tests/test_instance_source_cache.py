"""Offline source mirrors preserve real reader semantics and never fill online."""
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import build_instance_source_cache as builder
from suction20.instance_data import (InstanceDataset, load_observation, save_observation,
                                    _read_npz)
from suction20.instance_selection_cache import (reader_cache_stats, reset_reader_cache_stats,
                                              source_cache_location)
from test_instance_loading import _source
from test_instance_training import assert_nested_equal


def _build(source, output, route='a', **kwargs):
    return builder.build_instance_source_cache(source.manifest, output, route=route,
        max_bytes=64 * 1024**2, training_queries=4, **kwargs)


@pytest.mark.parametrize('route,workers', [('a', 1), ('b', 2)])
def test_source_mmap_matches_original_and_keeps_manifest_identity(tmp_path, monkeypatch, route, workers):
    source = _source(tmp_path / 'source')
    identity = dict(source.identity)
    manifest_before = source.manifest.read_bytes()
    expected = [source.load_uncached(i, _compact_source=True, source_route=route) for i in range(len(source))]
    output = tmp_path / 'npy'
    result = _build(source, output, route, workers=workers)
    assert result['converted'] == 4 and result['validation_reads']['source_hits'] == 4
    monkeypatch.setenv('SUCTION_INSTANCE_SOURCE_CACHE_' + route.upper(), str(output))
    reset_reader_cache_stats()
    for index, before in enumerate(expected):
        actual = source.load_uncached(index, _compact_source=True, source_route=route)
        assert_nested_equal(actual, before)
        assert isinstance(actual['observation']['rgb'], np.memmap)
        assert isinstance(actual['labels']['points'], np.memmap)
        actual['observation']['rgb'][0, 0] = -7
        again = source.load_uncached(index, _compact_source=True, source_route=route)
        np.testing.assert_array_equal(again['observation']['rgb'], before['observation']['rgb'])
    stats = reader_cache_stats()
    assert stats['source_hits'] == 8 and stats['source_fallbacks'] == stats['npz_reads'] == 0
    assert source.identity == identity and source.manifest.read_bytes() == manifest_before
    # Public full reads retain complete fields and do not use the route mirror.
    public = source.load_uncached(0)
    assert not isinstance(public['observation']['rgb'], np.memmap)
    assert reader_cache_stats()['npz_reads'] == 2


def test_source_cache_resume_budget_and_partial_publication(tmp_path, monkeypatch):
    source = _source(tmp_path / 'source')
    output = tmp_path / 'npy'
    with pytest.raises(ValueError, match='No arrays were converted'):
        builder.build_instance_source_cache(source.manifest, output, route='a',
            max_bytes=1, training_queries=4)
    assert not (output / 'cache.json').exists() and not list(output.rglob('*.npy'))
    original = builder._convert_archive
    calls = []
    def interrupted(task):
        if calls:
            raise RuntimeError('interrupted conversion')
        calls.append(task)
        return original(task)
    monkeypatch.setattr(builder, '_convert_archive', interrupted)
    with pytest.raises(RuntimeError, match='interrupted conversion'):
        _build(source, output, resume=True)
    assert not (output / 'cache.json').exists()
    monkeypatch.setattr(builder, '_convert_archive', original)
    result = _build(source, output, resume=True)
    assert result['reused'] == 1 and result['converted'] == 3
    assert _build(source, output, resume=True)['converted'] == 0
    with pytest.raises(FileExistsError, match='--resume'):
        _build(source, output)


def test_source_cache_invalid_geometry_never_publishes(tmp_path):
    source = _source(tmp_path / 'source')
    path = source.root / source.records[0]['observation_npz']
    obs = load_observation(path)
    obs['query_points'] = obs['query_points'].copy()
    obs['query_points'][0, 0] += .001
    save_observation(path, obs)
    output = tmp_path / 'npy'
    with pytest.raises(ValueError, match='raw source pixels'):
        _build(source, output)
    assert not (output / 'cache.json').exists()


def test_source_replacement_and_npy_modification_fall_back_without_online_writes(tmp_path, monkeypatch):
    source = _source(tmp_path / 'source')
    output = tmp_path / 'npy'
    _build(source, output)
    monkeypatch.setenv('SUCTION_INSTANCE_SOURCE_CACHE_A', str(output))
    path = (source.root / source.records[0]['observation_npz']).resolve()
    config = json.loads((output / 'cache.json').read_text())
    location = source_cache_location(output / 'data', path, config['source_root'])
    rgb = location / 'rgb.npy'
    before = {p: p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}
    stat = rgb.stat()
    os.utime(rgb, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    reset_reader_cache_stats()
    assert not isinstance(source.load_uncached(0, _compact_source=True, source_route='a')['observation']['rgb'], np.memmap)
    assert reader_cache_stats()['source_fallbacks'] == reader_cache_stats()['npz_reads'] == 1
    os.utime(rgb, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement = path.with_suffix('.new')
    shutil.copy2(path, replacement)
    replacement.replace(path)  # Same size/mtime, different inode must invalidate.
    reset_reader_cache_stats()
    assert not isinstance(source.load_uncached(0, _compact_source=True, source_route='a')['observation']['rgb'], np.memmap)
    assert reader_cache_stats()['source_fallbacks'] == 1
    assert before == {p: p.stat().st_mtime_ns for p in before}


def test_nested_source_paths_with_equal_parent_and_stem_remain_distinct(tmp_path, monkeypatch):
    source = _source(tmp_path / 'source')
    rows = []
    for index, original in enumerate(source.records):
        row = dict(original)
        for name in ('observation_npz', 'label_cache'):
            path = source.root / row[name]
            destination = source.root / f'nested_{index}' / path.parent.name / 'same.npz'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            row[name] = str(destination.relative_to(source.root))
        row['query_cache'] = row['observation_npz']
        rows.append(row)
    source.manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    source = InstanceDataset(source.manifest, split='train', training_queries=4)
    output = tmp_path / 'npy'
    _build(source, output)
    monkeypatch.setenv('SUCTION_INSTANCE_SOURCE_CACHE_A', str(output))
    reset_reader_cache_stats()
    for index in range(2):
        actual = source.load_uncached(index, _compact_source=True, source_route='a')
        assert actual['record']['frame_id'] == index
        assert isinstance(actual['observation']['rgb'], np.memmap)
    assert reader_cache_stats()['source_hits'] == 4


def test_source_cache_preserves_skipped_member_metadata_collision(tmp_path):
    source = _source(tmp_path / 'source')
    path = source.root / source.records[0]['observation_npz']
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    meta = json.loads(str(arrays['metadata_json']))
    meta['depth_m'] = 'ambiguous even though the route omits it'
    arrays['metadata_json'] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='array/metadata key collision'):
        _build(source, tmp_path / 'npy')


def test_source_cache_source_change_during_mapping_falls_back(tmp_path, monkeypatch):
    source = _source(tmp_path / 'source')
    output = tmp_path / 'npy'
    _build(source, output)
    path = (source.root / source.records[0]['observation_npz']).resolve()
    monkeypatch.setenv('SUCTION_INSTANCE_SOURCE_CACHE_A', str(output))
    original = np.load
    changed = []
    def replace_during_load(file, *args, **kwargs):
        value = original(file, *args, **kwargs)
        if str(file).endswith('rgb.npy') and not changed:
            changed.append(True)
            temporary = path.with_suffix('.new')
            shutil.copy2(path, temporary)
            temporary.replace(path)
        return value
    monkeypatch.setattr(np, 'load', replace_during_load)
    reset_reader_cache_stats()
    actual = source.load_uncached(0, _compact_source=True, source_route='a')
    assert not isinstance(actual['observation']['rgb'], np.memmap)
    assert reader_cache_stats()['source_fallbacks'] == 1


def test_source_cache_ignores_legacy_fields_without_recomputing_them(tmp_path, monkeypatch):
    source = _source(tmp_path / 'source')
    path = source.root / source.records[0]['label_cache']
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['sha256'] = np.asarray('unused legacy declaration')
    metadata = json.loads(str(arrays['metadata_json']))
    metadata['legacy_sha256'] = 'also unused'
    arrays['metadata_json'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    output = tmp_path / 'npy'
    _build(source, output)
    monkeypatch.setenv('SUCTION_INSTANCE_SOURCE_CACHE_A', str(output))
    reset_reader_cache_stats()
    item = source.load_uncached(0, _compact_source=True, source_route='a')
    assert isinstance(item['labels']['points'], np.memmap)
    assert 'sha256' not in item['labels'] and 'legacy_sha256' not in item['labels']
    assert reader_cache_stats()['source_hits'] == 2
    assert not list(output.rglob('sha256.npy'))


def test_sparse_npy_preserves_raw_bits_and_logical_budget(tmp_path, monkeypatch):
    # Put negative zero and two distinct NaN payloads in otherwise empty blocks.
    bits = np.zeros(131072, np.uint64)
    bits[[0, 8193, -1]] = [0x8000000000000000, 0x7FF8000000000001, 0xFFF8000000000002]
    value = bits.view(np.float64).reshape(512, 256)
    path = tmp_path / 'bits.npy'
    builder._save_sparse_npy(path, value)
    actual = np.load(path, allow_pickle=False, mmap_mode='c')
    np.testing.assert_array_equal(actual.view(np.uint64).reshape(-1), bits)
    assert actual.shape == value.shape and actual.dtype == value.dtype
    source = _source(tmp_path / 'source')
    result = _build(source, tmp_path / 'npy', sparse=True)
    assert result['logical_bytes'] <= result['reserved_bytes'] <= result['max_bytes']
    if result['allocated_bytes'] is not None:
        assert result['allocated_bytes'] <= result['reserved_bytes']
    # Dense fallback emulates a filesystem that does not retain holes. It is
    # still valid NPY and fits the same conservative logical-byte reservation.
    monkeypatch.setattr(builder, '_save_sparse_npy', lambda path, value: np.save(path, value, allow_pickle=False))
    dense = _build(source, tmp_path / 'no_holes', sparse=True)
    assert dense['logical_bytes'] <= dense['reserved_bytes'] <= dense['max_bytes']
    with pytest.raises(ValueError, match='No arrays were converted'):
        builder.build_instance_source_cache(source.manifest, tmp_path / 'small', route='a',
            max_bytes=1, training_queries=4, sparse=True)
