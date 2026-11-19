"""Local selection mirrors remain private to readers and fall back on source changes."""
import json
import os

import numpy as np
import pytest

from suction20.instance_data import _read_npz, _COMPACT_SOURCE_UNUSED_ARRAYS


def test_selection_mirror_private_and_stale_fallback(tmp_path, monkeypatch):
    source = tmp_path/'selection'/'observations'/'instance.npz'
    source.parent.mkdir(parents=True)
    values = dict(rgb=np.arange(12, dtype=np.float32), metadata_json=np.array('{"version":3}'))
    np.savez_compressed(source, **values)
    mirror = tmp_path/'local'
    instance = mirror/'observations'/'instance'
    instance.mkdir(parents=True)
    (mirror/'cache.json').write_text(json.dumps({
        'format':'instance-selection-npy-v1', 'source_root':str(tmp_path/'selection')}))
    stat = source.stat()
    (instance/'members.json').write_text(json.dumps({
        'source':[str(source),stat.st_size,stat.st_mtime_ns], 'members':list(values)}))
    for key,array in values.items():
        np.save(instance/(key+'.npy'),array,allow_pickle=False)
    monkeypatch.setenv('SUCTION_INSTANCE_SELECTION_CACHE',str(mirror))
    first = _read_npz(source)
    assert isinstance(first['rgb'], np.memmap)
    assert first['version'] == 3
    first['rgb'][0] = -7
    np.testing.assert_array_equal(_read_npz(source)['rgb'], values['rgb'])
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns+2_000_000_000))
    assert not isinstance(_read_npz(source)['rgb'], np.memmap)
    (instance/'rgb.npy').unlink()
    os.utime(source, ns=(stat.st_atime_ns,stat.st_mtime_ns))
    np.testing.assert_array_equal(_read_npz(source)['rgb'], values['rgb'])


@pytest.mark.parametrize('metadata_collision', [False, True])
def test_selection_mirror_excludes_arrays_without_losing_inventory(tmp_path, monkeypatch, metadata_collision):
    source = tmp_path/'selection'/'observations'/'instance.npz'
    source.parent.mkdir(parents=True)
    metadata = {'version': 3, **({'depth_m': 'ambiguous'} if metadata_collision else {})}
    values = dict(rgb=np.arange(12, dtype=np.float32), metadata_json=np.asarray(json.dumps(metadata)))
    values.update({key: np.arange(6) for key in _COMPACT_SOURCE_UNUSED_ARRAYS})
    np.savez_compressed(source, **values)
    mirror = tmp_path/'local'
    instance = mirror/'observations'/'instance'
    instance.mkdir(parents=True)
    (mirror/'cache.json').write_text(json.dumps({
        'format': 'instance-selection-npy-v1', 'source_root': str(tmp_path/'selection')}))
    stat = source.stat()
    (instance/'members.json').write_text(json.dumps({
        'source': [str(source), stat.st_size, stat.st_mtime_ns], 'members': list(values)}))
    for key, array in values.items():
        np.save(instance/(key+'.npy'), array, allow_pickle=False)
    monkeypatch.setenv('SUCTION_INSTANCE_SELECTION_CACHE', str(mirror))
    original = np.load
    opened = []

    def read_required(path, *args, **kwargs):
        path = str(path)
        opened.append(path)
        assert path.rsplit('/', 1)[-1] not in {key+'.npy' for key in _COMPACT_SOURCE_UNUSED_ARRAYS}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', read_required)
    if metadata_collision:
        with pytest.raises(ValueError, match='array/metadata key collision'):
            _read_npz(source, exclude_arrays=_COMPACT_SOURCE_UNUSED_ARRAYS)
    else:
        actual = _read_npz(source, exclude_arrays=_COMPACT_SOURCE_UNUSED_ARRAYS)
        assert actual.keys() == {'rgb', 'version'} and actual['version'] == 3
        assert isinstance(actual['rgb'], np.memmap)
        np.testing.assert_array_equal(actual['rgb'], values['rgb'])
    assert {path.rsplit('/', 1)[-1] for path in opened} == {'rgb.npy', 'metadata_json.npy'}
