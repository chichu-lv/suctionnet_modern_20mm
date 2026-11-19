"""Local selection mirrors remain private to readers and fall back on source changes."""
import json
import os

import numpy as np

from suction20.instance_data import _read_npz


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
