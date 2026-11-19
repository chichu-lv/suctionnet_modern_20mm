"""Uncompressed cache conversion preserves real manifests, labels and recovery."""
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import repack_instance_cache as repack
from suction20.instance_data import InstanceDataset, build_source_instances, load_observation, save_observation
from test_instance_data import _assets
from test_instance_loading import _source


@pytest.mark.parametrize('workers', [1, 2])
def test_repack_preserves_array_contract_prefix_and_relative_paths(tmp_path, workers):
    source = _source(tmp_path)
    originals = {path: (path.stat().st_size, path.stat().st_mtime_ns)
                 for path in source.root.rglob('*.npz')}
    result = repack.repack_instance_cache(source.manifest, tmp_path/'stored', workers=workers)
    output = InstanceDataset(result['manifest'], training_queries=4)
    assert result['archives'] == result['converted'] == 4 and not result['reused']
    for left, right in zip(source.records, output.records):
        for key in left:
            if key not in ('observation_npz', 'query_cache', 'label_cache'):
                assert left[key] == right[key]
        assert right['observation_npz'] == right['query_cache']
        for key in ('observation_npz', 'query_cache', 'label_cache'):
            assert not Path(right[key]).is_absolute()
            with zipfile.ZipFile(output.root/right[key]) as archive:
                assert all(entry.compress_type == zipfile.ZIP_STORED for entry in archive.infolist())
    for index in range(len(source)):
        before, after = source.load_uncached(index), output.load_uncached(index)
        for group in ('observation', 'labels'):
            assert before[group].keys() == after[group].keys()
            for key, value in before[group].items():
                if isinstance(value, np.ndarray):
                    assert np.array_equal(value, after[group][key], equal_nan=True), key
                else:
                    assert value == after[group][key], key
    old_summary = json.loads(source.manifest.with_suffix('.summary.json').read_text())
    new_summary = json.loads(Path(result['summary']).read_text())
    assert all(new_summary[key] == value for key, value in old_summary.items())
    assert new_summary['storage']['format'] == 'npz_stored'
    assert originals == {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in originals}
    assert json.loads((output.root/'repack_state.json').read_text())['complete']


def test_repack_resume_uses_completed_archives_and_rejects_changed_source(tmp_path):
    source = _source(tmp_path)
    destination = tmp_path/'stored'
    first = repack.repack_instance_cache(source.manifest, destination)
    archives = {p: p.stat().st_mtime_ns for p in destination.glob('archives/*.npz')}
    second = repack.repack_instance_cache(source.manifest, destination, resume=True)
    assert second['converted'] == 0 and second['reused'] == first['converted']
    assert archives == {p: p.stat().st_mtime_ns for p in archives}
    with pytest.raises(FileExistsError, match='--resume'):
        repack.repack_instance_cache(source.manifest, destination)
    original = source.root/source.records[0]['observation_npz']
    stat = original.stat()
    os.utime(original, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
    with pytest.raises(ValueError, match='changed since'):
        repack.repack_instance_cache(source.manifest, destination, resume=True)


def test_interrupted_repack_resumes_without_publishing_partial_manifest(tmp_path, monkeypatch):
    source = _source(tmp_path)
    destination = tmp_path/'stored'
    original = repack._copy_archive
    calls = []
    def interrupted(task):
        if calls:
            raise RuntimeError('simulated interrupted conversion')
        calls.append(task)
        return original(task)
    monkeypatch.setattr(repack, '_copy_archive', interrupted)
    with pytest.raises(RuntimeError, match='interrupted conversion'):
        repack.repack_instance_cache(source.manifest, destination)
    assert not (destination/source.manifest.name).exists()
    assert not json.loads((destination/'repack_state.json').read_text())['complete']
    journal = destination/'repack_progress.jsonl'
    with journal.open('a') as stream:
        stream.write('{"incomplete_tail":')
    monkeypatch.setattr(repack, '_copy_archive', original)
    result = repack.repack_instance_cache(source.manifest, destination, resume=True)
    assert result['reused'] == 1 and result['converted'] == 3
    assert len(InstanceDataset(result['manifest'], training_queries=4)) == 2
    assert len([json.loads(line) for line in journal.read_text().splitlines()]) == 4


def test_invalid_geometry_never_publishes_manifest(tmp_path):
    source = _source(tmp_path)
    observation_path = source.root/source.records[0]['observation_npz']
    observation = load_observation(observation_path)
    observation['query_points'] = observation['query_points'].copy()
    observation['query_points'][0, 0] += .001
    save_observation(observation_path, observation)
    destination = tmp_path/'stored'
    with pytest.raises(ValueError, match='raw source pixels'):
        repack.repack_instance_cache(source.manifest, destination)
    assert not (destination/source.manifest.name).exists()
    assert not json.loads((destination/'repack_state.json').read_text())['complete']


def test_unlabelled_and_empty_frame_evidence_survive_conversion(tmp_path):
    from PIL import Image
    root, release, _ = _assets(tmp_path, scene_id=100)
    Image.fromarray(np.zeros((65, 65), np.uint8)).save(root/'scenes/scene_0100/realsense/label/0000.png')
    manifest, _ = build_source_instances(root, release, tmp_path/'cache', split='test',
        scenes=[100], frames=[0], with_labels=False)
    result = repack.repack_instance_cache(manifest, tmp_path/'stored')
    assert result['archives'] == result['converted'] == 0
    output = InstanceDataset(result['manifest'], require_labels=False)
    assert len(output) == 0
    summary = json.loads(Path(result['summary']).read_text())
    assert summary['empty_gt_frames'] == [[100, 0]] and not summary['labels_computed']


def test_repack_rejects_source_directory_and_foreign_output(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match='separate output'):
        repack.repack_instance_cache(source.manifest, source.root)
    destination = tmp_path/'existing'
    destination.mkdir()
    (destination/'user_file.txt').write_text('retain this file')
    with pytest.raises(FileExistsError, match='not empty'):
        repack.repack_instance_cache(source.manifest, destination)
    assert (destination/'user_file.txt').read_text() == 'retain this file'
