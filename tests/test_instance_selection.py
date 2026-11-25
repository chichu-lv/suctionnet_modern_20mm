"""Fixed test-frame coverage, lazy asset access and selection identity."""
from copy import deepcopy
import json

import pytest

from suction20.instance_selection import (selection_frame_ids, select_source_evaluation,
                                         validate_selection_config)


class ManifestOnlyDataset:
    def __init__(self, frames, scenes=range(100, 190)):
        self.records = [
            {'scene_id': scene, 'frame_id': frame, 'split': 'test',
             'sample_id': f's{scene}_f{frame}_i0', 'instance_uid': f's{scene}_f{frame}_i0',
             'trainable': False, 'status': 'insufficient_support'}
            for scene in scenes for frame in frames]
        self.identity = {'normal_spec': 'fixed', 'teacher_identity': {'physics': {'radius_m': .02}}}
        self.training_queries = None
        self.require_labels = True
        self.loaded = []

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        self.loaded.append(index)
        # An opaque full-pool object makes any accidental slicing visible.
        return {'record': self.records[index], 'query_pool': 'all_original_queries', 'base_index': index}


def with_summary(dataset, tmp_path, empty=(), **fields):
    dataset.manifest = tmp_path/'manifest_test.jsonl'
    dataset.manifest.write_text('')
    summary = {'manifest': dataset.manifest.name, 'split': 'test', 'empty_gt_frames': list(empty)}
    summary.update(fields)
    dataset.manifest.with_suffix('.summary.json').write_text(json.dumps(summary))
    return dataset


def test_frame_sequence_is_fixed_spread_nested_and_complete():
    assert selection_frame_ids(8) == list(range(0, 256, 32))
    assert selection_frame_ids(16) == list(range(0, 256, 16))
    assert set(selection_frame_ids(8)) < set(selection_frame_ids(16))
    assert selection_frame_ids(256) == list(range(256))
    for count in (1, 3, 7, 31, 65, 255):
        frames = selection_frame_ids(count)
        assert len(frames) == len(set(frames)) == count
        assert frames == sorted(frames)
        assert set(frames) < set(selection_frame_ids(count + 1))
    assert validate_selection_config({}) == 256
    assert validate_selection_config({'selection_frames_per_scene': 8}) == 8


@pytest.mark.parametrize('value', [None, True, False, 0, -1, 257, 8.0, '8', [8]])
def test_invalid_selection_sizes_are_rejected(value):
    with pytest.raises(ValueError, match='selection_frames_per_scene'):
        selection_frame_ids(value)
    with pytest.raises(ValueError, match='selection_frames_per_scene'):
        validate_selection_config({'selection_frames_per_scene': value})


def test_full_manifest_selection_is_lazy_and_keeps_all_selected_instances():
    source = ManifestOnlyDataset(range(256))
    extra = deepcopy(source.records[0])
    extra.update(sample_id='extra_visible_fragment', instance_uid='extra_visible_fragment')
    source.records.append(extra)
    view = select_source_evaluation(source, 8)
    expected_indices = [i for i, row in enumerate(source.records) if row['frame_id'] % 32 == 0]
    assert list(view.indices) == expected_indices
    assert len(view) == 721
    assert source.loaded == []
    assert view.records == [source.records[index] for index in expected_indices]
    assert all(not row['trainable'] for row in view.records)  # no_pose keeps its denominator.
    item = view[-1]
    assert item['base_index'] == len(source.records) - 1
    assert item['query_pool'] == 'all_original_queries'
    assert source.loaded == [len(source.records) - 1]
    d = view.selection_descriptor
    assert d['scope'] == 'fixed_test_subset'
    assert d['complete_requested_selection'] is True
    assert d['selected_frame_count'] == d['expected_frame_count'] == 720
    assert d['frames_with_instances'] == 720
    assert d['selected_instance_count'] == 721
    assert d['available_instance_count'] == 23041


def test_manifest_containing_only_requested_frames_is_a_formal_selection():
    source = ManifestOnlyDataset(selection_frame_ids(8))
    view = select_source_evaluation(source, 8)
    assert len(view) == 720
    assert view.selection_descriptor['partial_manifests_allowed'] is False
    assert view.selection_descriptor['complete_requested_selection'] is True
    assert view.identity['teacher_identity'] == source.identity['teacher_identity']
    assert 'evaluation_selection' not in source.identity
    assert view.identity['evaluation_selection'] == view.selection_descriptor
    assert view.identity['selection_instance_ids'][0] == [100, 0, 's100_f0_i0', 's100_f0_i0']
    assert len(view.identity['selection_instance_ids']) == len(view)


def test_gt_empty_frame_counts_as_coverage_without_fabricated_instance(tmp_path):
    source = ManifestOnlyDataset(selection_frame_ids(8))
    source.records.pop(0)
    with_summary(source, tmp_path, [[100, 0]], scene_ids=list(range(100, 190)),
                 frame_ids=selection_frame_ids(8))
    view = select_source_evaluation(source, 8)
    assert len(view) == 719
    assert view.selection_descriptor['selected_frame_count'] == 720
    assert view.selection_descriptor['frames_with_instances'] == 719
    assert view.selection_descriptor['empty_gt_frames'] == [[100, 0]]
    assert view.manifest == view.manifest_path == source.manifest
    assert source.loaded == []


def test_missing_requested_frame_is_not_replaced_by_another_frame_or_summary_claim(tmp_path):
    source = ManifestOnlyDataset(selection_frame_ids(8))
    source.records[0].update(frame_id=1)
    with_summary(source, tmp_path, scene_ids=list(range(100, 190)), frame_ids=list(range(256)),
                 frame_count=23040, complete_requested_split=True)
    with pytest.raises(ValueError, match='incomplete: 1 missing.*100/0'):
        select_source_evaluation(source, 8)
    partial = select_source_evaluation(source, 8, allow_partial=True)
    assert len(partial) == 719
    assert partial.selection_descriptor['scope'] == 'partial_manifest_smoke'
    assert partial.selection_descriptor['missing_frame_count'] == 1
    assert partial.selection_descriptor['complete_requested_selection'] is False


def test_whole_missing_scene_is_rejected():
    source = ManifestOnlyDataset(selection_frame_ids(8), scenes=range(100, 189))
    with pytest.raises(ValueError, match='incomplete: 8 missing'):
        select_source_evaluation(source, 8)


def test_legacy_full_test_identity_and_order_are_unchanged():
    source = ManifestOnlyDataset(range(256))
    view = select_source_evaluation(source, 256)
    assert view.identity == source.identity
    assert view.identity is not source.identity
    assert view.records == source.records
    assert list(view.indices) == list(range(len(source)))
    assert view.selection_descriptor['scope'] == 'complete_test'
    assert view.selection_descriptor['is_subset'] is False
    assert view.selection_descriptor['selected_frame_count'] == 23040
    assert view[len(view)-1]['base_index'] == len(source)-1


def test_repeated_selection_is_idempotent_and_new_size_uses_original_dataset():
    source = ManifestOnlyDataset(range(256))
    eight = select_source_evaluation(source, 8)
    assert select_source_evaluation(eight, 8) is eight
    sixteen = select_source_evaluation(eight, 16)
    assert sixteen.dataset is source
    assert len(sixteen) == 1440
    assert set(eight.indices) < set(sixteen.indices)
    assert sixteen.identity['selection_instance_ids'] != eight.identity['selection_instance_ids']


def test_selection_identity_detects_membership_and_order_changes():
    source = ManifestOnlyDataset(selection_frame_ids(8))
    original = select_source_evaluation(source, 8)
    source.records[0], source.records[1] = source.records[1], source.records[0]
    reordered = select_source_evaluation(source, 8)
    assert original.identity['selection_instance_ids'] != reordered.identity['selection_instance_ids']
    source.records[0] = dict(source.records[0], sample_id='replacement', instance_uid='replacement')
    replaced = select_source_evaluation(source, 8)
    assert replaced.identity['selection_instance_ids'] != reordered.identity['selection_instance_ids']


@pytest.mark.parametrize('fields,match', [
    ({'manifest': 'other.jsonl'}, 'another manifest'),
    ({'split': 'train'}, 'another split'),
    ({'empty_gt_frames': [[100, 0], [100, 0]]}, 'Duplicate'),
    ({'empty_gt_frames': [[100, 256]]}, 'frame_id'),
    ({'empty_gt_frames': [[0, 0]]}, 'another split'),
    ({'empty_gt_frames': [[100, 0]]}, 'cannot contain instances'),
])
def test_bad_empty_frame_evidence_is_rejected(tmp_path, fields, match):
    source = with_summary(ManifestOnlyDataset(selection_frame_ids(8)), tmp_path, **fields)
    with pytest.raises(ValueError, match=match):
        select_source_evaluation(source, 8)


def test_combined_summary_can_have_empty_training_frames(tmp_path):
    source = ManifestOnlyDataset(selection_frame_ids(8))
    source.records.pop(0)
    with_summary(source, tmp_path, [[100, 0], [0, 1]], split='all')
    view = select_source_evaluation(source, 8)
    assert view.selection_descriptor['empty_gt_frames'] == [[100, 0]]


def test_empty_selected_set_is_not_a_zero_quality_evaluation(tmp_path):
    source = ManifestOnlyDataset([1])
    all_empty = [[scene, frame] for scene in range(100, 190) for frame in selection_frame_ids(8)]
    with_summary(source, tmp_path, all_empty)
    with pytest.raises(ValueError, match='No visible instances'):
        select_source_evaluation(source, 8)


@pytest.mark.parametrize('field,value,match', [
    ('scene_id', 99, 'scene_id'), ('scene_id', 100.0, 'scene_id'),
    ('frame_id', -1, 'frame_id'), ('split', 'train', 'another split'),
    ('sample_id', '', 'semantic IDs'),
])
def test_malformed_source_rows_are_rejected(field, value, match):
    source = ManifestOnlyDataset(selection_frame_ids(8))
    source.records[0][field] = value
    with pytest.raises(ValueError, match=match):
        select_source_evaluation(source, 8)


def test_selection_requires_full_query_pool_and_labels():
    source = ManifestOnlyDataset(selection_frame_ids(8))
    source.training_queries = 128
    with pytest.raises(ValueError, match='complete per-instance query pool'):
        select_source_evaluation(source, 8)
    source.training_queries = None
    source.require_labels = False
    with pytest.raises(ValueError, match='teacher labels'):
        select_source_evaluation(source, 8)
