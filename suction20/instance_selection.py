"""Fixed source-test frame selection for inexpensive checkpoint ranking.

Selection uses frame IDs and declared GT-empty frames, without loading any
observation or label archive. A formal subset retains every selected frame's
visible instances across all 90 test scenes; it never substitutes other frames.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from numbers import Integral
from pathlib import Path


SELECTION_SCHEMA = 'source_test_selection_v1'
FRAME_SAMPLER = 'bit_reverse8_prefix_v1'
TEST_SCENES = tuple(range(100, 190))


def _integer(value, field, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f'{field} must be an integer in [{low}, {high}]')
    return int(value)


def selection_frame_ids(frames_per_scene):
    """Return a sorted, nested and fixed prefix of 8-bit-reversed frame IDs."""
    count = _integer(frames_per_scene, 'selection_frames_per_scene', 1, 256)
    # Eight frames are 0,32,...,224; sixteen add 16,48,...,240. There is no
    # seed, draw, epoch dependence or quality-dependent frame selection.
    return sorted(int(f'{index:08b}'[::-1], 2) for index in range(count))


def validate_selection_config(pretrain_config):
    """Resolve the selection size; an omitted setting retains full-test use."""
    if not isinstance(pretrain_config, Mapping):
        raise ValueError('pretrain configuration must be a mapping')
    return _integer(pretrain_config.get('selection_frames_per_scene', 256),
                    'selection_frames_per_scene', 1, 256)


def _manifest_path(dataset):
    value = getattr(dataset, 'manifest_path', None)
    if value is None:
        value = getattr(dataset, 'manifest', None)
    return None if value is None else Path(value)


def _frame_pair(value, field):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{field} must contain [scene_id, frame_id] pairs')
    return (_integer(value[0], f'{field} scene_id', 0, 189),
            _integer(value[1], f'{field} frame_id', 0, 255))


def _empty_frames(dataset, observed):
    """Read explicit empty-frame evidence; frame declarations alone are not evidence."""
    path = _manifest_path(dataset)
    if path is None or not path.with_suffix('.summary.json').is_file():
        return set()
    summary = json.loads(path.with_suffix('.summary.json').read_text())
    if not isinstance(summary, dict):
        raise ValueError('Manifest coverage summary must be a mapping')
    if summary.get('manifest', path.name) != path.name:
        raise ValueError('Manifest coverage summary names another manifest')
    if summary.get('split', 'test') not in ('test', 'all'):
        raise ValueError('Selection coverage summary belongs to another split')
    raw = summary.get('empty_gt_frames', [])
    if not isinstance(raw, list):
        raise ValueError('empty_gt_frames must be a list')
    pairs = [_frame_pair(pair, 'empty_gt_frames') for pair in raw]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate GT-empty frame declaration')
    if summary.get('split', 'test') == 'test' and any(scene not in TEST_SCENES for scene, _ in pairs):
        raise ValueError('Selection GT-empty frame belongs to another split')
    # A combined source manifest can legitimately list empty training frames.
    empty = {pair for pair in pairs if pair[0] in TEST_SCENES}
    if empty & observed:
        raise ValueError('A frame cannot contain instances and be declared GT-empty')
    if 'frame_pairs' in summary:
        claimed = {_frame_pair(pair, 'frame_pairs') for pair in summary['frame_pairs']}
        if not set(pairs) <= claimed:
            raise ValueError('GT-empty frame is outside declared manifest coverage')
    elif 'scene_ids' in summary or 'frame_ids' in summary:
        if 'scene_ids' not in summary or 'frame_ids' not in summary:
            raise ValueError('Coverage summary must declare both scene_ids and frame_ids')
        scenes = {_integer(scene, 'summary scene_id', 0, 189) for scene in summary['scene_ids']}
        frames = {_integer(frame, 'summary frame_id', 0, 255) for frame in summary['frame_ids']}
        if any(scene not in scenes or frame not in frames for scene, frame in pairs):
            raise ValueError('GT-empty frame is outside declared manifest coverage')
    return empty


class SourceEvaluationSelection:
    """An order-preserving view; only indexing this view loads selected assets."""
    def __init__(self, dataset, indices, descriptor, sample_sequence):
        self.dataset = dataset
        self.indices = tuple(indices)
        self.records = [dataset.records[index] for index in self.indices]
        self.selection_descriptor = deepcopy(descriptor)
        self.identity = deepcopy(getattr(dataset, 'identity', None))
        # Preserve legacy full-test identity exactly for same-config resume.
        # A true subset must record the actual ordered sample membership.
        if descriptor['is_subset']:
            if self.identity is None:
                self.identity = {}
            if not isinstance(self.identity, dict):
                raise ValueError('Source selection dataset identity must be a mapping')
            self.identity['evaluation_selection'] = deepcopy(descriptor)
            self.identity['selection_instance_ids'] = sample_sequence
        path = _manifest_path(dataset)
        if path is not None:
            self.manifest = path
            self.manifest_path = path
        self.training_queries = getattr(dataset, 'training_queries', None)
        self.require_labels = getattr(dataset, 'require_labels', True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def load_uncached(self, index):
        read = getattr(self.dataset, 'load_uncached', self.dataset.__getitem__)
        return read(self.indices[index])


def select_source_evaluation(dataset, frames_per_scene, *, allow_partial=False):
    """Select a fixed test-frame set, validating its actual required coverage.

    Either a complete test manifest or a manifest containing just the required
    frames is valid. In both cases every requested scene/frame must have an
    instance row or an explicit GT-empty declaration. ``allow_partial`` remains
    an explicit smoke-test mode and is recorded as such in the descriptor.
    """
    frame_ids = selection_frame_ids(frames_per_scene)
    count = len(frame_ids)
    if not isinstance(allow_partial, bool):
        raise ValueError('allow_partial must be a boolean')
    if isinstance(dataset, SourceEvaluationSelection):
        old = dataset.selection_descriptor
        if old['frames_per_scene'] == count and old['partial_manifests_allowed'] == allow_partial:
            return dataset
        dataset = dataset.dataset
    if getattr(dataset, 'training_queries', None) is not None:
        raise ValueError('Source selection requires the complete per-instance query pool')
    if not getattr(dataset, 'require_labels', True):
        raise ValueError('Source selection requires teacher labels')
    records = dataset.records
    expected = {(scene, frame) for scene in TEST_SCENES for frame in frame_ids}
    observed, indices, sample_sequence = set(), [], []
    seen_samples = set()
    for index, row in enumerate(records):
        scene = _integer(row.get('scene_id'), 'selection scene_id', 100, 189)
        frame = _integer(row.get('frame_id'), 'selection frame_id', 0, 255)
        if row.get('split') != 'test':
            raise ValueError('Source selection manifest contains rows from another split')
        observed.add((scene, frame))
        if (scene, frame) not in expected:
            continue
        sample_id = row.get('sample_id', row.get('instance_uid'))
        instance_id = row.get('instance_uid', sample_id)
        if not isinstance(sample_id, str) or not sample_id or not isinstance(instance_id, str) or not instance_id:
            raise ValueError('Selected source instances require nonempty semantic IDs')
        if sample_id in seen_samples:
            raise ValueError('Duplicate selected source sample ID')
        seen_samples.add(sample_id)
        indices.append(index)
        sample_sequence.append([scene, frame, sample_id, instance_id])
    empty = _empty_frames(dataset, observed)
    selected_observed, selected_empty = observed & expected, empty & expected
    covered = selected_observed | selected_empty
    missing = expected - covered
    if missing and not allow_partial:
        examples = ', '.join(f'{scene}/{frame}' for scene, frame in sorted(missing)[:8])
        raise ValueError(f'Fixed source test selection is incomplete: {len(missing)} missing scene/frame pairs '
                         f'(examples {examples}); build the requested frames or use explicit smoke mode')
    if not indices:
        raise ValueError('No visible instances available in the requested source test selection')
    scope = ('partial_manifest_smoke' if allow_partial else
             'fixed_test_subset' if count < 256 else 'complete_test')
    descriptor = {
        'schema': SELECTION_SCHEMA, 'frame_sampler': FRAME_SAMPLER,
        'frames_per_scene': count, 'frame_ids': frame_ids, 'scene_ids': list(TEST_SCENES),
        'is_subset': count < 256, 'scope': scope,
        'expected_frame_count': len(expected), 'selected_frame_count': len(covered),
        'frames_with_instances': len(selected_observed),
        'empty_gt_frames': [list(pair) for pair in sorted(selected_empty)],
        'missing_frame_count': len(missing), 'complete_requested_selection': not missing,
        'selected_instance_count': len(indices), 'available_instance_count': len(records),
        'partial_manifests_allowed': allow_partial,
    }
    return SourceEvaluationSelection(dataset, indices, descriptor, sample_sequence)
