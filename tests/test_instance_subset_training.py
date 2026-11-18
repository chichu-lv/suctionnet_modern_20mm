"""Fixed-subset selection across the trainer, checkpoints and public CLIs."""
from copy import deepcopy
import json

import pytest
import torch

from scripts import evaluate_instance as evaluate_cli
from scripts import train_instance as train_cli
import suction20.instance_inference as inference
from suction20.instance_selection import selection_frame_ids, select_source_evaluation
from suction20.instance_training import InstanceTrainer, seed_all
from test_instance_evaluation import _metadata
from test_instance_inference import FixedScorer, checkpoint_state
from test_instance_training import (MemoryAttempts, MemorySource, TinyInstanceModel,
    assert_nested_equal, config as engine_config)


class FrameDataset(MemorySource):
    """Use actual engine observations while tracking loaded frame identities."""
    def __init__(self, *, frames, split='test', extra_unselected=False, empty=()):
        super().__init__(split)
        scenes = range(100) if split == 'train' else range(100, 190)
        empty = set(empty)
        self.records = []
        for scene in scenes:
            for frame in list(frames) + ([1] if extra_unselected else []):
                if (scene, frame) in empty:
                    continue
                uid = f'{split}_s{scene:04d}_f{frame:04d}_i000'
                self.records.append({'scene_id': scene, 'frame_id': frame, 'split': split,
                    'instance_uid': uid, 'sample_id': uid,
                    'test_subset': None if split == 'train' else
                        ('seen' if scene < 130 else 'similar' if scene < 160 else 'novel')})
        self.loaded = []

    def __getitem__(self, index):
        record = self.records[index]
        self.loaded.append((record['scene_id'], record['frame_id'], record['instance_uid']))
        item = super().__getitem__(index % 3)
        item['record'] = deepcopy(record)
        return item


def _summary(dataset, path, empty):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('')
    path.with_suffix('.summary.json').write_text(json.dumps({
        'manifest': path.name, 'split': 'test',
        'empty_gt_frames': [list(pair) for pair in empty]}))
    dataset.manifest = path
    return dataset


def subset_config(*, steps=2, every=1):
    config = engine_config()
    config['pretrain'].update(selection_frames_per_scene=8, steps=steps,
        evaluate_every=every, checkpoint_every=1, evaluation_records='none')
    return config


def _trainer(path, *, config=None, selection=None, stage='pretrain'):
    return InstanceTrainer(TinyInstanceModel(), config or subset_config(), MemorySource(),
        selection if selection is not None else (
            FrameDataset(frames=selection_frame_ids(8)) if stage == 'pretrain' else MemoryAttempts('val')),
        path, stage=stage, target_dataset=MemoryAttempts() if stage == 'finetune' else None)


def test_trainer_uses_the_same_formal_subset_twice_and_keeps_actual_denominators(tmp_path):
    empty = [(100, 32), (189, 224)]
    selection = _summary(FrameDataset(frames=selection_frame_ids(8),
        extra_unselected=True, empty=empty), tmp_path/'selection.jsonl', empty)
    trainer = _trainer(tmp_path/'run', selection=selection)
    assert not trainer.allow_partial
    assert trainer.policy['selection_scope'] == 'fixed_test_subset'
    assert trainer.policy['selection_frames_per_scene'] == 8
    assert trainer.policy['selection_frame_ids'] == selection_frame_ids(8)
    assert trainer.selection.selection_descriptor['selected_frame_count'] == 720
    assert trainer.selection.selection_descriptor['frames_with_instances'] == 718
    trainer.run()
    assert len(selection.loaded) == 2 * 718
    assert selection.loaded[:718] == selection.loaded[718:]
    assert all(frame in selection_frame_ids(8) for _, frame, _ in selection.loaded)
    for step in (1, 2):
        report = json.loads((trainer.run_dir/f'evaluation_step{step:06d}.json').read_text())
        assert report['metrics']['instance_count'] == 718
        assert report['metrics']['frame_count'] == 718
        assert report['metrics']['scene_count'] == 90
        assert report['timing']['observations'] == 718
        assert report['policy']['selection_scope'] == 'fixed_test_subset'
        assert set(report['metrics']['by_subset']) == {'seen', 'similar', 'novel'}
    state = torch.load(trainer.run_dir/'best.pt', weights_only=False)
    assert state['policy']['selection_scope'] == 'fixed_test_subset'
    assert state['identity']['selection']['length'] == 718
    assert state['checkpoint_selection_role'] == 'test_selected'


def test_fixed_subset_pause_resume_matches_continuous_training_exactly(tmp_path):
    config = subset_config(steps=4, every=2)
    seed_all(812)
    full = _trainer(tmp_path/'full', config=config)
    full.run()
    seed_all(812)
    paused = _trainer(tmp_path/'paused', config=config)
    paused.run(stop_after=1)
    assert not list(paused.run_dir.glob('evaluation_step*.json'))
    resumed = _trainer(tmp_path/'paused', config=config)
    resumed.resume(paused.run_dir/'last.pt')
    resumed.run()
    assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(full.best, resumed.best)
    assert full.selection.identity == resumed.selection.identity
    assert full.source_schedule.state_dict() == resumed.source_schedule.state_dict()
    assert full.supervised_source_instances == resumed.supervised_source_instances == 8
    assert sorted(p.name for p in resumed.run_dir.glob('evaluation_step*.json')) == [
        'evaluation_step000002.json', 'evaluation_step000004.json']


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('mutation', ['replace_id', 'reorder'])
def test_same_length_subset_changes_are_rejected_on_all_resume_paths(tmp_path, legacy, mutation):
    source = _trainer(tmp_path/'source')
    source.train_one_update()
    source.save()
    checkpoint = source.run_dir/'last.pt'
    if legacy:
        state = torch.load(checkpoint, weights_only=False)
        state['identity']['runtime_contract'] = 'historical_runtime_without_selection_extension'
        torch.save(state, checkpoint)
    changed = FrameDataset(frames=selection_frame_ids(8))
    if mutation == 'replace_id':
        changed.records[0].update(sample_id='replacement', instance_uid='replacement')
    else:
        changed.records[0], changed.records[1] = changed.records[1], changed.records[0]
    resumed = _trainer(tmp_path/'resumed', selection=changed)
    with pytest.raises(ValueError, match='fixed test selection definition differs'):
        resumed.resume(checkpoint)


def test_finetune_preserves_subset_source_policy_and_definition(tmp_path):
    source = _trainer(tmp_path/'source')
    source.train_one_update()
    source.evaluate()
    target = _trainer(tmp_path/'target', stage='finetune')
    target.initialize_finetune(source.run_dir/'best.pt')
    lineage = target.initialization
    assert lineage['selection_policy']['selection_scope'] == 'fixed_test_subset'
    assert lineage['selection_policy']['selection_frames_per_scene'] == 8
    assert lineage['source_definition']['selection'] == source.identity['selection']
    assert target.policy['selection_scope'] == 'complete_manifest'
    assert target.policy['selection_role'] == 'validation_selected'
    target.train_one_update()
    target.save()
    resumed = _trainer(tmp_path/'target', stage='finetune')
    resumed.resume(target.run_dir/'last.pt')
    assert resumed.initialization == lineage
    state = torch.load(source.run_dir/'best.pt', weights_only=False)
    state['policy']['selection_scope'] = 'complete_manifest'
    bad = tmp_path/'bad_source.pt'
    torch.save(state, bad)
    with pytest.raises(ValueError, match='fixed test selection definition'):
        _trainer(tmp_path/'bad_target', stage='finetune').initialize_finetune(bad)


def _subset_checkpoint():
    state = checkpoint_state(FixedScorer(), route='b')
    selection = select_source_evaluation(FrameDataset(frames=selection_frame_ids(8)), 8)
    state['config']['pretrain']['selection_frames_per_scene'] = 8
    state['identity']['config_definition'] = deepcopy(state['config'])
    state['identity']['partial_manifests_allowed'] = False
    state['identity']['selection'] = {'manifest_path': None, 'length': len(selection),
                                      'data_identity': deepcopy(selection.identity)}
    state['policy'].update(selection_scope='fixed_test_subset', selection_frames_per_scene=8,
        selection_frame_ids=selection_frame_ids(8),
        selection_frame_sampler=selection.selection_descriptor['frame_sampler'])
    return state


@pytest.mark.parametrize('mutation', ['config', 'descriptor', 'policy', 'ids', 'coverage', 'sampler'])
def test_loader_rejects_inconsistent_subset_before_constructing_model(tmp_path, monkeypatch, mutation):
    state = _subset_checkpoint()
    data = state['identity']['selection']['data_identity']
    if mutation == 'config':
        state['config']['pretrain']['selection_frames_per_scene'] = 16
        state['identity']['config_definition'] = deepcopy(state['config'])
    elif mutation == 'descriptor':
        data['evaluation_selection']['frame_ids'][0] = 1
    elif mutation == 'policy':
        state['policy']['selection_scope'] = 'complete_manifest'
    elif mutation == 'ids':
        data['selection_instance_ids'].pop()
    elif mutation == 'coverage':
        data['evaluation_selection']['selected_frame_count'] = 719
    else:
        state['policy']['selection_frame_sampler'] = 'unknown_sampler'
    checkpoint = tmp_path/'bad.pt'
    torch.save(state, checkpoint)
    calls = []
    monkeypatch.setattr(inference, 'create_instance_model', lambda *a, **k: calls.append(a) or FixedScorer())
    with pytest.raises(ValueError, match='fixed test selection'):
        inference.load_instance_checkpoint(checkpoint)
    assert calls == []


def test_loader_preserves_valid_subset_provenance(tmp_path, monkeypatch):
    state = _subset_checkpoint()
    checkpoint = tmp_path/'best.pt'
    torch.save(state, checkpoint)
    monkeypatch.setattr(inference, 'create_instance_model', lambda *a, **k: FixedScorer())
    _, metadata = inference.load_instance_checkpoint(checkpoint)
    assert metadata['selection_policy'] == state['policy']
    assert metadata['checkpoint_identity']['selection'] == state['identity']['selection']
    assert metadata['checkpoint_selection_role'] == 'test_selected'


def test_training_cli_accepts_only_required_test_frames_without_partial_flag(tmp_path, monkeypatch):
    source = FrameDataset(frames=range(256), split='train')
    selection = FrameDataset(frames=selection_frame_ids(8))
    requests = []
    def dataset(path, *, split, **kwargs):
        requests.append((split, kwargs))
        return source if split == 'train' else selection
    monkeypatch.setattr(train_cli, 'InstanceDataset', dataset)
    monkeypatch.setattr(train_cli, 'create_instance_model', lambda *a, **k: TinyInstanceModel())
    config = subset_config(steps=1)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    run = tmp_path/'run'
    train_cli.main(['--config', str(path), '--source-manifest', 'train.jsonl',
        '--selection-manifest', 'fixed_subset.jsonl', '--run-dir', str(run), '--device', 'cpu'])
    assert requests == [('train', {'training_queries': 128}), ('test', {})]
    state = torch.load(run/'best.pt', weights_only=False)
    assert state['identity']['partial_manifests_allowed'] is False
    assert state['policy']['selection_scope'] == 'fixed_test_subset'
    assert len(selection.loaded) == 720
    assert json.loads((run/'summary.json').read_text())['completed_budget'] is True


def test_full_evaluation_cli_does_not_reuse_checkpoint_selection_filter(tmp_path, monkeypatch):
    # All official frames are represented: two visible frames per scene and
    # 254 explicitly GT-empty frames. Frame 1 is outside the fixed 8-frame set.
    empty = [(scene, frame) for scene in range(100, 190) for frame in range(2, 256)]
    dataset = _summary(FrameDataset(frames=[0, 1]), tmp_path/'full_test.jsonl', empty)
    metadata = _metadata(dataset.identity)
    metadata['selection_policy'].update(selection_scope='fixed_test_subset',
        selection_frames_per_scene=8, selection_frame_ids=selection_frame_ids(8))
    monkeypatch.setattr(evaluate_cli, 'InstanceDataset', lambda *a, **k: dataset)
    monkeypatch.setattr(evaluate_cli, 'load_instance_checkpoint',
                        lambda *a, **k: (TinyInstanceModel(), deepcopy(metadata)))
    output = tmp_path/'full_report.json'
    report = evaluate_cli.main(['--checkpoint', 'subset_best.pt', '--manifest', str(dataset.manifest),
        '--domain', 'source', '--split', 'test', '--output', str(output), '--device', 'cpu'])
    assert report['evaluation_scope'] == 'complete_manifest'
    assert report['source_pretrain_selection_scope'] == 'fixed_test_subset'
    assert report['coverage']['scenes'] == 90
    assert report['coverage']['empty_gt_frames'] == 90 * 254
    assert report['metrics']['instance_count'] == len(dataset) == 180
    assert len(dataset.loaded) == 180
    assert sum(frame == 1 for _, frame, _ in dataset.loaded) == 90
    assert {row['frame_id'] for row in report['records']} == {0, 1}
