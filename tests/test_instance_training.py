"""Optimizer, loss, selection provenance and resume integration contracts."""
from copy import deepcopy
import json
import gzip

import numpy as np
import pytest
import torch
from torch import nn

from suction20.instance_training import (InstanceSchedule, InstanceTrainer,
    aggregate_instance_rows, check_source_splits, evaluate_instance_source, evaluate_instance_attempts,
    load_instance_config, scheduled_evaluation_steps, seed_all, source_loss)


class TinyInstanceModel(nn.Module):
    """Explicit engine fixture; never registered as a production architecture."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 8)
        self.decoder = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.SiLU())
        self.contact = nn.Linear(8, 8)
        self.heads = nn.Linear(8, 5)
        self.dropout = nn.Dropout(.2)

    def parameter_groups(self):
        return {name: list(getattr(self, name).parameters()) for name in ('encoder', 'decoder', 'contact', 'heads')}

    def encode_instance(self, observation):
        return observation

    def score_at_pose(self, encoded, points, normals, query_batch_index=None):
        device = next(self.parameters()).device
        x = torch.as_tensor(points, dtype=torch.float32, device=device)
        h = self.dropout(self.contact(self.decoder(self.encoder(x))))
        raw = self.heads(h)
        return {'q_inst': raw[:, 0].sigmoid(), 'seal': raw[:, 1].sigmoid(),
                'wrench': raw[:, 2].sigmoid(), 'ell': raw[:, 3].clamp(-6, 1),
                'task_logit': raw[:, 4], 'valid': torch.ones(len(x), dtype=torch.bool, device=device)}

    def forward(self, observation):
        return self.score_at_pose(observation, observation['query_points'], observation['query_normals'])


class MemorySource:
    def __init__(self, split='train', count=3):
        first = 0 if split == 'train' else 100
        self.records = [{'scene_id': first+i, 'frame_id': 0, 'split': split,
                         'test_subset': None if split == 'train' else 'seen',
                         'instance_uid': f'{split}-{i}'} for i in range(count)]
        self.identity = {'fixture': 'synthetic_only', 'split': split, 'dataset_revision': 'v1'}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        p = np.array([[0., 0., .6], [.01, 0., .6], [-.01, .01, .6]], np.float64)
        p[:, 0] += index * .002
        obs = {'query_points': p, 'query_normals': np.tile([0., 0., -1.], (3, 1)),
               'query_ids': np.array([0, 1, 2]), 'support_points': np.repeat(p, 40, axis=0)}
        labels = {'q_inst': np.array([.2, .8, .5]), 'seal': np.array([.25, .9, .6]),
                  'wrench': np.array([.8, .9, .8]), 'label_valid': np.ones(3, bool)}
        return {'observation': obs, 'labels': labels, 'record': deepcopy(self.records[index])}


class MemoryAttempts(MemorySource):
    def __init__(self, split='train'):
        super().__init__(split, 2)
        for row in self.records:
            row['split'] = split
    def __getitem__(self, index):
        item = super().__getitem__(index)
        p, n = item['observation']['query_points'], item['observation']['query_normals']
        item['attempts'] = {'point_cam_m': p, 'normal_out_cam': n,
                            'success': np.array([0, 1, 1]), 'label_valid': np.ones(3, bool),
                            'pose_support_valid': np.ones(3, bool)}
        return item


def config():
    return {'schema': 'instance_runtime_v1', 'route': 'b', 'quality_contract': 'gt_INST20_pose_v2',
            'seed': 42, 'rgb_augmentation': False, 'model': {}, 'max_grad_norm': 10.,
            'pretrain': {'train_scene_range': [0, 99], 'selection_split': 'test', 'steps': 2,
                'accumulate_instances': 2, 'queries': 128, 'peak_lr': .001, 'final_lr': .0001,
                'warmup_steps': 1, 'evaluate_every': 2, 'checkpoint_every': 2},
            'finetune': {'steps': 2, 'head_only_steps': 1, 'evaluate_every': 2, 'checkpoint_every': 2},
            'public_evaluation': {'status': 'deferred', 'radius_only_protocol': None}}


def trainer(path, *, stage='pretrain'):
    cfg = config()
    return InstanceTrainer(TinyInstanceModel(), cfg, MemorySource(),
        MemorySource('test') if stage == 'pretrain' else MemoryAttempts('val'), path,
        stage=stage, target_dataset=MemoryAttempts() if stage == 'finetune' else None,
        allow_partial=True)


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_laplace_loss_masks_unknown_before_arithmetic():
    q = torch.tensor([.3, float('nan')], requires_grad=True)
    ell = torch.tensor([-.5, float('nan')], requires_grad=True)
    pred = {'q_inst': q, 'seal': q, 'wrench': q, 'ell': ell, 'valid': torch.tensor([True, True])}
    y = np.array([.6, np.nan])
    labels = {'q_inst': y, 'seal': y, 'wrench': y, 'label_valid': np.array([True, False])}
    loss = source_loss(pred, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert q.grad[1] == 0 and ell.grad[1] == 0
    assert q.grad[0] != 0 and torch.isfinite(ell.grad[0])
    labels['label_valid'][:] = False
    assert source_loss(pred, labels) is None
    labels['label_valid'][:] = True
    with pytest.raises(ValueError, match='Known physical'):
        source_loss(pred, labels)


def test_runtime_config_100_train_test_selected_deferred(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for route in ('a', 'b'):
        path = root / f'configs/instance_{route}.yaml'
        if path.exists():
            c = load_instance_config(path)
            assert c['pretrain']['train_scene_range'] == [0, 99]
            assert c['pretrain']['selection_split'] == 'test'
            assert scheduled_evaluation_steps(c['pretrain']) == (4375, 8750)
            assert c['pretrain']['checkpoint_every'] == 875
            assert c['public_evaluation']['status'] == 'deferred'
    invalid = config(); invalid['pretrain']['selection_split'] = 'val'
    path = tmp_path/'invalid.json'; path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='100 train'):
        load_instance_config(path)


def test_sampler_exact_resume_and_no_test_in_train():
    rows = MemorySource().records
    one = InstanceSchedule(rows, 9)
    for _ in range(7):
        one.next()
    state = one.state_dict()
    two = InstanceSchedule(rows, 123); two.load_state_dict(state)
    assert [one.next() for _ in range(40)] == [two.next() for _ in range(40)]
    with pytest.raises(ValueError, match='full 100'):
        check_source_splits(MemorySource(), 'train')
    assert check_source_splits(MemorySource(), 'train', allow_partial=True)['scenes'] == 3
    with pytest.raises(ValueError, match='another split'):
        check_source_splits(MemorySource('test'), 'train', allow_partial=True)


@pytest.mark.parametrize('nodes', [[3, 2], [2, 2], [0], [5], [True], [1.5], '2,4'])
def test_invalid_explicit_evaluation_schedule_rejected(nodes):
    with pytest.raises(ValueError, match='evaluation_steps'):
        scheduled_evaluation_steps({'steps': 4, 'evaluation_steps': nodes})


def test_evaluation_schedule_has_one_mode_and_keeps_budget_endpoint():
    assert scheduled_evaluation_steps({'steps': 4, 'evaluation_steps': [2]}) == (2, 4)
    assert scheduled_evaluation_steps({'steps': 4, 'evaluation_steps': []}) == (4,)
    assert scheduled_evaluation_steps({'steps': 5, 'evaluate_every': 2}) == (2, 4, 5)
    with pytest.raises(ValueError, match='not both'):
        scheduled_evaluation_steps({'steps': 4, 'evaluation_steps': [2], 'evaluate_every': 2})


def test_sparse_full_selection_schedule_survives_pause_and_resume(tmp_path):
    cfg = config()
    cfg['pretrain'].pop('evaluate_every')
    cfg['pretrain'].update(steps=4, evaluation_steps=[2], checkpoint_every=1)
    def make(path):
        return InstanceTrainer(TinyInstanceModel(), cfg, MemorySource(), MemorySource('test'),
                               path, allow_partial=True)
    seed_all(91)
    full = make(tmp_path/'full')
    full.run()
    seed_all(91)
    part = make(tmp_path/'part')
    part.run(stop_after=1)
    assert not list((tmp_path/'part').glob('evaluation_step*.json'))
    resumed = make(tmp_path/'resumed')
    resumed.resume(tmp_path/'part/last.pt')
    resumed.run()
    assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(full.best, resumed.best)
    for directory in ('full', 'resumed'):
        reports = sorted((tmp_path/directory).glob('evaluation_step*.json'))
        assert [p.name for p in reports] == ['evaluation_step000002.json', 'evaluation_step000004.json']
        for path in reports:
            report = json.loads(path.read_text())
            assert report['metrics']['instance_count'] == len(full.selection)
            assert report['policy']['evaluation_steps'] == [2, 4]
            assert report['timing']['observations'] == len(full.selection)
            assert report['timing']['wall_seconds'] >= 0
    assert full.step == resumed.step == 4 and full.supervised_source_instances == 8


def test_frame_scene_macro_keeps_no_pose():
    rows = [{'scene_id': 100, 'frame_id': 0, 'hit': 1., 'selected_quality': .8, 'coverage': 1., 'selected_query': 0},
            {'scene_id': 100, 'frame_id': 0, 'hit': 0., 'selected_quality': 0., 'coverage': 1., 'selected_query': None},
            {'scene_id': 100, 'frame_id': 1, 'hit': 1., 'selected_quality': 1., 'coverage': 1., 'selected_query': 0},
            {'scene_id': 101, 'frame_id': 0, 'hit': 0., 'selected_quality': 0., 'coverage': 0., 'selected_query': None}]
    m = aggregate_instance_rows(rows)
    assert m['InstTop1_INST20_at_0_4'] == .375
    assert m['selected_Q_macro'] == .35
    assert m['no_pose_count'] == 2 and m['instance_count'] == 4


@pytest.mark.parametrize('stage', ['pretrain', 'finetune'])
def test_full_vs_resume_keeps_rng_parameters_adam_and_budgets(tmp_path, stage):
    torch.set_num_threads(1)
    seed_all(73)
    full = trainer(tmp_path/'full', stage=stage)
    full.run()
    seed_all(73)
    part = trainer(tmp_path/'part', stage=stage)
    part.run(stop_after=1)
    resumed = trainer(tmp_path/'resumed', stage=stage)
    resumed.resume(tmp_path/'part/last.pt')
    resumed.run()
    for attr in ('step', 'supervised_source_instances', 'supervised_target_observations'):
        assert getattr(full, attr) == getattr(resumed, attr)
    assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(full.source_schedule.state_dict(), resumed.source_schedule.state_dict())
    assert_nested_equal(full.best, resumed.best)
    ckpt = torch.load(tmp_path/'full/best.pt', weights_only=False)
    assert ckpt['policy']['independent_test_estimate'] is False
    assert ckpt['policy']['public_evaluation_status'] == 'deferred'
    assert ckpt['checkpoint_selection_role'] == ('test_selected' if stage == 'pretrain' else 'validation_selected')


def test_best_and_last_are_independent_and_policy_is_not_hidden(tmp_path, monkeypatch):
    import suction20.instance_training as module
    seed_all(5)
    t = trainer(tmp_path)
    t.config['pretrain']['evaluate_every'] = 1
    def fake_selection(model, data, **kwargs):
        # Ranking rule fixture: earlier checkpoint wins although training continues.
        return {'InstTop1_INST20_at_0_4': 1. if t.step == 1 else .2, 'selected_Q_macro': .5}, []
    monkeypatch.setattr(module, 'evaluate_instance_source', fake_selection)
    t.run()
    best = torch.load(tmp_path/'best.pt', weights_only=False)
    last = torch.load(tmp_path/'last.pt', weights_only=False)
    assert best['global_step'] == 1 and last['global_step'] == 2
    assert best['checkpoint_selection_role'] == 'test_selected'
    assert last['checkpoint_selection_role'] == 'budget_endpoint_or_progress'
    assert best['policy']['paper_comparison_ready'] is False


def test_finetune_phase_preserves_encoder_bn_and_head_adam(tmp_path):
    seed_all(32)
    t = trainer(tmp_path, stage='finetune')
    encoder = deepcopy(t.model.encoder.state_dict())
    decoder = deepcopy(t.model.decoder.state_dict())
    bn = deepcopy(t.model.decoder[1].state_dict())
    t.train_one_update()
    assert_nested_equal(t.model.encoder.state_dict(), encoder)
    assert_nested_equal(t.model.decoder.state_dict(), decoder)
    head = next(t.model.heads.parameters())
    assert int(t.optimizer.state[head]['step']) == 1
    t.train_one_update()
    assert_nested_equal(t.model.encoder.state_dict(), encoder)
    assert_nested_equal(t.model.decoder[1].state_dict(), bn)
    assert not torch.equal(t.model.decoder[0].weight, decoder['0.weight'])
    assert int(t.optimizer.state[head]['step']) == 2


def test_resume_rejects_data_asset_change(tmp_path):
    seed_all(1)
    a = trainer(tmp_path/'first'); a.run(stop_after=1)
    b = trainer(tmp_path/'second')
    b.identity['source']['data_identity']['dataset_revision'] = 'replaced'
    with pytest.raises(ValueError, match='Resume dataset definition'):
        b.resume(tmp_path/'first/last.pt')


def test_unknown_teacher_prevents_test_selection(tmp_path):
    class Unknown(MemorySource):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item['labels']['label_valid'][0] = False
            item['labels']['q_inst'][0] = np.nan
            return item
    with pytest.raises(ValueError, match='Incomplete teacher'):
        evaluate_instance_source(TinyInstanceModel(), Unknown('test'))


def test_initialization_does_not_accept_legacy_checkpoint(tmp_path):
    p = tmp_path/'legacy.pt'; torch.save({'model': {}}, p)
    t = trainer(tmp_path/'ft', stage='finetune')
    with pytest.raises(ValueError, match='same source'):
        t.initialize_finetune(p)


def test_source_test_rejects_known_teacher_physics_drift(tmp_path):
    source, selection = MemorySource(), MemorySource('test')
    source.identity['teacher_identity'] = {'physics': {'radius_m': .02}}
    selection.identity['teacher_identity'] = {'physics': {'radius_m': .01}}
    with pytest.raises(ValueError, match='physics definitions differ'):
        InstanceTrainer(TinyInstanceModel(), config(), source, selection, tmp_path, allow_partial=True)


@pytest.mark.parametrize('field', ['teacher_spec', 'schema_version', 'quality_contract'])
def test_source_test_rejects_known_teacher_contract_drift(tmp_path, field):
    source, selection = MemorySource(), MemorySource('test')
    source.identity['teacher_identity'] = {field: 'frozen_v2'}
    selection.identity['teacher_identity'] = {field: 'changed'}
    with pytest.raises(ValueError, match=f'{field} definitions differ'):
        InstanceTrainer(TinyInstanceModel(), config(), source, selection, tmp_path, allow_partial=True)


def test_finetune_keeps_source_selection_provenance_across_resume(tmp_path):
    source = trainer(tmp_path/'source')
    source.run()
    ft = trainer(tmp_path/'ft', stage='finetune')
    ft.initialize_finetune(tmp_path/'source/best.pt')
    lineage = deepcopy(ft.initialization)
    assert lineage['checkpoint_selection_role'] == 'test_selected'
    assert lineage['checkpoint_file_record']['size_bytes'] > 0
    assert lineage['selection_policy']['independent_test_estimate'] is False
    ft.run(stop_after=1)
    resumed = trainer(tmp_path/'resumed', stage='finetune')
    resumed.resume(tmp_path/'ft/last.pt')
    assert resumed.initialization == lineage
    assert resumed.state_dict()['initialization'] == lineage


@pytest.mark.parametrize('attempts', [False, True])
@pytest.mark.parametrize('raises', [False, True])
def test_evaluators_restore_all_mixed_modes_and_bn_statistics(attempts, raises):
    model = TinyInstanceModel()
    model.train()
    model.decoder.eval()
    model.decoder[2].train()  # Deliberate mixed descendant state.
    model.encoder.eval()
    before = {name: module.training for name, module in model.named_modules()}
    bn_before = deepcopy(model.decoder[1].state_dict())
    base = MemoryAttempts if attempts else MemorySource
    class Data(base):
        def __getitem__(self, index):
            if raises and index == 1:
                raise RuntimeError('intentional evaluation failure')
            return super().__getitem__(index)
    evaluator = evaluate_instance_attempts if attempts else evaluate_instance_source
    if raises:
        with pytest.raises(RuntimeError, match='intentional evaluation failure'):
            evaluator(model, Data('test'))
    else:
        evaluator(model, Data('test'))
    assert {name: module.training for name, module in model.named_modules()} == before
    # A caller may immediately run a forward without reapplying trainer phase.
    model(MemorySource()[0]['observation'])
    assert_nested_equal(model.decoder[1].state_dict(), bn_before)


@pytest.mark.parametrize('attempts', [False, True])
def test_streaming_evaluation_keeps_exact_metrics_without_record_list(attempts):
    model = TinyInstanceModel()
    data = MemoryAttempts('test') if attempts else MemorySource('test')
    evaluator = evaluate_instance_attempts if attempts else evaluate_instance_source
    expected, records = evaluator(model, data)
    streamed = []
    actual, empty = evaluator(model, data, collect_records=False, record_sink=streamed.append)
    assert actual == expected and streamed == records and empty == []


@pytest.mark.parametrize('threshold,has_prediction', [(20, True), (21, False), (150, False)])
def test_training_and_selection_use_configurable_support_threshold(tmp_path, threshold, has_prediction):
    class SmallSource(MemorySource):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item['observation']['support_points'] = item['observation']['support_points'][:20]
            return item
    model = TinyInstanceModel()
    model.min_support_points = threshold
    _, rows = evaluate_instance_source(model, SmallSource('test'))
    assert all((row['selected_query'] is not None) == has_prediction for row in rows)
    t = InstanceTrainer(model, config(), SmallSource(), SmallSource('test'), tmp_path, allow_partial=True)
    if has_prediction:
        t.train_one_update()
        assert t.step == 1 and t.supervised_source_instances == 2
    else:
        with pytest.raises(RuntimeError, match='Insufficient supervised source draws'):
            t.train_one_update()
        assert t.step == 0 and t.supervised_source_instances == 0


@pytest.mark.parametrize('retention,kept', [('best_and_final', [1, 3, 4]), ('all', [1, 2, 3, 4]), ('none', [])])
def test_periodic_records_are_streamed_and_retained_by_policy(tmp_path, monkeypatch, retention, kept):
    import suction20.instance_training as module
    cfg = config()
    cfg['pretrain'].update(steps=4, evaluate_every=1, evaluation_records=retention)
    t = InstanceTrainer(TinyInstanceModel(), cfg, MemorySource(), MemorySource('test'), tmp_path, allow_partial=True)
    values = [.8, .1, .9, .2]
    def selection(model, data, *, collect_records, record_sink):
        assert collect_records is False
        if record_sink is not None:
            for index in range(11):
                record_sink({'sample_id': index, 'step': t.step})
        return {'InstTop1_INST20_at_0_4': values[t.step-1], 'selected_Q_macro': .5}, []
    monkeypatch.setattr(module, 'evaluate_instance_source', selection)
    t.run()
    for step in range(1, 5):
        report = json.loads((tmp_path/f'evaluation_step{step:06d}.json').read_text())
        assert 'records' not in report
        assert report['record_details']['retained'] == (step in kept)
        details = tmp_path/f'evaluation_step{step:06d}.records.jsonl.gz'
        assert details.exists() == (step in kept)
        if step in kept:
            with gzip.open(details, 'rt') as stream:
                assert [json.loads(line) for line in stream] == [
                    {'sample_id': i, 'step': step} for i in range(11)]
    assert not list(tmp_path.glob('*.tmp'))
    assert torch.load(tmp_path/'best.pt', weights_only=False)['global_step'] == 3


def test_repeated_evaluation_retains_current_best_details_consistently(tmp_path):
    t = trainer(tmp_path)
    t.train_one_update()
    first, second = t.evaluate(), t.evaluate()
    details = tmp_path/second['record_details']['path']
    assert first['best_updated'] and not second['best_updated']
    assert second['record_details']['retained'] and details.exists()
    t.config['pretrain']['evaluation_records'] = 'none'
    disabled = t.evaluate()
    assert not disabled['record_details']['retained'] and not details.exists()


def test_finetune_rejects_task_definition_and_cross_manifest_capture_conflicts(tmp_path):
    target, selection = MemoryAttempts(), MemoryAttempts('val')
    target.identity['success_definition'] = 'seal_only'
    selection.identity['success_definition'] = 'lift_and_transport'
    with pytest.raises(ValueError, match='success_definition'):
        InstanceTrainer(TinyInstanceModel(), config(), MemorySource(), selection, tmp_path,
                        stage='finetune', target_dataset=target, allow_partial=True)
    selection.identity['success_definition'] = target.identity['success_definition']
    target.observation_owners = {'depth:/same-frame.png': ('train', 'capture-group')}
    selection.observation_owners = {'depth:/same-frame.png': ('val', 'capture-group')}
    with pytest.raises(ValueError, match='Shared target observation crosses'):
        InstanceTrainer(TinyInstanceModel(), config(), MemorySource(), selection, tmp_path,
                        stage='finetune', target_dataset=target, allow_partial=True)


def test_legacy_checksum_metadata_is_ignored_and_not_reemitted(tmp_path):
    original = trainer(tmp_path/'original')
    original.run(stop_after=1)
    state = torch.load(tmp_path/'original/last.pt', weights_only=False)
    state['identity'].pop('runtime_contract')
    state['identity'].pop('config_definition')
    state['identity']['code_sha256'] = 'obsolete-and-intentionally-wrong'
    state['config']['config_sha256'] = 'obsolete'
    state['source_schedule']['identity'] = 'opaque-legacy-checksum'
    state['initialization'] = {'kind': 'legacy', 'checkpoint_sha256': 'obsolete'}
    path = tmp_path/'legacy.pt'
    torch.save(state, path)
    resumed = trainer(tmp_path/'resumed')
    resumed.resume(path)
    resumed.run()
    saved = torch.load(tmp_path/'resumed/last.pt', weights_only=False)
    assert saved['global_step'] == 2
    assert saved['initialization'] == {'kind': 'legacy'}
    assert 'config_sha256' not in saved['config'] and 'code_sha256' not in saved['identity']


@pytest.mark.parametrize('dataset', ['source', 'selection', 'target'])
@pytest.mark.parametrize('field', ['length', 'data_identity'])
def test_legacy_resume_still_checks_available_dataset_fields(tmp_path, dataset, field):
    original = trainer(tmp_path/'original', stage='finetune')
    state = original.state_dict()
    state['identity'].pop('runtime_contract')
    state['source_schedule']['identity'] = 'opaque-legacy-checksum'
    state['target_schedule']['identity'] = 'opaque-legacy-checksum'
    if field == 'length':
        state['identity'][dataset][field] += 1
    else:
        state['identity'][dataset][field]['dataset_revision'] = 'different-known-data'
    path = tmp_path/'legacy.pt'
    torch.save(state, path)
    resumed = trainer(tmp_path/'resumed', stage='finetune')
    with pytest.raises(ValueError, match=f'Resume dataset definition differs: {dataset}'):
        resumed.resume(path)
