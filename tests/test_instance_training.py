"""Optimizer, loss, selection provenance and resume integration contracts."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch import nn

from suction20.instance_training import (InstanceSchedule, InstanceTrainer,
    aggregate_instance_rows, check_source_splits, evaluate_instance_source,
    load_instance_config, seed_all, source_loss)


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
        self.identity = {'fixture': 'synthetic_only', 'split': split, 'asset_hash': 'v1'}

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
    def fake_selection(model, data):
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
    b.identity['source']['data_identity']['asset_hash'] = 'replaced'
    with pytest.raises(ValueError, match='Strict resume identity'):
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


def test_source_test_rejects_shared_object_geometry_drift(tmp_path):
    source, selection = MemorySource(), MemorySource('test')
    source.identity['object_assets'] = {'7': {'dense_sha256': 'original', 'seal_candidates_sha256': 'same'}}
    selection.identity['object_assets'] = {'7': {'dense_sha256': 'replaced', 'seal_candidates_sha256': 'same'}}
    with pytest.raises(ValueError, match='assets differ for object 7'):
        InstanceTrainer(TinyInstanceModel(), config(), source, selection, tmp_path, allow_partial=True)


def test_finetune_keeps_source_selection_provenance_across_resume(tmp_path):
    source = trainer(tmp_path/'source')
    source.run()
    ft = trainer(tmp_path/'ft', stage='finetune')
    ft.initialize_finetune(tmp_path/'source/best.pt')
    lineage = deepcopy(ft.initialization)
    assert lineage['checkpoint_selection_role'] == 'test_selected'
    assert len(lineage['checkpoint_sha256']) == 64
    assert lineage['selection_policy']['independent_test_estimate'] is False
    ft.run(stop_after=1)
    resumed = trainer(tmp_path/'resumed', stage='finetune')
    resumed.resume(tmp_path/'ft/last.pt')
    assert resumed.initialization == lineage
    assert resumed.state_dict()['initialization'] == lineage
