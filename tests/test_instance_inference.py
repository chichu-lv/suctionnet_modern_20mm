"""Native postprocessing, checkpoint, and file input acceptance tests.

The small injected scorer isolates protocol behavior. These tests do not
substitute a CPU network for either production architecture.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from suction20.instance_data import save_observation
from suction20.instance_geometry import (NORMAL_SPEC, SAMPLER_SPEC, PRECISION_SPEC,
                                        prepare_instance)
from suction20.instance_inference import (contact_transform, infer_instance,
    load_instance_checkpoint, load_raw_observation, nms_pose_candidates, resolve_ranking)
from suction20.instance_training import CHECKPOINT_SCHEMA, QUALITY_CONTRACT
import suction20.instance_inference as inference
from scripts import infer_instance as cli


class FixedScorer(nn.Module):
    def __init__(self, *, scores=None, logits=None, valid=None, output_override=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.scores, self.logits, self.valid = scores, logits, valid
        self.output_override = output_override or {}
        self.encode_calls = 0
        self.last_points, self.last_normals = None, None

    def encode_instance(self, observation):
        self.encode_calls += 1
        return observation

    def score_at_pose(self, encoded, p, n):
        self.last_points, self.last_normals = np.array(p, copy=True), np.array(n, copy=True)
        count = len(p)
        scores = np.full(count, .5) if self.scores is None else np.asarray(self.scores)
        logits = np.full(count, 0.) if self.logits is None else np.asarray(self.logits)
        valid = np.ones(count, bool) if self.valid is None else np.asarray(self.valid, bool)
        result = {'q_inst': torch.tensor(scores, dtype=torch.float64),
                  'seal': torch.full((count,), .8), 'wrench': torch.full((count,), .7),
                  'ell': torch.full((count,), -2.), 'task_logit': torch.tensor(logits, dtype=torch.float64),
                  'valid': torch.tensor(valid), 'points_cam_m': torch.tensor(p),
                  'normals_out_cam': torch.tensor(n / np.linalg.norm(n, axis=1, keepdims=True))}
        result.update(self.output_override)
        return result


def observation(count=3, *, support=100):
    return {'query_points': np.column_stack((np.arange(count) * .02,
                                             np.zeros(count), np.full(count, .6))),
            'query_normals': np.tile([0., 0., -1.], (count, 1)),
            'query_ids': np.arange(count, dtype=np.int64),
            'query_source_ids': np.arange(count, dtype=np.int64) + 1000,
            'query_valid': np.ones(count, bool), 'support_points': np.zeros((support, 3)),
            'intrinsics': np.array([[600., 0, 30], [0, 600., 30], [0, 0, 1.]]),
            'normal_spec': NORMAL_SPEC, 'sampler_spec': SAMPLER_SPEC, 'precision_spec': PRECISION_SPEC,
            'instance_uid': 'test-object'}


@pytest.mark.parametrize('distance,angle,expected', [
    (.009, 0., 1), (.010, 0., 2), (.011, 0., 2),
    (0., 14.99, 1), (0., 15., 2), (0., 15.01, 2), (0., 180., 2),
])
def test_nms_strict_distance_and_directed_angle_boundaries(distance, angle, expected):
    theta = math.radians(angle)
    points = np.array([[0., 0., .6], [distance, 0., .6]])
    normals = np.array([[0., 0., 1.], [math.sin(theta), 0., math.cos(theta)]])
    kept = nms_pose_candidates(points, normals, [1., .9], np.array([8, 3]))
    assert len(kept) == expected
    if expected == 1:
        assert kept.tolist() == [0]


def test_nms_ties_use_query_identity_and_reject_bad_vectors():
    points = np.tile([0., 0., .6], (3, 1))
    normals = np.tile([0., 0., -1.], (3, 1))
    assert nms_pose_candidates(points, normals, [1., 1., 1.], np.array([20, 4, 9])).tolist() == [1]
    with pytest.raises(ValueError, match='nonzero'):
        nms_pose_candidates(points, np.zeros_like(normals), [1., 1., 1.], np.array([20, 4, 9]))


@pytest.mark.parametrize('normal', [[0, 0, -1], [1, 0, 0], [.3, -.4, .8]])
def test_se3_preserves_contact_and_uses_negative_outward_axis(normal):
    point = np.array([.037123456789, -.02, .600000000001])
    normal = np.asarray(normal, float)
    transform = contact_transform(point, normal)
    rotation = transform[:3, :3]
    np.testing.assert_array_equal(transform[:3, 3], point)
    np.testing.assert_allclose(rotation[:, 2], -normal / np.linalg.norm(normal), atol=2e-16)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=3e-16)
    assert np.linalg.det(rotation) == pytest.approx(1., abs=4e-16)


def test_inference_preserves_scored_pose_and_source_head_semantics():
    obs = observation()
    obs['query_points'][1] += [1.23e-10, 2.34e-10, 3.45e-10]
    obs['query_normals'][1] = [.6, .8, -.25]
    model = FixedScorer(scores=[.2, .9, .7], logits=[10., -10., 5.])
    model.train()
    report = infer_instance(model, obs, stage='pretrain')
    assert model.training and model.encode_calls == 1
    assert report['ranking'] == 'quality'
    assert [p['query_id'] for p in report['poses']] == [1, 2, 0]
    best = report['poses'][0]
    assert best['task_probability'] is None and best['task_logit'] is None
    assert best['source_point_id'] == 1001
    np.testing.assert_array_equal(model.last_points, obs['query_points'])
    np.testing.assert_array_equal(model.last_normals, obs['query_normals'])
    np.testing.assert_array_equal(best['point_cam_m'], obs['query_points'][1])
    np.testing.assert_allclose(best['normal_out_cam'], obs['query_normals'][1]/np.linalg.norm(obs['query_normals'][1]))
    assert report['public_evaluation']['status'] == 'deferred'
    assert report['counts']['before_nms'] == report['counts']['after_nms'] == 3
    json.dumps(report, allow_nan=False)


def test_task_probability_and_mixed_ranking_are_distinct():
    obs = observation(2)
    model = FixedScorer(scores=[.01, 1.], logits=[math.log(4.), math.log(1.5)])
    plain = infer_instance(model, obs, stage='finetune')
    mixed = infer_instance(model, obs, stage='finetune', ranking='mixed', alpha=.5)
    assert plain['poses'][0]['query_id'] == 0
    assert mixed['poses'][0]['query_id'] == 1
    row = next(p for p in mixed['poses'] if p['query_id'] == 0)
    assert row['task_probability'] == pytest.approx(.8)
    assert row['ranking_score'] == pytest.approx(.08)
    for mode in ('task', 'mixed'):
        with pytest.raises(ValueError, match='no trained task head'):
            infer_instance(model, obs, stage='pretrain', ranking=mode)
    with pytest.raises(ValueError, match='only meaningful'):
        resolve_ranking('finetune', 'auto', .5)


def test_top_five_ties_and_valid_empty_instances():
    obs = observation(8)
    obs['query_ids'] = np.array([11, 9, 15, 2, 8, 7, 1, 3])
    report = infer_instance(FixedScorer(), obs, stage='pretrain')
    assert [p['query_id'] for p in report['poses']] == [1, 2, 3, 7, 8]
    assert report['counts']['after_nms'] == 8 and report['counts']['returned'] == 5
    empty_model = FixedScorer()
    empty = infer_instance(empty_model, observation(0, support=0), stage='pretrain')
    assert empty['status'] == 'no_pose' and empty['poses'] == [] and empty_model.encode_calls == 0
    assert empty['reason'] == 'no_eligible_queries'
    unsupported = infer_instance(FixedScorer(valid=[False, False]), observation(2, support=20), stage='pretrain')
    assert unsupported['status'] == 'no_pose' and unsupported['reason'] == 'insufficient_support'
    assert unsupported['rejection_counts'] == {'insufficient_support': 2}


@pytest.mark.parametrize('threshold,expected_calls', [(20, 1), (21, 0), (150, 0)])
def test_inference_uses_model_support_threshold(threshold, expected_calls):
    model = FixedScorer()
    model.min_support_points = threshold
    report = infer_instance(model, observation(2, support=20), stage='pretrain')
    assert model.encode_calls == expected_calls
    assert report['status'] == ('ok' if expected_calls else 'no_pose')


@pytest.mark.parametrize('override,match', [
    ({'q_inst': torch.tensor([float('nan'), .5])}, 'nonfinite'),
    ({'q_inst': torch.tensor([1.1, .5])}, 'outside'),
    ({'q_inst': torch.tensor([.5])}, 'shape'),
    ({'valid': torch.ones(2)}, 'boolean'),
    ({'points_cam_m': torch.zeros((2, 3))}, 'moved'),
    ({'normals_out_cam': torch.tensor([[0., 0., 1.], [0., 0., 1.]])}, 'directed'),
])
def test_malformed_outputs_raise_instead_of_becoming_empty(override, match):
    with pytest.raises(ValueError, match=match):
        infer_instance(FixedScorer(output_override=override), observation(2), stage='pretrain')
    if match == 'nonfinite':
        with pytest.raises(ValueError, match=match):
            infer_instance(FixedScorer(valid=[False, False], output_override=override), observation(2), stage='pretrain')


def checkpoint_state(model, *, route='a', stage='pretrain'):
    config = {'schema': 'instance_runtime_v1', 'route': route, 'quality_contract': QUALITY_CONTRACT,
              'public_evaluation': {'status': 'deferred'}, 'model': {},
              'pretrain': {'selection_split': 'test', 'train_scene_range': [0, 99]}}
    policy = {'selection_split': 'test' if stage == 'pretrain' else 'val',
              'selection_role': 'test_selected' if stage == 'pretrain' else 'validation_selected',
              'independent_test_estimate': False, 'train_scene_range': [0, 99],
              'public_evaluation_status': 'deferred', 'paper_comparison_ready': False}
    identity = {'route': route, 'stage': stage, 'quality_contract': QUALITY_CONTRACT,
                'config_definition': deepcopy(config),
                'source': {'data_identity': {'normal_spec': NORMAL_SPEC, 'sampler_spec': SAMPLER_SPEC,
                            'precision_spec': PRECISION_SPEC, 'quality_contract': QUALITY_CONTRACT}}}
    return {'schema': CHECKPOINT_SCHEMA, 'config': config, 'identity': identity, 'policy': policy,
            'stage': stage, 'checkpoint_kind': 'best', 'checkpoint_selection_role': policy['selection_role'],
            'global_step': 7, 'model': model.state_dict(), 'rng': {'numpy': np.random.get_state()}}


def test_checkpoint_loading_preserves_policy_and_strict_weights(tmp_path, monkeypatch):
    model = FixedScorer()
    model.weight.data.fill_(4.2)
    state = checkpoint_state(model)
    path = tmp_path/'best.pt'
    torch.save(state, path)
    factory_calls = []

    def factory(config, device='cpu'):
        factory_calls.append(deepcopy(config))
        return FixedScorer().to(device)

    monkeypatch.setattr(inference, 'create_instance_model', factory)
    loaded, metadata = load_instance_checkpoint(path, backbone_root=tmp_path/'relocated-backbone')
    assert not loaded.training and loaded.weight.item() == pytest.approx(4.2)
    assert factory_calls[0]['model']['backbone_root'] == str((tmp_path/'relocated-backbone').resolve())
    assert metadata['selection_policy']['selection_role'] == 'test_selected'
    assert metadata['selection_policy']['independent_test_estimate'] is False
    assert metadata['checkpoint_selection_role'] == 'test_selected'
    assert metadata['initialization'] is None
    assert metadata['checkpoint_file']['path'] == str(path.resolve())
    assert metadata['checkpoint_file']['size_bytes'] == path.stat().st_size
    state['model']['unexpected_weight'] = torch.ones(1)
    torch.save(state, path)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        load_instance_checkpoint(path)


def test_finetune_parent_provenance_survives_without_parent_file(tmp_path, monkeypatch):
    state = checkpoint_state(FixedScorer(), stage='finetune')
    state['initialization'] = {'checkpoint_file_record': {'path': 'source/best.pt', 'size_bytes': 1234}, 'checkpoint_kind': 'best',
        'global_step': 2880, 'checkpoint_selection_role': 'test_selected',
        'selection_policy': {'selection_role': 'test_selected', 'independent_test_estimate': False}}
    path = tmp_path/'finetuned.pt'
    torch.save(state, path)
    monkeypatch.setattr(inference, 'create_instance_model', lambda config, device='cpu': FixedScorer().to(device))
    _, metadata = load_instance_checkpoint(path)
    assert metadata['stage'] == 'finetune'
    assert metadata['initialization'] == state['initialization']
    assert metadata['initialization']['checkpoint_selection_role'] == 'test_selected'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('mutate,match', [
    (lambda s: s.update(schema='legacy'), 'Expected'),
    (lambda s: s['identity'].update(quality_contract='gt_SW20'), 'quality'),
    (lambda s: s['identity']['source']['data_identity'].update(normal_spec='one-pixel'), 'normal/sampler'),
    (lambda s: s['policy'].update(independent_test_estimate=True), 'selection policy'),
    (lambda s: s['config']['model'].update(backbone_points=8), 'config definition'),
    (lambda s: s.update(global_step=0), 'completed optimizer'),
])
def test_checkpoint_mismatches_are_not_accepted(tmp_path, monkeypatch, mutate, match):
    state = checkpoint_state(FixedScorer())
    mutate(state)
    path = tmp_path/'bad.pt'
    torch.save(state, path)
    monkeypatch.setattr(inference, 'create_instance_model', lambda *args, **kwargs: pytest.fail('Invalid checkpoint reached model factory'))
    with pytest.raises(ValueError, match=match):
        load_instance_checkpoint(path)


def raw_files(tmp_path):
    shape = (31, 35)
    rgb = np.full((*shape, 3), 127, np.uint8)
    depth = np.full(shape, 600, np.uint16)
    mask = np.zeros(shape, np.uint8)
    mask[3:28, 4:31] = 7
    rgb_path, depth_path, mask_path = [tmp_path/name for name in ('rgb.png', 'depth.png', 'mask.png')]
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(depth).save(depth_path)
    Image.fromarray(mask).save(mask_path)
    intrinsics_path = tmp_path/'intrinsics.json'
    intrinsics_path.write_text(json.dumps({'fx': 600., 'fy': 600., 'cx': 17., 'cy': 15.}))
    return {'rgb_path': rgb_path, 'depth_path': depth_path, 'mask_path': mask_path,
            'intrinsics_path': intrinsics_path, 'depth_scale_m': .001, 'gravity': [0., 1., 0.],
            'instance_id': 7}


def test_raw_rgbd_requires_declared_mask_and_preserves_original_depth(tmp_path):
    options = raw_files(tmp_path)
    obs = load_raw_observation(**options)
    assert len(obs['support_points']) == 25*27 and len(obs['query_points']) == 256
    np.testing.assert_allclose(obs['query_points'][:, 2], .6)
    assert np.all(obs['query_normals'][:, 2] < -.999)
    with pytest.raises(ValueError, match='explicit instance_id'):
        load_raw_observation(**{**options, 'instance_id': None})
    with pytest.raises(ValueError, match='positive'):
        load_raw_observation(**{**options, 'depth_scale_m': 0})


def test_unlabeled_observation_cli_and_atomic_failure(tmp_path, monkeypatch):
    obs = load_raw_observation(**raw_files(tmp_path))
    prepared = tmp_path/'observation.npz'
    save_observation(prepared, obs)
    output = tmp_path/'predictions.json'
    metadata = {'stage': 'pretrain', 'selection_policy': {'selection_role': 'test_selected',
                 'independent_test_estimate': False}}
    monkeypatch.setattr(cli, 'load_instance_checkpoint', lambda *args, **kwargs: (FixedScorer(), metadata))
    cli.main(['--checkpoint', str(tmp_path/'own.pt'), '--observation', str(prepared),
              '--output', str(output), '--device', 'cpu'])
    result = json.loads(output.read_text())
    assert result['inputs']['labels_read'] is False
    assert result['summary']['instances'] == 1 and result['predictions'][0]['status'] == 'ok'
    assert result['checkpoint_selection_policy']['selection_role'] == 'test_selected'
    saved = output.read_bytes()

    def fail_after_one():
        yield {'status': 'ok'}
        raise ValueError('model crashed')

    with pytest.raises(ValueError, match='crashed'):
        cli._stream_report(output, {'schema': 'fixture'}, fail_after_one())
    assert output.read_bytes() == saved
    assert not list(tmp_path.glob('predictions.json.*.tmp'))


def test_cli_rejects_unprovided_raw_calibration(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(['--checkpoint', 'own.pt', '--rgb', 'rgb.png', '--depth', 'depth.png',
                  '--mask', 'mask.png', '--output', str(tmp_path/'out.json')])
