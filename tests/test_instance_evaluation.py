"""Native evaluator routing/provenance over actual manifest/teacher fixtures."""
from copy import deepcopy
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from scripts import evaluate_instance as cli
from suction20.instance_data import InstanceDataset, build_source_instances
from test_instance_data import _assets, _camera


class _FixedPrediction(torch.nn.Module):
    def forward(self, observation):
        count = len(observation['query_points'])
        return {'q_inst': torch.linspace(.2, .8, count), 'valid': torch.ones(count, dtype=torch.bool)}

    def encode_instance(self, observation):
        return observation

    def score_at_pose(self, encoded, points, normals):
        valid = np.isfinite(points).all(1) & np.isfinite(normals).all(1)
        valid &= np.linalg.norm(normals, axis=1) > 1e-8
        return {'task_logit': torch.ones(len(points)), 'valid': torch.as_tensor(valid)}


def _metadata(source_identity, stage='pretrain'):
    role, split = ('test_selected', 'test') if stage == 'pretrain' else ('validation_selected', 'val')
    return {'schema': 'instance_loaded_checkpoint_v1', 'route': 'b', 'stage': stage,
        'checkpoint_file': {'path': 'test.pt', 'size_bytes': 1}, 'quality_contract': 'gt_INST20_pose_v2',
        'checkpoint_kind': 'best', 'global_step': 1,
        'selection_policy': {'selection_split': split, 'selection_role': role,
            'independent_test_estimate': False, 'train_scene_range': [0, 99],
            'public_evaluation_status': 'deferred', 'paper_comparison_ready': False},
        'checkpoint_selection_role': role,
        'checkpoint_identity': {'source': {'data_identity': source_identity}},
        'public_evaluation': {'status': 'deferred', 'paper_comparison_ready': False}}


def _source_fixture(tmp_path):
    root, release, _ = _assets(tmp_path, scene_id=100)
    manifest, _ = build_source_instances(root, release, tmp_path / 'cache',
        split='test', scenes=[100], frames=[0])
    identity = InstanceDataset(manifest, split='test').identity
    return manifest, identity


def test_source_native_report_preserves_test_selection_and_rows(tmp_path, monkeypatch):
    manifest, identity = _source_fixture(tmp_path)
    metadata = _metadata(identity)
    monkeypatch.setattr(cli, 'load_instance_checkpoint', lambda *a, **k: (_FixedPrediction(), deepcopy(metadata)))
    path = tmp_path / 'report.json'
    report = cli.main(['--checkpoint', 'own.pt', '--manifest', str(manifest), '--domain', 'source',
        '--output', str(path), '--device', 'cpu', '--allow-partial-manifests'])
    assert json.loads(path.read_text()) == report
    assert report['test_selected'] and report['independent_test_estimate'] is False
    assert report['checkpoint_selection_policy'] == metadata['selection_policy']
    assert report['public_evaluation']['status'] == 'deferred'
    assert report['metrics']['instance_count'] == 1 and len(report['records']) == 1
    assert report['records'][0]['scene_id'] == 100
    assert report['coverage']['partial_manifests_allowed']
    assert not any(key.startswith('AP') for key in report['metrics'])
    metadata['checkpoint_kind'] = 'last'
    metadata['checkpoint_selection_role'] = 'budget_endpoint_or_progress'
    last = cli.main(['--checkpoint', 'last.pt', '--manifest', str(manifest), '--domain', 'source',
        '--output', str(tmp_path / 'last.json'), '--device', 'cpu', '--allow-partial-manifests'])
    assert not last['test_selected'] and last['source_run_uses_test_selection']
    assert last['independent_test_estimate'] is False


def test_source_partial_and_physics_mismatch_fail_explicitly(tmp_path, monkeypatch):
    manifest, identity = _source_fixture(tmp_path)
    metadata = _metadata(identity)
    monkeypatch.setattr(cli, 'load_instance_checkpoint', lambda *a, **k: (_FixedPrediction(), deepcopy(metadata)))
    args = ['--checkpoint', 'own.pt', '--manifest', str(manifest), '--domain', 'source',
        '--output', str(tmp_path / 'report.json'), '--device', 'cpu']
    with pytest.raises(ValueError, match='full 100-train/90-test'):
        cli.main(args)
    metadata['checkpoint_identity']['source']['data_identity'] = deepcopy(identity)
    metadata['checkpoint_identity']['source']['data_identity']['teacher_identity']['physics']['radius_m'] = .01
    with pytest.raises(ValueError, match='physics definitions differ'):
        cli.main(['--checkpoint', 'own.pt', '--manifest', str(manifest), '--domain', 'source',
                  '--output', str(tmp_path / 'bad.json'), '--device', 'cpu', '--allow-partial-manifests'])


def test_explicit_fp32_evaluation_batch_preserves_report(tmp_path, monkeypatch):
    manifest, identity = _source_fixture(tmp_path)
    class BatchedPrediction(_FixedPrediction):
        def __init__(self):
            super().__init__()
            self.batch_calls = []

        def forward_instances(self, observations):
            self.batch_calls.append(len(observations))
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            return [self(observation) for observation in observations]

    model = BatchedPrediction()
    monkeypatch.setattr(cli, 'load_instance_checkpoint',
                        lambda *a, **k: (model, _metadata(identity)))
    args = ['--checkpoint', 'own.pt', '--manifest', str(manifest), '--domain', 'source',
            '--device', 'cpu', '--allow-partial-manifests']
    original = cli.main(args + ['--output', str(tmp_path / 'original.json')])
    batched = cli.main(args + ['--output', str(tmp_path / 'batched.json'),
        '--microbatch-instances', '32', '--preparation-workers', '0'])
    assert model.batch_calls == [1]
    assert original['metrics'] == batched['metrics']
    assert original['records'] == batched['records']
    runtime = batched['checkpoint']['runtime_evaluation_execution']
    assert runtime['evaluation_microbatch_instances'] == 32
    assert runtime['evaluation_workers'] == 0 and runtime['precision'] == 'fp32'
    assert runtime['compile_mode'] is None and not runtime['allow_tf32']


def test_pretrain_cannot_report_target_probabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'load_instance_checkpoint', lambda *a, **k: (_FixedPrediction(), _metadata({})))
    with pytest.raises(ValueError, match='source task head is untrained'):
        cli.main(['--checkpoint', 'own.pt', '--manifest', str(tmp_path / 'unneeded.jsonl'),
            '--domain', 'target', '--output', str(tmp_path / 'report.json'), '--device', 'cpu'])


def test_target_native_report_keeps_unsupported_actions(tmp_path, monkeypatch):
    root, _, _ = _assets(tmp_path)
    base = root / 'scenes/scene_0000/realsense'
    observation = dict(format='rgbd', rgb_path=str(base / 'rgb/0000.png'),
        depth_path=str(base / 'depth/0000.png'), instance_mask_path=str(base / 'label/0000.png'),
        depth_scale_m=.001, intrinsics=_camera().tolist(), g_cam=[0., 0., -1.])
    rows = [dict(schema_version='suction20.target.v1', label_kind='attempt_success',
        sample_id=f'action{i}', group_id='group1', split='test', observation=observation,
        success_definition='pickup_and_transport', attempt=dict(point_cam_m=[0., 0., .6],
            normal_out_cam=[0., 0., -1.] if i < 2 else [0., 0., 0.], instance_id=1,
            success=i % 2, label_valid=True)) for i in range(3)]
    manifest = tmp_path / 'target.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    metadata = _metadata({}, stage='finetune')
    metadata['initialization'] = {'selection_policy': {'selection_role': 'test_selected'}}
    monkeypatch.setattr(cli, 'load_instance_checkpoint', lambda *a, **k: (_FixedPrediction(), metadata))
    report = cli.main(['--checkpoint', 'own.pt', '--manifest', str(manifest), '--domain', 'target',
        '--output', str(tmp_path / 'target_report.json'), '--device', 'cpu'])
    assert report['checkpoint_selection_role'] == 'validation_selected'
    assert report['source_run_uses_test_selection'] is True
    assert report['independent_test_estimate'] is False
    assert report['metrics']['supported_attempts'] == 2
    assert report['metrics']['unsupported_attempts'] == 1 and len(report['records']) == 3
    assert report['records'][-1]['probability'] is None
    assert report['records'][-1]['outcome'] is None
