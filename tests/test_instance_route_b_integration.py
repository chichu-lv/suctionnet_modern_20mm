"""One real route-B update over official-format synthetic RGB-D/teacher assets.

This verifies integration and checkpoint contracts, not real-dataset accuracy,
calibration, throughput, or completion of the source training protocol.
"""
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from scripts.evaluate_instance import main as evaluate_main
from suction20.instance_data import InstanceDataset, build_source_instances
from suction20.instance_inference import infer_instance, load_instance_checkpoint
from suction20.instance_teacher import empty_labels
from suction20.instance_training import (InstanceTrainer, create_instance_model,
    load_instance_config, seed_all, source_loss)
from test_instance_data import _assets


def test_actual_b_teacher_train_checkpoint_resume_infer_evaluate(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _actual_b_integration(tmp_path)
    finally:
        torch.set_num_threads(old_threads)


def _actual_b_integration(tmp_path):
    root, release, _ = _assets(tmp_path / 'train_fixture', scene_id=0)
    test_root, _, _ = _assets(tmp_path / 'test_fixture', scene_id=100)
    shutil.copytree(test_root / 'scenes/scene_0100', root / 'scenes/scene_0100')
    train_manifest, _ = build_source_instances(root, release, tmp_path / 'train_cache',
        split='train', scenes=[0], frames=[0], train_queries=8)
    test_manifest, _ = build_source_instances(root, release, tmp_path / 'test_cache',
        split='test', scenes=[100], frames=[0])
    source = InstanceDataset(train_manifest, split='train', training_queries=8)
    selection = InstanceDataset(test_manifest, split='test')
    assert len(source[0]['observation']['query_points']) == 8
    assert len(selection[0]['observation']['query_points']) == 256
    assert source.identity['teacher_identity'] == selection.identity['teacher_identity']

    config = load_instance_config(Path(__file__).resolve().parents[1] / 'configs/instance_b.yaml')
    config['pretrain'].update(steps=1, accumulate_instances=1, queries=8, warmup_steps=1,
                              evaluate_every=1, checkpoint_every=1)
    config['model']['query_chunk'] = 32
    config['rgb_augmentation'] = True
    seed_all(config['seed'])
    model = create_instance_model(config, 'cpu')
    assert type(model).__name__ == 'InstanceSuction2DFastV3'
    initial = model.output_heads['q_inst'][-1].weight.detach().clone()
    run_dir = tmp_path / 'run'
    trainer = InstanceTrainer(model, config, source, selection, run_dir, allow_partial=True)
    summary = trainer.run()
    assert summary['completed_budget'] and summary['completed_steps'] == 1
    assert summary['supervised_source_instances'] == 1
    assert summary['policy']['selection_role'] == 'test_selected'
    assert not summary['policy']['independent_test_estimate']
    assert not torch.equal(initial, model.output_heads['q_inst'][-1].weight)
    records = [json.loads(line) for line in (run_dir / 'training.jsonl').read_text().splitlines()]
    assert len(records) == 1 and np.isfinite(records[0]['source_loss'])
    assert np.isfinite(records[0]['grad_norm']) and records[0]['grad_norm'] > 0
    for filename, kind, role in [('last.pt', 'last', 'budget_endpoint_or_progress'),
                                 ('best.pt', 'best', 'test_selected')]:
        state = torch.load(run_dir / filename, map_location='cpu', weights_only=False)
        assert state['checkpoint_kind'] == kind and state['checkpoint_selection_role'] == role
        assert state['policy']['public_evaluation_status'] == 'deferred'
        assert state['identity']['partial_manifests_allowed']
        assert state['identity']['source']['data_identity']['teacher_identity'] == source.identity['teacher_identity']
        del state
    del trainer, model
    gc.collect()

    # A strict resume consumes the full real optimizer/RNG/data/code identity.
    resumed_model = create_instance_model(config, 'cpu')
    resumed = InstanceTrainer(resumed_model, config, source, selection, run_dir, allow_partial=True)
    resumed.resume(run_dir / 'last.pt')
    assert resumed.step == 1 and resumed.supervised_source_instances == 1
    assert resumed.best['step'] == 1
    del resumed, resumed_model
    gc.collect()

    loaded, metadata = load_instance_checkpoint(run_dir / 'best.pt', device='cpu')
    obs = selection[0]['observation']
    prediction = infer_instance(loaded, obs, stage=metadata['stage'], top_k=3)
    assert prediction['status'] == 'ok' and len(prediction['poses']) == 3
    assert prediction['ranking'] == 'quality'
    for pose in prediction['poses']:
        query = pose['query_id']
        assert np.array_equal(pose['point_cam_m'], obs['query_points'][query])
        assert pose['task_probability'] is None and pose['task_logit'] is None
        transform = np.asarray(pose['T_cam_contact'])
        assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
        assert np.linalg.det(transform[:3, :3]) > .999999

    # Recorded-axis queries retain p/n; zero normal and off-object p are rejected.
    p = np.array([[.001, -.002, .6], [0., 0., .6], [1., 1., .6]])
    n = np.array([[0., .6, -.8], [0., 0., 0.], [0., 0., -1.]])
    with torch.no_grad():
        scored = loaded.score_at_pose(loaded.encode_instance(obs), p, n)
        assert np.array_equal(scored['points_cam_m'].cpu().numpy(), p)
        assert scored['valid'].tolist() == [True, False, False]
        assert np.allclose(scored['normals_out_cam'][0].cpu().numpy(), n[0], atol=1e-15)
        training_obs = source[0]['observation']
        unknown = empty_labels(training_obs['query_points'], training_obs['query_normals'])
        assert source_loss(loaded(training_obs), unknown) is None
        assert np.isnan(unknown['q_inst']).all()
    del loaded
    gc.collect()

    # Exercise the actual saved-checkpoint CLI path, without a toy loader/model.
    report = evaluate_main(['--checkpoint', str(run_dir / 'best.pt'), '--manifest', str(test_manifest),
        '--domain', 'source', '--output', str(tmp_path / 'evaluation.json'), '--device', 'cpu',
        '--allow-partial-manifests'])
    assert report['test_selected'] and report['checkpoint_selection_role'] == 'test_selected'
    assert report['metrics']['instance_count'] == 1 and len(report['records']) == 1
    assert report['public_evaluation']['status'] == 'deferred'
    assert report['independent_test_estimate'] is False
    assert json.loads((tmp_path / 'evaluation.json').read_text()) == report
