"""Trainer checkpoint replay includes issued but unconsumed spawn tickets."""
from copy import deepcopy
import importlib.util
import json

import numpy as np
import pytest
import torch

from suction20.instance_data import InstanceAttemptDataset, InstanceDataset, build_source_instances
from suction20.instance_training import InstanceTrainer, seed_all
from test_instance_data import _assets, _camera
from test_instance_loading import _source
from test_instance_training import TinyInstanceModel, assert_nested_equal, config


ROUTE = 'b' if importlib.util.find_spec('models.instance_route_b') is not None else 'a'


class TinyBatchedModel(TinyInstanceModel):
    """Small real optimizer/BN/dropout fixture; production CPU preparation runs."""
    def __init__(self):
        super().__init__()
        self.min_support_points = 100
        if ROUTE == 'a':
            self.backbone_points, self.query_chunk_size = 32, 4
        self.source_trace, self.target_trace = [], []

    def configure_execution(self, *, precision, compile_mode, dense_query_chunk):
        assert precision == 'fp32' and compile_mode is None

    @staticmethod
    def _color(observation):
        return np.asarray(observation['support_rgb'], np.float32).mean(0)

    def score_at_pose(self, encoded, points, normals, query_batch_index=None):
        if encoded is not None:
            color = self._color(encoded)
            self.target_trace.append((encoded['sample_uid'], color.tolist()))
            points = np.asarray(points) + .05 * color
        return super().score_at_pose(encoded, points, normals, query_batch_index)

    def forward_instances(self, observations):
        counts, features = [], []
        for observation in observations:
            color = self._color(observation)
            self.source_trace.append((observation['sample_uid'], color.tolist()))
            counts.append(len(observation['query_points']))
            features.append(observation['query_points'] + .05 * color)
        packed = self.score_at_pose(None, np.concatenate(features), None)
        rows, offset = [], 0
        for count in counts:
            rows.append({key: value[offset:offset+count] for key, value in packed.items()})
            offset += count
        return rows


@pytest.fixture(scope='module')
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp('spawn_checkpoint_assets')
    source = _source(root/'train')
    raw, labels, _ = _assets(root/'test', scene_id=100)
    manifest, _ = build_source_instances(raw, labels, root/'test_cache', split='test',
                                         scenes=[100], frames=[0])
    selection = InstanceDataset(manifest, split='test')
    # Target observations use separate real captures per split, with both
    # recorded outcomes. No target pose is replaced with a sampled query.
    attempts = []
    for split, dataset in [('train', source), ('val', selection)]:
        for index, record in enumerate(dataset.records):
            paths = record['source_paths']
            observation = dict(format='rgbd', rgb_path=paths['rgb'], depth_path=paths['depth'],
                instance_mask_path=paths['label'], depth_scale_m=.001,
                intrinsics=_camera().tolist(), g_cam=[0., 0., -1.])
            for action, success in enumerate((0, 1)):
                attempts.append(dict(schema_version='suction20.target.v1', label_kind='attempt_success',
                    sample_id=f'{split}_{index}_{action}', group_id=f'{split}_{index}', split=split,
                    observation=observation, success_definition='pickup_and_transport',
                    attempt=dict(point_cam_m=[action*.005, 0., .6], normal_out_cam=[0., 0., -1.],
                                 instance_id=1, success=success, label_valid=True)))
    attempt_manifest = root/'attempts.jsonl'
    attempt_manifest.write_text(''.join(json.dumps(row)+'\n' for row in attempts))
    return source, selection, InstanceAttemptDataset(attempt_manifest, 'train'), InstanceAttemptDataset(attempt_manifest, 'val')


def make_trainer(path, datasets, *, stage='pretrain', microbatch=2):
    cfg = config()
    cfg.update(route=ROUTE, rgb_augmentation=True)
    cfg['pretrain'].update(steps=3, queries=4, evaluate_every=3, checkpoint_every=1)
    cfg['finetune'].update(steps=3, evaluate_every=3, checkpoint_every=1)
    cfg['execution'] = dict(microbatch_instances=microbatch, precision='fp32',
        train_workers=1, evaluation_workers=0, prefetch_factor=2,
        evaluation_microbatch_instances=2, main_threads=1)
    source, selection, target, target_selection = datasets
    return InstanceTrainer(TinyBatchedModel(), cfg, source,
        selection if stage == 'pretrain' else target_selection, path, stage=stage,
        target_dataset=target if stage == 'finetune' else None, allow_partial=True)


@pytest.mark.parametrize('stage', ['pretrain', 'finetune'])
def test_spawn_checkpoint_restores_pending_rgb_sampler_and_optimizer(tmp_path, datasets, stage):
    seed_all(573)
    full = make_trainer(tmp_path/'full', datasets, stage=stage)
    full.run()
    seed_all(573)
    partial = make_trainer(tmp_path/'partial', datasets, stage=stage)
    partial.run(stop_after=1)
    checkpoint = tmp_path/'partial/last.pt'
    state = torch.load(checkpoint, weights_only=False)
    assert len(state['preparation_state']['pending_source']) == 2
    assert len(state['preparation_state']['pending_target']) == (2 if stage == 'finetune' else 0)
    assert state['source_schedule']['draws'] == partial.supervised_source_instances + 2
    assert state['preparation_state']['pending_source'][0][1] is not None
    assert not partial._preparation_pools and not partial._preparation_streams

    resumed = make_trainer(tmp_path/'resumed', datasets, stage=stage)
    try:
        # Resume must first close an existing stream, not leave its old
        # pending-deque/RNG closure alive alongside the loaded state.
        next(resumed._draw_batch(resumed.source, resumed.source_schedule, 1))
        old_pool = resumed._preparation_pools['source']
        resumed.resume(checkpoint)
        assert old_pool.closed and not resumed._preparation_streams
        resumed.run()
    finally:
        resumed.close()
    assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(full.source_schedule.state_dict(), resumed.source_schedule.state_dict())
    assert_nested_equal(full.state_dict()['preparation_state'], resumed.state_dict()['preparation_state'])
    assert_nested_equal(full.augmentation_rng.bit_generator.state, resumed.augmentation_rng.bit_generator.state)
    assert_nested_equal(full.best, resumed.best)
    assert full.model.source_trace == partial.model.source_trace + resumed.model.source_trace
    assert full.model.target_trace == partial.model.target_trace + resumed.model.target_trace
    if stage == 'finetune':
        assert_nested_equal(full.target_schedule.state_dict(), resumed.target_schedule.state_dict())
    assert full.step == resumed.step == 3


def test_close_and_reopen_keeps_consumption_order_and_augmentation(tmp_path, datasets):
    reference = make_trainer(tmp_path/'continuous', datasets)
    restarted = make_trainer(tmp_path/'reopened', datasets)
    def consume(trainer, count):
        return [(item['record']['sample_id'], item['observation']['rgb'].copy())
                for item in trainer._draw_batch(trainer.source, trainer.source_schedule, count)]
    try:
        expected = consume(reference, 10)
        actual = consume(restarted, 3)
        pending = deepcopy(list(restarted._pending_tickets['source']))
        assert len(pending) == 2
        restarted.close()
        assert_nested_equal(pending, list(restarted._pending_tickets['source']))
        actual += consume(restarted, 7)
        assert_nested_equal(expected, actual)
        assert_nested_equal(reference.source_schedule.state_dict(), restarted.source_schedule.state_dict())
        assert_nested_equal(reference.augmentation_rng.bit_generator.state,
                            restarted.augmentation_rng.bit_generator.state)
        assert_nested_equal(reference.state_dict()['preparation_state'], restarted.state_dict()['preparation_state'])
    finally:
        reference.close()
        restarted.close()


def test_mid_update_worker_failure_does_not_overwrite_last_checkpoint(tmp_path, datasets, monkeypatch):
    seed_all(49)
    trainer = make_trainer(tmp_path/'failure', datasets, microbatch=1)
    trainer.run(stop_after=1)
    path = tmp_path/'failure/last.pt'
    previous = torch.load(path, weights_only=False)
    previous_stat = (path.stat().st_size, path.stat().st_mtime_ns)
    original = trainer.model.forward_instances
    partial_calls = []
    def lose_worker_after_first_microbatch(observations):
        output = original(observations)
        if not partial_calls:
            partial_calls.append(True)
            process = trainer._preparation_pools['source']._processes[0]
            process.terminate()
            process.join(timeout=5)
        return output
    monkeypatch.setattr(trainer.model, 'forward_instances', lose_worker_after_first_microbatch)
    with pytest.raises(RuntimeError, match='worker exited'):
        trainer.run(stop_after=2)
    assert partial_calls and trainer.step == 1
    assert any(parameter.grad is not None for parameter in trainer.model.parameters())
    assert not trainer._preparation_pools and not trainer._preparation_streams
    assert previous_stat == (path.stat().st_size, path.stat().st_mtime_ns)
    after = torch.load(path, weights_only=False)
    assert_nested_equal(previous, after)
    assert after['global_step'] == 1
    # Failed work did mutate in-memory BN: the completed checkpoint, not that
    # partial state, is the correct recovery source.
    assert trainer.model.decoder[1].num_batches_tracked > after['model']['decoder.1.num_batches_tracked']


def test_batched_checkpoint_without_pending_state_is_rejected(tmp_path, datasets):
    trainer = make_trainer(tmp_path/'source', datasets)
    trainer.run(stop_after=1)
    checkpoint = torch.load(tmp_path/'source/last.pt', weights_only=False)
    checkpoint.pop('preparation_state')
    path = tmp_path/'missing_pending.pt'
    torch.save(checkpoint, path)
    resumed = make_trainer(tmp_path/'resumed', datasets)
    try:
        with pytest.raises(ValueError, match='preparation ticket state'):
            resumed.resume(path)
        assert resumed.step == 0 and not resumed._preparation_streams
    finally:
        resumed.close()
