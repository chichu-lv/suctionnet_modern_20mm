"""CPU faults exercise real trainer transactions with a tiny stochastic model."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from scripts import train_pretrain as entry
from suction20 import pretrain_runtime as runtime, validation


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.bn = torch.nn.BatchNorm1d(3)

    def set_epoch(self, epoch):
        pass

    def forward(self, inputs, **kwargs):
        # Consume the training RNG; interrupted validation must not shift it.
        value = self.linear(self.bn(inputs['xyz_cam_m']) + torch.rand(4, 3))
        return {'value': value, 'seed_valid': torch.ones(4, dtype=torch.bool)}


@pytest.fixture
def harness(monkeypatch):
    class Dataset:
        keys = [(0, 'realsense', i) for i in range(2)]
        data_cfg = {'radius': .02}

        def get_sample(self, index, seed):
            return {'inputs': {'xyz_cam_m': torch.tensor(np.random.default_rng(seed).normal(size=(4, 3)), dtype=torch.float32),
                               'voxel_coords': torch.zeros(4, 4)}, 'targets': {}}

    monkeypatch.setattr(entry, 'make_source_dataset', lambda *a, **k: Dataset())
    monkeypatch.setattr(entry, 'make_model', lambda *a: TinyModel())
    monkeypatch.setattr(entry, 'collate_public', lambda samples, voxel: samples[0])
    monkeypatch.setattr(entry, 'source_losses', lambda prediction, *a, **k: {'total': prediction['value'].square().mean()})
    monkeypatch.setattr(entry, 'worktree_state', lambda: 'synthetic-fixed-source')
    calls = []

    def evaluate(*args, progress=None, **kwargs):
        calls.append(True)
        random.random()
        np.random.rand(9)
        torch.rand(13)
        if progress:
            progress(0, 1, None)
            progress(1, 1, [9, 'realsense', 0])
        return {'instance_Top1_Q20_at_0_4': .4, 'AP_R20': .2,
                'frames': [{'scene_id': 9, 'frame_id': 0}]}

    monkeypatch.setattr(validation, 'evaluate_model', evaluate)
    config = {'experiment': 'E1', 'seed': 42, 'model': {'quality_target': 'gt_B'},
              'data': {'smoke_frame_keys': [[0, 0], [0, 1]], 'smoke_val_frame_keys': [[9, 0]]},
              'training': {'max_optimizer_steps': 4, 'microbatch_size': 1,
                           'gradient_accumulation_steps': 2, 'effective_batch_size': 2,
                           'validate_every_optimizer_steps': 2, 'amp': False},
              'runtime': {'cpu_threads': 1}}
    return config, calls, evaluate


def state(directory):
    return torch.load(directory / 'last.pt', weights_only=False)


def canonical_rows(directory):
    rows = []
    for segment in state(directory)['runtime_state']['ledger']['lineage']:
        data = (directory / segment['path']).read_bytes()[:segment['end_offset']]
        rows.extend(json.loads(line) for line in data.splitlines())
    return rows


def same_training(actual, expected):
    assert actual['sampler_state'] == expected['sampler_state']
    assert actual['global_step'] == expected['global_step']
    for name, tensor in expected['model_state_dict'].items():
        torch.testing.assert_close(actual['model_state_dict'][name], tensor, rtol=0, atol=0)
    for key, values in expected['optimizer_state_dict']['state'].items():
        for name, tensor in values.items():
            torch.testing.assert_close(actual['optimizer_state_dict']['state'][key][name], tensor, rtol=0, atol=0)
    torch.testing.assert_close(actual['rng_state']['torch'], expected['rng_state']['torch'], rtol=0, atol=0)


def test_pause_100_resume_keeps_budget_and_has_no_extra_validation(tmp_path, harness):
    config, calls, _ = harness
    config['training'].update(max_optimizer_steps=101, validate_every_optimizer_steps=101)
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    assert entry.train(config, full, torch.device('cpu')) == 101
    calls.clear()
    assert entry.train(config, resumed, torch.device('cpu'), pause_at_step=100) == 100
    assert calls == []
    assert runtime.read_json(resumed / 'control/status.json')['phase'] == 'paused'
    prefix = runtime.read_json(resumed / 'checkpoint-manifest.json')['nodes']['100']
    assert entry.train(config, resumed, torch.device('cpu'), resume=resumed / 'last.pt', pause_at_step=100) == 101
    assert runtime.read_json(resumed / 'checkpoint-manifest.json')['nodes']['100'] == prefix
    assert runtime.file_metadata(resumed / prefix['path']) == {'size_bytes': prefix['size_bytes']}
    same_training(state(resumed), state(full))
    assert len(calls) == 1
    assert [row['global_step'] for row in canonical_rows(resumed)] == list(range(1, 102))


@pytest.mark.parametrize('crash_at', [2, 4])
def test_validation_interruption_recovers_before_next_update_even_at_final(tmp_path, monkeypatch, harness, crash_at):
    config, calls, evaluate = harness
    full, crashed = tmp_path / 'full', tmp_path / 'crashed'
    entry.train(config, full, torch.device('cpu'))

    def interrupted(*args, **kwargs):
        if state(crashed)['global_step'] == crash_at:
            torch.rand(123)
            raise RuntimeError('validation power loss')
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(validation, 'evaluate_model', interrupted)
    with pytest.raises(RuntimeError, match='power loss'):
        entry.train(config, crashed, torch.device('cpu'))
    assert state(crashed)['runtime_state']['pending_validation']['step'] == crash_at
    monkeypatch.setattr(validation, 'evaluate_model', evaluate)
    entry.train(config, crashed, torch.device('cpu'), resume=crashed / 'last.pt')
    same_training(state(crashed), state(full))
    assert sorted(state(crashed)['runtime_state']['validations']) == ['2', '4']
    assert state(crashed)['best_metric']['global_step'] == 2  # full ties keep earlier
    assert len(canonical_rows(crashed)) == 4


def test_crash_after_result_before_best_commit_is_idempotent(tmp_path, monkeypatch, harness):
    config, calls, _ = harness
    run = tmp_path / 'run'
    original = runtime.CheckpointStore.publish
    armed = True

    def fail(self, writer, **kwargs):
        nonlocal armed
        if armed and kwargs['step'] == 2 and kwargs['pending'] is None:
            armed = False
            raise RuntimeError('best publication crash')
        return original(self, writer, **kwargs)

    monkeypatch.setattr(runtime.CheckpointStore, 'publish', fail)
    with pytest.raises(RuntimeError, match='publication'):
        entry.train(config, run, torch.device('cpu'))
    assert len(calls) == 1
    entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt')
    assert len(calls) == 2  # reused complete step-2 result; step 4 ran once
    assert state(run)['best_metric']['global_step'] == 2
    assert torch.load(run / 'best.pt', weights_only=False)['global_step'] == 2


def test_crash_tail_preserved_but_replayed_updates_not_double_counted(tmp_path, monkeypatch, harness):
    config, _, _ = harness
    run, full = tmp_path / 'run', tmp_path / 'full'
    entry.train(config, full, torch.device('cpu'))
    original = runtime.AttemptLog.append

    def fail(self, row):
        original(self, row)
        if row['global_step'] == 1:
            raise RuntimeError('after log before checkpoint')

    monkeypatch.setattr(runtime.AttemptLog, 'append', fail)
    with pytest.raises(RuntimeError, match='after log'):
        entry.train(config, run, torch.device('cpu'), attempt_id='old')
    tail = (run / 'attempts/old/updates.jsonl').read_bytes()
    monkeypatch.setattr(runtime.AttemptLog, 'append', original)
    entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt', attempt_id='new')
    assert (run / 'attempts/old/updates.jsonl').read_bytes() == tail
    identity = runtime.read_json(run / 'attempts/new/identity.json')
    assert identity['abandoned_tails'][0]['start_offset'] == 0
    assert [row['global_step'] for row in canonical_rows(run)] == [1, 2, 3, 4]
    same_training(state(run), state(full))


def test_corrupt_last_recovers_verified_predecessor_and_records_rollback(tmp_path, harness):
    config, _, _ = harness
    run = tmp_path / 'run'
    entry.train(config, run, torch.device('cpu'), pause_at_step=1)
    manifest = runtime.read_json(run / 'checkpoint-manifest.json')
    (run / manifest['current']['path']).write_bytes(b'corrupt')
    entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt', attempt_id='recover')
    recovery = runtime.read_json(run / 'attempts/recover/identity.json')['checkpoint_recovery']
    assert recovery['rollback_steps'] == 1
    assert [row['global_step'] for row in canonical_rows(run)] == [1, 2, 3, 4]


def test_identity_drift_and_cross_run_resume_are_rejected(tmp_path, monkeypatch, harness):
    config, _, _ = harness
    run = tmp_path / 'run'
    entry.train(config, run, torch.device('cpu'), pause_at_step=1)
    changed = copy.deepcopy(config)
    changed['experiment'] = 'E2'
    with pytest.raises(ValueError, match='semantics|experiment'):
        entry.train(changed, run, torch.device('cpu'), resume=run / 'last.pt')
    monkeypatch.setattr(entry, 'worktree_state', lambda: 'changed-source')
    with pytest.raises(ValueError, match='identity'):
        entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt')
    with pytest.raises(ValueError, match='this run'):
        entry.train(config, run, torch.device('cpu'), resume=tmp_path / 'other/last.pt')


def test_waiting_gate_honors_pause_and_rejects_stale_grant(tmp_path):
    progress = runtime.Progress(tmp_path, run=tmp_path, experiment='E1', attempt_id='new', target_step=4)
    runtime.atomic_json(tmp_path / 'validation.grant.json', {'request_id': 'stale', 'token': 'token'})
    # Request writing is the observation point: a coordinator can pause waiters.
    original = runtime.atomic_json
    from unittest.mock import patch
    def publish(path, value):
        original(path, value)
        if str(path).endswith('validation.request.json'):
            (tmp_path / 'pause.request').touch()
    try:
        with patch.object(runtime, 'atomic_json', publish):
            assert progress.validation_grant({'generation': 'checkpoint-generation'}, 2) is None
        assert progress.state['phase'] == 'waiting_validation'
    finally:
        progress.close()


def test_duplicate_run_lock_rejected(tmp_path):
    lock = runtime.RunLock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='Another trainer'):
            runtime.RunLock(tmp_path)
    finally:
        lock.close()


def test_cpu_trainer_does_not_claim_cuda_context(tmp_path, harness):
    import os
    config, _, _ = harness
    entry.train(config, tmp_path, torch.device('cpu'), pause_at_step=1)
    status = runtime.read_json(tmp_path / 'control/status.json')
    assert status['trainer_pid'] == os.getpid()
    assert status['cuda_context_ready'] is False
    assert 'gpu_uuid' not in status


def test_explicit_null_disables_cuda_cap_and_numeric_legacy_limits_still_apply(monkeypatch):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: SimpleNamespace(total_memory=128 * 1024**3))
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction',
                        lambda fraction, device: calls.append((fraction, device)))
    device = torch.device('cuda:0')
    entry.apply_gpu_memory_limit({'runtime': {'gpu_memory_limit_gib': None}}, device)
    assert calls == []
    entry.apply_gpu_memory_limit({'runtime': {'gpu_memory_limit_gib': 32}}, device)
    entry.apply_gpu_memory_limit({}, device)
    assert calls == [(.25, device), (.5, device)]


@pytest.mark.parametrize('prefix', ['', 'GPU-'])
def test_torch_and_nvml_uuid_formats_have_one_identity(prefix):
    value = '1ec9b2a9-e0aa-6b31-d39f-35ad07f58653'
    assert runtime.normalize_gpu_uuid(prefix + value) == 'GPU-' + value


@pytest.mark.parametrize('requested, expected', [('cuda', 'cuda:3'), ('cuda:0', 'cuda:0'), ('cpu', 'cpu')])
def test_runtime_device_resolves_bare_cuda_without_changing_explicit_device(monkeypatch, requested, expected):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    calls = []
    def current_device():
        calls.append(True)
        return 3
    monkeypatch.setattr(torch.cuda, 'current_device', current_device)
    resolved = entry.require_device(requested)
    assert str(resolved) == expected
    assert bool(calls) == (requested == 'cuda')
    if resolved.type == 'cuda':
        # Exercise Torch's real allocator argument validation without a CUDA
        # context: only lazy initialization and the final C call are replaced.
        allocated = []
        monkeypatch.setattr(torch.cuda.memory, '_lazy_init', lambda: None)
        monkeypatch.setattr(torch._C, '_cuda_setMemoryFraction',
                            lambda fraction, index: allocated.append((fraction, index)), raising=False)
        torch.cuda.set_per_process_memory_fraction(.5, resolved)
        assert allocated == [(.5, resolved.index)]


def test_time_cadence_saves_skipped_boundary_and_restores_sampler(tmp_path, monkeypatch, harness):
    config, _, _ = harness
    run = tmp_path / 'run'
    clock = [0.]
    monkeypatch.setattr(entry.time, 'monotonic', lambda: clock[0])
    original = entry.optimizer_update

    def skipped(optimizer, scaler, parameters, max_norm):
        optimizer.zero_grad(set_to_none=True)
        clock[0] = 901.
        return False, float('inf')

    monkeypatch.setattr(entry, 'optimizer_update', skipped)
    publish = runtime.CheckpointStore.publish
    def stop_after_save(self, writer, **kwargs):
        result = publish(self, writer, **kwargs)
        if clock[0] == 901.:
            raise RuntimeError('power loss after timed checkpoint')
        return result
    monkeypatch.setattr(runtime.CheckpointStore, 'publish', stop_after_save)
    with pytest.raises(RuntimeError, match='timed checkpoint'):
        entry.train(config, run, torch.device('cpu'))
    saved = state(run)
    assert saved['global_step'] == 0
    assert saved['runtime_state']['consecutive_skips'] == 1
    assert saved['runtime_state']['ledger']['skipped'] == 1
    assert saved['runtime_state']['ledger']['frames'] == 2
    sampler = copy.deepcopy(saved['sampler_state'])
    monkeypatch.setattr(entry, 'optimizer_update', original)
    monkeypatch.setattr(runtime.CheckpointStore, 'publish', publish)
    entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt')
    rows = canonical_rows(run)
    assert [row['global_step'] for row in rows] == [0, 1, 2, 3, 4]
    assert state(run)['runtime_state']['ledger']['attempts'] == 5
    assert state(run)['sampler_state']['epoch'] > sampler['epoch']


@pytest.mark.parametrize('after_manifest', [False, True])
def test_checkpoint_commit_is_atomic_across_alias_failure(tmp_path, monkeypatch, harness, after_manifest):
    config, _, _ = harness
    run = tmp_path / 'run'
    original_atomic = runtime.atomic_json
    original_alias = runtime.CheckpointStore._alias
    armed = True

    def atomic(path, value):
        nonlocal armed
        if armed and Path(path).name == 'checkpoint-manifest.json' and value['current']['global_step'] == 1:
            armed = False
            raise RuntimeError('before manifest')
        return original_atomic(path, value)

    def alias(self, name, value):
        nonlocal armed
        if armed and value['global_step'] == 1:
            armed = False
            raise RuntimeError('after manifest')
        return original_alias(self, name, value)

    from pathlib import Path
    monkeypatch.setattr(runtime, 'atomic_json', original_atomic if after_manifest else atomic)
    monkeypatch.setattr(runtime.CheckpointStore, '_alias', alias if after_manifest else original_alias)
    with pytest.raises(RuntimeError, match='manifest'):
        entry.train(config, run, torch.device('cpu'), pause_at_step=1)
    manifest = runtime.read_json(run / 'checkpoint-manifest.json')
    assert manifest['current']['global_step'] == int(after_manifest)
    entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt')
    assert [row['global_step'] for row in canonical_rows(run)] == [1, 2, 3, 4]


def test_corrupt_manifest_never_silently_uses_alias(tmp_path, harness):
    config, _, _ = harness
    run = tmp_path / 'run'
    entry.train(config, run, torch.device('cpu'), pause_at_step=1)
    (run / 'checkpoint-manifest.json').write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt')


def test_validation_pause_keeps_pending_and_preserves_rng(tmp_path, monkeypatch, harness):
    config, _, evaluate = harness
    run, full = tmp_path / 'run', tmp_path / 'full'
    entry.train(config, full, torch.device('cpu'))
    def request_pause(*args, progress=None, **kwargs):
        torch.rand(33)
        (run / 'control/pause.request').touch()
        progress(1, 1, [9, 'realsense', 0])
        pytest.fail('pause callback should stop validation')
    monkeypatch.setattr(validation, 'evaluate_model', request_pause)
    assert entry.train(config, run, torch.device('cpu')) == 2
    assert runtime.read_json(run / 'control/status.json')['phase'] == 'paused'
    assert state(run)['runtime_state']['pending_validation']['step'] == 2
    assert not list((run / 'validations').glob('*.json'))
    (run / 'control/pause.request').unlink()
    monkeypatch.setattr(validation, 'evaluate_model', evaluate)
    entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt')
    same_training(state(run), state(full))


def test_matching_validation_grant_is_required(tmp_path):
    progress = runtime.Progress(tmp_path, run=tmp_path, experiment='E1', attempt_id='attempt', target_step=4)
    original = runtime.atomic_json
    from unittest.mock import patch
    def coordinator(path, value):
        original(path, value)
        if str(path).endswith('validation.request.json'):
            original(tmp_path / 'validation.grant.json', dict(value, token='assignment-token'))
    try:
        with patch.object(runtime, 'atomic_json', coordinator):
            grant = progress.validation_grant({'generation': 'checkpoint-generation'}, 2)
        assert grant['token'] == 'assignment-token'
        assert grant['attempt_id'] == 'attempt' and grant['step'] == 2
    finally:
        progress.close()


def test_skipped_update_after_best_does_not_replace_evaluated_bn_state(tmp_path, monkeypatch, harness):
    config, _, _ = harness
    run = tmp_path / 'run'
    original = entry.optimizer_update
    attempts = [0]
    def update(optimizer, scaler, parameters, max_norm):
        attempts[0] += 1
        if attempts[0] == 3:
            optimizer.zero_grad(set_to_none=True)
            (run / 'control/pause.request').touch()
            return False, float('inf')
        return original(optimizer, scaler, parameters, max_norm)
    monkeypatch.setattr(entry, 'optimizer_update', update)
    assert entry.train(config, run, torch.device('cpu')) == 2
    current = state(run)
    best = torch.load(run / 'best.pt', weights_only=False)
    assert best['global_step'] == current['global_step'] == 2
    assert best['runtime_state']['ledger']['attempts'] == 2
    assert current['runtime_state']['ledger']['attempts'] == 3
    assert not torch.equal(best['model_state_dict']['bn.running_mean'], current['model_state_dict']['bn.running_mean'])


@pytest.mark.parametrize('crash_at', [2, 4])
def test_committed_validation_survives_crash_before_supervisor_done(tmp_path, monkeypatch, harness, crash_at):
    config, calls, _ = harness
    full, run = tmp_path / 'full', tmp_path / 'run'
    entry.train(config, full, torch.device('cpu'))
    calls.clear()
    original = runtime.atomic_json
    armed = True

    def coordinator_and_fault(path, value):
        nonlocal armed
        if str(path).endswith('validation.done.json') and armed and value['step'] == crash_at:
            armed = False
            raise RuntimeError('crash before validation acknowledgement')
        original(path, value)
        if str(path).endswith('validation.request.json'):
            original(path.parent / 'validation.grant.json', dict(value, token=value['attempt_id']))

    monkeypatch.setattr(runtime, 'atomic_json', coordinator_and_fault)
    monkeypatch.setattr(entry, 'atomic_json', coordinator_and_fault)
    with pytest.raises(RuntimeError, match='acknowledgement'):
        entry.train(config, run, torch.device('cpu'), validation_gate=True,
                    control_dir=run / 'old-control', attempt_id='old')
    committed = state(run)
    assert committed['global_step'] == crash_at
    assert committed['runtime_state']['pending_validation'] is None
    assert str(crash_at) in committed['runtime_state']['validations']
    entry.train(config, run, torch.device('cpu'), resume=run / 'last.pt',
                validation_gate=True, control_dir=run / 'new-control', attempt_id='new')
    assert len(calls) == 2  # Already committed nodes must not be evaluated again.
    same_training(state(run), state(full))
    assert [row['global_step'] for row in canonical_rows(run)] == [1, 2, 3, 4]


def test_v3_profile_and_budget_cannot_resume_v2_or_different_profile():
    from suction20.training import validate_resume_config
    from suction20.pretrain_budget import resolve_config, CONTRACT_VERSION
    legacy={'training':{'max_optimizer_steps':57600,'effective_batch_size':4}}
    modern=resolve_config({'training':{'budget_contract':CONTRACT_VERSION},
                           'runtime_profile':{'profile_id':'H20-E5-b4','training_version':'v3','selection_status':'selected'}})
    with pytest.raises(ValueError,match='semantics'):
        validate_resume_config(legacy,modern)
    changed=copy.deepcopy(modern);changed['runtime_profile']['profile_id']='another-profile'
    with pytest.raises(ValueError,match='semantics'):
        validate_resume_config(modern,changed)


def test_analysis_manifest_counts_events_and_completed_validation_survive_resume(tmp_path,harness):
    config,_,_=harness
    run=tmp_path/'run'
    entry.train(config,run,torch.device('cpu'),pause_at_step=1,attempt_id='first')
    entry.train(config,run,torch.device('cpu'),resume=run/'last.pt',attempt_id='second')
    saved=state(run);manifest=runtime.read_json(run/'checkpoint-manifest.json')
    analysis=manifest['current']['analysis'];status=runtime.read_json(run/'control/status.json')
    assert analysis['ledger']==saved['runtime_state']['ledger']
    assert analysis['counters']=={'effective_samples_completed':8,'consumed_samples':8,'skipped_samples':0,'optimizer_attempts':4}
    assert status['last_completed_validation']['step']==4
    assert status['last_completed_validation']==saved['runtime_state']['last_completed_validation']
    assert status['completed_validation_nodes']==[{'step':2,'effective_samples':4,'frame_count':1},{'step':4,'effective_samples':8,'frame_count':1}]
    rows=canonical_rows(run)
    assert status['contract_version']=='suction20.pretrain.updates.legacy'
    assert manifest['current']['contract_version']==status['contract_version']
    # Later validation must not rewrite a retained publication's public aliases.
    retained = manifest['best']
    assert retained['global_step'] == 2
    assert retained['completed_validation_nodes'] == [{'step': 2, 'effective_samples': 4, 'frame_count': 1}]
    assert retained['completed_validation_nodes'] == retained['analysis']['completed_validation_nodes']
    assert retained['last_completed_validation']['step'] == 2
    assert rows[-1]['effective_samples_completed']==8 and rows[-1]['consumed_samples']==8
    assert rows[-1]['peak_gpu_bytes'] is None and rows[-1]['gpu_memory_unavailable_reason']=='cpu_device'
    assert all(frame['valid_seed_count']==4 for frame in rows[-1]['frame_details'])
    assert len(rows[0]['frame_details'])==2
    assert all('augmentation_seed' in frame and 'voxel_count' in frame for frame in rows[0]['frame_details'])
    events=[json.loads(line) for path in sorted((run/'attempts').glob('*/events.jsonl')) for line in path.read_text().splitlines()]
    kinds={event['kind'] for event in events}
    assert {'checkpoint_started','checkpoint_published','pause_requested','paused','resume','validation_started','validation_completed','completed'} <= kinds
    assert all(event['utc'] and event['monotonic']>0 for event in events)
    # Operational event tails remain separate from accepted optimizer updates.
    metadata=runtime.read_json(run/'attempts/second/identity.json')
    assert metadata['abandoned_event_tails'] and not metadata['abandoned_tails']


def test_v3_e8_prefix_skips_credit_only_success_and_preserves_sampling_regime(tmp_path,monkeypatch,harness):
    from suction20.pretrain_budget import CONTRACT_VERSION
    from suction20.training import FrameSchedule
    config,calls,_=harness
    config=copy.deepcopy(config)
    config['data']={'label_root':str(tmp_path/'labels'),'camera':'realsense','views_per_train_scene_per_epoch':64}
    (tmp_path/'labels').mkdir()
    (tmp_path/'labels/config_frozen.json').write_text(json.dumps({'radius':.02}))
    (tmp_path/'labels/manifest.json').write_text('{}')
    config['training']={'budget_contract':CONTRACT_VERSION,'microbatch_size':1,'gradient_accumulation_steps':8,'effective_batch_size':8,'amp':False}
    config['runtime_profile']={'profile_id':'synthetic-E8','selection_status':'selected','training_version':'v3'}
    monkeypatch.setattr(entry,'FrameSchedule',lambda *args,**kwargs:FrameSchedule([0],frame_keys=[(0,0),(0,1)]))
    epochs=[]
    monkeypatch.setattr(TinyModel,'set_epoch',lambda self,epoch:epochs.append(epoch))
    real=entry.optimizer_update;attempts=[0]
    def skip_first(optimizer,scaler,parameters,max_norm):
        attempts[0]+=1
        if attempts[0]==1:
            optimizer.zero_grad(set_to_none=True)
            return False,float('inf')
        return real(optimizer,scaler,parameters,max_norm)
    monkeypatch.setattr(entry,'optimizer_update',skip_first)
    run=tmp_path/'run'
    assert entry.train(config,run,torch.device('cpu'),pause_at_step=50,attempt_id='prefix')==50
    assert not calls  #400 credited samples is an off-cycle recovery boundary.
    saved=state(run); rows=canonical_rows(run)
    assert saved['runtime_state']['effective_samples_completed']==400
    assert saved['runtime_state']['ledger']['frames']==408
    assert saved['runtime_state']['ledger']['skipped']==1
    assert all(epoch==0 for epoch in epochs)  #physical sampler already crossed200 epochs
    assert rows[0]['effective_samples_completed']==0 and rows[0]['skipped_samples']==8
    assert rows[0]['lr']==rows[1]['lr'] and rows[0]['skip_reason']=='nonfinite_gradients'
    assert rows[-1]['effective_samples_completed']==400 and rows[-1]['consumed_samples']==408
    assert len(rows[-1]['frame_details'])==8
    prefix=runtime.read_json(run/'checkpoint-manifest.json')['nodes']['50']
    assert prefix['budget']['target_updates']==28800 and prefix['effective_samples_completed']==400
    assert entry.train(config,run,torch.device('cpu'),resume=run/'last.pt',pause_at_step=51,attempt_id='resumed')==51
    manifest=runtime.read_json(run/'checkpoint-manifest.json')
    assert manifest['nodes']['50']==prefix
    assert manifest['current']['analysis']['counters']['effective_samples_completed']==408
    assert manifest['current']['analysis']['counters']['consumed_samples']==416
    assert not calls


def test_explicit_bf16_normal_pause_resume_without_scaling(tmp_path, harness):
    config, _, _ = harness
    config["training"]["precision"] = "bf16"
    config["optimizer"] = {"fused": True}
    run = tmp_path / "bf16"
    assert entry.train(config, run, torch.device("cpu"), pause_at_step=1) == 1
    first = state(run)
    assert first["scaler_state_dict"] == {}
    assert entry.train(config, run, torch.device("cpu"), resume=run / "last.pt", pause_at_step=3) == 3
    saved = state(run)
    assert saved["global_step"] == 3 and saved["scaler_state_dict"] == {}
    assert saved["optimizer_state_dict"]["param_groups"][0]["fused"] is False
    rows = canonical_rows(run)
    assert [row["global_step"] for row in rows] == [1, 2, 3]
    assert all(row["updated"] and row["precision"]["dense"] == "bf16" for row in rows)
    assert all(row["amp_scale"] == 1 for row in rows)


def test_file_metadata_requires_only_stat_and_checks_committed_length(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / 'payload.bin'
    path.write_bytes(b'unchanged length')
    size = path.stat().st_size
    def refuse_open(*args, **kwargs):
        raise AssertionError('metadata must not open or read file contents')
    monkeypatch.setattr(Path, 'open', refuse_open)
    assert runtime.file_metadata(path) == {'size_bytes': size}
    assert runtime.file_metadata(path, size - 1) == {'size_bytes': size - 1}
    with pytest.raises(ValueError, match='Truncated'):
        runtime.file_metadata(path, size + 1)


def test_legacy_identity_resume_ignores_retired_fields_and_republishes_structure(tmp_path, harness):
    config, _, _ = harness
    full, resumed = tmp_path / 'full', tmp_path / 'legacy'
    entry.train(config, full, torch.device('cpu'))
    entry.train(config, resumed, torch.device('cpu'), pause_at_step=1)
    old = state(resumed)
    identity = old['runtime_state']['identity']
    for key in ('source_label_definition', 'semantic_config', 'worktree_snapshot'):
        identity.pop(key)
    identity['semantic_config_sha256'] = 'retired-value-ignored'
    old['source_label_config_sha256'] = 'retired-value-ignored'
    old['training_config']['runtime']['raw_config_sha256'] = 'retired-value-ignored'
    manifest = runtime.read_json(resumed / 'checkpoint-manifest.json')
    current_path = manifest['current']['path']
    torch.save(old, resumed / current_path)
    for candidate in [manifest.get(key) for key in ('current', 'predecessor', 'best')] + list(manifest.get('nodes', {}).values()):
        if candidate and candidate['path'] == current_path:
            candidate['size_bytes'] = (resumed / current_path).stat().st_size
    runtime.atomic_json(resumed / 'checkpoint-manifest.json', manifest)
    assert entry.train(config, resumed, torch.device('cpu'), resume=resumed / 'last.pt') == 4
    restored = state(resumed)
    same_training(restored, state(full))
    assert 'source_label_config_sha256' not in restored
    assert 'semantic_config_sha256' not in restored['runtime_state']['identity']
    assert restored['runtime_state']['identity']['source_label_definition'] == {'radius': .02}
