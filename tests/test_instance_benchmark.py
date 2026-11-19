"""Benchmark orchestration contracts using an explicit disposable trainer stub."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_instance_training as benchmark


@pytest.fixture
def runtime(monkeypatch):
    config = {'schema': 'instance_runtime_v1', 'route': 'b', 'seed': 42, 'model': {},
              'max_grad_norm': 10., 'pretrain': {'steps': 300, 'accumulate_instances': 32,
                  'queries': 128, 'warmup_steps': 20, 'peak_lr': .00042, 'final_lr': .000015},
              'execution': {'schema': 'instance_batched_execution_v1',
                  'microbatch_instances': 8, 'train_workers': 4, 'precision': 'fp32',
                  'compile_mode': None, 'dense_query_chunk': 4096}}
    state = {'config': config, 'clock': 0., 'models': [], 'trainers': [], 'seeds': [],
             'split_checks': [], 'selection_calls': 0, 'failure': None}

    class Dataset:
        def __init__(self, path, *, split, training_queries=None):
            self.path, self.split, self.training_queries = path, split, training_queries

        def __len__(self):
            return 43 if self.split == 'train' else 11

    class Trainer:
        def __init__(self, model, cfg, source, selection, run_dir, *, stage, allow_partial):
            assert stage == 'pretrain'
            self.config, self.model = deepcopy(cfg), model
            self.source, self.selection, self.run_dir = source, selection, run_dir
            self.allow_partial = allow_partial
            self.step = self.supervised_source_instances = 0
            self.closed = False
            state['trainers'].append(self)

        def train_one_update(self):
            micro = self.config['execution']['microbatch_instances']
            if state['failure'] == 'oom' and micro == 16:
                raise torch.cuda.OutOfMemoryError('explicit test OOM')
            state['clock'] += 32. / micro
            self.step += 1
            self.supervised_source_instances += self.config['pretrain']['accumulate_instances']
            norm = float('nan') if state['failure'] == 'nan' and micro == 16 else 12.
            return {'source_loss': .2, 'grad_norm': norm, 'step': self.step,
                    'supervised_source_instances': self.supervised_source_instances}

        def close(self):
            self.closed = True

        def run(self, *args, **kwargs):
            pytest.fail('Benchmark must not execute the complete training/selection loop')

        def evaluate(self, *args, **kwargs):
            pytest.fail('Benchmark must not execute selection')

        def resume(self, *args, **kwargs):
            pytest.fail('Every benchmark trial must start fresh')

        def save(self, *args, **kwargs):
            pytest.fail('Disposable benchmark trials must not write model checkpoints')

    def create_model(cfg, device):
        model = object()
        state['models'].append(model)
        return model

    def select(dataset, config, *, allow_partial):
        assert dataset.split == 'test'
        state['selection_calls'] += 1
        return dataset

    monkeypatch.setattr(benchmark, 'load_instance_config', lambda path: deepcopy(config))
    monkeypatch.setattr(benchmark, 'seed_all', lambda seed: state['seeds'].append(seed))
    monkeypatch.setattr(benchmark, 'InstanceDataset', Dataset)
    monkeypatch.setattr(benchmark, 'check_source_splits',
                        lambda source, split, **kwargs: state['split_checks'].append((source.split, split, kwargs)))
    monkeypatch.setattr(benchmark, 'validate_selection_config', lambda config: config)
    monkeypatch.setattr(benchmark, 'select_source_evaluation', select)
    monkeypatch.setattr(benchmark, 'create_instance_model', create_model)
    monkeypatch.setattr(benchmark, 'InstanceTrainer', Trainer)
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: state['clock'])
    monkeypatch.setattr(benchmark, '_rss_snapshot', lambda: {
        'parent_rss_bytes': 100, 'children_rss_bytes': 400, 'child_process_count': 4, 'source': 'stub'})
    monkeypatch.setattr(benchmark, '_environment', lambda device: {'device': str(device), 'test_only': True})
    return state


def _argv(tmp_path, *options):
    return ['--config', str(tmp_path / 'config.yaml'),
            '--source-manifest', str(tmp_path / 'train.jsonl'),
            '--selection-manifest', str(tmp_path / 'test.jsonl'),
            '--output', str(tmp_path / 'benchmark'), '--device', 'cpu',
            '--warmup-updates', '2', '--measure-updates', '3', *options]


def _report(tmp_path):
    return json.loads((tmp_path / 'benchmark' / 'benchmark.json').read_text())


def test_default_trial_uses_config_batch_and_counts_only_measurements(tmp_path, runtime):
    assert benchmark.main(_argv(tmp_path)) == 0
    report = _report(tmp_path)
    assert len(report['trials']) == 1
    trial = report['trials'][0]
    assert trial['microbatch_instances'] == 8 and trial['train_workers'] == 4
    assert trial['measurement']['updates'] == 3
    assert trial['measurement']['effective_instances'] == 96
    assert trial['measurement']['effective_instances_per_second'] == 8.
    assert trial['measurement']['seconds'] == 12.
    assert trial['measurement']['p50_seconds_per_update'] == 4.
    assert trial['measurement']['p95_seconds_per_update'] == 4.
    assert trial['measurement']['updates_above_clip_threshold'] == 3
    assert trial['measurement']['all_gradients_finite'] is True
    assert trial['warmup']['effective_instances'] == 64
    assert trial['disposable_final_step'] == 5
    assert trial['disposable_final_source_instances'] == 160
    assert not trial['training_budget_completed'] and not trial['selection_executed']
    assert trial['checkpoint_loaded'] is None and trial['checkpoint_saved'] is None
    assert trial['measured_gpu_memory']['max_memory_allocated_bytes'] is None
    assert trial['sampled_parent_rss_max_bytes'] == 100
    assert trial['sampled_children_rss_max_bytes'] == 400
    assert runtime['trainers'][0].closed
    assert runtime['split_checks'] == [('train', 'train', {'allow_partial': False})]
    assert not list((tmp_path / 'benchmark').rglob('*.pt'))


def test_trial_grid_is_fresh_and_preserves_effective_batch_and_schedule(tmp_path, runtime):
    assert benchmark.main(_argv(tmp_path, '--microbatches', '8,16', '--worker-counts', '0,4',
                                '--precision', 'bf16_dense', '--compile-mode', 'reduce-overhead',
                                '--allow-tf32', '--allow-partial-manifests')) == 0
    report = _report(tmp_path)
    assert len(report['trials']) == len(runtime['models']) == len(runtime['trainers']) == 4
    assert len({id(model) for model in runtime['models']}) == 4
    assert runtime['seeds'] == [42] * 4
    assert len({trainer.run_dir for trainer in runtime['trainers']}) == 4
    for trainer, trial in zip(runtime['trainers'], report['trials']):
        assert trainer.closed and trainer.allow_partial
        assert trial['actual_config']['pretrain'] == runtime['config']['pretrain']
        assert trial['actual_config']['execution']['precision'] == 'bf16_dense'
        assert trial['actual_config']['execution']['compile_mode'] == 'reduce-overhead'
        assert trial['actual_config']['execution']['allow_tf32'] is True
        assert trial['measured_update_records'][0]['step'] == 3
        assert trial['effective_batch_instances'] == 32
    assert report['fastest_successful_trial'] == 'trial_03_micro16_workers0'
    assert runtime['config']['execution']['precision'] == 'fp32'


@pytest.mark.parametrize('failure, expected_status', [('oom', 'out_of_memory'), ('nan', 'failed')])
def test_failed_trials_release_workers_and_do_not_win(tmp_path, runtime, failure, expected_status):
    runtime['failure'] = failure
    assert benchmark.main(_argv(tmp_path, '--microbatches', '16,8', '--warmup-updates', '0')) == 0
    report = _report(tmp_path)
    assert report['trials'][0]['status'] == expected_status
    assert report['trials'][0]['measurement']['updates'] == 0
    assert report['trials'][1]['status'] == 'ok'
    assert report['fastest_successful_trial'] == 'trial_02_micro8_workers4'
    assert all(trainer.closed for trainer in runtime['trainers'])
    assert runtime['seeds'] == [42, 42]


def test_no_success_returns_failure_and_preserves_diagnostics(tmp_path, runtime):
    runtime['failure'] = 'nan'
    assert benchmark.main(_argv(tmp_path, '--microbatches', '16')) == 1
    report = _report(tmp_path)
    assert report['fastest_successful_trial'] is None and report['successful_trials'] == 0
    assert report['trials'][0]['error']['type'] == 'FloatingPointError'
    assert runtime['trainers'][0].closed


@pytest.mark.parametrize('options', [
    ['--microbatches', '64'], ['--microbatches', '0'], ['--worker-counts', '-1'],
    ['--microbatches', '1,2,4,8,16', '--worker-counts', '0,1,2,4'],
    ['--warmup-updates', '-1'], ['--measure-updates', '0'], ['--measure-updates', '300'],
    ['--compile-mode', 'default', '--warmup-updates', '0'],
])
def test_invalid_trial_options_fail_before_starting_model(tmp_path, runtime, options):
    with pytest.raises(SystemExit) as error:
        benchmark.main(_argv(tmp_path, *options))
    assert error.value.code == 2
    assert not runtime['models']


def test_existing_training_directory_is_never_overwritten(tmp_path, runtime):
    directory = tmp_path / 'benchmark'
    directory.mkdir()
    checkpoint = directory / 'last.pt'
    checkpoint.write_bytes(b'keep existing training state')
    with pytest.raises(SystemExit):
        benchmark.main(_argv(tmp_path))
    assert checkpoint.read_bytes() == b'keep existing training state'
    assert not runtime['trainers']


def test_cuda_timing_brackets_each_update_and_reports_peak_allocated_reserved(tmp_path, runtime, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, 'synchronize', lambda device: calls.append(('sync', str(device))))
    monkeypatch.setattr(benchmark.torch.cuda, 'reset_peak_memory_stats', lambda device: calls.append(('reset', str(device))))
    monkeypatch.setattr(benchmark.torch.cuda, 'max_memory_allocated', lambda device: 4096)
    monkeypatch.setattr(benchmark.torch.cuda, 'max_memory_reserved', lambda device: 8192)
    monkeypatch.setattr(benchmark.torch.cuda, 'empty_cache', lambda: calls.append(('empty', None)))
    assert benchmark.main(_argv(tmp_path, '--device', 'cuda:1')) == 0
    trial = _report(tmp_path)['trials'][0]
    assert trial['measured_gpu_memory'] == {'max_memory_allocated_bytes': 4096, 'max_memory_reserved_bytes': 8192}
    # Two synchronizations per update, plus setup and the warmup boundary.
    assert calls.count(('sync', 'cuda:1')) == 12
    assert calls.count(('reset', 'cuda:1')) == 2
    assert calls[-1] == ('empty', None)


def test_hundred_update_stability_mode_is_an_explicit_single_trial(tmp_path, runtime):
    assert benchmark.main(_argv(tmp_path, '--measure-updates', '100', '--warmup-updates', '5')) == 0
    report = _report(tmp_path)
    assert len(report['trials']) == 1
    assert report['trials'][0]['measurement']['updates'] == 100
    assert report['trials'][0]['measurement']['effective_instances'] == 3200
    assert report['trials'][0]['disposable_final_step'] == 105


@pytest.mark.parametrize('field,value', [
    ('schema', 'unsupported_profile'), ('hidden_fast_mode', True),
    ('prefetch_factor', False), ('train_workers', -1), ('dense_query_chunk', 1.5),
])
def test_invalid_execution_profile_rejected_before_model_or_directory(tmp_path, runtime, field, value):
    runtime['config']['execution'][field] = value
    with pytest.raises(SystemExit) as error:
        benchmark.main(_argv(tmp_path))
    assert error.value.code == 2
    assert not runtime['models']
    assert not (tmp_path / 'benchmark').exists()


def test_old_config_materializes_complete_explicit_benchmark_defaults(tmp_path, runtime):
    runtime['config'].pop('execution')
    assert benchmark.main(_argv(tmp_path)) == 0
    execution = _report(tmp_path)['trials'][0]['actual_config']['execution']
    assert execution['schema'] == 'instance_batched_execution_v1'
    assert execution['microbatch_instances'] == 1
    assert execution['train_workers'] == execution['evaluation_workers'] == 0
    assert execution['precision'] == 'fp32' and execution['compile_mode'] is None
    assert execution['dense_query_chunk'] is None and execution['main_threads'] == 1
    assert execution['allow_tf32'] is False


def test_trial_grid_override_is_itself_strictly_validated(tmp_path, runtime):
    args = SimpleNamespace(output=tmp_path / 'benchmark')
    with pytest.raises(ValueError, match='cannot exceed'):
        benchmark._run_trial(1, runtime['config'], args, 64, 4, torch.device('cpu'))
    assert not runtime['models']
    assert not args.output.exists()


def test_environment_records_real_legacy_worker_names(monkeypatch):
    monkeypatch.setenv('SUCTION20_INSTANCE_TRAIN_WORKERS', '8')
    monkeypatch.setenv('SUCTION20_INSTANCE_EVAL_WORKERS', '4')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    monkeypatch.setenv('SUCTION20_INSTANCE_PREPARE_WORKERS', 'wrong_old_name')
    environment = benchmark._environment(torch.device('cpu'))['environment']
    assert environment['SUCTION20_INSTANCE_TRAIN_WORKERS'] == '8'
    assert environment['SUCTION20_INSTANCE_EVAL_WORKERS'] == '4'
    assert environment['OPENBLAS_NUM_THREADS'] == '1'
    assert 'SUCTION20_INSTANCE_PREPARE_WORKERS' not in environment
