import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from suction20.final_pretraining import ROOT, finalize_pretraining, select_final_setting
from suction20.from_scratch import from_scratch_config
from scripts.benchmark_geometry_overlap import prepare_plan


def candidate_plan(tmp_path, hardware='H20'):
    experiment = 'E5' if hardware == 'H20' else 'E3'
    microbatch = 4 if hardware == 'H20' else 1
    base = json.loads((ROOT / f'configs/runtime_profiles/cd009-v3/{hardware}-{experiment}.json').read_text())
    config, _ = from_scratch_config(base, hardware_preset=hardware, microbatch_size=microbatch,
                                  effective_batch_size=8, geometry_pipeline_depth=2)
    config['data'].update(source_root='/real/input', label_root='/real/labels')
    config['model']['backbone_root'] = '/real/backbone'
    train_file = tmp_path / 'candidate.json'
    train_file.write_text(json.dumps(config))
    variants = []
    for name, role, mode, chunk in [('baseline', 'baseline', 'chunk', 32),
                                    ('optimized', 'optimization', 'zone', 128)]:
        variants.append(dict(name=name, role=role, microbatch_size=microbatch,
                             gradient_accumulation_steps=8 // microbatch, effective_batch_size=8,
                             geometry_pipeline_depth=2, contact_feature_gather_mode=mode,
                             pose_chunk_size=chunk, train_config=str(train_file)))
    plan = {'suite': 'final', 'hardware': hardware, 'experiment': experiment, 'variants': variants}
    summary = {'hardware': hardware, 'results': [
        {'name': 'baseline', 'execution_status': 'completed', 'frames_per_second': 100.},
        {'name': 'optimized', 'execution_status': 'completed', 'frames_per_second': 104.},
    ]}
    return plan, summary


def measured_combo_plan(tmp_path, hardware='H20'):
    plan = prepare_plan(hardware, tmp_path / 'benchmark', suite='batch-combo',
                        source_root='/real/input', label_root='/real/labels',
                        backbone_root='/real/backbone')
    public_results = json.loads((ROOT / 'docs/performance/CD010_BATCH_ACCUMULATION_RESULTS.json').read_text())
    return plan, public_results['hosts'][hardware]


def test_final_selection_uses_fastest_baseline_and_three_percent_rule(tmp_path):
    plan, summary = candidate_plan(tmp_path)
    selected = select_final_setting(plan, summary)
    assert selected['selected_variant'] == 'optimized'
    assert selected['selected_gain_percent'] == pytest.approx(4.)
    summary['results'][1]['frames_per_second'] = 102.
    assert select_final_setting(plan, summary)['selected_variant'] == 'baseline'
    summary['results'][1].update(execution_status='error', frames_per_second=None)
    assert select_final_setting(plan, summary)['selected_variant'] == 'baseline'
    plan['variants'].append({**plan['variants'][0], 'name': 'second_baseline', 'microbatch_size': 2,
                             'gradient_accumulation_steps': 4})
    summary['results'].append({'name': 'second_baseline', 'execution_status': 'completed',
                               'frames_per_second': 105.})
    assert select_final_setting(plan, summary)['selected_variant'] == 'second_baseline'


@pytest.mark.parametrize('hardware,suite,expected_experiments,effective,selected_variant', [
    ('H20', 'final', ['E5'], 8, 'optimized'),
    ('L20', 'final', ['E0', 'E1', 'E2', 'E3', 'E4'], 8, 'optimized'),
    ('H20', 'batch-combo', ['E5'], 48, 'b4_a12_e48_zone128'),
    ('L20', 'batch-combo', ['E0', 'E1', 'E2', 'E3', 'E4'], 12, 'b1_a12_e12_zone128'),
])
def test_final_export_preserves_sample_budget_and_builds_host_recipe(
        hardware, suite, expected_experiments, effective, selected_variant, tmp_path):
    plan, summary = (measured_combo_plan(tmp_path, hardware) if suite == 'batch-combo'
                     else candidate_plan(tmp_path, hardware))
    report = finalize_pretraining(plan, summary, tmp_path / 'formal', python='/train/python')
    assert report['status'] == 'prepared; no training launched'
    assert report['training_mode'] == 'from_scratch'
    assert report['selection']['selected_variant'] == selected_variant
    if suite == 'batch-combo':
        assert report['selection']['selection_policy'] == 'current_completed_max'
        assert report['selection']['threshold_percent'] is None
        baseline_fps = summary['results'][0]['frames_per_second']
        assert report['selection']['baseline']['frames_per_second'] == baseline_fps
        assert report['selection']['selected_gain_percent'] == pytest.approx(
            (summary['ranking'][0]['frames_per_second'] / baseline_fps - 1) * 100)
    assert report['effective_batch_size'] == effective
    assert report['target_optimizer_updates_per_experiment'] == 230400 // effective
    assert report['warmup_optimizer_updates'] == 11520 // effective
    assert report['validation_interval_updates'] == 23040 // effective
    assert report['validation_nodes'] == 10
    assert [item['experiment'] for item in report['experiments']] == expected_experiments
    assert Path(report['selection_path']).is_file() and Path(report['manifest_path']).is_file()
    for record in report['experiments']:
        config = json.loads(Path(record['train_config']).read_text())
        performance = json.loads(Path(record['performance_config']).read_text())
        assert config['training']['effective_batch_size'] == effective
        assert config['training']['target_effective_samples'] == 230400
        assert config['training']['max_optimizer_steps'] == record['target_optimizer_updates'] == 230400 // effective
        assert config['scheduler']['warmup_optimizer_steps'] == record['warmup_optimizer_updates'] == 11520 // effective
        assert config['training']['validate_every_optimizer_steps'] == record['validation_interval_updates'] == 23040 // effective
        assert config['training']['microbatch_size'] == (4 if hardware == 'H20' else 1)
        assert config['training']['gradient_accumulation_steps'] == effective // config['training']['microbatch_size']
        assert config['seed'] == 42
        assert config['optimizer']['lr_peak'] == .001
        assert config['runtime']['gpu_memory_limit_gib'] is None
        assert config['data']['source_root'] == '/real/input'
        assert config['model']['backbone_root'] == '/real/backbone'
        assert not Path(record['run_dir']).exists()
        assert not any(flag in record['command'] for flag in ('--resume', '--init-checkpoint', '--pause-at-step', '--stop-at-step'))
        contact = record['experiment'] in ('E3', 'E4', 'E5')
        assert config['training']['geometry_pipeline_depth'] == (2 if contact else 1)
        assert performance['contact_feature_gather_mode'] == ('zone' if contact else 'chunk')
        assert config['model']['pose_chunk_size'] == (128 if contact else 32)
        assert performance['loader']['num_workers'] == (6 if record['experiment'] in ('E0', 'E1') else 4)
        assert config['training']['precision'] == ('bf16' if record['experiment'] == 'E2' else 'fp16')


def test_launch_saves_manifest_before_spawn_and_reports_only_pids(tmp_path, monkeypatch):
    plan, summary = measured_combo_plan(tmp_path)
    training_root = tmp_path / 'formal'
    calls = []

    def fake_popen(command, **kwargs):
        before = json.loads((training_root / 'manifests/H20.json').read_text())
        assert before['status'] == 'prepared; no training launched'
        assert before['effective_batch_size'] == 48
        assert before['target_optimizer_updates_per_experiment'] == 4800
        assert all(Path(item['train_config']).is_file() for item in before['experiments'])
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '7'
        assert kwargs['start_new_session'] is True
        assert '--resume' not in command and '--init-checkpoint' not in command
        calls.append(command)
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr('suction20.final_pretraining.subprocess.Popen', fake_popen)
    report = finalize_pretraining(plan, summary, training_root, launch=True,
                                  resource_map={'E5': {'gpu': '7', 'cpus': '32-47'}})
    assert report['status'] == 'processes_spawned'
    assert report['experiments'][0]['process_spawned']
    assert report['experiments'][0]['pid'] == 4321
    assert 'not verified optimizer updates' in report['process_observation']
    with pytest.raises(FileExistsError):
        finalize_pretraining(plan, summary, training_root, launch=True)
    assert len(calls) == 1
