import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from suction20.final_pretraining import ROOT, finalize_pretraining, select_final_setting
from suction20.from_scratch import from_scratch_config


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


@pytest.mark.parametrize('hardware,expected_experiments', [('H20', ['E5']), ('L20', ['E0', 'E1', 'E2', 'E3', 'E4'])])
def test_final_export_preserves_sample_budget_and_builds_host_recipe(hardware, expected_experiments, tmp_path):
    plan, summary = candidate_plan(tmp_path, hardware)
    report = finalize_pretraining(plan, summary, tmp_path / 'formal', python='/train/python')
    assert report['status'] == 'prepared; no training launched'
    assert [item['experiment'] for item in report['experiments']] == expected_experiments
    assert Path(report['selection_path']).is_file() and Path(report['manifest_path']).is_file()
    for record in report['experiments']:
        config = json.loads(Path(record['train_config']).read_text())
        performance = json.loads(Path(record['performance_config']).read_text())
        assert config['training']['effective_batch_size'] == 8
        assert config['training']['target_effective_samples'] == 230400
        assert config['training']['max_optimizer_steps'] == 28800
        assert config['scheduler']['warmup_optimizer_steps'] == 1440
        assert config['training']['validate_every_optimizer_steps'] == 2880
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
    plan, summary = candidate_plan(tmp_path)
    training_root = tmp_path / 'formal'
    calls = []

    def fake_popen(command, **kwargs):
        before = json.loads((training_root / 'manifests/H20.json').read_text())
        assert before['status'] == 'prepared; no training launched'
        assert all(Path(item['train_config']).is_file() for item in before['experiments'])
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '7'
        assert kwargs['start_new_session'] is True
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
