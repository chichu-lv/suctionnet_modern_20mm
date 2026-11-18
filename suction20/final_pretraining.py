"""Select the last performance candidate, export a fixed recipe, and spawn trainers.

This is a one-shot local-host handoff, not a scheduler. Ordinary trainer logging,
checkpoints and the ten validation nodes remain in train_pretrain.py.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .from_scratch import from_scratch_config
from .pretrain_budget import resolve_budget
from .metadata import without_legacy_checksums


ROOT = Path(__file__).resolve().parents[1]
MIN_OPTIMIZATION_GAIN = .03
LEGACY_FINAL_EFFECTIVE_BATCH = 8
HOST_EXPERIMENTS = {'H20': ('E5',), 'L20': ('E0', 'E1', 'E2', 'E3', 'E4')}
DEFAULT_RESOURCES = {
    'E0': {'gpu': '1', 'cpus': '8-15'},
    'E1': {'gpu': '0', 'cpus': '0-7'},
    'E2': {'gpu': '5', 'cpus': '56-63'},
    'E3': {'gpu': '3', 'cpus': '24-39'},
    'E4': {'gpu': '4', 'cpus': '40-55'},
    'E5': {'gpu': '0', 'cpus': '0-15'},
}
_SETTING_FIELDS = ('name', 'role', 'microbatch_size', 'gradient_accumulation_steps',
                   'effective_batch_size', 'geometry_pipeline_depth',
                   'contact_feature_gather_mode', 'pose_chunk_size')


def select_final_setting(plan, summary):
    """Select a completed candidate for a new, independently initialized run.

    The historical final suite keeps its effective-batch8/3% rule. The later
    batch-combo suite chooses the highest measured throughput across its tested
    effective batches, with no extra adoption threshold or repeat comparison.
    """
    hardware = plan.get('hardware')
    suite = plan.get('suite')
    if suite not in ('final', 'batch-combo') or hardware not in HOST_EXPERIMENTS:
        raise ValueError('Fresh training selection requires a final or batch-combo H20/L20 plan')
    if summary.get('hardware', hardware) != hardware:
        raise ValueError('Summary hardware differs from its plan')
    variants = {item['name']: item for item in plan['variants']}
    completed = []
    for result in summary['results']:
        fps = result.get('frames_per_second')
        if (result.get('execution_status') != 'completed' or type(fps) not in (int, float)
                or not math.isfinite(fps) or fps <= 0):
            continue
        variant = variants[result['name']]
        if suite == 'final' and variant.get('effective_batch_size') != LEGACY_FINAL_EFFECTIVE_BATCH:
            raise ValueError('Final performance candidates must use effective batch8')
        completed.append({**{key: variant[key] for key in _SETTING_FIELDS},
                          'frames_per_second': float(fps)})
    if suite == 'batch-combo':
        if not completed:
            raise ValueError('No completed batch-combo candidate is available')
        selected = max(completed, key=lambda item: item['frames_per_second'])
        baseline = next((item for item in completed if item['role'] == 'current_baseline'), None)
        optimized = [item for item in completed if item['role'] == 'candidate']
        best_optimization = max(optimized, key=lambda item: item['frames_per_second']) if optimized else None
        gain = (100 * (selected['frames_per_second'] / baseline['frames_per_second'] - 1)
                if baseline else None)
        optimization_gain = (100 * (best_optimization['frames_per_second'] / baseline['frames_per_second'] - 1)
                             if baseline and best_optimization else None)
        return {
            'hardware': hardware, 'source_benchmark_suite': suite,
            'selection_policy': 'current_completed_max',
            'selected_variant': selected['name'], 'selected': selected,
            **{key: selected[key] for key in _SETTING_FIELDS if key not in ('name', 'role')},
            'baseline': baseline, 'best_optimization': best_optimization,
            'threshold_percent': None, 'optimization_gain_percent': optimization_gain,
            'selected_gain_percent': gain,
            'reason': 'highest measured frames/s among completed batch-combo candidates',
            'decision_scope': 'Use the selected host-specific batch for a fresh start, with no new performance test.',
        }
    baselines = [item for item in completed if item['role'] == 'baseline']
    if not baselines:
        raise ValueError('No completed baseline is available; resolve the actual trainer failure before launch')
    baseline = max(baselines, key=lambda item: item['frames_per_second'])
    optimized = [item for item in completed if item['role'] == 'optimization']
    best_optimization = max(optimized, key=lambda item: item['frames_per_second']) if optimized else None
    gain = (100 * (best_optimization['frames_per_second'] / baseline['frames_per_second'] - 1)
            if best_optimization else None)
    adopt = (best_optimization is not None and best_optimization['frames_per_second'] >=
             baseline['frames_per_second'] * (1 + MIN_OPTIMIZATION_GAIN))
    selected = best_optimization if adopt else baseline
    return {
        'hardware': hardware, 'source_benchmark_suite': suite,
        'selection_policy': 'legacy_final_three_percent',
        'selected_variant': selected['name'], 'selected': selected,
        **{key: selected[key] for key in _SETTING_FIELDS if key not in ('name', 'role')},
        'baseline': baseline, 'best_optimization': best_optimization,
        'threshold_percent': MIN_OPTIMIZATION_GAIN * 100,
        'optimization_gain_percent': gain, 'selected_gain_percent': gain if adopt else 0.,
        'reason': ('new head meets the 3% throughput adoption rule' if adopt else
                   'keep the fastest completed baseline; no completed new head meets the 3% rule'),
        'decision_scope': 'Last performance selection; no repeat test or new optimization gate before fresh training.',
    }


def _write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(without_legacy_checksums(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _resources(experiments, overrides):
    if overrides is not None and (not isinstance(overrides, dict) or set(overrides) - set(experiments)):
        raise ValueError('Resource map must contain only this host\'s experiment IDs')
    selected = {}
    for experiment in experiments:
        resource = dict(DEFAULT_RESOURCES[experiment])
        override = (overrides or {}).get(experiment, {})
        if not isinstance(override, dict) or set(override) - {'gpu', 'cpus'}:
            raise ValueError('Each resource override supports gpu and cpus only')
        resource.update(override)
        gpu = resource['gpu']
        if type(gpu) is int and gpu >= 0:
            gpu = str(gpu)
        if not isinstance(gpu, str) or not gpu.strip() or ',' in gpu:
            raise ValueError('Each experiment needs one GPU ID or UUID')
        cpus = resource['cpus']
        if not isinstance(cpus, str):
            raise ValueError('cpus must be a taskset CPU list, such as 24-39')
        cpu_ids = set()
        for part in cpus.split(','):
            bounds = part.strip().split('-')
            if len(bounds) not in (1, 2) or not all(item.isdigit() for item in bounds):
                raise ValueError('cpus must be a taskset CPU list, such as 24-39')
            start, end = int(bounds[0]), int(bounds[-1])
            if end < start:
                raise ValueError('CPU ranges must increase')
            cpu_ids.update(range(start, end + 1))
        selected[experiment] = {'gpu': gpu.strip(), 'cpus': cpus,
                                'cpu_affinity_count': len(cpu_ids)}
    if len({item['gpu'] for item in selected.values()}) != len(selected):
        raise ValueError('Concurrent formal experiments require distinct GPUs on their host')
    return selected


def finalize_pretraining(plan, summary, training_root, *, resource_map=None,
                         launch=False, python=sys.executable):
    """Prepare all formal runs for this host, optionally spawning them once.

    Use a new training_root. Configurations and the manifest are saved before
    any process starts. A returned PID means process spawn, not verified updates.
    """
    selection = select_final_setting(plan, summary)
    hardware = selection['hardware']
    effective_batch = selection['effective_batch_size']
    experiments = HOST_EXPERIMENTS[hardware]
    resources = _resources(experiments, resource_map)
    chosen = next(item for item in plan['variants'] if item['name'] == selection['selected_variant'])
    candidate = json.loads(Path(chosen['train_config']).read_text())
    output = Path(training_root).resolve()
    config_root = output / 'configs' / hardware
    manifest_path = output / 'manifests' / f'{hardware}.json'
    selection_path = output / 'selection' / f'{hardware}.json'
    if config_root.exists() or manifest_path.exists() or selection_path.exists():
        raise FileExistsError('This host already has a final export; use its saved commands or choose a new training root')
    for experiment in experiments:
        if (output / 'runs' / f'{experiment}_seed42').exists():
            raise FileExistsError(f'{experiment} already has a run directory; final training must start fresh')

    # Exclusive host directory also prevents two local finalizers from racing.
    config_root.parent.mkdir(parents=True, exist_ok=True)
    config_root.mkdir()
    records = []
    supervisor_entries = {'profiles': {hardware: {}}, 'performance_configs': {hardware: {}}}
    for experiment in experiments:
        resource = resources[experiment]
        contact = experiment in ('E3', 'E4', 'E5')
        microbatch = selection['microbatch_size'] if contact else 1
        base = json.loads((ROOT / 'configs/runtime_profiles/cd009-v3' / f'{hardware}-{experiment}.json').read_text())
        config, performance = from_scratch_config(
            base, hardware_preset=hardware, effective_batch_size=effective_batch,
            microbatch_size=microbatch,
            geometry_pipeline_depth=selection['geometry_pipeline_depth'] if contact else 1,
            precision='bf16' if experiment == 'E2' else 'fp16',
            dense_chunk_size=65536, pose_chunk_size=selection['pose_chunk_size'] if contact else 32,
            contact_feature_gather_mode=selection['contact_feature_gather_mode'] if contact else 'chunk',
            workers=6 if experiment in ('E0', 'E1') else 4,
            geometry_threads=8 if contact else 4,
            cpu_affinity_count=resource['cpu_affinity_count'], cpu_threads=8,
            no_gpu_memory_limit=True)
        config['seed'] = 42
        for field in ('source_root', 'label_root'):
            config['data'][field] = candidate['data'][field]
        config['model']['backbone_root'] = candidate['model']['backbone_root']
        budget = resolve_budget(config)
        profile = config['runtime_profile']
        profile['profile_id'] = profile['profile_id'].replace('cd010-fresh-', 'cd010-final-', 1)
        profile['selection_status'] = 'selected'
        profile['selection_basis'] = (
            f'Fresh start, effective-batch{effective_batch}; source_suite={plan["suite"]}, '
            f'selection_policy={selection["selection_policy"]}. {hardware} contact execution selected by '
            f'{plan["experiment"]} performance comparison ({selection["selected_variant"]}). '
            'Other experiment definitions and sample budgets retained; no per-experiment speed claim.')
        directory = config_root / experiment
        train_path, performance_path = directory / 'train.json', directory / 'performance.json'
        _write_json(train_path, config, exclusive=True)
        _write_json(performance_path, performance, exclusive=True)
        run_dir = output / 'runs' / f'{experiment}_seed42'
        log_path = output / 'logs' / hardware / f'{experiment}.log'
        command = ['taskset', '--cpu-list', resource['cpus'], str(python),
                   str(ROOT / 'scripts/train_pretrain.py'), '--config', str(train_path),
                   '--performance-config', str(performance_path), '--run-dir', str(run_dir),
                   '--device', 'cuda:0']
        environment = {'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': resource['gpu'],
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        records.append({
            'experiment': experiment, **resource,
            'train_config': str(train_path), 'performance_config': str(performance_path),
            'run_dir': str(run_dir), 'console_log': str(log_path),
            'command': command, 'environment': environment,
            'shell_command': shlex.join(['env', *(f'{key}={value}' for key, value in environment.items()), *command]),
            'microbatch_size': microbatch, 'gradient_accumulation_steps': effective_batch // microbatch,
            'effective_batch_size': effective_batch,
            'geometry_pipeline_depth': config['training']['geometry_pipeline_depth'],
            'target_optimizer_updates': budget['target_updates'],
            'warmup_optimizer_updates': budget['warmup_updates'],
            'validation_interval_updates': budget['validation_interval_updates'],
            'pose_chunk_size': config['model']['pose_chunk_size'],
            'contact_feature_gather_mode': performance['contact_feature_gather_mode'],
            'precision': config['training']['precision'], 'process_spawned': False,
        })
        for section, path in (('profiles', train_path), ('performance_configs', performance_path)):
            supervisor_entries[section][hardware][experiment] = {
                'config_path': str(path), 'config_definition': json.loads(path.read_text())}

    report = {
        'schema_version': 'suction20.final_pretraining.v1',
        'hardware': hardware, 'training_root': str(output), 'selection': selection,
        'training_mode': 'from_scratch', 'source_benchmark_suite': plan['suite'],
        'selection_path': str(selection_path), 'manifest_path': str(manifest_path),
        'status': 'prepared; no training launched', 'launch_requested': bool(launch),
        'working_directory': str(ROOT), 'experiments': records,
        'effective_batch_size': effective_batch,
        'target_effective_samples_per_experiment': budget['target_samples'],
        'target_optimizer_updates_per_experiment': budget['target_updates'],
        'warmup_optimizer_updates': budget['warmup_updates'],
        'validation_interval_updates': budget['validation_interval_updates'],
        'validation_nodes': len(budget['validation_steps']),
        'supervisor_entries': supervisor_entries,
        'process_observation': 'PID/process_spawned reports process creation only, not verified optimizer updates.',
        'service_scope': 'Direct trainers run existing logs/checkpoints/validation. No coordinator, backup daemon, or old run is started.',
    }
    _write_json(selection_path, selection, exclusive=True)
    _write_json(manifest_path, report, exclusive=True)
    if launch:
        for record in records:
            log_path = Path(record['console_log'])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open('x') as stream:
                    process = subprocess.Popen(
                        record['command'], cwd=ROOT, env={**os.environ, **record['environment']},
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                record.update(process_spawned=True, pid=process.pid)
                report['status'] = 'processes_spawned' if all(item['process_spawned'] for item in records) else 'launching'
            except OSError as error:
                record['launch_error'] = str(error)
                report['status'] = 'launch_failed; inspect manifest for already spawned processes'
                _write_json(manifest_path, report)
                raise
            _write_json(manifest_path, report)
    return report
