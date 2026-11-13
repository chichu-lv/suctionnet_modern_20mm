"""Compare serial and overlapped geometry using one experiment on one GPU.

H20 uses E5/B2A2; L20 uses E3/B1A4. By default this only writes two fresh
configurations and their commands. --run executes those production trainers;
--summarize-existing reads their logs without launching anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from suction20.from_scratch import from_scratch_config
from suction20.training import load_config


ROOT = Path(__file__).resolve().parents[1]
WARMUP_STEPS = 10
STOP_STEP = 30
HOSTS = {
    'H20': {'experiment': 'E5', 'microbatch_size': 2, 'gpu_memory_limit_gib': 112},
    'L20': {'experiment': 'E3', 'microbatch_size': 1, 'gpu_memory_limit_gib': 36},
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prepare_plan(hardware, out_dir, *, source_root=None, label_root=None,
                 backbone_root=None, cpu_list=None, python=sys.executable):
    """Two configurations, one experiment, one intentional difference: depth."""
    selected = HOSTS[hardware]
    output = Path(out_dir).resolve()
    if (output / 'plan.json').exists() or any((output / name).exists() for name in ('serial', 'overlap')):
        raise ValueError('Choose a new output directory; benchmark runs are fresh starts')
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / f'configs/runtime_profiles/cd009-v3/{hardware}-{selected["experiment"]}.json')
    variants = []
    for name, depth in (('serial', 1), ('overlap', 2)):
        config, performance = from_scratch_config(base,
            hardware_preset=hardware, microbatch_size=selected['microbatch_size'],
            geometry_pipeline_depth=depth, dense_chunk_size=65536, pose_chunk_size=32,
            geometry_threads=8, workers=4, cpu_affinity_count=16, cpu_threads=8,
            gpu_memory_limit_gib=selected['gpu_memory_limit_gib'])
        config['seed'] = 42
        for key, value in (('source_root', source_root), ('label_root', label_root)):
            if value is not None:
                config['data'][key] = str(Path(value).resolve())
        if backbone_root is not None:
            config['model']['backbone_root'] = str(Path(backbone_root).resolve())
        directory = output / name
        directory.mkdir()
        train_path, performance_path = directory / 'train.json', directory / 'performance.json'
        write_json(train_path, config)
        write_json(performance_path, performance)
        command = [str(python), str(ROOT / 'scripts/train_pretrain.py'),
                   '--config', str(train_path), '--performance-config', str(performance_path),
                   '--run-dir', str(directory / 'run'), '--pause-at-step', str(STOP_STEP)]
        if cpu_list:
            command = ['taskset', '-c', cpu_list, *command]
        variants.append({'name': name, 'geometry_pipeline_depth': depth,
                         'train_config': str(train_path), 'performance_config': str(performance_path),
                         'run_dir': str(directory / 'run'), 'command': command})
    environment = {'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    for variant in variants:
        variant['shell_command'] = shlex.join(['env', *(f'{key}={value}' for key, value in environment.items()),
                                               *variant['command']])
    plan = {
        'schema_version': 'suction20.geometry_overlap_benchmark.v1',
        'hardware': hardware, 'experiment': selected['experiment'],
        'microbatch_size': selected['microbatch_size'],
        'gradient_accumulation_steps': 4 // selected['microbatch_size'],
        'effective_batch_size': 4, 'warmup_successful_steps': WARMUP_STEPS,
        'stop_at_successful_step': STOP_STEP, 'timed_credited_frames': 80,
        'working_directory': str(ROOT), 'environment': environment,
        'gpu_selection': 'Inherit CUDA_VISIBLE_DEVICES; no device discovery or reassignment',
        'cuda_visible_devices_at_prepare': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cpu_list': cpu_list,
        'cpu_allocation': '16 allocated logical CPUs; supply --cpu-list or launch under the intended affinity',
        'measurement_scope': 'Step 10 end to step 30 end; real forward/loss/backward/AdamW, including skips and intervening waits; excludes startup and final checkpoint/cleanup',
        'variants': variants,
    }
    write_json(output / 'plan.json', plan)
    return plan


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    values = [float(value) for value in values if value is not None]
    return {'median': statistics.median(values), 'p95': percentile(values, .95)} if values else None


def summarize_rows(rows, *, warmup_steps=WARMUP_STEPS, stop_step=STOP_STEP):
    """Use boundary-to-boundary wall time, including failed attempts in between."""
    start = next((i for i, row in enumerate(rows)
                  if row['updated'] and row['global_step'] == warmup_steps), None)
    end = next((i for i, row in enumerate(rows)
                if row['updated'] and row['global_step'] == stop_step), None)
    if start is None or end is None or end <= start:
        return {'measurement_status': 'incomplete', 'frames_per_second': None,
                'successful_updates_recorded': sum(bool(row['updated']) for row in rows),
                'reason': f'Need successful step {warmup_steps} and {stop_step} boundaries'}
    anchor, last = rows[start], rows[end]
    window = rows[start + 1:end + 1]
    # Both boundaries belong to the same fresh attempt, so monotonic time is
    # unaffected by wall-clock corrections and includes logging/save gaps.
    wall = last['ended_monotonic'] - anchor['ended_monotonic']
    if wall <= 0:
        raise ValueError('Update boundary timestamps must increase')
    frames = last['effective_samples_completed'] - anchor['effective_samples_completed']
    pipeline_depth = last.get('geometry_pipeline', {}).get('depth', 1)
    result = {
        'measurement_status': 'complete', 'warmup_successful_steps': warmup_steps,
        'stop_at_successful_step': stop_step, 'credited_frames': frames,
        'wall_seconds': wall, 'frames_per_second': frames / wall,
        'successful_updates': sum(bool(row['updated']) for row in window),
        'skipped_updates': sum(not row['updated'] for row in window),
        'update_seconds': distribution(row['elapsed_seconds'] for row in window),
        'loader_wait_seconds': distribution(row.get('loader_wait_s') for row in window),
        'geometry_pipeline_depth': pipeline_depth,
        # Depth 1 has no background job/future. A literal 0 would incorrectly
        # suggest that the serial baseline has no geometry work or waiting.
        'geometry_job_wall_seconds': distribution(
            row.get('geometry_pipeline', {}).get('geometry_job_wall_s') for row in window) if pipeline_depth > 1 else None,
        'geometry_future_wait_seconds': distribution(
            row.get('geometry_pipeline', {}).get('geometry_wait_host_s') for row in window) if pipeline_depth > 1 else None,
        'geometry_timing_scope': 'Job wall is summed overlapped background time; future wait is exposed host wait. Neither is pure CPU kernel time. Depth-1 job/future fields are not applicable.',
        'synchronous_geometry_host_seconds': distribution(
            sum(frame.get('grouping_timings', {}).get(name, 0.)
                for frame in row.get('frame_stats', [])
                for name in ('sampler_cpu_s', 'spatial_index_seconds', 'query_cpu_seconds'))
            for row in window) if pipeline_depth == 1 else None,
        'gradients_finite': all(row.get('gradients_finite', True) for row in window),
    }
    for source, target in (('peak_gpu_bytes', 'peak_allocated_gib'),
                           ('peak_gpu_reserved_bytes', 'peak_reserved_gib')):
        values = [row[source] for row in window if row.get(source) is not None]
        result[target] = max(values) / 1024**3 if values else None
    return result


def summarize_run(run_dir):
    directory = Path(run_dir)
    paths = list((directory / 'attempts').glob('*/updates.jsonl'))
    if not paths:
        return {'measurement_status': 'not_run', 'frames_per_second': None}
    if len(paths) != 1:
        raise ValueError('This benchmark summarizes one fresh attempt per variant; use a new run directory')
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    result = summarize_rows(rows)
    status_path = directory / 'control/status.json'
    result['trainer_phase'] = json.loads(status_path.read_text()).get('phase') if status_path.exists() else None
    return result


def summarize_plan(plan, out_dir):
    results = []
    for variant in plan['variants']:
        result = {'name': variant['name'], **summarize_run(variant['run_dir'])}
        execution_path = Path(variant['run_dir']).parent / 'execution.json'
        if execution_path.exists():
            result.update(json.loads(execution_path.read_text()))
        result['normal_pause'] = (result.get('trainer_phase') == 'paused' and result['exit_code'] == 75
                                  if 'exit_code' in result else None)
        results.append(result)
    base_fps, candidate_fps = (result.get('frames_per_second') for result in results)
    change = (candidate_fps / base_fps - 1.) * 100. if base_fps and candidate_fps else None
    summary = {'hardware': plan['hardware'], 'experiment': plan['experiment'],
               'throughput_change_percent': change, 'results': results,
               'scope': plan['measurement_scope'],
               'conclusion_scope': 'One short performance comparison; does not measure convergence. No historical checkpoint/gradient equivalence gate or automatic production switch.'}
    output = Path(out_dir)
    write_json(output / 'summary.json', summary)
    def number(value, digits=3):
        return '—' if value is None else f'{value:.{digits}f}'
    def pair(result, key):
        value = result.get(key)
        return '—' if not value else f'{number(value["median"])} / {number(value["p95"])}'
    lines = [f'# {plan["hardware"]} / {plan["experiment"]}: geometry overlap', '',
             '| Variant | frames/s | Update median / p95 (s) | Allocated / reserved (GiB) | Geometry job median / p95 (s) | Exposed future wait median / p95 (s) | Loader wait median / p95 (s) | Success / skip | Normal pause |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for result in results:
        pause = 'unobserved' if result['normal_pause'] is None else str(result['normal_pause'])
        lines.append(f'| {result["name"]} | {number(result.get("frames_per_second"), 6)} | '
                     f'{pair(result, "update_seconds")} | {number(result.get("peak_allocated_gib"))} / {number(result.get("peak_reserved_gib"))} | '
                     f'{pair(result, "geometry_job_wall_seconds")} | {pair(result, "geometry_future_wait_seconds")} | '
                     f'{pair(result, "loader_wait_seconds")} | {result.get("successful_updates", "—")} / {result.get("skipped_updates", "—")} | {pause} |')
    lines.extend(['', f'Throughput change: {number(change, 2)}%.', '', plan['measurement_scope'] + '.', '',
                  'Serial geometry has no background future, so its job/future cells are not applicable. Summed background job time overlaps GPU work and must not be added to update wall time. The JSON also reports serial geometry host intervals.', '',
                  summary['conclusion_scope'], ''])
    (output / 'summary.md').write_text('\n'.join(lines))
    return summary


def run_plan(plan, out_dir):
    environment = dict(os.environ, **plan['environment'])
    for variant in plan['variants']:
        console_path = Path(variant['run_dir']).parent / 'console.log'
        print(f'Running {plan["hardware"]}/{plan["experiment"]}: {variant["name"]}; log: {console_path}', flush=True)
        with console_path.open('x') as stream:
            completed = subprocess.run(variant['command'], cwd=plan['working_directory'],
                                       env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
        write_json(console_path.parent / 'execution.json', {
            'exit_code': completed.returncode,
            'cuda_visible_devices_at_run': os.environ.get('CUDA_VISIBLE_DEVICES')})
        if completed.returncode != 75:
            # Keep a partial measurement (e.g. shutdown failure) visible; do
            # not turn the benchmark into automatic retries or OOM searches.
            summary = summarize_plan(plan, out_dir)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 1
    print(json.dumps(summarize_plan(plan, out_dir), indent=2, ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', required=True, choices=tuple(HOSTS))
    parser.add_argument('--out-dir', required=True, help='New benchmark directory, or existing one with --summarize-existing')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Prepare and sequentially run just the two real training candidates')
    mode.add_argument('--summarize-existing', action='store_true', help='Read plan/logs; launch no process')
    parser.add_argument('--source-root')
    parser.add_argument('--label-root')
    parser.add_argument('--backbone-root')
    parser.add_argument('--cpu-list', help='Actual allocated CPU IDs for taskset, e.g. an assigned 16-core range; otherwise inherit affinity')
    args = parser.parse_args()
    if args.run and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Set CUDA_VISIBLE_DEVICES to the GPU allocated for this short test')
    if args.summarize_existing:
        plan = json.loads((Path(args.out_dir) / 'plan.json').read_text())
        if plan['hardware'] != args.hardware:
            parser.error('Hardware differs from the existing benchmark plan')
        print(json.dumps(summarize_plan(plan, args.out_dir), indent=2, ensure_ascii=False))
        return
    plan = prepare_plan(args.hardware, args.out_dir, source_root=args.source_root,
                        label_root=args.label_root, backbone_root=args.backbone_root, cpu_list=args.cpu_list)
    if args.run:
        sys.exit(run_plan(plan, args.out_dir))
    print(json.dumps({'status': 'prepared; no training launched', **plan}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
