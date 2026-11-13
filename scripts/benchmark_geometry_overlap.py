"""Compare geometry scheduling or batch capacity on one experiment/GPU.

The default suite compares serial/overlap: H20 E5/B2A2, L20 E3/B1A4.
--suite memory keeps overlap and compares H20 B2A4 to B4A2 (effective batch
8 in both), or L20 B1A4 to B2A2 (effective batch stays 4).
Those pairs warm up on 40 credited frames and time 80.
--suite batch-scan runs multiple microbatches at one effective batch per host,
with 48 warmup and 240 timed credited frames. --suite contact-head keeps the
measured H20 B4A6 / L20 B1A12 depth2 choices and compares per-chunk vs per-zone
feature gathers and pose chunks32/64/128. New configurations have no
artificial GPU memory cap. By default only prepare; --run executes the plan.
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
WARMUP_FRAMES = 40
TIMED_FRAMES = 80
HOSTS = {
    'H20': {'experiment': 'E5', 'microbatch_size': 2},
    'L20': {'experiment': 'E3', 'microbatch_size': 1},
}
SCAN_SETTINGS = {
    'H20': {'effective_batch_size': 24, 'batch_sizes': (1, 2, 3, 4, 6, 8)},
    'L20': {'effective_batch_size': 12, 'batch_sizes': (1, 2, 3, 4)},
}
CONTACT_HEAD_SETTINGS = {
    'H20': {'microbatch_size': 4, 'effective_batch_size': 24},
    'L20': {'microbatch_size': 1, 'effective_batch_size': 12},
}
SCAN_WARMUP_FRAMES = 48
SCAN_TIMED_FRAMES = 240


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prepare_plan(hardware, out_dir, *, source_root=None, label_root=None,
                 backbone_root=None, cpu_list=None, python=sys.executable,
                 suite='overlap', gpu_memory_limit_gib=None, no_gpu_memory_limit=False,
                 batch_sizes=None, effective_batch_size=None, overlap_only=False):
    """Fresh configurations with equal credited-frame windows within a suite."""
    selected = HOSTS[hardware]
    if suite not in ('overlap', 'memory', 'batch-scan', 'contact-head'):
        raise ValueError('suite must be overlap, memory, batch-scan or contact-head')
    if suite != 'batch-scan' and (batch_sizes is not None or effective_batch_size is not None or overlap_only):
        raise ValueError('Batch list, effective batch and overlap-only overrides require --suite batch-scan')
    warmup_frames, timed_frames = WARMUP_FRAMES, TIMED_FRAMES
    head_settings = {}
    if suite == 'overlap':
        specifications = [('serial', selected['microbatch_size'], 4, 1),
                          ('overlap', selected['microbatch_size'], 4, 2)]
        comparison_note = 'Only geometry pipeline depth changes; microbatch and effective batch stay fixed.'
    elif suite == 'memory':
        larger_microbatch, larger_effective = (4, 8) if hardware == 'H20' else (2, 4)
        specifications = [('baseline', selected['microbatch_size'], larger_effective, 2),
                          ('larger_batch', larger_microbatch, larger_effective, 2)]
        comparison_note = (
            'Both H20 variants use effective batch8 and the same AdamW update count. Only '
            'microbatch/accumulation grouping changes, including BN/loss numerical behavior. '
            'Effective batch8 differs from the previously measured batch4 recipe; its published '
            'speed is a historical reference, not this comparison baseline.' if hardware == 'H20' else
            'L20 effective batch stays 4, but microbatch grouping changes BN/loss numerical behavior. '
            'This comparison measures throughput, not convergence.')
    elif suite == 'batch-scan':
        scan = SCAN_SETTINGS[hardware]
        effective = scan['effective_batch_size'] if effective_batch_size is None else effective_batch_size
        batches = list(scan['batch_sizes'] if batch_sizes is None else batch_sizes)
        if type(effective) is not int or effective < 1 or not batches:
            raise ValueError('Choose a positive effective batch and a nonempty batch list')
        if any(type(batch) is not int or batch < 1 or effective % batch or effective // batch < 2
               for batch in batches):
            raise ValueError('Every microbatch must divide the effective batch with at least two accumulation steps')
        batches = sorted(set(batches))
        specifications = [(f'b{batch}_a{effective // batch}_d2', batch, effective, 2) for batch in batches]
        if not overlap_only:
            largest = batches[-1]
            specifications.append((f'b{largest}_a{effective // largest}_d1', largest, effective, 1))
        warmup_frames = math.ceil(SCAN_WARMUP_FRAMES / effective) * effective
        timed_frames = math.ceil(SCAN_TIMED_FRAMES / effective) * effective
        comparison_note = (
            f'All {hardware} candidates use effective batch {effective}, preserving AdamW update '
            'frequency within this scan. Depth2 varies microbatch. '
            + ('The final depth1 candidate compares one larger resident graph against CPU/GPU overlap. '
               if not overlap_only else '') + 'This effective batch is '
            'a throughput-screening setting, not an automatic change to the final training recipe. '
            'BN/loss grouping changes; historical effective-batch4/8 speeds are references only.')
    else:
        selected_head = CONTACT_HEAD_SETTINGS[hardware]
        microbatch, effective = selected_head['microbatch_size'], selected_head['effective_batch_size']
        head_settings = {'chunk_pose32': ('chunk', 32), 'zone_pose32': ('zone', 32),
                         'zone_pose64': ('zone', 64), 'zone_pose128': ('zone', 128)}
        specifications = [(name, microbatch, effective, 2) for name in head_settings]
        warmup_frames, timed_frames = SCAN_WARMUP_FRAMES, SCAN_TIMED_FRAMES
        comparison_note = (
            f'Keep the measured {hardware} B{microbatch}A{effective // microbatch}/depth2 '
            f'performance setting at effective batch {effective}. chunk_pose32 is the existing '
            'path; zone_pose32 isolates one gather per zone, and zone_pose64/128 change only '
            'the contact encoder chunk on that path. Geometry, dense chunk, resources and '
            'frame windows stay fixed. New operator speedups require these GPU measurements; '
            'the scan effective batch is not an automatically selected final training recipe.')
    memory_limit = gpu_memory_limit_gib
    if no_gpu_memory_limit and memory_limit is not None:
        raise ValueError('Choose unrestricted memory or a numeric limit, not both')
    output = Path(out_dir).resolve()
    if (output / 'plan.json').exists() or any((output / spec[0]).exists() for spec in specifications):
        raise ValueError('Choose a new output directory; benchmark runs are fresh starts')
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / f'configs/runtime_profiles/cd009-v3/{hardware}-{selected["experiment"]}.json')
    variants = []
    for name, microbatch, effective_batch, depth in specifications:
        gather_mode, pose_chunk_size = head_settings.get(name, ('chunk', 32))
        warmup_steps = warmup_frames // effective_batch
        stop_step = (warmup_frames + timed_frames) // effective_batch
        config, performance = from_scratch_config(base,
            hardware_preset=hardware, microbatch_size=microbatch,
            effective_batch_size=effective_batch,
            geometry_pipeline_depth=depth, dense_chunk_size=65536, pose_chunk_size=pose_chunk_size,
            contact_feature_gather_mode=gather_mode,
            geometry_threads=8, workers=4, cpu_affinity_count=16, cpu_threads=8,
            gpu_memory_limit_gib=memory_limit, no_gpu_memory_limit=memory_limit is None)
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
                   '--run-dir', str(directory / 'run'), '--pause-at-step', str(stop_step)]
        if cpu_list:
            command = ['taskset', '-c', cpu_list, *command]
        variants.append({'name': name, 'geometry_pipeline_depth': depth,
                         'microbatch_size': microbatch,
                         'gradient_accumulation_steps': effective_batch // microbatch,
                         'effective_batch_size': effective_batch,
                         'contact_feature_gather_mode': gather_mode,
                         'pose_chunk_size': pose_chunk_size,
                         'warmup_successful_steps': warmup_steps,
                         'stop_at_successful_step': stop_step,
                         'warmup_credited_frames': warmup_frames,
                         'timed_credited_frames': timed_frames,
                         'gpu_memory_limit_gib': memory_limit,
                         'train_config': str(train_path), 'performance_config': str(performance_path),
                         'run_dir': str(directory / 'run'), 'command': command})
    environment = {'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    for variant in variants:
        variant['shell_command'] = shlex.join(['env', *(f'{key}={value}' for key, value in environment.items()),
                                               *variant['command']])
    plan = {
        'schema_version': ('suction20.geometry_overlap_benchmark.v1' if suite == 'overlap'
                           else 'suction20.geometry_overlap_benchmark.v4' if suite == 'contact-head'
                           else 'suction20.geometry_overlap_benchmark.v3' if suite == 'batch-scan'
                           else 'suction20.geometry_overlap_benchmark.v2'),
        'suite': suite,
        'hardware': hardware, 'experiment': selected['experiment'],
        'warmup_credited_frames': warmup_frames, 'timed_credited_frames': timed_frames,
        'total_credited_frames_per_variant': warmup_frames + timed_frames,
        'gpu_memory_limit_gib': memory_limit,
        'memory_policy': 'No artificial allocator cap; use available device VRAM' if memory_limit is None else f'Explicit {memory_limit:g} GiB allocator cap',
        'continue_on_cuda_oom': suite in ('batch-scan', 'contact-head'),
        'comparison_note': comparison_note,
        'working_directory': str(ROOT), 'environment': environment,
        'gpu_selection': 'Inherit CUDA_VISIBLE_DEVICES; no device discovery or reassignment',
        'cuda_visible_devices_at_prepare': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cpu_list': cpu_list,
        'cpu_allocation': '16 allocated logical CPUs; supply --cpu-list or launch under the intended affinity',
        'measurement_scope': (f'After {warmup_frames} warmup credited frames, time the next {timed_frames}; '
                              'exact successful-step boundaries are recorded per variant. '
                              'Real forward/loss/backward/AdamW, including skips and '
                              'intervening waits; excludes startup and final checkpoint/cleanup. Compare '
                              'frames/s, not seconds/update across different effective batches.'),
        'variants': variants,
    }
    if suite == 'overlap':
        # Retain the original plan fields for old callers and saved summaries.
        plan.update(microbatch_size=selected['microbatch_size'],
                    gradient_accumulation_steps=4 // selected['microbatch_size'],
                    effective_batch_size=4, warmup_successful_steps=WARMUP_STEPS,
                    stop_at_successful_step=STOP_STEP)
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
        'stage_host_seconds': {
            name: distribution(row.get('timing_seconds', {}).get(name) for row in window)
            for name in ('h2d_host', 'forward_host', 'loss_host', 'backward_host', 'optimizer_host', 'diagnostics_host')},
        'stage_timing_scope': 'Existing host intervals with incidental synchronization, not isolated GPU kernel times; no extra synchronization or trace is inserted.',
    }
    for source, target in (('peak_gpu_bytes', 'peak_allocated_gib'),
                           ('peak_gpu_reserved_bytes', 'peak_reserved_gib')):
        values = [row[source] for row in window if row.get(source) is not None]
        result[target] = max(values) / 1024**3 if values else None
    return result


def summarize_run(run_dir, *, warmup_steps=WARMUP_STEPS, stop_step=STOP_STEP):
    directory = Path(run_dir)
    paths = list((directory / 'attempts').glob('*/updates.jsonl'))
    if not paths:
        return {'measurement_status': 'not_run', 'frames_per_second': None}
    if len(paths) != 1:
        raise ValueError('This benchmark summarizes one fresh attempt per variant; use a new run directory')
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    result = summarize_rows(rows, warmup_steps=warmup_steps, stop_step=stop_step)
    status_path = directory / 'control/status.json'
    result['trainer_phase'] = json.loads(status_path.read_text()).get('phase') if status_path.exists() else None
    return result


def summarize_plan(plan, out_dir):
    results = []
    for variant in plan['variants']:
        warmup = variant.get('warmup_successful_steps', plan.get('warmup_successful_steps', WARMUP_STEPS))
        stop = variant.get('stop_at_successful_step', plan.get('stop_at_successful_step', STOP_STEP))
        result = {'name': variant['name'],
                  **{key: variant.get(key, plan.get(key)) for key in
                     ('microbatch_size', 'gradient_accumulation_steps', 'effective_batch_size',
                      'geometry_pipeline_depth', 'gpu_memory_limit_gib',
                      'contact_feature_gather_mode', 'pose_chunk_size')},
                  **summarize_run(variant['run_dir'], warmup_steps=warmup, stop_step=stop)}
        execution_path = Path(variant['run_dir']).parent / 'execution.json'
        if execution_path.exists():
            result.update(json.loads(execution_path.read_text()))
        result['normal_pause'] = (result.get('trainer_phase') == 'paused' and result['exit_code'] == 75
                                  if 'exit_code' in result else None)
        if result.get('outcome') == 'oom':
            result['execution_status'] = 'oom'
        elif result.get('exit_code') not in (None, 75):
            result['execution_status'] = 'error'
        elif result['normal_pause'] and result['measurement_status'] == 'complete':
            result['execution_status'] = 'completed'
        else:
            result['execution_status'] = result['measurement_status']
        results.append(result)
    base_fps = results[0].get('frames_per_second')
    for result in results:
        fps = result.get('frames_per_second')
        result['throughput_change_from_first_percent'] = (fps / base_fps - 1.) * 100. if base_fps and fps else None
    ranked = sorted((result for result in results if result['execution_status'] == 'completed'),
                    key=lambda result: result['frames_per_second'], reverse=True)
    ranking = [{key: result.get(key) for key in
                ('name', 'microbatch_size', 'gradient_accumulation_steps', 'effective_batch_size',
                 'geometry_pipeline_depth', 'contact_feature_gather_mode', 'pose_chunk_size',
                 'frames_per_second', 'peak_allocated_gib', 'peak_reserved_gib')}
               for result in ranked]
    change = results[1]['throughput_change_from_first_percent'] if len(results) == 2 else None
    summary = {'hardware': plan['hardware'], 'experiment': plan['experiment'],
               'suite': plan.get('suite', 'overlap'),
               'throughput_change_percent': change, 'results': results,
               'ranking': ranking, 'best_variant': ranking[0]['name'] if ranking else None,
               'gpu_memory_limit_gib': plan.get('gpu_memory_limit_gib'),
               'memory_policy': plan.get('memory_policy'),
               'scope': plan['measurement_scope'],
               'comparison_note': plan.get('comparison_note'),
               'conclusion_scope': 'One-pass throughput/capacity screening; only complete windows with observed normal trainer exits enter ranking. Does not measure convergence or automatically change the final training recipe.'}
    output = Path(out_dir)
    write_json(output / 'summary.json', summary)
    def number(value, digits=3):
        return '—' if value is None else f'{value:.{digits}f}'
    def pair(result, key):
        value = result.get(key)
        return '—' if not value else f'{number(value["median"])} / {number(value["p95"])}'
    lines = [f'# {plan["hardware"]} / {plan["experiment"]}: {plan.get("suite", "overlap")} comparison', '',
             plan.get('memory_policy', 'See per-variant allocator configuration.') + '.', '',
             '| Variant | Microbatch / accumulation / effective | Depth | Gather / pose chunk | Outcome | frames/s | Change from first (%) | Update median / p95 (s) | Allocated / reserved (GiB) | Success / skip |',
             '| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for result in results:
        batch = ' / '.join(str(result.get(key) or '—') for key in
                           ('microbatch_size', 'gradient_accumulation_steps', 'effective_batch_size'))
        lines.append(f'| {result["name"]} | {batch} | {result.get("geometry_pipeline_depth", "—")} | '
                     f'{result.get("contact_feature_gather_mode") or "—"} / {result.get("pose_chunk_size") or "—"} | '
                     f'{result["execution_status"]} | {number(result.get("frames_per_second"), 6)} | '
                     f'{number(result["throughput_change_from_first_percent"], 2)} | '
                     f'{pair(result, "update_seconds")} | {number(result.get("peak_allocated_gib"))} / {number(result.get("peak_reserved_gib"))} | '
                     f'{result.get("successful_updates", "—")} / {result.get("skipped_updates", "—")} |')
    lines.extend(['', f'Fastest completed candidate: {summary["best_variant"] or "none"}.', '',
                  '## Existing stage diagnostics', '',
                  '| Variant | Future wait median / p95 (s) | Loader wait median / p95 (s) | Forward host median / p95 (s) | Backward host median / p95 (s) | Optimizer host median / p95 (s) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for result in results:
        stages = result.get('stage_host_seconds', {})
        lines.append(f'| {result["name"]} | {pair(result, "geometry_future_wait_seconds")} | '
                     f'{pair(result, "loader_wait_seconds")} | {pair(stages, "forward_host")} | '
                     f'{pair(stages, "backward_host")} | {pair(stages, "optimizer_host")} |')
    lines.extend(['', 'Host stages include incidental synchronization; they are not isolated CUDA kernel times. No additional trace or synchronization is required.', '',
                  plan['measurement_scope'] + '.', '',
                  'Serial geometry has no background future, so its job/future cells are not applicable. Summed background job time overlaps GPU work and must not be added to update wall time. The JSON also reports serial geometry host intervals.', '',
                  summary.get('comparison_note') or '', '', summary['conclusion_scope'], ''])
    failures = [result for result in results if result.get('error_message')]
    if failures:
        lines.extend(['## Failed candidates', ''])
        for result in failures:
            message = result['error_message'].replace('\n', ' ')
            lines.extend([f'- `{result["name"]}`: {message}', f'  Full console: `{result.get("console_log")}`'])
        lines.append('')
    (output / 'summary.md').write_text('\n'.join(lines))
    return summary


def cuda_oom_message(console_path):
    """Recognize CUDA allocation failures; CPU OOM and unrelated errors stop."""
    for line in reversed(Path(console_path).read_text(errors='replace').splitlines()):
        lowered = line.lower()
        if any(marker in lowered for marker in
               ('cuda out of memory', 'cuda error: out of memory', 'cudaerrormemoryallocation')):
            return line.strip()
    return None


def run_plan(plan, out_dir):
    environment = dict(os.environ, **plan['environment'])
    for variant in plan['variants']:
        console_path = Path(variant['run_dir']).parent / 'console.log'
        print(f'Running {plan["hardware"]}/{plan["experiment"]}: {variant["name"]}; log: {console_path}', flush=True)
        with console_path.open('x') as stream:
            completed = subprocess.run(variant['command'], cwd=plan['working_directory'],
                                       env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
        oom = cuda_oom_message(console_path) if completed.returncode != 75 else None
        execution = {
            'exit_code': completed.returncode,
            'outcome': 'completed' if completed.returncode == 75 else 'oom' if oom else 'error',
            'console_log': str(console_path),
            'cuda_visible_devices_at_run': os.environ.get('CUDA_VISIBLE_DEVICES')}
        if completed.returncode != 75:
            execution['error_message'] = oom or f'Trainer exited with code {completed.returncode}; see console.log'
        write_json(console_path.parent / 'execution.json', execution)
        # Persist progress after every candidate, including CUDA OOM. A new
        # process starts the next candidate with its own allocator/context.
        summary = summarize_plan(plan, out_dir)
        if oom and plan.get('continue_on_cuda_oom', False):
            print(f'{variant["name"]}: CUDA OOM; continuing the fixed candidate list.', flush=True)
            continue
        if completed.returncode != 75:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary['ranking'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', required=True, choices=tuple(HOSTS))
    parser.add_argument('--suite', choices=('overlap', 'memory', 'batch-scan', 'contact-head'), default='overlap',
                        help='overlap: serial/depth2 pair; memory: larger-batch pair; batch-scan: full-VRAM scan; contact-head: selected batches with gather/chunk candidates')
    parser.add_argument('--out-dir', required=True, help='New benchmark directory, or existing one with --summarize-existing')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Prepare and sequentially run all planned candidates; batch-scan/contact-head continue after CUDA OOM')
    mode.add_argument('--summarize-existing', action='store_true', help='Read plan/logs; launch no process')
    parser.add_argument('--source-root')
    parser.add_argument('--label-root')
    parser.add_argument('--backbone-root')
    parser.add_argument('--cpu-list', help='Actual allocated CPU IDs for taskset, e.g. an assigned 16-core range; otherwise inherit affinity')
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument('--gpu-memory-limit-gib', type=float, help='Optional finite GiB allocator cap; default is unrestricted')
    memory.add_argument('--no-gpu-memory-limit', action='store_true', help='Explicitly use all available GPU memory (already the default)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', help='batch-scan microbatch list; defaults H20 1 2 3 4 6 8 / L20 1 2 3 4')
    parser.add_argument('--effective-batch-size', type=int, help='batch-scan common effective batch; defaults H20 24 / L20 12')
    parser.add_argument('--overlap-only', action='store_true', help='batch-scan: omit the largest-batch depth1 scheduling comparison')
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
                        label_root=args.label_root, backbone_root=args.backbone_root, cpu_list=args.cpu_list,
                        suite=args.suite, gpu_memory_limit_gib=args.gpu_memory_limit_gib,
                        no_gpu_memory_limit=args.no_gpu_memory_limit, batch_sizes=args.batch_sizes,
                        effective_batch_size=args.effective_batch_size, overlap_only=args.overlap_only)
    if args.run:
        sys.exit(run_plan(plan, args.out_dir))
    print(json.dumps({'status': 'prepared; no training launched', **plan}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
