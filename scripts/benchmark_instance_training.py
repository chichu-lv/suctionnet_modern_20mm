#!/usr/bin/env python3
"""Measure disposable source-training trials on real manifests, without selecting weights.

Each trial constructs a fresh model, optimizer, seeded sampler and worker pool.
Only microbatch size and worker count vary in the optional bounded grid; the
effective optimizer batch, learning-rate schedule and source budget stay fixed.
No checkpoint is loaded or written and no selection evaluation is performed.
The optional wall-clock mode counts only synchronized updates ending within its
fixed window; an optimizer update crossing the deadline is finished and logged
separately, without contributing to window throughput or latency percentiles.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import importlib
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from suction20.instance_config import apply_model_runtime_overrides
from suction20.instance_data import InstanceDataset
from suction20.instance_execution import resolve_execution
from suction20.instance_selection import select_source_evaluation, validate_selection_config
from suction20.instance_training import (
    InstanceTrainer, atomic_json, check_source_splits, clean_metadata,
    create_instance_model, load_instance_config, seed_all,
)


MAX_TRIALS = 16
_ENVIRONMENT_NAMES = (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF',
    'SUCTION20_INSTANCE_TRAIN_WORKERS', 'SUCTION20_INSTANCE_EVAL_WORKERS',
    'SUCTION20_INSTANCE_B_MICROBATCH',
    'SUCTION_COMPACT_SOURCE_IPC', 'SUCTION_INSTANCE_SELECTION_CACHE',
    'SUCTION_INSTANCE_STATIC_CACHE_MB',
    'SUCTION_INSTANCE_SOURCE_CACHE_A', 'SUCTION_INSTANCE_SOURCE_CACHE_B',
    'SUCTION_INSTANCE_GEOMETRY_CACHE_A', 'SUCTION_INSTANCE_GEOMETRY_CACHE_B',
    'SUCTION_A_CONTACT_BACKEND', 'SUCTION_A_CONTACT_UNION',
)


def _integer_choices(text, *, minimum, name):
    try:
        values = [int(value.strip()) for value in text.split(',')]
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f'{name} must be comma-separated integers') from error
    if not values or any(value < minimum for value in values):
        raise argparse.ArgumentTypeError(f'{name} entries must be at least {minimum}')
    return list(dict.fromkeys(values))


def _microbatches(text):
    return _integer_choices(text, minimum=1, name='microbatches')


def _worker_counts(text):
    return _integer_choices(text, minimum=0, name='worker-counts')


def _rss_snapshot():
    """Report resident bytes, with optional live child totals when psutil exists."""
    try:
        import psutil
        process = psutil.Process()
        children = []
        for child in process.children(recursive=True):
            try:
                children.append(int(child.memory_info().rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {'parent_rss_bytes': int(process.memory_info().rss),
                'children_rss_bytes': sum(children), 'child_process_count': len(children),
                'source': 'psutil_resident_bytes'}
    except ImportError:
        # /proc is read-only and does not require an extra runtime dependency.
        try:
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    return {'parent_rss_bytes': int(line.split()[1]) * 1024,
                            'children_rss_bytes': None, 'child_process_count': None,
                            'source': 'proc_self_status'}
        except (OSError, ValueError):
            pass
        return {'parent_rss_bytes': None, 'children_rss_bytes': None,
                'child_process_count': None, 'source': 'unavailable'}


def _environment(device):
    result = {'python': sys.version.split()[0], 'platform': platform.platform(),
              'torch': str(torch.__version__), 'numpy': str(np.__version__),
              'cuda_runtime': torch.version.cuda, 'cpu_logical_count': os.cpu_count(),
              'device': str(device),
              'preparation_capabilities': _preparation_capabilities(),
              'environment': {name: os.environ[name] for name in _ENVIRONMENT_NAMES if name in os.environ}}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        result['cuda_device'] = {'name': properties.name,
                                 'total_memory_bytes': int(properties.total_memory),
                                 'compute_capability': [properties.major, properties.minor]}
        result['cudnn_version'] = torch.backends.cudnn.version()
    return result


def _preparation_capabilities():
    """Read loaded CPU preparation versions, without constructing a CUDA model.

    An available entry point is a capability, not evidence that a measured
    update called it. Explicit native mode and preparation cache counters are
    reported separately to make the actual execution path reviewable.
    """
    result = {'scope': 'module_capabilities_only_not_observed_invocations',
              'route_a': None, 'route_b': None, 'import_errors': {}}
    try:
        geometry = importlib.import_module('models.instance_route_a_geometry')
        native = getattr(geometry, '_spatial_index_cpu', None)
        fps = getattr(geometry, '_fps_cpu', None)
        axes = getattr(geometry, '_FPS_AXES', None)
        result['route_a'] = {
            'geometry_module': getattr(geometry, '__file__', None),
            'native_extension': getattr(native, '__file__', None),
            'native_contact_entry_available': callable(getattr(native, 'route_a_contact_query', None)),
            'ROUTE_A_CONTACT_SOURCE_VERSION': getattr(native, 'ROUTE_A_CONTACT_SOURCE_VERSION', None),
            'ROUTE_A_CONTACT_ABI_VERSION': getattr(native, 'ROUTE_A_CONTACT_ABI_VERSION', None),
            'ROUTE_A_CONTACT_PROFILE_VERSION': getattr(native, 'ROUTE_A_CONTACT_PROFILE_VERSION', None),
            'fps_extension': getattr(fps, '__file__', None),
            'compiled_fps_supported': callable(getattr(fps, 'fps_positions', None)) and axes is not None,
            'fps_axis_order': list(map(int, axes)) if axes is not None else None,
        }
    except (ImportError, OSError) as error:
        result['import_errors']['route_a'] = f'{type(error).__name__}: {error}'
    try:
        preparation = importlib.import_module('models.instance_route_b_preparation')
        result['route_b'] = {'STATIC_INPUTS_SPEC': preparation.STATIC_INPUTS_SPEC,
                             'preparation_module': getattr(preparation, '__file__', None)}
    except (ImportError, OSError) as error:
        result['import_errors']['route_b'] = f'{type(error).__name__}: {error}'
    return result


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _gpu_memory(device):
    if device.type != 'cuda':
        return {'max_memory_allocated_bytes': None, 'max_memory_reserved_bytes': None}
    return {'max_memory_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
            'max_memory_reserved_bytes': int(torch.cuda.max_memory_reserved(device))}


def _timed_update(trainer, device, *, window_start=None):
    """Include data waiting, host work and completed GPU execution in every time."""
    before_step = int(trainer.step)
    before_instances = int(trainer.supervised_source_instances)
    start = time.perf_counter()
    _synchronize(device)
    metrics = trainer.train_one_update()
    _synchronize(device)
    finished = time.perf_counter()
    seconds = finished - start
    instances = int(trainer.supervised_source_instances) - before_instances
    if int(trainer.step) != before_step + 1 or instances <= 0:
        raise RuntimeError('Benchmark update did not complete one successful optimizer update')
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError('Benchmark wall-clock interval must be finite and positive')
    for name in ('source_loss', 'grad_norm'):
        if name not in metrics or not math.isfinite(float(metrics[name])):
            raise FloatingPointError(f'Benchmark update has absent/nonfinite {name}')
    result = {'step': int(trainer.step), 'seconds': seconds, 'effective_instances': instances,
              'metrics': clean_metadata(metrics)}
    if window_start is not None:
        result.update(window_start_offset_seconds=start - window_start,
                      window_end_offset_seconds=finished - window_start)
    return result


def _summary(updates, clipping_threshold, *, throughput_seconds=None):
    if not updates:
        return {'updates': 0, 'effective_instances': 0, 'seconds': 0.,
                'throughput_denominator_seconds': throughput_seconds,
                'effective_instances_per_second': 0. if throughput_seconds is not None else None,
                'mean_seconds_per_update': None,
                'p50_seconds_per_update': None, 'p95_seconds_per_update': None,
                'preclip_grad_norm_max': None, 'updates_above_clip_threshold': 0,
                'all_gradients_finite': False}
    seconds = np.asarray([row['seconds'] for row in updates], dtype=np.float64)
    norms = np.asarray([row['metrics']['grad_norm'] for row in updates], dtype=np.float64)
    instances = sum(row['effective_instances'] for row in updates)
    denominator = float(seconds.sum()) if throughput_seconds is None else float(throughput_seconds)
    return {'updates': len(updates), 'effective_instances': instances, 'seconds': float(seconds.sum()),
            'throughput_denominator_seconds': denominator,
            'effective_instances_per_second': instances / denominator,
            'mean_seconds_per_update': float(seconds.mean()),
            'p50_seconds_per_update': float(np.percentile(seconds, 50)),
            'p95_seconds_per_update': float(np.percentile(seconds, 95)),
            'preclip_grad_norm_max': float(norms.max()),
            'updates_above_clip_threshold': int((norms > clipping_threshold).sum()),
            'all_gradients_finite': bool(np.isfinite(norms).all())}


def _training_budget_exhausted(trainer):
    config = trainer.config['pretrain']
    if int(trainer.step) >= int(config['steps']):
        return True
    budget = config.get('successful_instance_budget')
    return budget is not None and int(trainer.supervised_source_instances) >= int(budget)


def _run_trial(index, config, args, microbatch, workers, device):
    trial_id = f'trial_{index:02d}_micro{microbatch}_workers{workers}'
    trial_dir = args.output / trial_id
    trial_config = deepcopy(config)
    execution = resolve_execution(trial_config) or {}
    execution.update(microbatch_instances=microbatch, train_workers=workers)
    trial_config['execution'] = execution
    execution = resolve_execution(trial_config)
    trial_config['execution'] = execution
    trial_dir.mkdir(parents=True, exist_ok=False)
    trainer = model = source = selection = None
    warmup, measured, overrun, rss = [], [], [], []
    window_start = None
    window_complete = False
    window_stop_reason = 'not_started'
    measure_seconds = getattr(args, 'measure_seconds', None)
    result = {'trial_id': trial_id, 'status': 'initializing', 'role': 'disposable_training_benchmark',
              'training_budget_completed': False, 'selection_executed': False,
              'checkpoint_loaded': None, 'checkpoint_saved': None,
              'requested_config': clean_metadata(trial_config), 'seed': int(trial_config.get('seed', 42)),
              'effective_batch_instances': int(trial_config['pretrain']['accumulate_instances']),
              'microbatch_instances': microbatch, 'train_workers': workers,
              'compile_mode': execution.get('compile_mode'),
              'compile_timing': 'Compilation and worker startup occur in setup/warmup; dynamic recompilation may also affect measurement.',
              'warmup_updates_requested': args.warmup_updates,
              'measured_updates_requested': args.measure_updates,
              'measurement_mode': 'wall_clock_window' if measure_seconds is not None else 'updates',
              'measurement_seconds_requested': measure_seconds,
              'partial_manifests_allowed': bool(args.allow_partial_manifests)}
    atomic_json(trial_dir / 'config.json', trial_config)
    setup_start = time.perf_counter()
    try:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        seed_all(result['seed'])
        source = InstanceDataset(args.source_manifest, split='train',
                                 training_queries=int(trial_config['pretrain']['queries']))
        check_source_splits(source, 'train', allow_partial=args.allow_partial_manifests)
        selection = InstanceDataset(args.selection_manifest, split='test')
        selection = select_source_evaluation(selection, validate_selection_config(trial_config['pretrain']),
                                             allow_partial=args.allow_partial_manifests)
        model = create_instance_model(trial_config, str(device))
        trainer = InstanceTrainer(model, trial_config, source, selection, trial_dir / 'run',
                                  stage='pretrain', allow_partial=args.allow_partial_manifests)
        _synchronize(device)
        result['setup_seconds'] = time.perf_counter() - setup_start
        result['actual_config'] = clean_metadata(trainer.config)
        result['source_instances_available'] = len(source)
        result['selection_instances_available'] = len(selection)
        result['rss_after_setup'] = _rss_snapshot()
        for _ in range(args.warmup_updates):
            warmup.append(_timed_update(trainer, device))
        _synchronize(device)
        result['warmup_gpu_memory'] = _gpu_memory(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        rss.append(_rss_snapshot())
        if measure_seconds is None:
            for _ in range(args.measure_updates):
                measured.append(_timed_update(trainer, device))
                rss.append(_rss_snapshot())
        else:
            window_start = time.perf_counter()
            window_stop_reason = 'error'
            while time.perf_counter() - window_start < measure_seconds:
                if _training_budget_exhausted(trainer):
                    window_stop_reason = 'training_budget_exhausted'
                    raise RuntimeError('Configured training budget exhausted before the requested measurement window ended')
                row = _timed_update(trainer, device, window_start=window_start)
                if row['window_end_offset_seconds'] <= measure_seconds:
                    measured.append(row)
                else:
                    # Finish the optimizer update, but exclude all its samples
                    # and timing from the fixed-window performance statistics.
                    overrun.append(row)
                if (row['window_end_offset_seconds'] < measure_seconds
                        and _training_budget_exhausted(trainer)):
                    window_stop_reason = 'training_budget_exhausted'
                    raise RuntimeError('Configured training budget exhausted before the requested measurement window ended')
                rss.append(_rss_snapshot())
                if overrun:
                    break
            window_complete = True
            window_stop_reason = 'deadline_reached'
            if not measured:
                raise RuntimeError('No complete optimizer update ended within the requested measurement window')
        result['status'] = 'ok'
    except Exception as error:
        result['status'] = 'out_of_memory' if isinstance(error, torch.cuda.OutOfMemoryError) else 'failed'
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if measure_seconds is not None:
            actual_wall = 0. if window_start is None else time.perf_counter() - window_start
            result['measurement_window'] = {
                'requested_seconds': measure_seconds, 'completed': window_complete,
                'stop_reason': window_stop_reason, 'actual_wall_seconds': actual_wall,
                'overrun_seconds': max(0., actual_wall - measure_seconds),
                'update_inclusion': 'synchronized completion at or before the deadline',
                'throughput_denominator': 'requested_seconds',
                'actual_wall_scope': 'measurement loop including synchronization, RSS sampling and any overrun update; excludes cleanup',
                'memory_sample_scope': 'measurement loop including any overrun update',
            }
        result['measured_gpu_memory'] = _gpu_memory(device)
        result['disposable_final_step'] = int(trainer.step) if trainer is not None else 0
        result['disposable_final_source_instances'] = int(trainer.supervised_source_instances) if trainer is not None else 0
        result['warmup'] = _summary(warmup, float(trial_config.get('max_grad_norm', 10.)))
        result['measurement'] = _summary(measured, float(trial_config.get('max_grad_norm', 10.)),
                                          throughput_seconds=measure_seconds)
        result['warmup_update_records'], result['measured_update_records'] = warmup, measured
        result['overrun_update_records'] = overrun
        result['rss_samples'] = rss
        result['sampled_parent_rss_max_bytes'] = max(
            (row['parent_rss_bytes'] for row in rss if row['parent_rss_bytes'] is not None), default=None)
        result['sampled_children_rss_max_bytes'] = max(
            (row['children_rss_bytes'] for row in rss if row['children_rss_bytes'] is not None), default=None)
        result['preparation_caches'] = {name: pool.preparation_cache_stats()
            for name, pool in getattr(trainer, '_preparation_pools', {}).items()}
        try:
            if trainer is not None:
                trainer.close()
        except Exception as error:
            result['cleanup_error'] = {'type': type(error).__name__, 'message': str(error)}
            result['status'] = 'failed'
        trainer = model = source = selection = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        atomic_json(trial_dir / 'trial.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--selection-manifest', required=True, type=Path,
                        help='Validated for the configured test selection, but never evaluated in the benchmark')
    parser.add_argument('--backbone-root', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True, type=Path,
                        help='New or empty benchmark directory; never use an existing training run directory')
    parser.add_argument('--allow-partial-manifests', action='store_true')
    parser.add_argument('--microbatches', type=_microbatches,
                        help='Optional comma-separated GPU microbatches; defaults to the config value')
    parser.add_argument('--worker-counts', type=_worker_counts,
                        help='Optional comma-separated worker counts; defaults to the config value')
    parser.add_argument('--warmup-updates', type=int, default=5)
    measurement = parser.add_mutually_exclusive_group()
    measurement.add_argument('--measure-updates', type=int,
                             help='Completed updates to measure (default 20 when no time window is given)')
    measurement.add_argument('--measure-seconds', type=float,
                             help='Fixed wall-clock window; only synchronized updates ending within it count')
    parser.add_argument('--precision', choices=['fp32', 'bf16_dense'])
    parser.add_argument('--compile-mode', choices=['none', 'default', 'reduce-overhead'],
                        help='Explicit override; none disables compilation')
    parser.add_argument('--allow-tf32', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.measure_updates is None and args.measure_seconds is None:
        args.measure_updates = 20
    if args.warmup_updates < 0 or (args.measure_updates is not None and args.measure_updates < 1):
        parser.error('--warmup-updates must be nonnegative and --measure-updates must be positive')
    if args.measure_seconds is not None and (not math.isfinite(args.measure_seconds) or args.measure_seconds <= 0):
        parser.error('--measure-seconds must be finite and positive')
    try:
        config = apply_model_runtime_overrides(load_instance_config(args.config), backbone_root=args.backbone_root)
        device = torch.device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable; use --device cpu only for explicit CPU checks')
    if config['route'] == 'a' and config['model'].get('backbone_root') is None:
        parser.error('Route A requires --backbone-root or model.backbone_root in the config')
    # An old config becomes an explicit default execution profile for this
    # disposable benchmark. Unknown/invalid profile fields never get silently
    # ignored or coerced by the trial-grid overrides.
    execution = deepcopy(config.get('execution'))
    if execution is None:
        execution = {}
    if not isinstance(execution, dict):
        parser.error('execution must be a mapping')
    if args.precision is not None:
        execution['precision'] = args.precision
    if args.compile_mode is not None:
        execution['compile_mode'] = None if args.compile_mode == 'none' else args.compile_mode
    if args.allow_tf32 is not None:
        execution['allow_tf32'] = args.allow_tf32
    config['execution'] = execution
    try:
        execution = resolve_execution(config)
    except ValueError as error:
        parser.error(str(error))
    config['execution'] = execution
    if execution.get('compile_mode') is not None and args.warmup_updates == 0:
        parser.error('Compiled trials require at least one warmup update to separate cold startup')
    microbatches = args.microbatches or [int(execution.get('microbatch_instances', 1))]
    worker_counts = args.worker_counts or [int(execution.get('train_workers', 0))]
    effective_batch = int(config['pretrain']['accumulate_instances'])
    if any(value > effective_batch for value in microbatches):
        parser.error('Every microbatch must be <= pretrain.accumulate_instances; benchmarking never changes effective batch')
    if len(microbatches) * len(worker_counts) > MAX_TRIALS:
        parser.error(f'At most {MAX_TRIALS} trials are allowed per invocation; narrow the configuration grid')
    if args.measure_updates is not None and args.warmup_updates + args.measure_updates > int(config['pretrain']['steps']):
        parser.error('Warmup plus measurement must fit within the configured pretrain steps')
    if args.measure_seconds is not None and args.warmup_updates >= int(config['pretrain']['steps']):
        parser.error('Warmup must leave at least one configured update for the measurement window')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be a new or empty directory; existing run/artifacts are never overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'instance_training_benchmark_v1', 'role': 'disposable_training_benchmark',
              'training_budget_completed': False, 'selection_executed': False,
              'environment': _environment(device),
              'source_manifest': str(args.source_manifest.resolve()),
              'selection_manifest': str(args.selection_manifest.resolve()),
              'config_path': str(args.config.resolve()),
              'microbatches': microbatches, 'worker_counts': worker_counts,
              'effective_batch_instances': effective_batch, 'trials': [],
              'measurement_mode': 'wall_clock_window' if args.measure_seconds is not None else 'updates',
              'measurement_seconds_requested': args.measure_seconds,
              'measured_updates_requested': args.measure_updates,
              'comparison': 'Same effective optimizer batch, seed, source data and configured schedule. '
                            'Rank successful trials by measured effective instances/second; GPU utilization '
                            'and occupied memory are not optimization targets.'}
    for microbatch in microbatches:
        for workers in worker_counts:
            index = len(report['trials']) + 1
            trial = _run_trial(index, config, args, microbatch, workers, device)
            report['trials'].append(trial)
            atomic_json(args.output / 'benchmark.json', report)
            rate = trial['measurement']['effective_instances_per_second']
            speed = f'{rate:.3f} effective instances/s' if rate is not None else 'no completed measurements'
            print(f"{trial['trial_id']}: {trial['status']}; {speed}", flush=True)
    successful = [trial for trial in report['trials'] if trial['status'] == 'ok'
                  and (trial.get('measurement_window', {}).get('completed', False)
                       if args.measure_seconds is not None else trial['measurement']['updates'] == args.measure_updates)
                  and trial['measurement']['all_gradients_finite']]
    fastest = max(successful, key=lambda trial: trial['measurement']['effective_instances_per_second'], default=None)
    report['fastest_successful_trial'] = fastest['trial_id'] if fastest else None
    report['successful_trials'] = len(successful)
    atomic_json(args.output / 'benchmark.json', report)
    print(f"Benchmark report: {(args.output / 'benchmark.json').resolve()}", flush=True)
    return 0 if successful else 1


if __name__ == '__main__':
    raise SystemExit(main())
