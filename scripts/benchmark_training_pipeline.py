"""Compare four real frames from one checkpoint in isolated, disposable processes.

This does not update optimizer parameters or write any formal run state. Run on
an authorized free device, with an immutable checkpoint snapshot. Reports local
stage timings separately from actual training throughput (measured in run logs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def child(args):
    # Resolve all project imports from exactly one checkout, including the old
    # unmodified checkout when this newer script is used as its measurement tool.
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import copy
    import torch
    from suction20.datasets import collate_public, frame_key, split_scene_ids
    from suction20.losses import source_losses
    from suction20.pretrain_budget import resolve_budget, resolve_config, sampling_epoch
    from suction20.runner import make_model, make_source_dataset, override_paths
    from suction20.training import (FrameSchedule, capture_rng, restore_rng,
                                   load_config, move_to_device, seed_everything,
                                   code_revision, worktree_digest)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = override_paths(resolve_config(load_config(args.config)), args)
    performance = None
    if args.phase == 'optimized':
        performance = json.loads(Path(args.performance_config).read_text())
        performance['detailed_cpu_timing'] = True  # Diagnostic process only.
        config.setdefault('runtime', {})['performance'] = performance
    seed_everything(config.get('seed', 42))
    torch.set_num_threads(config.get('runtime', {}).get('cpu_threads', 8))
    device = torch.device(args.device)
    if device.type == 'cuda':
        device = torch.device('cuda', device.index or 0)
        if not torch.cuda.is_available():
            raise RuntimeError('Real CUDA comparison requires CUDA')
        cap = config['runtime']['gpu_memory_limit_gib'] * 1024**3
        torch.cuda.set_per_process_memory_fraction(min(1., cap / torch.cuda.get_device_properties(device).total_memory), device)
    dataset = make_source_dataset(config, augment=True)
    keys = {key: i for i, key in enumerate(dataset.keys)}
    schedule = FrameSchedule(split_scene_ids('train'), seed=config.get('sampling_seed', 20261110),
                             views_per_scene=config['data'].get('views_per_train_scene_per_epoch', 64),
                             frame_keys=config['data'].get('smoke_frame_keys'))
    schedule.load_state_dict(state['sampler_state'])
    budget = resolve_budget(config)
    if budget['effective_batch_size'] != 4 or config['training']['microbatch_size'] != 1:
        raise ValueError('This bounded comparison requires the approved 1 x 4 profile')
    model = make_model(config, dataset.data_cfg, device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.train()
    model.zero_grad(set_to_none=True)
    amp = config['training'].get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    scaler.load_state_dict(state['scaler_state_dict'])
    restore_rng(state['rng_state'])
    records, query_results = [], []
    group = getattr(model, 'contact_grouping', None)
    if group is not None:
        original_query = group.query

        def capture_query(*pos, **kw):
            before = capture_rng()
            value = original_query(*pos, **kw)
            after = capture_rng()
            row = {'result': fingerprint(value)}
            saved_timings = copy.deepcopy(getattr(group, 'last_timings', {}))
            saved_query_timings = copy.deepcopy(getattr(group, 'last_query_timings', {}))
            if not query_results:
                # Same poses and roll RNG, two local repetitions after the real
                # query warmed this path. No additional model forward/backward.
                repetitions = []
                for _ in range(2):
                    restore_rng(before)
                    started = time.perf_counter()
                    repeated = original_query(*pos, **kw)
                    repetitions.append(time.perf_counter() - started)
                    if fingerprint(repeated) != row['result']:
                        raise ValueError('Repeated fixed-pose query changed its result')
                row['query_repeated_seconds'] = repetitions
                if args.phase == 'optimized':
                    chunk, workers = group.query_chunk_size, group.query_workers
                    variants = {}
                    for label, query, size, count in (
                        ('compiled_scalar_shared', original_query, None, 1),
                        ('compiled_batch64_worker1', original_query, 64, 1),
                        ('compiled_batch64_worker4', original_query, 64, 4),
                    ):
                        group.query_chunk_size, group.query_workers = size, count
                        durations = []
                        for _ in range(2):
                            restore_rng(before)
                            started = time.perf_counter()
                            variant = query(*pos, **kw)
                            durations.append(time.perf_counter() - started)
                            if fingerprint(variant) != row['result']:
                                raise ValueError(f'{label} changed fixed-pose geometry')
                        variants[label] = {'seconds': durations,
                                           'timings': copy.deepcopy(group.last_query_timings)}
                    group.query_chunk_size, group.query_workers = chunk, workers
                    row['query_schedule_comparison'] = variants
            restore_rng(after)
            group.last_timings = saved_timings
            if hasattr(group, 'last_query_timings'):
                group.last_query_timings = saved_query_timings
            query_results.append(row)
            return value

        group.query = capture_query

    def timed(fn):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return result, time.perf_counter() - start

    for index in range(4):
        key, aug_seed = schedule.next()
        normalized = frame_key(key, config['data'].get('camera', 'realsense'))
        sample, read_s = timed(lambda: dataset.get_sample(keys[normalized], aug_seed))
        packed, collate_s = timed(lambda: collate_public([sample], config['model'].get('voxel_size_m', .002)))
        data_identity = fingerprint({'inputs': packed['inputs'], 'targets': packed['targets'], 'meta': packed['meta']})
        kwargs = {}
        if performance is not None:
            from suction20.cpu_geometry import CpuGeometry, SamplerTargetsCPU
            kwargs = {'cpu_geometry': CpuGeometry.from_inputs(packed['inputs']),
                      'sampler_targets_cpu': SamplerTargetsCPU.from_targets(packed['targets'])}
        (inputs, targets), transfer_s = timed(lambda: (move_to_device(packed['inputs'], device),
                                                       move_to_device(packed['targets'], device)))
        model.set_epoch(sampling_epoch(budget, state['global_step'], schedule.epoch))

        def forward():
            with torch.autocast(device_type=device.type, enabled=amp):
                return model(inputs, sampler_mode='train', targets=targets, **kwargs)

        prediction, forward_s = timed(forward)
        losses, loss_s = timed(lambda: source_losses(prediction, targets, inputs,
            quality_target=config['model']['quality_target'],
            beta=config.get('loss', {}).get('smooth_l1_beta', .05),
            weights=config.get('loss', {}).get('weights')))
        if not torch.isfinite(losses['total']):
            raise ValueError('Non-finite comparison loss')
        _, backward_s = timed(lambda: scaler.scale(losses['total'] / 4).backward())
        outputs = {k: v.detach().cpu() for k, v in prediction.items() if isinstance(v, torch.Tensor)}
        torch.save(outputs, out / f'{args.phase}-frame-{index}.pt')
        records.append({'frame_key': list(normalized), 'augmentation_seed': aug_seed,
                        'data': data_identity, 'loss': {k: float(v.detach()) for k, v in losses.items()},
                        'seconds': {'prepare': read_s, 'collate': collate_s, 'h2d': transfer_s,
                                    'forward_including_diagnostic_repeats': forward_s,
                                    'loss': loss_s, 'backward': backward_s},
                        'grouping': dict(getattr(model, 'last_grouping_timings', {})),
                        'production': packed.get('production_timing', {})})
        print(json.dumps({'phase': args.phase, 'frame': index, 'seconds': records[-1]['seconds']}), flush=True)
        del sample, packed, prediction, outputs, losses, inputs, targets, kwargs
    gradients = {name: p.grad.detach().cpu() if p.grad is not None else None for name, p in model.named_parameters()}
    if any(g is not None and not torch.isfinite(g).all() for g in gradients.values()):
        raise ValueError('Non-finite comparison gradients')
    torch.save({'gradients': gradients, 'buffers': {k: v.detach().cpu() for k, v in model.named_buffers()},
                'rng': capture_rng()}, out / f'{args.phase}-state.pt')
    report = {'phase': args.phase, 'repo': str(Path(args.repo).resolve()), 'checkpoint_sha256': digest(args.checkpoint),
              'code_revision': code_revision(), 'worktree_sha256': worktree_digest(),
              'checkpoint_step': state['global_step'], 'frames': records, 'queries': query_results,
              'sampler_after': schedule.state_dict(), 'amp': bool(amp), 'optimizer_updates': 0,
              'gradient_scale': scaler.get_scale(),
              'diagnostic_performance': performance,
              'model_execution': getattr(model, 'execution_settings', {}),
              'performance_config_sha256': digest(args.performance_config) if performance else None}
    (out / f'{args.phase}.json').write_text(json.dumps(report, indent=2) + '\n')


def compare(args):
    import numpy as np
    import torch
    out = Path(args.out_dir)
    old, new = [json.loads((out / f'{name}.json').read_text()) for name in ('baseline', 'optimized')]
    failures, numeric = [], {}
    atol, rtol = (2e-4, 2e-3) if old['amp'] else (1e-6, 1e-4)

    def check(a, b, path):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                failures.append(path + ': tensor contract'); return
            if a.is_floating_point():
                delta = (a.double() - b.double()).abs()
                numeric[path] = {'max_abs': float(delta.max()) if a.numel() else 0.,
                                 'max_rel': float((delta / a.double().abs().clamp_min(1e-12)).max()) if a.numel() else 0.}
                if not torch.allclose(a, b, atol=atol, rtol=rtol): failures.append(path)
            elif not torch.equal(a, b): failures.append(path)
        elif isinstance(a, np.ndarray):
            if not np.array_equal(a, b): failures.append(path)
        elif isinstance(a, dict):
            if a.keys() != b.keys(): failures.append(path + ': keys'); return
            for k in a: check(a[k], b[k], path + '.' + str(k))
        elif isinstance(a, (tuple, list)):
            if len(a) != len(b): failures.append(path + ': length'); return
            for i, (x, y) in enumerate(zip(a, b)): check(x, y, f'{path}.{i}')
        elif a != b: failures.append(path)

    for k in ('checkpoint_sha256', 'checkpoint_step', 'sampler_after', 'amp'):
        check(old[k], new[k], k)
    for i, (a, b) in enumerate(zip(old['frames'], new['frames'])):
        for k in ('frame_key', 'augmentation_seed', 'data'): check(a[k], b[k], f'frame.{i}.{k}')
        for k in a['loss']:
            check(torch.tensor(a['loss'][k]), torch.tensor(b['loss'][k]), f'frame.{i}.loss.{k}')
        check(torch.load(out / f'baseline-frame-{i}.pt', weights_only=False),
              torch.load(out / f'optimized-frame-{i}.pt', weights_only=False), f'outputs.{i}')
    check([q['result'] for q in old['queries']], [q['result'] for q in new['queries']], 'query')
    check(torch.load(out / 'baseline-state.pt', weights_only=False),
          torch.load(out / 'optimized-state.pt', weights_only=False), 'state')
    report = {'schema_version': 'suction20.pipeline-comparison.v1', 'passed': not failures,
              'failures': failures, 'atol': atol, 'rtol': rtol, 'numeric_errors': numeric,
              'baseline': old, 'optimized': new,
              'limitations': ['Four fixed real frames, no optimizer update, no quality/convergence claim.',
                  'Explicit stage CUDA synchronization; diagnostic query repeats inflate forward timing.',
                  'Both processes compare gradients with the same checkpoint AMP scale; gradients are not unscaled.',
                  'Both processes upload all input fields; selective H2D and prefetch gains require normal logs.',
                  'Normal training logs, not this benchmark, establish combined pipeline throughput.']}
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'failures': failures, 'report': str(out / 'comparison.json')}))
    return 0 if report['passed'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-repo')
    parser.add_argument('--candidate-repo', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--repo', help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=('baseline', 'optimized'), help=argparse.SUPPRESS)
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--performance-config', required=True)
    parser.add_argument('--source-root')
    parser.add_argument('--label-root')
    parser.add_argument('--backbone-root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if args.phase:
        child(args); return 0
    if args.compare_only:
        return compare(args)
    if not args.baseline_repo:
        parser.error('--baseline-repo is required')
    if Path(args.out_dir).exists() and any(Path(args.out_dir).iterdir()):
        parser.error('Use a new comparison output directory')
    for phase, repo in (('baseline', args.baseline_repo), ('optimized', args.candidate_repo)):
        command = [sys.executable, str(Path(__file__).resolve()), '--phase', phase, '--repo', repo,
                   '--checkpoint', args.checkpoint, '--config', args.config,
                   '--performance-config', args.performance_config, '--device', args.device,
                   '--out-dir', args.out_dir]
        for name in ('source_root', 'label_root', 'backbone_root'):
            if getattr(args, name): command += ['--' + name.replace('_', '-'), getattr(args, name)]
        subprocess.run(command, check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    return compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
