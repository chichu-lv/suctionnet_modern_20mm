#!/usr/bin/env python3
"""Disposable optimized-pipeline trial: 1 warmup + 3 measured E=4 blocks.

No optimizer update, checkpoint access, validation or formal-run write. Uses the
production ticket dataset/sampler/collate/consume path, bounded to 16 real frames,
CPU geometry, selective nonblocking upload and compiled FPS. Fresh seed42 model;
BN/RNG evolve locally, parameters never update. Adam moments and update cost are
EXCLUDED. Different batch partitions change BN, loss reduction and RNG ordering.

Host stage intervals retain incidental synchronization, as in train_pretrain;
only block boundaries explicitly synchronize CUDA. They are not kernel timings.
Producer work overlaps training. Throughput includes exposed loader wait, gradient
checks and instrument overhead. Parent must verify idle authorized GPU/CPU cores,
launch trials sequentially and enforce external timeout <=1200 seconds including
cleanup (Python SIGALRM cannot preempt a stuck native call). No automatic retries.
"""
from __future__ import annotations

import argparse
import fcntl
from functools import partial
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid


class LimitedBatches:
    """Bound production issue-ahead without altering ticket identity/consumption."""
    def __init__(self, sampler, count):
        self.sampler, self.count = sampler, count

    def __iter__(self):
        yield from itertools.islice(iter(self.sampler), self.count)

    def __len__(self):
        return self.count


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(blocks):
    rows = [b for b in blocks if b['kind'] == 'measured' and b.get('complete')]
    seconds = sum(b['total_seconds'] for b in rows)
    return dict(measured_blocks=len(rows), measured_frames=sum(b['frames'] for b in rows),
                measured_seconds=seconds,
                frames_per_second=sum(b['frames'] for b in rows) / seconds if seconds else None,
                median_block_seconds=statistics.median(b['total_seconds'] for b in rows) if rows else None,
                all_gradients_finite=all(b['gradients_finite'] for b in rows) if rows else None)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(value.dtype).encode() + value.numpy().tobytes())
    return digest.hexdigest()


class Monitor:
    """Read-only existing resource collectors; no invented host/NVML PID mapping."""
    def __init__(self, gpu_uuid, workers):
        from suction20.training_telemetry import ResourceSampler
        self.sampler = ResourceSampler()
        self.gpu_uuid, self.workers = gpu_uuid, workers
        self.worker_samplers, self.samples = {}, []
        self.phase = 'startup'
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        from suction20.training_telemetry import sample_loader_workers
        while not self.stop.is_set():
            try:
                row = self.sampler.sample(pid=os.getpid(), gpu_uuid=self.gpu_uuid, phase=self.phase)
                row['loader_workers'] = sample_loader_workers(self.worker_samplers, self.workers, os.getpid())
                self.samples.append(row)
            except Exception as error:
                self.samples.append(dict(unix=time.time(), phase=self.phase, error=repr(error)))
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)


def run_trial(args, report):
    repo = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo))
    # Spawned production workers inherit this explicit frozen checkout path.
    import torch
    from torch.utils.data import DataLoader
    from models.geometry_ops import fps_backend_identity
    from suction20.cpu_geometry import CpuGeometry, SamplerTargetsCPU
    from suction20.datasets import frame_key, split_scene_ids
    from suction20.losses import source_losses
    from suction20.performance import attach_performance, build_runtime_identity
    from suction20.prefetch import (TicketDataset, TicketBatchSampler, collate_tickets,
        worker_init_cpu_only, consume_tickets, close_ticket_loader)
    from suction20.pretrain_runtime import file_digest, normalize_gpu_uuid
    from suction20.runner import make_model, make_source_dataset, override_paths
    from suction20.run_supervisor import process_identity
    from suction20.training import (load_config, seed_everything, FrameSchedule, content_digest,
        code_revision, worktree_digest, move_to_device, select_training_gpu_inputs,
        source_supervision_counts, decay_groups)

    require(Path(sys.modules['suction20.training'].__file__).resolve().is_relative_to(repo), 'Wrong package import root')
    revision, source_hash = code_revision(), worktree_digest()
    require(revision == args.expected_code_revision and source_hash == args.expected_worktree_sha256,
            'Frozen code revision/worktree digest mismatch')
    require(len(os.sched_getaffinity(0)) == args.cpu_threads, 'Launch with taskset: affinity width must equal requested CPU threads')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Launch with OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1')
    gpu_uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(gpu_uuid == args.gpu_uuid and gpu_uuid.startswith('GPU-') and ',' not in gpu_uuid,
            'Expose the single parent-verified GPU UUID')
    def smi(query):
        return subprocess.check_output(['nvidia-smi', '-i', gpu_uuid, query, '--format=csv,noheader,nounits'],
                                       text=True, timeout=5).strip()
    require(not smi('--query-compute-apps=gpu_uuid,pid,used_gpu_memory'), 'GPU has existing compute processes')
    baseline = smi('--query-gpu=uuid,index,name,memory.total,memory.used,driver_version')
    fields = [v.strip() for v in baseline.split(',')]
    require(fields[0] == gpu_uuid and int(fields[4]) < 1024, 'GPU mismatch or >=1GiB already occupied')
    report['gpu_preflight'] = baseline
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one available CUDA device required')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    require(normalize_gpu_uuid(props.uuid) == gpu_uuid, 'Torch/NVML UUID mismatch')
    config = override_paths(load_config(args.config), args)
    require(config.get('seed', 42) == 42 and config.get('sampling_seed', 20261110) == 20261110, 'Expected shared seeds42/20261110')
    require(not config.get('init_checkpoint') and not config['data'].get('adapter_cache') and
            not any('smoke' in k for k in config['data']), 'No smoke, initialization checkpoint or prepared cache')
    require(config['training'].get('amp', True), 'AMP must remain enabled')
    require(config['model'].get('voxel_size_m', .002) == .002 and config['model'].get('seed_count', 512) == 512,
            'Full geometry/candidate resolution required')
    config = attach_performance(config, args.performance_config)
    require(config['runtime']['performance']['fps_backend'] == 'compiled', 'Compiled FPS required')
    base_config = json.loads(json.dumps(config))
    config['training'].update(microbatch_size=args.microbatch, gradient_accumulation_steps=args.accumulation,
                              effective_batch_size=4)
    config['runtime']['cpu_threads'] = args.cpu_threads
    config['runtime']['raw_config_sha256'] = file_digest(args.config)
    # Preserve validated sidecar identity. This research-only override cannot be
    # supplied to the frozen formal launcher, whose schema accepts workers0/2.
    loader_settings = dict(config['runtime']['performance']['loader'], num_workers=args.workers)
    require(not config['runtime']['performance']['detailed_cpu_timing'], 'Use production detailed_cpu_timing=false')
    fraction = min(args.memory_fraction, args.gpu_memory_limit_gib * 1024**3 / props.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.set_num_threads(args.cpu_threads)
    seed_everything(42)
    schedule = FrameSchedule(split_scene_ids('train'), seed=20261110,
        views_per_scene=config['data'].get('views_per_train_scene_per_epoch', 64))
    plan_schedule = FrameSchedule(split_scene_ids('train'), seed=20261110,
        views_per_scene=config['data'].get('views_per_train_scene_per_epoch', 64))
    planned = [plan_schedule.next() for _ in range(16)]
    dataset = make_source_dataset(config, frame_keys=list(dict.fromkeys(k for k, _ in planned)), augment=True)
    report['planned_frames'] = [dict(sample_uid=list(frame_key(k, 'realsense')), augmentation_seed=s) for k, s in planned]
    report['planned_frames_sha256'] = content_digest(report['planned_frames'])
    report['identity'] = dict(runtime=build_runtime_identity(base_config, dataset.data_cfg),
        original_config=base_config, actual_model_config=config, loader_override=loader_settings,
        config_raw_sha256=file_digest(args.config), helper_sha256=file_digest(__file__),
        performance_raw_sha256=file_digest(args.performance_config), fps=fps_backend_identity('compiled'),
        code_revision=revision, worktree_sha256=source_hash, python=sys.version, host=platform.node(),
        gpu_uuid=gpu_uuid, gpu_name=props.name, total_gpu_bytes=props.total_memory,
        allocator_fraction=fraction, allocator_limit_bytes=int(props.total_memory * fraction),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch_cpu_threads=torch.get_num_threads(),
        torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        detailed_cpu_timing_override=False, sampler_regime_epoch=args.sampler_epoch,
        effective_profile_sha256=content_digest(dict(config=config, loader=loader_settings)))
    model = make_model(config, dataset.data_cfg, device).train()
    report['identity']['model_execution'] = getattr(model, 'execution_settings', {
        'profile_dense_chunk_size': config['model'].get('dense_chunk_size', 16384),
        'effective_dense_chunk_size': model.chunk_size,
        'dense_chunk_override_source': 'profile', 'chunk_input_mode': 'slice'})
    model.set_epoch(args.sampler_epoch)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(decay_groups(model, weight_decay=config.get('optimizer', {}).get('weight_decay', 1e-4)))
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    report['estimated_excluded_adam_moment_bytes'] = 2 * sum(p.numel() * p.element_size() for p in parameters)
    report['initial_parameters_sha256'] = parameter_digest(model)
    # Same production loader components/options, except finite issuance and the
    # explicit worker candidate. No extra prepared frames beyond the16 tickets.
    indices = {frame_key(key, 'realsense'): i for i, key in enumerate(dataset.keys)}
    batches = LimitedBatches(TicketBatchSampler(schedule.state_dict(), 0, args.microbatch, indices),
                             4 * args.accumulation)
    generator = torch.Generator(device='cpu').manual_seed(20261110)
    loader = DataLoader(TicketDataset(dataset), batch_sampler=batches,
        collate_fn=partial(collate_tickets, voxel_size_m=.002), generator=generator,
        worker_init_fn=worker_init_cpu_only, **loader_settings)
    iterator, monitor = None, None
    consumed = 0
    try:
        loader_started = time.perf_counter()
        iterator = iter(loader)
        report['loader_startup_host_seconds'] = time.perf_counter() - loader_started
        workers = [process_identity(w.pid) for w in getattr(iterator, '_workers', [])]
        report['loader_workers'] = workers
        monitor = Monitor(gpu_uuid, workers)
        monitor.thread.start()
        optimizer.zero_grad(set_to_none=True)
        for block_index in range(4):
            kind = 'warmup' if block_index == 0 else 'measured'
            monitor.phase = f'{kind}:{block_index}'
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            block = dict(index=block_index, kind=kind, frames=4, complete=False,
                         amp_scale_before=scaler.get_scale(), microbatches=[])
            report['blocks'].append(block)
            for micro_index in range(args.accumulation):
                timer = time.perf_counter()
                packed = next(iterator)
                micro = dict(index=micro_index, loader_wait_s=time.perf_counter() - timer,
                    production_timing=packed.get('production_timing', {}), frames=[], timing_host_seconds={})
                block['microbatches'].append(micro)
                for i, ticket in enumerate(packed['tickets']):
                    offsets = packed['inputs']['point_offsets']
                    micro['frames'].append(dict(sample_uid=list(ticket.frame_key), augmentation_seed=ticket.augmentation_seed,
                        sequence_id=ticket.sequence_id, point_count=int(offsets[i+1]-offsets[i]),
                        voxel_count=int((packed['inputs']['voxel_coords'][:,0] == i).sum())))
                timer = time.perf_counter()
                cpu_kwargs = dict(cpu_geometry=CpuGeometry.from_inputs(packed['inputs']),
                                  sampler_targets_cpu=SamplerTargetsCPU.from_targets(packed['targets']))
                inputs = move_to_device(select_training_gpu_inputs(packed['inputs']), device, non_blocking=True)
                targets = move_to_device(packed['targets'], device, non_blocking=True)
                micro['timing_host_seconds']['context_and_h2d'] = time.perf_counter()-timer
                timer = time.perf_counter()
                with torch.autocast(device_type='cuda', enabled=True):
                    prediction = model(inputs, sampler_mode='train', targets=targets, **cpu_kwargs)
                micro['timing_host_seconds']['forward'] = time.perf_counter()-timer
                timer = time.perf_counter()
                losses = source_losses(prediction, targets, inputs, quality_target=config['model']['quality_target'],
                    beta=config.get('loss', {}).get('smooth_l1_beta', .05), weights=config.get('loss', {}).get('weights'))
                require(bool(torch.isfinite(losses['total'])), 'Nonfinite loss')
                micro['timing_host_seconds']['loss_and_finite_check'] = time.perf_counter()-timer
                timer = time.perf_counter()
                scaler.scale(losses['total'] / args.accumulation).backward()
                micro['losses'] = {k: float(v.detach()) for k,v in losses.items()}
                micro['timing_host_seconds']['backward_and_loss_readback'] = time.perf_counter()-timer
                timer = time.perf_counter()
                micro['supervised_frame_counts'] = source_supervision_counts(prediction, packed)
                micro['valid_seed_count_by_frame'] = prediction['seed_valid'].sum(dim=-1).detach().cpu().tolist()
                micro['grouping_timings'] = dict(getattr(model, 'last_grouping_timings', {}))
                consumed = consume_tickets(schedule, packed['tickets'], consumed)
                micro['timing_host_seconds']['diagnostics'] = time.perf_counter()-timer
                del packed, inputs, targets, cpu_kwargs, prediction, losses
            timer = time.perf_counter()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.get('optimizer', {}).get('grad_clip_norm', 10.))
            finite = bool(torch.isfinite(norm))
            scaler.update()  # consumes finite checks only; NEVER steps optimizer
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            block.update(complete=True, gradients_finite=finite, grad_norm=float(norm) if finite else None,
                amp_scale_after=scaler.get_scale(), unscale_clip_zero_sync_seconds=time.perf_counter()-timer,
                total_seconds=time.perf_counter()-start, max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                max_reserved_bytes=torch.cuda.max_memory_reserved(device), consumed_frames=consumed,
                allocator_reserved_headroom_bytes=int(props.total_memory*fraction)-torch.cuda.max_memory_reserved(device),
                estimated_reserved_headroom_after_adam_moments_bytes=int(props.total_memory*fraction)
                    -torch.cuda.max_memory_reserved(device)-report['estimated_excluded_adam_moment_bytes'])
            block['loader_wait_s'] = sum(m['loader_wait_s'] for m in block['microbatches'])
            print(json.dumps(dict(block=block_index, kind=kind, seconds=block['total_seconds'],
                                 frames_per_second=4/block['total_seconds'], optimizer_updates=0)), flush=True)
            require(finite, 'Nonfinite gradients: trial stops without retry or parameter update')
        require(consumed == 16, 'Consumption cardinality mismatch')
        observed = [dict(sample_uid=f['sample_uid'], augmentation_seed=f['augmentation_seed'])
                    for b in report['blocks'] for m in b['microbatches'] for f in m['frames']]
        require(observed == report['planned_frames'], 'Consumed frame/augmentation order differs from plan')
        report['consumed_frames_sha256'] = content_digest(observed)
        report['final_parameters_sha256'] = parameter_digest(model)
        require(report['final_parameters_sha256'] == report['initial_parameters_sha256'], 'Unexpected parameter update')
        require(not optimizer.state, 'Unexpected Adam moment state')
        report['status'] = 'completed'
    finally:
        if monitor:
            monitor.close()
            report['resources'] = monitor.samples
        close_ticket_loader(loader, iterator)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    for flag in ('repo-root', 'config', 'performance-config', 'expected-code-revision', 'expected-worktree-sha256', 'gpu-uuid', 'gpu-lock', 'out'):
        p.add_argument('--'+flag)
    p.add_argument('--microbatch', type=int, choices=(1,2,4), default=1)
    p.add_argument('--accumulation', type=int, choices=(1,2,4), default=4)
    p.add_argument('--cpu-threads', type=int, choices=(8,16), default=8)
    p.add_argument('--workers', type=int, choices=(2,4), default=2)
    p.add_argument('--sampler-epoch', type=int, choices=(0,2), default=0)
    p.add_argument('--memory-fraction', type=float, default=.85)
    p.add_argument('--gpu-memory-limit-gib', type=float, required=False)
    p.add_argument('--max-wall-seconds', type=int, default=1200)
    for flag in ('source-root', 'label-root', 'backbone-root'):
        p.add_argument('--'+flag)
    return p


def main():
    args = parser().parse_args()
    if args.self_test:
        source = iter(range(100))
        assert list(LimitedBatches(source, 4)) == [0,1,2,3] and next(source) == 4
        rows = [dict(kind='warmup', complete=True, frames=4, total_seconds=99, gradients_finite=True)]
        rows += [dict(kind='measured', complete=True, frames=4, total_seconds=s, gradients_finite=True) for s in (2,3,5)]
        result = summarize(rows)
        assert result['frames_per_second'] == 1.2 and result['median_block_seconds'] == 3
        print('Bounded issuance and warmup-excluding throughput self-test passed (CPU/stdlib).')
        return 0
    for field in ('repo_root','config','performance_config','expected_code_revision','expected_worktree_sha256','gpu_uuid','gpu_lock','out','gpu_memory_limit_gib'):
        require(getattr(args, field) is not None, f'--{field.replace("_", "-")} required')
    require(args.microbatch * args.accumulation == 4, 'Fixed effective batch4 required')
    require(0 < args.memory_fraction <= .85 and args.gpu_memory_limit_gib > 0, 'Invalid allocator cap')
    require(1 <= args.max_wall_seconds <= 1200, 'Maximum trial wall is1200 seconds')
    require(Path(args.gpu_lock).name == args.gpu_uuid + '.lock', 'GPU lock basename must match visible UUID')
    out, repo = Path(args.out).resolve(), Path(args.repo_root).resolve()
    require(out.suffix == '.json' and not out.is_relative_to(repo), 'Output must be a new JSON outside frozen checkout')
    require(not any((p/'checkpoint-manifest.json').exists() for p in out.parents), 'Output cannot be inside a formal run')
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version='suction20.optimized-disposable-profile.v1', trial_id=uuid.uuid4().hex,
        classification='profiling_only_zero_optimizer_updates', optimizer_updates=0,
        checkpoint_loaded=False, checkpoint_written=False, status='starting', arguments=vars(args),
        started_unix=time.time(), blocks=[], errors=[], limitations=[
            'Fresh seed42 model; no optimizer update or formal run state. BN and local RNG advance then are discarded.',
            'Adam moments/update, checkpoint and validation are excluded; reported moment estimate is not a complete safety margin.',
            'Equal effective4 partitions may differ in BN statistics, loss weighting and RNG interleaving; no numerical equivalence claim.',
            'Host stage times include incidental sync; block boundaries synchronize. Producer work overlaps training; do not add it to wall.',
            'Only16 tickets are issued/prepared; one warmup and three measured blocks. Existing bounded metadata caches/OS cache remain.',
            'Resource observations are sparse1s snapshots; maxima can miss peaks. NVML process mapping unavailable, device values reported.',
            'CPU workers share selected affinity, each uses one Torch/BLAS thread. Four workers are research-only outside formal schema.',
            'SIGALRM cannot interrupt stuck native calls; caller must enforce external process-group timeout <=1200s.'])
    with out.open('x') as stream:
        json.dump(report, stream, indent=2)
    def timeout(signum, frame):
        raise TimeoutError('Trial wall bound reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
    gpu_lock = None
    try:
        gpu_lock = Path(args.gpu_lock).open('rb')
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['gpu_lock'] = dict(path=str(Path(args.gpu_lock).resolve()), held=True, content_modified=False)
        run_trial(args, report)
    except Exception as error:
        report['status'] = 'oom' if 'out of memory' in str(error).lower() else 'failed'
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report['finished_unix'] = time.time()
        report['wall_seconds'] = report['finished_unix'] - report['started_unix']
        if gpu_lock is not None:
            gpu_lock.close()
        report['summary'] = summarize(report['blocks'])
        temporary = out.with_suffix('.tmp')
        with temporary.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(temporary, out)
    print(json.dumps(dict(status=report['status'], out=str(out), summary=report['summary'], errors=report['errors'])), flush=True)
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
