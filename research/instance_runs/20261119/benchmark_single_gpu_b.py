"""Measure B's production update on real partial manifests in an exclusive CUDA window.

Run with CUDA_VISIBLE_DEVICES=<one physical index>, PYTHONPATH=<B repo>,
OMP_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=1. No training checkpoints written.
"""
import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from suction20.instance_data import InstanceDataset
from suction20.instance_training import InstanceTrainer, create_instance_model, load_instance_config, seed_all


def verify_exclusive():
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not physical.isdigit():
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to one physical GPU index')
    processes = subprocess.check_output(
        ['nvidia-smi', '-i', physical, '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if processes.strip():
        raise RuntimeError(f'GPU {physical} has an active compute process; benchmark requires a free card')


def measure(args, cfg, source, selection, workers, microbatch):
    os.environ['SUCTION20_INSTANCE_TRAIN_WORKERS'] = str(workers)
    os.environ['SUCTION20_INSTANCE_B_MICROBATCH'] = str(microbatch)
    source._cache.clear()
    seed_all(cfg['seed'])
    model = create_instance_model(cfg, 'cuda')
    trainer = InstanceTrainer(model, cfg, source, selection, args.output.parent/'benchmark_state',
                              allow_partial=True)
    timings = []
    for i in range(args.updates + 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        record = trainer.train_one_update()
        torch.cuda.synchronize()
        timings.append(dict(step=record['step'], warmup=i==0, seconds=time.perf_counter()-start,
                            loss=record['source_loss'], grad_norm=record['grad_norm'],
                            draws=record['source_draws'], skipped=record['skipped_no_supervision'],
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    return dict(workers=workers, microbatch=microbatch, records=timings,
                median_s=float(np.median([r['seconds'] for r in timings[1:]])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/instance_b.yaml'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=3)
    parser.add_argument('--workers',type=int,nargs='+',default=[4,8,16])
    parser.add_argument('--microbatches',type=int,nargs='+',default=[2,4,8])
    args=parser.parse_args()
    if args.updates < 1 or any(n<1 for n in args.workers+args.microbatches):
        parser.error('Use positive numbers of workers, updates, and microbatches')
    verify_exclusive()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        parser.error('Output exists; choose a fresh output file')
    cfg=load_instance_config(args.config)
    source=InstanceDataset(args.source,split='train',training_queries=cfg['pretrain']['queries'])
    selection=InstanceDataset(args.selection,split='test')
    result=dict(gpu=torch.cuda.get_device_name(), config=str(args.config), source=str(args.source),
                selection=str(args.selection), scope='partial_real_successful_updates_including_loader_and_optimizer',
                torch=torch.__version__, cuda=torch.version.cuda, shape_policy='context_batch_equal_texture_hw', trials=[])
    reference=None
    def run(workers,micro):
        nonlocal reference
        gc.collect(); torch.cuda.empty_cache()
        row=measure(args,cfg,source,selection,workers,micro)
        if reference is None:
            reference=row['records']
        row['same_draw_and_skip_counts']=all((a['draws'],a['skipped'])==(b['draws'],b['skipped'])
                                              for a,b in zip(reference,row['records']))
        row['first_update_loss_difference']=abs(row['records'][0]['loss']-reference[0]['loss'])
        row['first_update_grad_norm_difference']=abs(row['records'][0]['grad_norm']-reference[0]['grad_norm'])
        result['trials'].append(row)
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2))
        temporary.replace(args.output)
        print('workers',workers,'microbatch',micro,'seconds/update',row['median_s'],flush=True)
        return row
    run(0,1)
    loader_trials=[run(w,1) for w in args.workers]
    best=min(loader_trials,key=lambda row:row['median_s'])['workers']
    for micro in args.microbatches:
        run(best,micro)
    run(best,1)

if __name__=='__main__':main()
