"""Measure warmed native per-instance CUDA inference from original validation RGB-D."""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from suction20.instance_data import InstanceAttemptDataset
from suction20.instance_inference import load_instance_checkpoint, load_raw_observation, infer_instance


def fixed_cases(dataset):
    records = sorted(dataset.records, key=lambda r: r['sample_id'])
    if len(records) < 200:
        raise ValueError('Timing protocol needs 200 separate validation instances')
    indices = np.linspace(0, len(records)-1, 200, dtype=int)
    return [records[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint', 'manifest', 'output'):
        parser.add_argument('--'+k, required=True, type=Path)
    parser.add_argument('--backbone-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = fixed_cases(InstanceAttemptDataset(args.manifest, 'val', cache_size=0))
    k_dir = args.output/'intrinsics'
    k_dir.mkdir(exist_ok=True)
    for i, item in enumerate(samples):
        (k_dir/f'{i}.json').write_text(json.dumps(item['observation']['intrinsics']))
    model, metadata = load_instance_checkpoint(args.checkpoint, 'cuda', backbone_root=args.backbone_root)
    model.eval()
    timing = {}
    info = {}

    def capture(name):
        original = getattr(model, name)
        def timed(*arguments, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*arguments, **kwargs)
            end.record()
            timing[name] = start, end
            if name == 'encode_instance' and isinstance(result, dict):
                for key in ('texture_roi_hw','texture_padded_hw'):
                    if key in result:
                        info[key] = list(map(int, result[key]))
            return result
        setattr(model, name, timed)

    capture('encode_instance')
    capture('score_at_pose')
    elapsed = []
    measured = args.output/'samples.jsonl'
    with measured.with_suffix('.tmp').open('w') as stream:
        for step, i in enumerate(list(range(50)) + list(range(200))):
            item = samples[i]
            src = item['observation']
            timing.clear()
            info.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            obs = load_raw_observation(rgb_path=src['rgb_path'], depth_path=src['depth_path'],
                mask_path=src['instance_mask_path'], intrinsics_path=k_dir/f'{i}.json',
                depth_scale_m=src['depth_scale_m'], gravity=src['g_cam'], instance_id=item['instance_id'],
                depth_min_m=src.get('depth_min_m', .2), depth_max_m=src.get('depth_max_m', 1.2))
            prepared = time.perf_counter()
            obs.update(sample_uid=item['sample_id'], instance_uid=item['sample_id'])
            output = infer_instance(model, obs, stage=metadata['stage'])
            torch.cuda.synchronize()
            end = time.perf_counter()
            if step < 50:
                continue
            durations = {key:a.elapsed_time(b) for key,(a,b) in timing.items()}
            row = dict(sample_id=item['sample_id'], slot=i, input_size_hw=obs['xyz_map'].shape[:2] if 'xyz_map' in obs else src.get('image_size_hw'),
                       support_points=len(obs['support_points']), status=output['status'], counts=output['counts'],
                       e2e_ms=1000*(end-start), raw_preprocessing_ms=1000*(prepared-start),
                       model_ms=sum(durations.values()), model_stages_ms=durations,
                       native_timing_ms=output['timing_ms'], peak_allocated_bytes=torch.cuda.max_memory_allocated(), **info)
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            elapsed.append(row)
    measured.with_suffix('.tmp').replace(measured)
    result = dict(warmup_count=50, measured_instances=200, selection='uniform index in sample_id-sorted target val distinct captured instances',
        device=torch.cuda.get_device_name(0), gpu_id=os.environ.get('CUDA_VISIBLE_DEVICES'), python=platform.python_version(),
        torch=torch.__version__, cuda=torch.version.cuda, driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0],
        cpu_threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)), dtype=str(next(model.parameters()).dtype),
        model_config=metadata['runtime_model_config'], checkpoint=str(args.checkpoint), opencv_camera_and_segmentation_included=False,
        e2e_scope='read original RGB/depth/mask/K, normal/candidates, H2D, native encode+score, Top5/pose and synchronization; disk export excluded',
        model_scope='CUDA event intervals of native encode_instance plus score_at_pose, including host launch gaps, copies, internal geometry',
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in elapsed), no_pose=sum(r['status']=='no_pose' for r in elapsed))
    for key in ('model_ms','e2e_ms','raw_preprocessing_ms'):
        result[key] = dict(zip(('p50','p95'),np.percentile([r[key] for r in elapsed], [50,95]).tolist()))
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__':
    main()
