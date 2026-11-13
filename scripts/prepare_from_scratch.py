"""Write independent CD-010 fresh-start configs, without launching a training job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

from suction20.from_scratch import from_scratch_config
from suction20.training import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', required=True,
                        help='Existing E0–E5 profile supplies experiment and hardware limits')
    parser.add_argument('--out-dir', required=True, help='New directory for train.json and performance.json')
    parser.add_argument('--run-dir', required=True, help='New, independent training run directory')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp16',
                        help='BF16 uses an FP32 sparse backbone on spconv 2.3.6; select explicitly')
    parser.add_argument('--dense-chunk-size', type=int, default=65536)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--geometry-threads', type=int, default=4)
    parser.add_argument('--geometry-precision', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--scene-cache-size', type=int, default=90)
    parser.add_argument('--object-cache-size', type=int, default=88)
    parser.add_argument('--instance-broadcast-mode', choices=('segment', 'indexed'), default='segment')
    parser.add_argument('--compile-dense-heads', action='store_true', help='Optional local MLP compile; off by default')
    parser.add_argument('--no-voxel-heads', action='store_true')
    parser.add_argument('--no-fused-optimizer', action='store_true')
    parser.add_argument('--source-root')
    parser.add_argument('--label-root')
    parser.add_argument('--backbone-root')
    args = parser.parse_args()
    config, performance = from_scratch_config(load_config(args.base_config),
        precision=args.precision, dense_chunk_size=args.dense_chunk_size,
        workers=args.workers, geometry_threads=args.geometry_threads,
        geometry_precision=args.geometry_precision, scene_cache_size=args.scene_cache_size,
        object_cache_size=args.object_cache_size, voxel_heads=not args.no_voxel_heads,
        instance_broadcast_mode=args.instance_broadcast_mode,
        compile_dense_heads=args.compile_dense_heads, fused_optimizer=not args.no_fused_optimizer)
    for key in ('source_root', 'label_root'):
        value = getattr(args, key)
        if value:
            config['data'][key] = str(Path(value).resolve())
    if args.backbone_root:
        config['model']['backbone_root'] = str(Path(args.backbone_root).resolve())
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_file, performance_file = output / 'train.json', output / 'performance.json'
    if train_file.exists() or performance_file.exists():
        parser.error('Output configs already exist; choose a new directory to preserve the existing run settings')
    train_file.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    performance_file.write_text(json.dumps(performance, indent=2, ensure_ascii=False) + '\n')
    command = [sys.executable, 'scripts/train_pretrain.py', '--config', str(train_file),
               '--performance-config', str(performance_file), '--run-dir', str(Path(args.run_dir).resolve())]
    print(json.dumps({'train_config': str(train_file), 'performance_config': str(performance_file),
                      'training_command': shlex.join(command),
                      'supervisor_entries': {
                          'profiles': {config['runtime_profile']['host']: {config['experiment']: {
                              'config_path': str(train_file),
                              'config_sha256': hashlib.sha256(train_file.read_bytes()).hexdigest()}}},
                          'performance_configs': {config['runtime_profile']['host']: {config['experiment']: {
                              'config_path': str(performance_file),
                              'config_sha256': hashlib.sha256(performance_file.read_bytes()).hexdigest()}}}},
                      'precision': config['training']['precision'],
                      'status': 'prepared; no training launched'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
