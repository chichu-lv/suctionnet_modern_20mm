"""Build retryable scene shards, then publish the CLI-selected manifest."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def publish(path, text):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('selection','train','test'))
    phase = p.parse_args().phase
    root = Path('/home/ubuntu/work/instance-routes-20261118')
    repo = root / 'A/repo'
    source = Path('/home/ubuntu/data/datasets/pose/suctionnet-r20-v1/full-20261110')
    cache = Path('/home/ubuntu/data/datasets/pose/instance-routes-20261118') / phase
    log = root / 'source-logs'
    records = log / (phase + '-shards')
    records.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    split = 'train' if phase == 'train' else 'test'
    scenes = range(100) if phase == 'train' else range(100, 190)
    selection_frames = range(0, 256, 32)
    frames = {
        'train': list(range(256)),
        'selection': list(selection_frames),
        'test': [f for f in range(256) if f not in selection_frames],
    }[phase]
    frame_args = [] if phase == 'train' else ['--frames', ','.join(map(str,frames))]
    workers = {'train':25, 'selection':6, 'test':15}[phase]
    environ = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONPATH=str(repo))
    def build(start):
        receipt = records / f'{start:04d}-{start+1:04d}.json'
        if receipt.is_file():
            path = Path(json.loads(receipt.read_text())['manifest'])
            if not path.is_file() or not path.with_suffix('.summary.json').is_file():
                raise FileNotFoundError(f'Missing published shard: {path}')
            return str(path)
        cmd = [sys.executable, '-u', 'scripts/build_instance_dataset.py',
               '--dataset-root',str(source/'input_raw'), '--label-root',str(source/'output'),
               '--output',str(cache/'parts'), '--split',split, '--camera','realsense',
               '--scenes',f'{start}-{start+1}', '--workers','2', *frame_args]
        receipt.with_suffix('.command.json').write_text(json.dumps(cmd))
        with receipt.with_suffix('.err').open('a') as err:
            result = subprocess.run(cmd, cwd=repo, env=environ, stdout=subprocess.PIPE, stderr=err, check=True)
        text = result.stdout.decode()
        info = json.loads(text)
        assert info['scene_ids'] == [start,start+1]
        assert info['frame_count'] == 2*len(frames)
        publish(receipt, text)
        return info['manifest']
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(build, scenes[::2]))
    if phase == 'test':
        selection_receipt = json.loads((log / 'selection.json').read_text())
        paths.append(selection_receipt['manifest'])
    target = log / f'{phase}.json'
    cmd = [sys.executable, '-u', 'scripts/merge_instance_manifests.py', '--output', str(cache/'published'),
           '--split',split, '--manifests', *paths]
    if phase == 'selection':
        cmd.append('--allow-partial')
    with (log / f'{phase}-merge.err').open('a') as err:
        res = subprocess.run(cmd, cwd=repo, env=environ, stdout=subprocess.PIPE, stderr=err, check=True)
    text = res.stdout.decode()
    info = json.loads(text)
    expected = {(s,f) for s in scenes for f in (range(0,256,32) if phase == 'selection' else range(256))}
    if set(map(tuple,info['frame_pairs'])) != expected:
        raise ValueError('Final coverage does not equal requested scene/frame definition')
    publish(target, text)
    publish(log / f'{phase}.completed', json.dumps(dict(manifest=info['manifest'], frames=len(expected),
            elapsed_s=time.time()-started, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))


if __name__ == '__main__':
    main()
