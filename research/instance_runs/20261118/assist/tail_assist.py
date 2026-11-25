"""Populate reusable cache entries for unvisited tail frames without overwriting H20 outputs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

base = Path('/home/ubuntu/work/instance-routes-20261118')
repo = base/'repo'
raw = Path('/home/ubuntu/data/datasets/pose/suctionnet-r20-v1/full-20261110')
cache = Path('/home/ubuntu/data/datasets/pose/instance-routes-20261118/train')
out = cache/'tail-assist-v1'
logs = base/'tail-logs'
logs.mkdir(exist_ok=True)
lock = (logs/'agent.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.chdir(repo)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='364132a5f892745c9fff05e6ed4a33b173a11ea5'
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='1',PYTHONPATH=str(repo))

def highwater(folder):
    return max((int(p.name.split('_f')[1][:4]) for p in folder.glob('*.npz') if not p.is_symlink()),default=-1)

def fill(item):
    scene=item['scene']
    original, = (cache/'parts').glob(f'shards/realsense/train/*/*/shards/*/shards/realsense/train/scenes_{scene:04d}/frames_0000-0255')
    begin=item['start']
    while begin<256:
        key=f's{scene:04d}-f{begin:04d}'
        stop=min(255,begin+(0 if begin==item['start'] else 15))
        receipt=logs/(key+'.json')
        if not receipt.exists():
            hw=highwater(original/'observations')
            low=max(begin,hw+3)
            if low<=stop:
                cmd=[sys.executable,'-u','scripts/build_instance_dataset.py','--dataset-root',str(raw/'input_raw'),
                     '--label-root',str(raw/'output'),'--output',str(out),'--camera','realsense','--split','train',
                     '--scenes',str(scene),'--frames',f'{low}-{stop}']
                (logs/(key+'.command.json')).write_text(json.dumps(cmd))
                with (logs/(key+'.err')).open('a') as err:
                    info=json.loads(subprocess.check_output(cmd,stderr=err,text=True))
                assert info['labels_computed'] and info['train_queries']==128 and info['scene_ids']==[scene]
                assert info['frame_ids']==list(range(low,stop+1))
                manifest=Path(info['manifest'])
                rows=[json.loads(line) for line in manifest.read_text().splitlines()]
                hw=highwater(original/'observations')
                links=0
                for row in rows:
                    if row['frame_id']<=hw+1: continue
                    for field,kind in [('observation_npz','observations'),('label_cache','labels')]:
                        rel=Path(row[field])
                        assert rel==Path(kind)/(row['sample_id']+'.npz')
                        artifact=manifest.parent/rel
                        assert artifact.is_file()
                        dest=original/rel
                        try:
                            dest.symlink_to(os.path.relpath(artifact,dest.parent))
                        except FileExistsError:
                            continue
                        assert dest.is_file()
                        links+=1
                tmp=receipt.with_suffix('.tmp')
                tmp.write_text(json.dumps(dict(scene=scene,first=low,last=stop,instances=len(rows),
                                               links=links,manifest=str(manifest),epoch=time.time())))
                tmp.replace(receipt)
        begin=stop+1

with ThreadPoolExecutor(max_workers=32) as pool:
    jobs=[pool.submit(fill,item) for item in json.loads((base/'tail-plan.json').read_text())]
    for job in as_completed(jobs): job.result()
