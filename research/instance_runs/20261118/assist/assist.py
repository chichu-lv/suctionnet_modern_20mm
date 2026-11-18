"""Build disjoint real train scenes with the original builder, and merge real receipts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import traceback

role = sys.argv[1]
base = Path('/home/ubuntu/work/instance-routes-20261118')
cache = Path('/home/ubuntu/data/datasets/pose/instance-routes-20261118/train/assist-v1')
ctrl = cache/'control'
ctrl.mkdir(parents=True, exist_ok=True)
repo = base/('A/repo' if role=='H20' else 'repo')
fixed = '364132a5f892745c9fff05e6ed4a33b173a11ea5'
assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==fixed
os.chdir(repo)
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='1', PYTHONPATH=str(repo))
raw = Path('/home/ubuntu/data/datasets/pose/suctionnet-r20-v1/full-20261110')
logs = base/'assist-logs'
logs.mkdir(exist_ok=True)
lock = (logs/'worker.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
assert role in ('L20','H20')
first,last,width = (72,100,86) if role=='L20' else (50,72,128)

def save(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)

def run(name,command):
    if (ctrl/'fallback').exists(): raise RuntimeError('fallback active')
    receipt=logs/(name+'.json')
    if receipt.is_file():
        result=json.loads(receipt.read_text())
        if Path(result['manifest']).is_file(): return result
    command=[sys.executable,'-u','scripts/'+command[0],*map(str,command[1:])]
    save(logs/(name+'.command.json'),command)
    with (logs/(name+'.stderr')).open('a') as err:
        info=json.loads(subprocess.check_output(command,stderr=err,text=True))
    save(receipt,info)
    return info

def build(task):
    s,lo=task
    hi=min(lo+width,256)-1
    info=run(f's{s:04d}-f{lo:03d}', ['build_instance_dataset.py','--dataset-root',raw/'input_raw','--label-root',raw/'output','--output',cache/'parts','--camera','realsense','--split','train','--scenes',s,'--frames',f'{lo}-{hi}'])
    assert info['labels_computed'] and info['scene_ids']==[s] and info['frame_count']==hi-lo+1
    return info['manifest']

try:
    scenes=range(first,last)
    offsets=range(0,256,width)
    with ThreadPoolExecutor(max_workers=len(scenes)*len(offsets)) as pool:
        work={(s,lo):pool.submit(build,(s,lo)) for s in scenes for lo in offsets}
        for even in range(first,last,2):
            name=f'{even:04d}-{even+1:04d}'
            manifests=[work[(s,lo)].result() for s in (even,even+1) for lo in offsets]
            info=run('merge-'+name,['merge_instance_manifests.py','--output',cache/'pairs'/name,'--split','train','--allow-partial','--manifests',*manifests])
            assert info['labels_computed'] and set(map(tuple,info['frame_pairs']))=={(s,f) for s in (even,even+1) for f in range(256)}
            save(ctrl/(name+'.json'),info)
except BaseException:
    (ctrl/(role+'.failed')).write_text(traceback.format_exc())
    raise
