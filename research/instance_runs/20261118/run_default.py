"""Run the authorized route A baseline on H20. Restart from published states."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

base = Path(__file__).resolve().parent
root = base.parent
repo = base / 'repo'
os.chdir(repo)
os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
py = base / 'env/bin/python'
backbone = base / 'upstream/graspness_modern'
target = root / 'attempts_rgbd.jsonl'
logs = base / 'logs'
run_source = base / 'runs/source100'
run_target = base / 'runs/finetune'
lock = (logs / 'default.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = logs / 'default-state.json'
stage = 'initialization'


def status(phase, **details):
    global stage
    stage = phase
    state = dict(stage=phase, pid=os.getpid(), device='H20 GPU0',
                 time=datetime.now(timezone.utc).isoformat(), **details)
    temp = state_path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(state_path)


def manifest(name):
    marker = root / 'source-logs' / (name+'.completed')
    status('waiting_for_'+name, ready_marker=str(marker))
    while not marker.is_file():
        time.sleep(30)
    filename = Path(json.loads(marker.with_suffix('.json').read_text())['manifest'])
    if not filename.is_file():
        raise FileNotFoundError(filename)
    return filename


def execute(name, args):
    argv = [str(v) for v in [py, '-u', *args]]
    status(name, argv=argv)
    with (logs / (name+'.log')).open('a') as log:
        log.write(datetime.now(timezone.utc).isoformat()+' '+shlex.join(argv)+'\n')
        log.flush()
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)


def complete(run, steps):
    summary = run / 'summary.json'
    if not summary.is_file():
        return False
    result = json.loads(summary.read_text())
    return (result['completed_budget'] and result['completed_steps']==steps
            and result['planned_steps']==steps and (run/'best.pt').is_file() and (run/'last.pt').is_file())


def train(name, run, steps, extra):
    if complete(run, steps):
        return
    args = ['scripts/train_instance.py','--config','configs/instance_a.yaml','--stage',name,
            '--source-manifest',source,'--run-dir',run,'--device','cuda','--backbone-root',backbone,*extra]
    if (run/'last.pt').is_file():
        args += ['--resume',run/'last.pt']
    elif name=='finetune':
        args += ['--init-checkpoint',run_source/'best.pt']
    execute(name,args)
    if not complete(run,steps):
        raise RuntimeError(f'{name} did not finish {steps} updates')


def evaluate(domain, run, data):
    output=run / (domain+'_full_test.json')
    if not output.is_file():
        execute('evaluate_'+domain,['scripts/evaluate_instance.py','--checkpoint',run/'best.pt',
                '--manifest',data,'--domain',domain,'--split','test','--device','cuda','--output',output])


try:
    if not complete(base/'runs/partial_source',2) or not complete(base/'runs/partial_target',2):
        raise RuntimeError('A real sample chain has not finished')
    if not (base/'runs/partial_source/source_evaluation.json').is_file():
        raise RuntimeError('A partial native evaluation missing')
    counts = Counter(row['split'] for row in map(json.loads,target.open()))
    if counts != dict(train=1676,val=399,test=87):
        raise ValueError(f'target actions differ: {counts}')
    source=manifest('train')
    selection=manifest('selection')
    train('pretrain',run_source,8750,['--selection-manifest',selection])
    evaluate('source',run_source,manifest('test'))
    train('finetune',run_target,1000,['--selection-manifest',target,'--target-manifest',target])
    evaluate('target',run_target,target)
    execute('time_instances',[root/'time_instances.py','--checkpoint',run_target/'best.pt',
            '--manifest',target,'--output',run_target/'timing','--backbone-root',backbone])
    status('complete')
except Exception as e:
    status('failed', failed_stage=stage, reason=repr(e))
    raise
