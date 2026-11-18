"""Keep existing H20 workers running; resume queued dispatch once assisted pairs are published."""
import json, os, signal, sys, time, traceback
from pathlib import Path
root=Path('/home/ubuntu/work/instance-routes-20261118')
control=Path('/home/ubuntu/data/datasets/pose/instance-routes-20261118/train/assist-v1/control')
control.mkdir(parents=True,exist_ok=True)
records=root/'source-logs/train-shards'
pid=int(sys.argv[1])
proc=Path('/proc')/str(pid)
expected=['/home/ubuntu/work/cd009-pretraining-v3/venv/bin/python','publish-source.py','train']
assert (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')==expected
assert (proc/'cwd').resolve()==root
pending=[f'{s:04d}-{s+1:04d}' for s in range(50,100,2)]
assert not any((records/(key+'.command.json')).exists() for key in pending)
def release(*args):
    raise RuntimeError('Gate interrupted')
signal.signal(signal.SIGTERM,release)
signal.signal(signal.SIGHUP,signal.SIG_IGN)
start=time.monotonic()
try:
    os.kill(pid,signal.SIGSTOP)
    print('Paused train dispatch only, existing builders continue',pid,flush=True)
    while pending:
        if list(control.glob('*.failed')) or time.monotonic()-start>4*3600:
            raise RuntimeError('Assist error or timeout')
        for key in list(pending):
            src=control/(key+'.json')
            if not src.is_file(): continue
            result=json.loads(src.read_text())
            manifest=Path(result['manifest'])
            assert manifest.is_file() and manifest.with_suffix('.summary.json').is_file()
            assert result['labels_computed'] and result['frame_count']==512
            tmp=records/(key+'.handoff.tmp')
            tmp.write_text(json.dumps(result))
            tmp.replace(records/(key+'.json'))
            pending.remove(key)
        if pending: time.sleep(10)
    print('All 25 delegated pairs registered',flush=True)
except BaseException:
    (control/'fallback').write_text(traceback.format_exc())
    raise
finally:
    os.kill(pid,signal.SIGCONT)
    print('Resumed original train publisher',flush=True)
