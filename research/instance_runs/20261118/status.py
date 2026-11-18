"""Display stage markers and most recent successful update. Run on the execution host."""
import json
import os
from pathlib import Path
import subprocess

root = Path('/home/ubuntu/work/instance-routes-20261118')
def tail(path):
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        lines = stream.read().decode(errors='replace').splitlines()
    return lines[-1][:1500] if lines else None

def read_last_update(path):
    line = tail(path)
    try:
        row = json.loads(line)
        return {k:v for k,v in row.items() if isinstance(v,(float,int,bool,str))}
    except (ValueError, TypeError):
        return line

for stage in ('train','selection','test'):
    log = root / 'source-logs'
    for suffix in ('completed','failed','started'):
        file = log / f'{stage}.{suffix}'
        if file.is_file():
            print('source cache', stage, suffix, tail(file)); break
    else:
        print('source cache', stage, 'awaiting publication')
for route in ('A','B'):
    for sub in ('runs','run'):
        for path in sorted((root/route/sub).glob('**/training.jsonl')):
            print(route, path.relative_to(root), read_last_update(path))
    for file in sorted((root/route/'logs').glob('*status*')):
        if file.is_file(): print(route, file.name, tail(file))
subprocess.run(['nvidia-smi','-i','0,1','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv'])
