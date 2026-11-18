"""Count published label filenames by frame window without opening any data file."""
from pathlib import Path
import json,re,time,sys
cache=Path('/home/databeyond/data/datasets/pose/instance-routes-20261118/train')
entries=[]
for folder in sorted(cache.glob('parts/shards/realsense/train/*/*/shards/*/shards/realsense/train/*/*/labels'))+sorted(cache.glob('assist-v1/parts/shards/realsense/train/*/*/labels')):
    s=int(folder.parts[-3].split('_')[1]); lo,hi=map(int,folder.parts[-2].replace('frames_','').split('-'))
    ids=set()
    n=0
    for p in folder.iterdir():
        if p.name.endswith('.npz'):
            ids.add(int(p.name.split('_f')[1][:4])); n+=1
    entries.append([s,lo,hi,len(ids),n])
sample={'time':time.time(),'windows':entries}
Path(sys.argv[1]).write_text(json.dumps(sample))
for a,b in [(0,49),(50,71),(72,99)]:
    rows=[e for e in entries if a<=e[0]<=b];print(a,b,len(rows),'frame IDs reached',sum(e[3] for e in rows),'instances',sum(e[4] for e in rows),'window min/median/max', [sorted(e[3] for e in rows)[i] for i in [0,len(rows)//2,-1]] if rows else None)
