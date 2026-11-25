import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
row=next(r for r in map(json.loads,(root.parent/'attempts_rgbd.jsonl').open()) if r['split']=='train')
obs=row['observation']
k=root/'config/raw-train-k.json'
k.write_text(json.dumps(obs['intrinsics']))
run=root/'runs/partial_source'
args=[sys.executable,str(root/'repo/scripts/infer_instance.py'),'--checkpoint',str(run/'best.pt'),'--device','cuda','--rgb',obs['rgb_path'],'--depth',obs['depth_path'],'--mask',obs['instance_mask_path'],'--intrinsics',str(k),'--depth-scale-m',str(obs['depth_scale_m']),'--gravity',*map(str,obs['g_cam']),'--instance-id',str(row['attempt']['instance_id']),'--output',str(run/'raw-train-prediction.json')]
print(json.dumps(args),flush=True)
subprocess.run(args,check=True)
