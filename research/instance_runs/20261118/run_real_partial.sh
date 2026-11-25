#!/usr/bin/env bash
set -euo pipefail
root=/home/ubuntu/work/instance-routes-20261118
base=$root/A
cd "$base/repo"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=$base/env/bin/python
cfg=$base/config/partial.yaml
run=$base/runs/partial_source
ft=$base/runs/partial_target
subset=$base/config/partial_attempts.jsonl
mkdir -p "$base/config"
"$py" - <<'PY'
import yaml,json
from pathlib import Path
from collections import Counter
root=Path.cwd().parents[1]
cfg=yaml.safe_load(Path('configs/instance_a.yaml').read_text())
cfg['pretrain'].update(steps=2,warmup_steps=1,evaluate_every=2,checkpoint_every=1)
cfg['finetune'].update(steps=2,head_only_steps=1,evaluate_every=2,checkpoint_every=1)
(root/'A/config/partial.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
counts=Counter()
with (root/'A/config/partial_attempts.jsonl').open('w') as out:
 for line in (root/'attempts_rgbd.jsonl').open():
  row=json.loads(line); key=(row['split'],row['attempt']['success'])
  if row['split'] in ('train','val') and counts[key]<4:
   out.write(line);counts[key]+=1
print('partial_target_counts',dict(counts),flush=True)
PY
train=$1
test=$2
options=(--device cuda --source-manifest "$train" --backbone-root "$base/upstream/graspness_modern" --config "$cfg" --allow-partial-manifests)
src=(scripts/train_instance.py "${options[@]}" --stage pretrain --selection-manifest "$test" --run-dir "$run")
"$py" -u "${src[@]}" --stop-after 1
"$py" -u "${src[@]}" --resume "$run/last.pt"
"$py" -u scripts/infer_instance.py --device cuda --checkpoint "$run/best.pt" --manifest "$test" --split test --max-instances 1 --output "$run/instance.json"
"$py" -u scripts/evaluate_instance.py --device cuda --checkpoint "$run/best.pt" --manifest "$test" --domain source --split test --allow-partial-manifests --output "$run/source_evaluation.json"
"$py" -u scripts/train_instance.py "${options[@]}" --stage finetune --init-checkpoint "$run/best.pt" --target-manifest "$subset" --selection-manifest "$subset" --run-dir "$ft"
