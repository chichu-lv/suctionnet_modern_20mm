#!/usr/bin/env bash
set -euo pipefail
set -x
base=/home/ubuntu/work/instance-routes-20261118
route=$base/B
cd "$route/repo"
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONNOUSERSITE=1
py=$route/env/bin/python
train_manifest=$1
selection_manifest=$2
partial=$route/run/partial
mkdir -p "$partial"
"$py" - "$partial" "$base/attempts_rgbd.jsonl" <<'PY'
import json,sys,yaml
from collections import Counter
from pathlib import Path
partial=Path(sys.argv[1])
config=yaml.safe_load(Path('configs/instance_b.yaml').read_text())
config['pretrain'].update(steps=2,accumulate_instances=2,warmup_steps=1,evaluate_every=2,checkpoint_every=1)
config['finetune'].update(steps=2,head_only_steps=1,evaluate_every=2,checkpoint_every=1)
(partial/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
counts=Counter()
with open(sys.argv[2]) as source,(partial/'target.jsonl').open('w') as output:
    for line in source:
        row=json.loads(line)
        key=row['split'],row['attempt']['success']
        if counts[key]<3:
            output.write(line)
            counts[key]+=1
assert len(counts)==6
PY
common=(--device cuda --config "$partial/config.yaml" --allow-partial-manifests --source-manifest "$train_manifest")
source_train=(scripts/train_instance.py "${common[@]}" --stage pretrain --selection-manifest "$selection_manifest" --run-dir "$partial/source")
"$py" "${source_train[@]}" --stop-after 1
"$py" "${source_train[@]}" --resume "$partial/source/last.pt"
"$py" scripts/infer_instance.py --checkpoint "$partial/source/best.pt" --manifest "$selection_manifest" --max-instances 1 --output "$partial/source_prediction.json"
"$py" scripts/evaluate_instance.py --domain source --checkpoint "$partial/source/best.pt" --manifest "$selection_manifest" --output "$partial/source_evaluation.json" --allow-partial-manifests
"$py" scripts/train_instance.py "${common[@]}" --stage finetune --target-manifest "$partial/target.jsonl" --selection-manifest "$partial/target.jsonl" --run-dir "$partial/target" --init-checkpoint "$partial/source/best.pt"
"$py" scripts/evaluate_instance.py --domain target --checkpoint "$partial/target/best.pt" --manifest "$partial/target.jsonl" --output "$partial/target_evaluation.json"
