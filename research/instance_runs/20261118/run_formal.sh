#!/usr/bin/env bash
# H20 route B. Safe to invoke again after a failed stage.
set -Eeuo pipefail
base=/home/ubuntu/work/instance-routes-20261118
route=$base/B
cd "$route/repo"
exec 9>"$route/run/formal.lock"
flock -n 9 || exit 2
py=$route/env/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONPATH="$route/repo"
config=configs/instance_b.yaml
target="$base/attempts_rgbd.jsonl"
run="$route/run"
status="$run/status.txt"
mark() { date -u "+%FT%TZ pid=$$ $*" > "$status.tmp"; mv "$status.tmp" "$status"; }
trap 'rc=$?; mark "FAILED rc=$rc line=$LINENO stage=$stage"; exit $rc' ERR
stage=waiting_for_formal_caches
mark "$stage"
[[ -s "$run/partial/target_evaluation.json" ]]
await_manifest() {
    local split=$1
    until [[ -s "$base/source-logs/$split.json" && -e "$base/source-logs/$split.completed" ]]; do
        sleep 60
    done
    "$py" -c 'import json,sys;from pathlib import Path;s=json.load(open(sys.argv[1]));p=Path(s["manifest"]);assert s["labels_computed"] and s["camera"]=="realsense" and p.is_file();print(p)' "$base/source-logs/$split.json"
}
claim_idle_device() {
    local gpu_uuid busy
    gpu_uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
    while true; do
        busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
        if ! grep -Fxq "$gpu_uuid" <<< "$busy"; then break; fi
        mark "waiting_for_GPU1 stage=$stage"
        sleep 30
    done
}
budget_done() {
    [[ -f "$1/summary.json" ]] && "$py" -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(not (d["completed_budget"] and d["completed_steps"]==int(sys.argv[2])))' "$1/summary.json" "$2"
}
command_log() {
    mark "$stage"
    printf '%q ' "$@" >> "$route/logs/formal-commands.txt"
    printf '\n' >> "$route/logs/formal-commands.txt"
    "$@"
}
train=$(await_manifest train)
select=$(await_manifest selection)
stage=source_pretrain
if ! budget_done "$run/source" 8750; then
    claim_idle_device
    resume=()
    if [[ -s "$run/source/last.pt" ]]; then resume=(--resume "$run/source/last.pt"); fi
    command_log "$py" -u scripts/train_instance.py --stage pretrain --source-manifest "$train" --selection-manifest "$select" --config "$config" --run-dir "$run/source" --device cuda "${resume[@]}"
fi
stage=wait_for_full_test
mark "$stage"
full=$(await_manifest test)
stage=source_test
if [[ ! -s "$run/source-test.json" ]]; then
    claim_idle_device
    command_log "$py" -u scripts/evaluate_instance.py --domain source --manifest "$full" --split test --checkpoint "$run/source/best.pt" --output "$run/source-test.json" --device cuda
fi
stage=target_finetune
if ! budget_done "$run/target" 1000; then
    claim_idle_device
    init=(--init-checkpoint "$run/source/best.pt")
    if [[ -s "$run/target/last.pt" ]]; then init=(--resume "$run/target/last.pt"); fi
    command_log "$py" -u scripts/train_instance.py --stage finetune --source-manifest "$train" --selection-manifest "$target" --target-manifest "$target" --config "$config" --run-dir "$run/target" --device cuda "${init[@]}"
fi
stage=target_test
if [[ ! -s "$run/target-test.json" ]]; then
    claim_idle_device
    command_log "$py" -u scripts/evaluate_instance.py --domain target --manifest "$target" --split test --checkpoint "$run/target/best.pt" --output "$run/target-test.json" --device cuda
fi
stage=timing
if [[ ! -s "$run/timing/summary.json" ]]; then
    claim_idle_device
    command_log "$py" -u "$base/time_instances.py" --manifest "$target" --checkpoint "$run/target/best.pt" --output "$run/timing"
fi
mark complete
