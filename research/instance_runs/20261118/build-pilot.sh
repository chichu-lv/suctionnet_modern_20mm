#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/home/ubuntu/work/instance-routes-20261118
release=/home/ubuntu/data/datasets/pose/suctionnet-r20-v1/full-20261110
python=/home/ubuntu/work/cd009-pretraining-v3/venv/bin/python
cd "$root/A/repo"
export PYTHONPATH="$PWD"
mkdir -p "$root/source-pilot/logs"
"$python" scripts/build_instance_dataset.py --help > "$root/source-pilot/logs/builder-help.txt"
"$python" scripts/merge_instance_manifests.py --help > "$root/source-pilot/logs/merge-help.txt"
build() {
 local split=$1 scenes=$2 frames=$3
 "$python" -u scripts/build_instance_dataset.py --dataset-root "$release/input_raw" --label-root "$release/output" --output "$root/source-pilot/$split" --camera realsense --split "$split" --scenes "$scenes" --frames "$frames" --workers 2 > "$root/source-pilot/logs/$split.json" 2> "$root/source-pilot/logs/$split.err"
}
build train 0,44 0,64,128,192 & train_job=$!
build test 100,160 0 & test_job=$!
train_rc=0; test_rc=0
wait "$train_job" || train_rc=$?
wait "$test_job" || test_rc=$?
echo "train_rc=$train_rc test_rc=$test_rc"
(( train_rc == 0 && test_rc == 0 ))
