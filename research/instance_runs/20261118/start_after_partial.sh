#!/usr/bin/env bash
set -euo pipefail
base=/home/ubuntu/work/instance-routes-20261118/A
cd "$base"
partial_pid=$(cat logs/partial.pid)
while kill -0 "$partial_pid" 2>/dev/null && test "$(ps -o stat= -p "$partial_pid" | tr -d ' ' | cut -c1)" != Z; do
  sleep 10
done
if ! test -s runs/partial_target/best.pt; then echo 'Partial target failed; inspect logs/partial.log' >&2; exit 1; fi
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1
env/bin/python raw_cli_probe.py > logs/raw-cli.log 2>&1
exec env/bin/python run_default.py
