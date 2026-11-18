#!/usr/bin/env bash
set -euo pipefail
root=/home/ubuntu/work/instance-routes-20261118
cd "$root"
mkdir -p source-logs
exec 9>source-logs/manager.lock
flock -n 9 || exit 2
echo "$$" > source-logs/manager.pid
python=/home/ubuntu/work/cd009-pretraining-v3/venv/bin/python
phase() {
 local stage=$1
 if [[ -s "source-logs/$stage.completed" ]]; then return 0; fi
 date -Is > "source-logs/$stage.started"
 if "$python" publish-source.py "$stage" > "source-logs/$stage.manager.log" 2>&1; then
    return 0
 else
    local rc=$?
    echo "stage=$stage rc=$rc at=$(date -Is)" > "source-logs/$stage.failed"
    return "$rc"
 fi
}
phase train & train_pid=$!
phase selection & selection_pid=$!
wait "$selection_pid"
phase test & test_pid=$!
wait "$train_pid"
wait "$test_pid"
