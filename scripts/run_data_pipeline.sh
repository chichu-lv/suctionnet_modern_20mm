#!/usr/bin/env bash
set -euo pipefail
: "${DATASET:?Set DATASET to SuctionNet root}"
: "${UPSTREAM:?Set UPSTREAM to local suction_annotation checkout}"
: "${OUT:?Set OUT to output directory}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m suction20.full_cli run-all \
  --dataset "$DATASET" --upstream "$UPSTREAM" --output "$OUT" \
  --config configs/suction20.json --mode poisson \
  --objects all --scenes all --frames all
