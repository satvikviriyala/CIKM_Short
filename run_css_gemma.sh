#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-run_20260521}"; MODEL="gemma4_e4b"
HOST="http://127.0.0.1:1234/v1"
echo "=== CSS variants: $MODEL | run_id=$RUN_ID | host=$HOST ==="
for SAMPLE in 0 1 2; do
  echo ">>> sample=$SAMPLE  $(date -u +%H:%M:%SZ)"
  python3 -u scripts/generate_criterion_variants.py \
    --backend local --model "$MODEL" --all-variants \
    --sample-idx "$SAMPLE" --run-id "$RUN_ID" \
    --host "$HOST" --concurrency 1
done
echo "=== $MODEL CSS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
