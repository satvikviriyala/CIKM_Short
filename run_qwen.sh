#!/usr/bin/env bash
# Qwen 3.5 9B — localhost:1234 — all regimes × all samples
# Run this after deploying Qwen 3.5 9B in LM Studio.
# Usage: bash run_qwen.sh [run_id]
set -euo pipefail

RUN_ID="${1:-run_20260521}"
SCRIPT="python3 -u scripts/generate_responses_local.py"
HOST="http://127.0.0.1:1234/v1"
MODEL="qwen3_local"

echo "=== Qwen 3.5 9B | run_id=$RUN_ID | host=$HOST ==="
for REGIME in vanilla utility_first neutrality_oriented clarification_first; do
  for SAMPLE in 0 1 2; do
    echo ""
    echo ">>> regime=$REGIME  sample=$SAMPLE  $(date -u +%H:%M:%SZ)"
    $SCRIPT \
      --model "$MODEL" \
      --regime "$REGIME" \
      --sample-idx "$SAMPLE" \
      --run-id "$RUN_ID" \
      --host "$HOST" \
      --concurrency 1
  done
done

echo ""
echo "=== Qwen 3.5 9B COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
