#!/usr/bin/env bash
# Llama 3.3 70B — 172.16.144.45:1234 — all regimes × all samples
# Usage: bash run_llama.sh [run_id]
set -euo pipefail

RUN_ID="${1:-run_20260521}"
SCRIPT="python3 -u scripts/generate_responses_local.py"
HOST="http://172.16.144.45:1234/v1"
MODEL="llama3_70b"

echo "=== Llama 3.3 70B | run_id=$RUN_ID | host=$HOST ==="
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
echo "=== Llama 3.3 70B COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
