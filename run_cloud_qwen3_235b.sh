#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-run_20260521}"
MODEL="qwen3_235b"
PROJECT="gcp-learn-483706"
echo "=== $MODEL | run_id=$RUN_ID ==="
for REGIME in vanilla utility_first neutrality_oriented clarification_first; do
  for SAMPLE in 0 1 2; do
    echo ">>> regime=$REGIME  sample=$SAMPLE  $(date -u +%H:%M:%SZ)"
    python3 -u scripts/generate_responses.py \
      --model "$MODEL" --regime "$REGIME" --sample-idx "$SAMPLE" \
      --run-id "$RUN_ID" --project-id "$PROJECT" --concurrency 1
  done
done
echo "=== $MODEL COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
