#!/usr/bin/env bash
# Cloud models — Vertex AI MaaS — all regimes × all samples
# Usage: bash run_cloud.sh [run_id]
set -euo pipefail

RUN_ID="${1:-run_20260521}"
SCRIPT="python3 scripts/generate_responses.py"
PROJECT="gcp-learn-483706"

echo "=== Cloud models | run_id=$RUN_ID | project=$PROJECT ==="
for MODEL in qwen3_235b gpt_oss_120b grok_4_20; do
  echo ""
  echo "========== Model: $MODEL =========="
  for REGIME in vanilla utility_first neutrality_oriented clarification_first; do
    for SAMPLE in 0 1 2; do
      echo ""
      echo ">>> model=$MODEL  regime=$REGIME  sample=$SAMPLE  $(date -u +%H:%M:%SZ)"
      $SCRIPT \
        --model "$MODEL" \
        --regime "$REGIME" \
        --sample-idx "$SAMPLE" \
        --run-id "$RUN_ID" \
        --project-id "$PROJECT" \
        --concurrency 1
    done
  done
done

echo ""
echo "=== Cloud models COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
