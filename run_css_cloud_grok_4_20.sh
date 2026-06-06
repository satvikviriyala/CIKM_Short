#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-run_20260521}"; MODEL="grok_4_20"; PROJECT="gcp-learn-483706"
echo "=== CSS variants: $MODEL | run_id=$RUN_ID ==="
for SAMPLE in 0 1 2; do
  echo ">>> sample=$SAMPLE  $(date -u +%H:%M:%SZ)"
  python3 -u scripts/generate_criterion_variants.py \
    --backend cloud --model "$MODEL" --all-variants \
    --sample-idx "$SAMPLE" --run-id "$RUN_ID" \
    --project-id "$PROJECT" --concurrency 1
done
echo "=== $MODEL CSS COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
