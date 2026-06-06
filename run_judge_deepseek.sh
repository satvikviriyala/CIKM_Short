#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-run_20260521}"
JUDGE="deepseek"
PROJECT="gcp-learn-483706"
echo "=== Judge: $JUDGE | run_id=$RUN_ID ==="
for f in runs/$RUN_ID/responses/*.jsonl; do
  [[ "$f" == *_variant_* ]] && continue
  echo ">>> $(basename $f)  $(date -u +%H:%M:%SZ)"
  python3 -u eval_gcp_models.py --judge-key "$JUDGE" --run-id "$RUN_ID" \
    --input "$f" --project-id "$PROJECT" --concurrency 1
done
echo "=== $JUDGE COMPLETE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
