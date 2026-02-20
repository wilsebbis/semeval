#!/usr/bin/env bash
# Run inference and generate submission file
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# HF cache: avoid macOS sandbox permission errors on ~/.cache
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTORCH_ENABLE_MPS_FALLBACK=1

CKPT="${1:-checkpoints/deberta_v3_large/best_model.pt}"
DATA="${2:-data/test.csv}"
OUT="${3:-submissions/sub.csv}"

python -m clarity.predict \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --out "$OUT" \
    --evaluate \
    --metrics_out "submissions/metrics.json"

echo "Submission written to: $OUT"
