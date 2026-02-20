#!/usr/bin/env bash
# Train with DeBERTa-v3-large (competitive config)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# HF cache: avoid macOS sandbox permission errors on ~/.cache
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m clarity.train \
    --config configs/deberta_v3_large.yaml \
    --data data/train.csv \
    --dev data/dev.csv \
    --task evasion
