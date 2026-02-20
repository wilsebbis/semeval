#!/usr/bin/env bash
# Ensemble: train 3 seeds, average logits at prediction time
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# HF cache: avoid macOS sandbox permission errors on ~/.cache
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTORCH_ENABLE_MPS_FALLBACK=1

CONFIG="${1:-configs/deberta_v3_large.yaml}"
SEEDS=(42 123 2026)

# Train with different seeds
for SEED in "${SEEDS[@]}"; do
    echo "=== Training seed $SEED ==="
    python -m clarity.train \
        --config "$CONFIG" \
        --data data/train.csv \
        --dev data/dev.csv \
        --task evasion \
        --seed "$SEED" \
        --output_dir "checkpoints/ensemble_seed${SEED}"
done

# Ensemble prediction
echo "=== Ensemble prediction ==="
CKPTS=""
for SEED in "${SEEDS[@]}"; do
    CKPTS="$CKPTS checkpoints/ensemble_seed${SEED}/best_model.pt"
done

python -m clarity.predict \
    --ckpt checkpoints/ensemble_seed42/best_model.pt \
    --ensemble_ckpts $CKPTS \
    --data data/test.csv \
    --out submissions/ensemble_sub.csv \
    --evaluate \
    --metrics_out submissions/ensemble_metrics.json

echo "Ensemble submission written to: submissions/ensemble_sub.csv"
