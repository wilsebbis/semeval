#!/bin/bash
# Ensemble: average logits from 3 seeds → final prediction
set -euo pipefail
export HF_HOME=${HF_HOME:-/content/hf_cache}

DATA=${1:-data/dev.csv}
OUT=${2:-submissions/ensemble_predictions.csv}

python -m clarity.ensemble \
    --ckpts \
        checkpoints/deberta_v3_large/best_model.pt \
        checkpoints/deberta_v3_large_seed43/best_model.pt \
        checkpoints/deberta_v3_large_seed44/best_model.pt \
    --data "$DATA" \
    --out "$OUT" \
    --task evasion \
    --evaluate
