#!/bin/bash
# Train DeBERTa-v3-large seed 44 on A100
set -euo pipefail
export HF_HOME=${HF_HOME:-/content/hf_cache}

python -m clarity.train \
    --config configs/deberta_v3_large_seed44.yaml \
    --data data/train.csv \
    --dev data/dev.csv \
    --task evasion
