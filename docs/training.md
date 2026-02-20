# Training Guide

> Configurations, hyperparameters, and recipes for CLARITY training.

## Quick Start

```bash
# RoBERTa-base (fast baseline)
bash scripts/run_train_roberta.sh

# DeBERTa-v3-base (local workhorse)
bash scripts/run_train_deberta.sh

# DeBERTa-v3-large (competitive, needs CUDA)
bash scripts/run_train_deberta_large.sh

# Multitask (both evasion + clarity heads)
python -m clarity.train --config configs/deberta_multitask.yaml \
    --data data/train.csv --dev data/dev.csv --task multitask
```

## Configuration Reference

All configs live in `configs/` and are loaded via `--config`.

| Config | Model | Max Len | Batch | LR | Epochs | Label Smooth | Expected Task2 F1 | VRAM |
|--------|-------|---------|-------|-----|--------|--------------|-------------------|------|
| `roberta_base.yaml` | `roberta-base` | 256 | 16 | 2e-5 | 5 | 0.05 | ~0.45–0.52 | ~4GB |
| `deberta_v3_base.yaml` | `deberta-v3-base` | 384 | 8×2 | 2e-5 | 4 | 0.05 | ~0.48–0.56 | ~6GB |
| `deberta_v3_large.yaml` | `deberta-v3-large` | 384 | 4×4 | 1e-5 | 5 | 0.05 | ~0.56–0.65 | ~16GB |
| `deberta_multitask.yaml` | `deberta-v3-base` | 384 | 8×2 | 1.5e-5 | 5 | 0.05 | ~0.48–0.55 | ~6GB |

**Batch** column: `batch_size × grad_accum` = effective batch size.

## CLI Arguments

Every config key can be overridden via CLI:

```bash
python -m clarity.train \
    --config configs/deberta_v3_base.yaml \
    --data data/train.csv \
    --dev data/dev.csv \
    --task evasion \
    --lr 3e-5 \
    --label_smoothing 0.1 \
    --device cpu \
    --seed 123
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | — | YAML config file path |
| `--data` | — | Training CSV/parquet path |
| `--dev` | — | Dev CSV/parquet path |
| `--task` | `evasion` | `evasion`, `clarity`, or `multitask` |
| `--device` | auto | `cpu`, `cuda`, `mps` (auto-detected) |
| `--fp16` | config | Mixed precision (CUDA only) |
| `--label_smoothing` | 0.0 | Label smoothing factor (0.05–0.1 recommended) |
| `--seed` | 42 | Random seed |

## Training Recipe Recommendations

### For Imbalanced 9-Way Classification

1. **Start with label smoothing** (0.05–0.1) + class weights
2. **Avoid extreme class weights** — they destabilize learning
3. **Use max_length ≥ 384** — evasion cues appear later in answers
4. **LR sweep**: 1e-5 (large) to 3e-5 (base)

### For Top-10 Push

1. DeBERTa-v3-large on CUDA
2. max_length 384–512
3. 3–5 seed ensemble (logit averaging via `scripts/run_ensemble.sh`)
4. Optionally add metadata tokens (president, date)

## Ensemble Training

```bash
# Train 3 seeds, then average logits at prediction time
bash scripts/run_ensemble.sh configs/deberta_v3_large.yaml
```

Seeds: 42, 123, 2026. Logit averaging typically adds +1–3 F1 points.

## Checkpoints

Saved automatically to `output_dir/`:
- `best_model.pt` — best dev F1 checkpoint
- `metrics.json` — epoch-by-epoch training history
- `train.log` — full training log

Checkpoint contents:
```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": int,
    "best_f1": float,
    "args": dict,  # full config for reproducibility
}
```
