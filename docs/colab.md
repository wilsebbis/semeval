# Colab A100 Training Guide

## Quick Start

### 1. Setup (run once)

```python
# Clone repo
!git clone https://github.com/wilsebbis/semeval.git /content/semeval
%cd /content/semeval
%pip install -e ".[dev]"

# Prepare data
!git clone https://huggingface.co/datasets/ailsntua/QEvasion
!bash scripts/prepare_data.sh
```

### 2. Single Run (DeBERTa-v3-large)

```bash
export HF_HOME=/content/hf_cache

python -m clarity.train \
    --config configs/deberta_v3_large.yaml \
    --data data/train.csv \
    --dev data/dev.csv \
    --task evasion \
    --debug_text
```

Expected logs on A100:
```
Precision: bf16 | Attention: sdpa | Effective batch: 16
CUDA: TF32 enabled for matmul and cuDNN
CUDA: Flash/mem-efficient/math SDP all enabled
GPU: NVIDIA A100-SXM4-80GB (85.1 GB)
```

### 3. Three-Seed Ensemble Training

```bash
export HF_HOME=/content/hf_cache

# Seed 42 (default)
bash scripts/run_train_large_seed42.sh

# Seed 43
bash scripts/run_train_large_seed43.sh

# Seed 44
bash scripts/run_train_large_seed44.sh
```

Or inline:
```bash
for SEED in 42 43 44; do
    python -m clarity.train \
        --config configs/deberta_v3_large_seed${SEED}.yaml \
        --data data/train.csv \
        --dev data/dev.csv \
        --task evasion
done
```

### 4. Ensemble Predict

```bash
python -m clarity.ensemble \
    --ckpts \
        checkpoints/deberta_v3_large/best_model.pt \
        checkpoints/deberta_v3_large_seed43/best_model.pt \
        checkpoints/deberta_v3_large_seed44/best_model.pt \
    --data data/test.csv \
    --out submissions/task2.csv \
    --task evasion \
    --evaluate
```

Or use the script: `bash scripts/run_ensemble.sh data/test.csv submissions/task2.csv`

## Expected GPU Memory

| Config | Precision | Batch | VRAM |
|--------|-----------|-------|------|
| DeBERTa-v3-base | bf16 | 16 | ~6 GB |
| DeBERTa-v3-large | bf16 | 8 | ~12-14 GB |
| DeBERTa-v3-large | fp32 | 4 | ~28 GB |

## Precision Options

The `precision` config key controls mixed precision:

| Value | Behavior |
|-------|----------|
| `auto` | bf16 on A100/H100, fp32 on MPS/CPU |
| `bf16` | Force bfloat16 (A100/H100 only) |
| `fp16` | Force float16 (T4/V100 compatible) |
| `fp32` | No mixed precision |

Legacy `fp16: true` in configs is still supported and maps to `precision=fp16`.

## Troubleshooting

**Low F1 (< 0.3)?** Check logs for:
- `Precision: fp32` → should be `bf16` on CUDA
- `Attention: eager` → should be `sdpa` on CUDA
- `Text columns: question='question'` → should be `interview_question`

**OOM?** Reduce `batch_size` and increase `grad_accum` to keep effective batch constant.
