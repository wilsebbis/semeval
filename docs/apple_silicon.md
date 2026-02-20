# Apple Silicon (MPS) Guide

> Workarounds, CPU fallback, and HF cache fixes for training on macOS.

## Summary

| Model | MPS Status | Recommendation |
|-------|-----------|----------------|
| RoBERTa-base | ✅ Works | Use MPS (fastest local option) |
| DeBERTa-v3-base | ⚠️ Supported | MPS with eager attention + float32 |
| DeBERTa-v3-large | ❌ Crashes | CPU fallback (or use CUDA) |

## DeBERTa + MPS

DeBERTa uses **disentangled attention** which involves complex tensor operations that can crash Apple's Metal kernels, especially with mixed dtypes.

### Automatic Handling

The training pipeline automatically:
1. **DeBERTa-large on MPS** → falls back to CPU with a warning
2. **DeBERTa-base on MPS** → allowed with safety hardening:
   - `attn_implementation="eager"` (no SDPA/flash attention)
   - All parameters cast to `float32` via `model.float()`
   - `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops
3. **RoBERTa on MPS** → works natively, no special handling

### Override

Force a specific device:
```bash
python -m clarity.train --config configs/deberta_v3_base.yaml \
    --device cpu  # force CPU even when MPS is available
```

## HF Cache Permissions

macOS sandbox blocks writes to `~/.cache/huggingface/hub`. All scripts set:

```bash
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
```

This redirects model downloads to `/tmp/hf_cache`, which is always writable.

### Manual Fix

If you encounter `PermissionError` on `~/.cache/huggingface`:
```bash
# Option 1: Fix permissions
chmod -R u+w ~/.cache/huggingface 2>/dev/null

# Option 2: Use writable cache (recommended)
export HF_HOME=/tmp/hf_cache
```

## Environment Variables

Set automatically by all `scripts/*.sh`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HOME` | `/tmp/hf_cache` | Writable HF model cache |
| `TRANSFORMERS_CACHE` | `$HF_HOME` | Transformers-specific cache |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | CPU fallback for unsupported MPS ops |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.0` | Disable MPS memory limit |

Set in code (`train.py`, `predict.py`):

| Variable | Purpose |
|----------|---------|
| `TOKENIZERS_PARALLELISM` | Set `false` to avoid deadlocks |

## FP16

Mixed precision (`fp16: true`) is **only supported on CUDA**. On MPS/CPU, the training loop automatically disables it with a warning.

## Performance Expectations

On Apple M-series (CPU fallback for DeBERTa):
- DeBERTa-v3-base: ~8–12 min/epoch (2930 train examples)
- DeBERTa-v3-large: ~25–40 min/epoch

On MPS (RoBERTa):
- RoBERTa-base: ~3–5 min/epoch
