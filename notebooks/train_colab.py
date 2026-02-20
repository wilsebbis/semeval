# %% [markdown]
# # CLARITY: SemEval 2026 — Training Notebook
#
# Works on **both Colab (GPU) and local**.
#
# | Config | Model | VRAM | Expected Task2 F1 |
# |--------|-------|------|-------------------|
# | `deberta_v3_base.yaml` | DeBERTa-v3-base | ~6GB | ~0.48–0.56 |
# | `deberta_v3_large.yaml` | DeBERTa-v3-large | ~12-14GB (bf16) | ~0.56–0.65 |

# %% [markdown]
# ## 1. Setup Environment

# %%
# Check GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"VRAM: {vram / 1e9:.1f} GB")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("MPS (Apple Silicon) available")
else:
    print("CPU only")

# %%
import os, sys

# Detect environment
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Environment: {'Colab' if IN_COLAB else 'Local'}")

if IN_COLAB:
    REPO_URL = "https://github.com/wilsebbis/semeval.git"
    if not os.path.exists('/content/semeval'):
        get_ipython().system(f'git clone {REPO_URL} /content/semeval')
    else:
        get_ipython().system('cd /content/semeval && git pull')
    os.chdir('/content/semeval')
    get_ipython().system('pip install -e ".[dev]"')       # noqa: !pip
    os.environ['HF_HOME'] = '/content/hf_cache'
else:
    if os.path.basename(os.getcwd()) == 'notebooks':
        os.chdir(os.path.join(os.getcwd(), '..'))

# Add src/ to path as fallback
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"Working dir: {os.getcwd()}")

# Verify
from clarity.labels import EVASION_LABELS, CLARITY_LABELS, validate_labels
validate_labels()
print(f"✓ {len(EVASION_LABELS)} evasion labels, {len(CLARITY_LABELS)} clarity labels")

# %% [markdown]
# ## 2. Prepare Data

# %%
from pathlib import Path

if not Path('data/train.csv').exists():
    get_ipython().system('git clone https://huggingface.co/datasets/ailsntua/QEvasion 2>/dev/null || echo "Already cloned"')
    get_ipython().system('bash scripts/prepare_data.sh')
else:
    print("Data already prepared.")

# %%
import pandas as pd

train = pd.read_csv("data/train.csv")
dev = pd.read_csv("data/dev.csv")
print(f"Train: {len(train)} rows | Dev: {len(dev)} rows")
print(f"\nEvasion distribution:")
print(train["evasion_label"].value_counts().to_string())

# %% [markdown]
# ## 3. Train

# %%
# ── Choose config ────────────────────────────────────────
# CONFIG = "configs/deberta_v3_base.yaml"           # T4/Mac OK
CONFIG = "configs/deberta_v3_large.yaml"             # A100/L4
# CONFIG = "configs/deberta_v3_large_a100.yaml"      # A100 dedicated
# ─────────────────────────────────────────────────────────

import yaml
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)
print(f"Config: {CONFIG}")
for k, v in sorted(cfg.items()):
    print(f"  {k}: {v}")

# %%
get_ipython().system(
    f'python -m clarity.train '
    f'--config {CONFIG} '
    f'--data data/train.csv '
    f'--dev data/dev.csv '
    f'--task evasion '
    f'--debug_text'
)

# %% [markdown]
# ## 4. Evaluate

# %%
import json
from pathlib import Path

output_dir = cfg["output_dir"]
metrics_path = Path(output_dir) / "metrics.json"

if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    history = metrics.get("history", metrics if isinstance(metrics, list) else [])
    for m in history:
        ep = m.get('epoch', '?')
        ev = m.get('evasion_macro_f1', 0)
        cl = m.get('clarity_macro_f1', 0)
        loss = m.get('val_loss', 0)
        print(f"  Epoch {ep}: Task2 F1={ev:.4f}, Task1 F1={cl:.4f}, loss={loss:.4f}")
else:
    print("No metrics found — check training output above.")

# %% [markdown]
# ## 5. Predict

# %%
CKPT = f"{output_dir}/best_model.pt"
DATA = "data/dev.csv"  # Change to data/test.csv for submission

get_ipython().system(
    f'python -m clarity.predict '
    f'--ckpt {CKPT} '
    f'--data {DATA} '
    f'--out submissions/predictions.csv '
    f'--evaluate'
)

# %%
preds = pd.read_csv("submissions/predictions.csv")
print(f"Predictions: {len(preds)} rows")
print(preds["evasion_pred"].value_counts().to_string())

# %% [markdown]
# ## 6. Ensemble (Optional)
#
# Train 3 seeds, average logits → +1–3 F1 points.

# %%
# Train seeds 43 and 44 (seed 42 already trained above)
for seed_cfg in ["configs/deberta_v3_large_seed43.yaml",
                 "configs/deberta_v3_large_seed44.yaml"]:
    print(f"\n{'='*60}")
    print(f"Training: {seed_cfg}")
    print(f"{'='*60}")
    get_ipython().system(
        f'python -m clarity.train '
        f'--config {seed_cfg} '
        f'--data data/train.csv '
        f'--dev data/dev.csv '
        f'--task evasion'
    )

# %%
# Ensemble prediction
get_ipython().system(
    'python -m clarity.ensemble '
    '--ckpts '
    'checkpoints/deberta_v3_large/best_model.pt '
    'checkpoints/deberta_v3_large_seed43/best_model.pt '
    'checkpoints/deberta_v3_large_seed44/best_model.pt '
    '--data data/dev.csv '
    '--out submissions/ensemble_predictions.csv '
    '--task evasion '
    '--evaluate'
)

# %% [markdown]
# ## 7. Download (Colab only)

# %%
if IN_COLAB:
    from google.colab import files
    files.download("submissions/predictions.csv")
    if Path("submissions/ensemble_predictions.csv").exists():
        files.download("submissions/ensemble_predictions.csv")
else:
    print(f"Results: {os.path.abspath('submissions/predictions.csv')}")
