#!/usr/bin/env bash
# Convert QEvasion parquet data to CSV and split train into train/dev
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

mkdir -p "$DATA_DIR"

python3 -c "
import pandas as pd
import sys
from pathlib import Path

qe_dir = Path('$PROJECT_DIR/QEvasion/data')
if not qe_dir.exists():
    print('ERROR: QEvasion dataset not found. Clone it first:')
    print('  git clone https://huggingface.co/datasets/ailsntua/QEvasion')
    sys.exit(1)

# Load parquet
train_pq = list(qe_dir.glob('train-*.parquet'))
test_pq = list(qe_dir.glob('test-*.parquet'))

if not train_pq:
    print('ERROR: No train parquet files found')
    sys.exit(1)

train = pd.read_parquet(train_pq[0])
print(f'Loaded train: {len(train)} rows, columns: {list(train.columns)}')

# Print label distribution
if 'clarity_label' in train.columns:
    print(f'Clarity distribution:')
    print(train['clarity_label'].value_counts().to_string())
if 'evasion_label' in train.columns:
    print(f'Evasion distribution:')
    print(train['evasion_label'].value_counts().to_string())

# Create train/dev split (85/15 stratified)
from sklearn.model_selection import train_test_split

if 'evasion_label' in train.columns:
    stratify_col = train['evasion_label']
else:
    stratify_col = train['clarity_label']

train_split, dev_split = train_test_split(
    train, test_size=0.15, random_state=42, stratify=stratify_col
)

print(f'Train split: {len(train_split)} rows')
print(f'Dev split:   {len(dev_split)} rows')

# Save
out = Path('$DATA_DIR')
train_split.to_csv(out / 'train.csv', index=False)
dev_split.to_csv(out / 'dev.csv', index=False)
train.to_csv(out / 'full_train.csv', index=False)

if test_pq:
    test = pd.read_parquet(test_pq[0])
    test.to_csv(out / 'test.csv', index=False)
    print(f'Test: {len(test)} rows')

print('Done! Files saved to $DATA_DIR/')
"

echo "Data preparation complete."
echo "  Train: $DATA_DIR/train.csv"
echo "  Dev:   $DATA_DIR/dev.csv"
echo "  Test:  $DATA_DIR/test.csv"
