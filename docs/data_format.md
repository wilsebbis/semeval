# Data Format

> CSV schema, label taxonomy, and normalization rules.

## Input Data

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `question` or `interview_question` | str | Sub-question text |
| `interview_answer` or `answer` | str | Full answer text |

Column names are auto-detected via aliases (see `data.py`).

### Label Columns (Train/Dev)

| Column | Type | Classes | Description |
|--------|------|---------|-------------|
| `evasion_label` | str | 9 | Fine-grained evasion technique |
| `clarity_label` | str | 3 | Coarse clarity level |

`clarity_label` is optional — if absent, it's derived from `evasion_label` via the taxonomy hierarchy.

### Annotator Columns (Test)

| Column | Type | Description |
|--------|------|-------------|
| `annotator1` | str | First annotator's evasion label |
| `annotator2` | str | Second annotator's evasion label |
| `annotator3` | str | Third annotator's evasion label |

Used for multi-annotator evaluation. May be empty/NaN on train/dev.

### Other Columns

| Column | Used | Description |
|--------|------|-------------|
| `index` | ID field | Example identifier for submissions |
| `title` | Ignored | Interview title |
| `date` | Ignored | Interview date |
| `president` | Ignored | Interviewee |
| `gpt3.5_prediction` | Ignored | Baseline GPT-3.5 prediction |
| `annotator_id` | Ignored | Annotator identifier |

## Label Taxonomy

### Evasion Labels (9-way)

| ID | Label | Parent (Clarity) |
|----|-------|-------------------|
| 0 | Explicit | Clear Reply |
| 1 | Implicit | Ambivalent Reply |
| 2 | General | Ambivalent Reply |
| 3 | Partial/half-answer | Ambivalent Reply |
| 4 | Dodging | Ambivalent Reply |
| 5 | Deflection | Ambivalent Reply |
| 6 | Declining to answer | Clear Non-Reply |
| 7 | Claims ignorance | Clear Non-Reply |
| 8 | Clarification | Clear Non-Reply |

### Clarity Labels (3-way)

| ID | Label | Evasion Children |
|----|-------|-----------------|
| 0 | Clear Reply | Explicit |
| 1 | Ambivalent Reply | Implicit, General, Partial/half-answer, Dodging, Deflection |
| 2 | Clear Non-Reply | Declining to answer, Claims ignorance, Clarification |

### Class Distribution (Train, n=2930)

**Evasion (heavily imbalanced):**

| Label | Count | % |
|-------|-------|---|
| Explicit | 894 | 30.5% |
| Implicit | 415 | 14.2% |
| General | 328 | 11.2% |
| Partial/half-answer | 67 | 2.3% |
| Dodging | 600 | 20.5% |
| Deflection | 324 | 11.1% |
| Declining to answer | 123 | 4.2% |
| Claims ignorance | 101 | 3.4% |
| Clarification | 78 | 2.7% |

**Clarity:**

| Label | Count | % |
|-------|-------|---|
| Ambivalent Reply | 1734 | 59.2% |
| Clear Reply | 894 | 30.5% |
| Clear Non-Reply | 302 | 10.3% |

## Label Normalization

All string labels are normalized via `normalize_label()` before use:

| Input | Output | Rule |
|-------|--------|------|
| `"Explicit"` | `"Explicit"` | Canonical (fast path) |
| `"explicit"` | `"Explicit"` | Case-insensitive |
| `"Partial"` | `"Partial/half-answer"` | Alias |
| `"Ambivalent"` | `"Ambivalent Reply"` | Standalone alias |
| `"Ambiguous"` | `"Ambivalent Reply"` | Website variant |
| `""` or `None` | `""` | Empty |

Unknown labels pass through unchanged (with a warning in data loading).

## Data Preparation

```bash
# Clone the QEvasion dataset
git clone https://huggingface.co/datasets/ailsntua/QEvasion

# Convert parquet → CSV and create 85/15 stratified train/dev split
bash scripts/prepare_data.sh
```

Output:
- `data/train.csv` — 2930 examples
- `data/dev.csv` — 518 examples
- `data/full_train.csv` — full 3448 examples (before split)
- `data/test.csv` — test set (if available)

## Submission Format

| Column | Description |
|--------|-------------|
| `id` | Example index |
| `clarity_pred` | Task 1 prediction (3-way) |
| `evasion_pred` | Task 2 prediction (9-way) |
