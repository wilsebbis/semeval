# Evaluation

> Scoring methodology, multi-annotator logic, and metrics for CLARITY tasks.

## Overview

| Task | Classes | Metric | Note |
|------|---------|--------|------|
| Task 1 — Clarity | 3-way | Macro F1 | Derived from Task 2 via taxonomy |
| Task 2 — Evasion | 9-way | Macro F1 | Multi-annotator scoring on test |

## Multi-Annotator Scoring (Task 2)

The test set provides labels from up to 3 annotators per example. The official SemEval scoring accepts a prediction as **correct if it matches ANY annotator**.

### Algorithm

For each example:
1. Build `gold_set = {annotator1, annotator2, annotator3} ∩ valid_labels`
2. If `pred ∈ gold_set` → `effective_gold = pred` (counts as true positive for that class)
3. Else → `effective_gold = sorted(gold_set)[0]` (deterministic fallback for FP/FN accounting)
4. If `gold_set` is empty → **skip the example entirely** (no annotator labels available)

Then compute standard macro F1 over `(filtered_preds, effective_golds)`.

> **Important:** Examples with empty annotator labels are skipped, not counted as correct. This prevents gold-set leakage where predictions are trivially "correct" against an empty set.

### Strict vs Multi-Annotator

During training on dev (which typically lacks multi-annotator labels), only **strict scoring** is used — comparing predictions against the single `evasion_label` column.

Multi-annotator scoring activates automatically when the annotator columns contain actual data.

## Early Stopping

The training loop uses the **multi-annotator F1** for early stopping when available (official metric). Falls back to strict F1 when annotator labels are absent.

## Majority Vote

For comparison, we also compute majority-vote strict F1:
- For each example, take the mode of annotator labels (ties broken alphabetically)
- Compute standard macro F1 against this single gold label

## Implementation

All scoring logic lives in [`src/clarity/eval.py`](../src/clarity/eval.py):

| Function | Purpose |
|----------|---------|
| `compute_clarity_f1` | Task 1 macro F1 |
| `compute_evasion_f1_strict` | Task 2 strict macro F1 (single gold) |
| `compute_evasion_f1_multiannotator` | Task 2 multi-annotator macro F1 |
| `majority_vote_gold` | Compute majority-vote from annotator labels |
| `evaluate_all` | Combined evaluation orchestrator |

## Label Normalization

All labels are normalized before scoring via `normalize_label()` in [`labels.py`](../src/clarity/labels.py):
- Case-insensitive matching
- Alias resolution (e.g., "Partial" → "Partial/half-answer", "Ambivalent" → "Ambivalent Reply")
- Canonical labels pass through unchanged (fast path)
