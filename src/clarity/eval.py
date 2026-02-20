"""
Evaluation utilities for CLARITY tasks.

Task 1 (Clarity): standard macro F1 on 3-way classification.
Task 2 (Evasion): macro F1 with multi-annotator acceptance.

Multi-annotator scoring for Task 2:
  - Each test instance may have gold labels from up to 3 annotators.
  - A prediction is "correct" if it matches ANY annotator label.
  - For confusion matrix accounting:
    * If pred ∈ gold_set: assign effective_gold = pred (true positive for that class).
    * Else: assign effective_gold = annotator1 (or first non-empty annotator) as
      the reference label for FP/FN accounting.
  - Also reports "strict" score against majority-vote gold when possible.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from clarity.labels import (
    CLARITY_ID2LABEL,
    CLARITY_LABELS,
    EVASION_ID2LABEL,
    EVASION_LABELS,
    EVASION_TO_CLARITY,
    normalize_label,
)

logger = logging.getLogger("clarity")


# ─── Task 1: Standard macro F1 ───────────────────────────────────────────────


def compute_clarity_f1(
    predictions: List[str],
    gold_labels: List[str],
) -> Dict[str, Any]:
    """
    Compute macro F1 for Task 1 (3-way clarity classification).

    Args:
        predictions: predicted clarity labels (strings)
        gold_labels: ground-truth clarity labels (strings)

    Returns:
        Dict with macro_f1, per-class metrics, and confusion matrix.
    """
    assert len(predictions) == len(gold_labels), (
        f"Mismatch: {len(predictions)} predictions vs {len(gold_labels)} gold labels"
    )

    preds_norm = [normalize_label(p) for p in predictions]
    golds_norm = [normalize_label(g) for g in gold_labels]

    macro_f1 = f1_score(golds_norm, preds_norm, labels=CLARITY_LABELS, average="macro")

    report = classification_report(
        golds_norm,
        preds_norm,
        labels=CLARITY_LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(golds_norm, preds_norm, labels=CLARITY_LABELS)

    return {
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in CLARITY_LABELS
            if label in report
        },
        "confusion_matrix": cm.tolist(),
        "labels": CLARITY_LABELS,
    }


# ─── Task 2: Multi-annotator macro F1 ────────────────────────────────────────


def _resolve_effective_gold(
    pred: str,
    gold_set: Set[str],
) -> str:
    """
    Resolve the effective gold label for confusion matrix accounting.

    If prediction matches any annotator → effective_gold = pred (TP).
    Else → pick the first annotator label deterministically.
    """
    if pred in gold_set:
        return pred
    # Deterministic fallback: first alphabetically
    return sorted(gold_set)[0]


def compute_evasion_f1_multiannotator(
    predictions: List[str],
    annotator_labels: List[List[str]],
) -> Dict[str, Any]:
    """
    Compute macro F1 for Task 2 with multi-annotator gold.

    For each example:
      - gold_set = union of non-empty annotator labels
      - If pred ∈ gold_set: effective_gold = pred (counts as TP)
      - Else: effective_gold = first sorted annotator label

    Then compute standard macro F1 on (predictions, effective_golds).

    Args:
        predictions: predicted evasion labels (strings)
        annotator_labels: list of lists, each inner list = set of annotator labels

    Returns:
        Dict with macro_f1, per-class metrics, accuracy, confusion matrix.
    """
    assert len(predictions) == len(annotator_labels)

    preds_norm = [normalize_label(p) for p in predictions]

    effective_golds = []
    filtered_preds = []
    correct = 0
    total = 0
    skipped = 0
    for pred, golds_raw in zip(preds_norm, annotator_labels):
        gold_set = {normalize_label(g) for g in golds_raw if g.strip()}
        if not gold_set:
            # No annotator labels for this example — skip it entirely
            # (do NOT count as correct; this would leak pred into gold)
            skipped += 1
            continue

        eff_gold = _resolve_effective_gold(pred, gold_set)
        effective_golds.append(eff_gold)
        filtered_preds.append(pred)
        if pred in gold_set:
            correct += 1
        total += 1

    if skipped > 0:
        logger.info(
            f"Multi-annotator eval: {skipped}/{len(predictions)} examples "
            f"skipped (no annotator labels)"
        )

    if total == 0:
        logger.warning("No examples with annotator labels — cannot compute multi-annotator F1")
        return {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "labels": EVASION_LABELS,
            "skipped": skipped,
        }

    # Standard macro F1 on effective golds (using only annotated examples)
    macro_f1 = f1_score(
        effective_golds, filtered_preds, labels=EVASION_LABELS, average="macro", zero_division=0
    )

    accuracy = correct / total

    report = classification_report(
        effective_golds,
        filtered_preds,
        labels=EVASION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(effective_golds, filtered_preds, labels=EVASION_LABELS)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in EVASION_LABELS
            if label in report
        },
        "confusion_matrix": cm.tolist(),
        "labels": EVASION_LABELS,
    }


def compute_evasion_f1_strict(
    predictions: List[str],
    gold_labels: List[str],
) -> Dict[str, Any]:
    """
    Compute strict macro F1 for Task 2 (single gold label).

    Args:
        predictions: predicted evasion labels
        gold_labels: single ground-truth evasion labels

    Returns:
        Dict with macro_f1, per-class metrics.
    """
    preds_norm = [normalize_label(p) for p in predictions]
    golds_norm = [normalize_label(g) for g in gold_labels]

    macro_f1 = f1_score(
        golds_norm, preds_norm, labels=EVASION_LABELS, average="macro", zero_division=0
    )

    report = classification_report(
        golds_norm,
        preds_norm,
        labels=EVASION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in EVASION_LABELS
            if label in report
        },
    }


# ─── Majority vote gold ──────────────────────────────────────────────────────


def majority_vote_gold(annotator_labels: List[List[str]]) -> List[str]:
    """
    Compute majority-vote gold from multi-annotator labels.
    Ties broken alphabetically.
    """
    results = []
    for golds_raw in annotator_labels:
        normed = [normalize_label(g) for g in golds_raw if g.strip()]
        if not normed:
            results.append("")
            continue
        counts = Counter(normed)
        max_count = max(counts.values())
        # Among ties, pick alphabetically first
        candidates = sorted(k for k, v in counts.items() if v == max_count)
        results.append(candidates[0])
    return results


# ─── Confusion matrix formatting ─────────────────────────────────────────────


def format_confusion_matrix(cm: List[List[int]], labels: List[str]) -> str:
    """Format a confusion matrix as a readable string table."""
    header = "Pred→  " + "  ".join(f"{l[:8]:>8}" for l in labels)
    lines = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        lines.append(f"{labels[i][:8]:>8}  {row_str}")
    return "\n".join(lines)


# ─── Combined evaluation ─────────────────────────────────────────────────────


def evaluate_all(
    evasion_preds: List[str],
    clarity_preds: List[str],
    evasion_golds: Optional[List[str]] = None,
    clarity_golds: Optional[List[str]] = None,
    annotator_labels: Optional[List[List[str]]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation for both tasks.

    Args:
        evasion_preds: Task 2 predictions
        clarity_preds: Task 1 predictions
        evasion_golds: Task 2 single gold (train/dev) — optional
        clarity_golds: Task 1 gold labels — optional
        annotator_labels: Task 2 multi-annotator gold (test) — optional
        output_path: save results JSON here

    Returns:
        Combined metrics dict.
    """
    results: Dict[str, Any] = {}

    # Task 1: Clarity
    if clarity_golds:
        t1 = compute_clarity_f1(clarity_preds, clarity_golds)
        results["task1_clarity"] = t1
        logger.info(f"Task 1 Macro F1: {t1['macro_f1']:.4f}")

    # Task 2: Evasion
    if evasion_golds:
        t2_strict = compute_evasion_f1_strict(evasion_preds, evasion_golds)
        results["task2_evasion_strict"] = t2_strict
        logger.info(f"Task 2 Strict Macro F1: {t2_strict['macro_f1']:.4f}")

    if annotator_labels:
        t2_multi = compute_evasion_f1_multiannotator(evasion_preds, annotator_labels)
        results["task2_evasion_multiannotator"] = t2_multi
        logger.info(f"Task 2 Multi-Annotator Macro F1: {t2_multi['macro_f1']:.4f}")
        logger.info(f"Task 2 Multi-Annotator Accuracy: {t2_multi['accuracy']:.4f}")

        # Also compute majority-vote strict
        mv_golds = majority_vote_gold(annotator_labels)
        valid_pairs = [
            (p, g) for p, g in zip(evasion_preds, mv_golds) if g
        ]
        if valid_pairs:
            mv_preds, mv_golds_filtered = zip(*valid_pairs)
            t2_mv = compute_evasion_f1_strict(list(mv_preds), list(mv_golds_filtered))
            results["task2_evasion_majority_vote"] = t2_mv
            logger.info(f"Task 2 Majority-Vote Macro F1: {t2_mv['macro_f1']:.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Metrics saved to {output_path}")

    return results
