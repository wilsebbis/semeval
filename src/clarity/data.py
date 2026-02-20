"""
Dataset loading, tokenization, and collation for CLARITY tasks.

Supports loading from:
  - CSV / TSV files
  - Parquet files
  - HuggingFace datasets hub (ailsntua/QEvasion)

Constructs input as:  "Question: {question}\nAnswer: {answer}"
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from clarity.labels import (
    CLARITY_LABEL2ID,
    EVASION_LABEL2ID,
    EVASION_TO_CLARITY,
    normalize_label,
)

logger = logging.getLogger("clarity")

# ─── Column name detection ────────────────────────────────────────────────────

REQUIRED_COLUMNS_TRAIN = {"question", "interview_answer"}
OPTIONAL_COLUMNS = {
    "clarity_label",
    "evasion_label",
    "annotator1",
    "annotator2",
    "annotator3",
    "index",
    "question_order",
}

# Prefer interview_question (full) over question (short/preprocessed)
ANSWER_COLUMN_ALIASES = ["interview_answer", "answer", "response"]
QUESTION_COLUMN_ALIASES = ["interview_question", "question"]


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Detect the question and answer column names."""
    q_col = None
    for alias in QUESTION_COLUMN_ALIASES:
        if alias in df.columns:
            q_col = alias
            break
    if q_col is None:
        raise ValueError(
            f"Could not find question column. Available: {list(df.columns)}. "
            f"Expected one of: {QUESTION_COLUMN_ALIASES}"
        )

    a_col = None
    for alias in ANSWER_COLUMN_ALIASES:
        if alias in df.columns:
            a_col = alias
            break
    if a_col is None:
        raise ValueError(
            f"Could not find answer column. Available: {list(df.columns)}. "
            f"Expected one of: {ANSWER_COLUMN_ALIASES}"
        )

    return q_col, a_col


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a dataframe from CSV, TSV, or Parquet."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(p)
    elif suffix == ".tsv":
        df = pd.read_csv(p, sep="\t")
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(p)
    else:
        # Try CSV as fallback
        df = pd.read_csv(p)

    logger.info(f"Loaded {len(df)} rows from {path}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def load_hf_dataset(split: str = "train") -> pd.DataFrame:
    """Load the QEvasion dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ailsntua/QEvasion", split=split)
        return ds.to_pandas()
    except ImportError:
        raise ImportError(
            "Install the `datasets` library to load from HuggingFace: "
            "pip install datasets"
        )


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────


class ClarityDataset(Dataset):
    """
    PyTorch Dataset for CLARITY classification.

    Input format: "Question: {question}\nAnswer: {answer}"

    Supports three modes:
      - "evasion" (Task 2): 9-way classification
      - "clarity" (Task 1): 3-way classification
      - "multitask": both labels simultaneously
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        task: str = "evasion",  # "evasion" | "clarity" | "multitask"
        is_test: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.is_test = is_test

        # Detect columns — prefers interview_question over question
        q_col, a_col = _detect_columns(df)
        self.q_col = q_col  # Track which column was selected
        self.a_col = a_col
        self.questions = df[q_col].fillna("").astype(str).tolist()
        self.answers = df[a_col].fillna("").astype(str).tolist()
        logger.info(f"Text columns: question='{q_col}', answer='{a_col}'")

        # IDs for submission
        if "index" in df.columns:
            self.ids = df["index"].tolist()
        elif "question_order" in df.columns:
            self.ids = df["question_order"].tolist()
        else:
            self.ids = list(range(len(df)))

        # Labels
        self.evasion_labels: Optional[List[int]] = None
        self.clarity_labels: Optional[List[int]] = None

        if not is_test:
            # Evasion labels
            if "evasion_label" in df.columns:
                raw_ev = df["evasion_label"].fillna("").astype(str).tolist()
                normed = [normalize_label(l) for l in raw_ev]
                missing = [l for l in normed if l and l not in EVASION_LABEL2ID]
                if missing:
                    unique_missing = set(missing)
                    logger.warning(
                        f"Unknown evasion labels (will be skipped): {unique_missing}"
                    )
                self.evasion_labels = [
                    EVASION_LABEL2ID.get(l, -1) for l in normed
                ]

            # Clarity labels
            if "clarity_label" in df.columns:
                raw_cl = df["clarity_label"].fillna("").astype(str).tolist()
                normed = [normalize_label(l) for l in raw_cl]
                missing = [l for l in normed if l and l not in CLARITY_LABEL2ID]
                if missing:
                    unique_missing = set(missing)
                    logger.warning(
                        f"Unknown clarity labels (will be skipped): {unique_missing}"
                    )
                self.clarity_labels = [
                    CLARITY_LABEL2ID.get(l, -1) for l in normed
                ]
            elif self.evasion_labels is not None:
                # Derive clarity from evasion via hierarchy
                self.clarity_labels = []
                for eid in self.evasion_labels:
                    if eid == -1:
                        self.clarity_labels.append(-1)
                    else:
                        from clarity.labels import EVASION_ID2LABEL
                        ev_str = EVASION_ID2LABEL[eid]
                        cl_str = EVASION_TO_CLARITY[ev_str]
                        self.clarity_labels.append(CLARITY_LABEL2ID[cl_str])

        # Multi-annotator gold for test evaluation
        self.annotator_labels: Optional[List[List[str]]] = None
        if "annotator1" in df.columns:
            annots = []
            for _, row in df.iterrows():
                golds = set()
                for col in ["annotator1", "annotator2", "annotator3"]:
                    if col in df.columns:
                        val = str(row.get(col, "")).strip()
                        if val and val.lower() != "nan":
                            golds.add(normalize_label(val))
                annots.append(list(golds))
            self.annotator_labels = annots

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        q = self.questions[idx]
        a = self.answers[idx]

        # Format: "Question: {question}\nAnswer: {answer}"
        text = f"Question: {q}\nAnswer: {a}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "idx": self.ids[idx],
        }

        if self.evasion_labels is not None:
            item["evasion_label"] = self.evasion_labels[idx]
        if self.clarity_labels is not None:
            item["clarity_label"] = self.clarity_labels[idx]

        return item


def get_class_weights(
    labels: List[int], num_classes: int, device: torch.device
) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    from collections import Counter
    counts = Counter(l for l in labels if l >= 0)
    total = sum(counts.values())
    weights = torch.ones(num_classes, device=device)
    for cls_id, cnt in counts.items():
        if cls_id < num_classes:
            weights[cls_id] = total / (num_classes * cnt)
    return weights


def build_datasets(
    train_path: str,
    dev_path: Optional[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
    task: str = "evasion",
) -> Tuple[ClarityDataset, Optional[ClarityDataset]]:
    """Build train/dev datasets from file paths."""
    train_df = load_dataframe(train_path)
    train_ds = ClarityDataset(train_df, tokenizer, max_length, task, is_test=False)

    dev_ds = None
    if dev_path:
        dev_df = load_dataframe(dev_path)
        dev_ds = ClarityDataset(dev_df, tokenizer, max_length, task, is_test=False)

    return train_ds, dev_ds
