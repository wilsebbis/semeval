"""
Ensemble prediction via logit averaging for CLARITY tasks.

Usage:
    python -m clarity.ensemble \\
        --ckpts ckpt1.pt ckpt2.pt ckpt3.pt \\
        --data data/test.csv \\
        --out submissions/ensemble.csv \\
        --task evasion

Averages raw logits from N checkpoints and decodes via argmax.
"""

import argparse
import logging
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from clarity.data import ClarityDataset, load_dataframe
from clarity.labels import (
    EVASION_ID2LABEL,
    EVASION_TO_CLARITY,
    normalize_label,
)
from clarity.eval import evaluate_all
from clarity.predict import load_model_from_checkpoint, predict_single
from clarity.utils import get_device, seed_everything, setup_logging

logger = logging.getLogger("clarity")


def average_logits(logits_list: List[torch.Tensor]) -> torch.Tensor:
    """Average logits from multiple models."""
    stacked = torch.stack(logits_list, dim=0)  # (N, batch, classes)
    return stacked.mean(dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble prediction via logit averaging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpts", nargs="+", required=True,
        help="Checkpoint paths (.pt files)"
    )
    parser.add_argument("--data", type=str, required=True, help="Input data path")
    parser.add_argument("--out", type=str, default="submissions/ensemble.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--task", type=str, default="evasion",
        choices=["evasion", "clarity", "multitask"],
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate if gold labels exist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    setup_logging()

    logger.info(f"Ensemble: {len(args.ckpts)} checkpoints")
    for i, p in enumerate(args.ckpts):
        logger.info(f"  [{i+1}] {p}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    logger.info(f"Using device: {device}")

    # Load model config from first checkpoint
    first_ckpt = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
    model_name = first_ckpt["args"].get("model_name", "roberta-base")
    max_length = first_ckpt["args"].get("max_length", 256)
    task = first_ckpt["args"].get("task", args.task)

    # Tokenizer
    use_fast = "deberta" not in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    # Dataset
    df = load_dataframe(args.data)
    dataset = ClarityDataset(
        df, tokenizer, max_length=max_length, task=task, is_test=True
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # Run inference for each checkpoint, collect logits
    all_logits = None
    all_ids = None

    for ckpt_path in args.ckpts:
        logger.info(f"Loading {ckpt_path}...")
        model = load_model_from_checkpoint(ckpt_path, device)
        results = predict_single(model, dataloader, device)

        if all_logits is None:
            all_logits = results["logits"]
            all_ids = results["ids"]
        else:
            all_logits = all_logits + results["logits"]

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Average
    all_logits = all_logits / len(args.ckpts)

    # Decode
    ev_pred_ids = all_logits.argmax(dim=-1).numpy()
    evasion_preds = [EVASION_ID2LABEL[int(i)] for i in ev_pred_ids]
    clarity_preds = [EVASION_TO_CLARITY[ev] for ev in evasion_preds]

    # Build submission
    sub_df = pd.DataFrame({
        "id": all_ids,
        "clarity_pred": clarity_preds,
        "evasion_pred": evasion_preds,
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    logger.info(f"Ensemble predictions saved: {out_path} ({len(sub_df)} rows)")

    # Evaluate if requested
    if args.evaluate:
        ev_golds = None
        if "evasion_label" in df.columns:
            raw = df["evasion_label"].fillna("").astype(str).tolist()
            ev_golds = [normalize_label(l) for l in raw]
            if all(g == "" for g in ev_golds):
                ev_golds = None

        cl_golds = None
        if "clarity_label" in df.columns:
            raw = df["clarity_label"].fillna("").astype(str).tolist()
            cl_golds = [normalize_label(l) for l in raw]
            if all(g == "" for g in cl_golds):
                cl_golds = None

        if ev_golds or cl_golds:
            metrics = evaluate_all(
                evasion_preds=evasion_preds,
                clarity_preds=clarity_preds,
                evasion_golds=ev_golds,
                clarity_golds=cl_golds,
            )
            for key, val in metrics.items():
                if isinstance(val, dict) and "macro_f1" in val:
                    logger.info(f"  {key}: macro_f1 = {val['macro_f1']:.4f}")

    # Distribution
    logger.info(f"Evasion: {pd.Series(evasion_preds).value_counts().to_dict()}")
    logger.info(f"Clarity: {pd.Series(clarity_preds).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
