"""
Inference and submission generation for CLARITY tasks.

Usage:
    python -m clarity.predict \\
        --ckpt checkpoints/best_model.pt \\
        --data data/test.parquet \\
        --out submissions/sub.csv

Produces a CSV with columns: id, clarity_pred, evasion_pred
Optionally evaluates against gold labels if present.
"""

import argparse
import json
import logging
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from clarity.data import ClarityDataset, load_dataframe
from clarity.eval import evaluate_all
from clarity.labels import (
    CLARITY_ID2LABEL,
    EVASION_ID2LABEL,
    EVASION_TO_CLARITY,
    NUM_EVASION_CLASSES,
    normalize_label,
)
from clarity.models import build_model
from clarity.utils import get_device, seed_everything, setup_logging

logger = logging.getLogger("clarity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CLARITY inference and generate submission file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Test data path")
    parser.add_argument("--out", type=str, default="submissions/sub.csv", help="Output path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate if gold labels exist")
    parser.add_argument("--metrics_out", type=str, default=None, help="Save metrics JSON")
    parser.add_argument("--seed", type=int, default=42)
    # Ensemble: pass multiple checkpoints
    parser.add_argument(
        "--ensemble_ckpts", nargs="+", default=None,
        help="Multiple checkpoint paths for logit averaging"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device: 'cpu', 'cuda', 'mps'. Auto-detected if not set."
    )

    return parser.parse_args()


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, List]:
    """Run inference with a single model."""
    model.eval()
    all_ids = []
    all_logits = []

    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

        all_logits.append(logits.cpu())
        all_ids.extend(batch["idx"])

    all_logits = torch.cat(all_logits, dim=0)
    return {"ids": all_ids, "logits": all_logits}


def load_model_from_checkpoint(
    ckpt_path: str, device: torch.device
) -> torch.nn.Module:
    """Load a model from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    attn_impl = "sdpa" if device.type == "cuda" else "eager"
    model = build_model(
        model_name=args.get("model_name", "roberta-base"),
        task=args.get("task", "evasion"),
        dropout=args.get("dropout", 0.1),
        use_focal_loss=args.get("use_focal_loss", False),
        focal_gamma=args.get("focal_gamma", 2.0),
        alpha=args.get("alpha", 0.7),
        consistency_beta=args.get("consistency_beta", 0.1),
        label_smoothing=args.get("label_smoothing", 0.0),
        attn_implementation=attn_impl,
    )

    # strict=False: checkpoints with class_weights save loss_fn.weight
    # which won't exist in a fresh model. Safe to ignore.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    if device.type != "cuda":
        model = model.float()
    logger.info(f"Loaded model from {ckpt_path}")
    return model


def main():
    args = parse_args()
    seed_everything(args.seed)
    setup_logging()

    # Load checkpoint(s)
    ckpt_paths = args.ensemble_ckpts if args.ensemble_ckpts else [args.ckpt]

    # Load config from first checkpoint to get model_name
    first_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    model_name = first_ckpt["args"].get("model_name", "roberta-base")
    max_length = first_ckpt["args"].get("max_length", 256)
    task = first_ckpt["args"].get("task", "evasion")

    # Device selection (must come after model_name is known)
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
        if device.type == "mps" and "deberta" in model_name.lower():
            if "large" in model_name.lower():
                logger.warning(
                    "DeBERTa-large crashes on MPS. Falling back to CPU."
                )
                device = torch.device("cpu")
            else:
                logger.info("DeBERTa-base on MPS: using eager attention + float32.")
    logger.info(f"Using device: {device}")

    use_fast = "deberta" not in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    # Load data
    df = load_dataframe(args.data)
    dataset = ClarityDataset(
        df, tokenizer, max_length=max_length, task=task, is_test=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Predict (ensemble or single)
    if len(ckpt_paths) > 1:
        logger.info(f"Ensemble mode: averaging logits from {len(ckpt_paths)} models")
        all_logits = None
        all_ids = None

        for ckpt_path in ckpt_paths:
            model = load_model_from_checkpoint(ckpt_path, device)
            results = predict_single(model, dataloader, device)

            if all_logits is None:
                all_logits = results["logits"]
                all_ids = results["ids"]
            else:
                all_logits = all_logits + results["logits"]

            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

        all_logits = all_logits / len(ckpt_paths)
    else:
        model = load_model_from_checkpoint(ckpt_paths[0], device)
        results = predict_single(model, dataloader, device)
        all_logits = results["logits"]
        all_ids = results["ids"]

    # Decode predictions
    ev_pred_ids = all_logits.argmax(dim=-1).numpy()
    evasion_preds = [EVASION_ID2LABEL[int(i)] for i in ev_pred_ids]
    clarity_preds = [EVASION_TO_CLARITY[ev] for ev in evasion_preds]

    # Build submission DataFrame
    sub_df = pd.DataFrame({
        "id": all_ids,
        "clarity_pred": clarity_preds,
        "evasion_pred": evasion_preds,
    })

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    logger.info(f"Submission saved to {out_path} ({len(sub_df)} rows)")

    # Evaluate if requested
    if args.evaluate:
        logger.info("Running evaluation on predictions...")

        # Check for gold labels
        ev_golds = None
        cl_golds = None
        annot_labels = None

        if "evasion_label" in df.columns:
            raw = df["evasion_label"].fillna("").astype(str).tolist()
            ev_golds = [normalize_label(l) for l in raw]
            # Filter empty
            if all(g == "" for g in ev_golds):
                ev_golds = None

        if "clarity_label" in df.columns:
            raw = df["clarity_label"].fillna("").astype(str).tolist()
            cl_golds = [normalize_label(l) for l in raw]
            if all(g == "" for g in cl_golds):
                cl_golds = None

        if dataset.annotator_labels:
            annot_labels = dataset.annotator_labels

        metrics = evaluate_all(
            evasion_preds=evasion_preds,
            clarity_preds=clarity_preds,
            evasion_golds=ev_golds,
            clarity_golds=cl_golds,
            annotator_labels=annot_labels,
            output_path=args.metrics_out,
        )

        # Print summary
        for key, val in metrics.items():
            if isinstance(val, dict) and "macro_f1" in val:
                logger.info(f"  {key}: macro_f1 = {val['macro_f1']:.4f}")

    # Print prediction distribution
    logger.info("Prediction distribution:")
    logger.info(f"  Evasion: {pd.Series(evasion_preds).value_counts().to_dict()}")
    logger.info(f"  Clarity: {pd.Series(clarity_preds).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
