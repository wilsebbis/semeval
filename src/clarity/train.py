"""
Training CLI for CLARITY tasks.

Usage:
    python -m clarity.train --config configs/roberta_base.yaml \\
        --data data/train.parquet --dev data/test.parquet --task evasion

Supports:
    - Hierarchical (evasion-only, clarity derived via mapping)
    - Multitask (shared encoder, two heads)
    - Early stopping on dev macro F1
    - Class-weighted / focal loss
    - Gradient accumulation
    - Mixed precision (bf16/fp16 on CUDA, fp32 on MPS/CPU)
    - Checkpoint saving
"""

import argparse
import json
import logging
import os

# MPS hardening: prevent Metal kernel crashes
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from clarity.data import ClarityDataset, build_datasets, get_class_weights, load_dataframe
from clarity.eval import evaluate_all
from clarity.labels import (
    CLARITY_ID2LABEL,
    EVASION_ID2LABEL,
    EVASION_LABELS,
    EVASION_TO_CLARITY,
    NUM_CLARITY_CLASSES,
    NUM_EVASION_CLASSES,
)
from clarity.models import build_model
from clarity.utils import (
    count_parameters,
    get_device,
    load_config,
    save_metrics,
    seed_everything,
    setup_logging,
)

logger = logging.getLogger("clarity")


# ─── Precision / CUDA helpers ─────────────────────────────────────────────────


def _resolve_precision(precision: str, device: torch.device) -> str:
    """Resolve 'auto' precision and validate choice for the target device."""
    if precision == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return "bf16"
            return "fp16"
        return "fp32"  # MPS and CPU always fp32

    # Validate device compatibility
    if precision in ("bf16", "fp16") and device.type != "cuda":
        logger.warning(
            f"Mixed precision '{precision}' requires CUDA but device is '{device.type}'. "
            f"Falling back to fp32."
        )
        return "fp32"

    return precision


def _setup_cuda_optimizations(precision: str) -> None:
    """Enable CUDA-specific performance optimizations."""
    # TF32 — free ~2x speedup on Ampere+ with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("CUDA: TF32 enabled for matmul and cuDNN")

    # SDPA / flash attention — enable on CUDA (disabled on MPS for stability)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("CUDA: Flash/mem-efficient/math SDP all enabled")
    except Exception:
        logger.info("CUDA: SDP backend control not available (older PyTorch)")


def _setup_mps_guards() -> None:
    """Disable SDPA on MPS to prevent Metal kernel crashes."""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _get_autocast_dtype(precision: str) -> Optional[torch.dtype]:
    """Map precision string to torch dtype for autocast."""
    if precision == "bf16":
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    return None


# ─── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLARITY classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--dev", type=str, default=None, help="Dev/validation data path")
    parser.add_argument(
        "--task",
        type=str,
        default="evasion",
        choices=["evasion", "clarity", "multitask"],
        help="Task type",
    )
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.7, help="Multitask loss weight")
    parser.add_argument(
        "--consistency_beta", type=float, default=0.1,
        help="Multitask consistency reg weight"
    )
    parser.add_argument("--num_workers", type=int, default=0)
    # Legacy fp16 flag — mapped to precision=fp16
    parser.add_argument("--fp16", action="store_true", help="(Legacy) Use fp16 mixed precision")
    parser.add_argument(
        "--precision", type=str, default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Mixed precision mode. 'auto' = bf16 on CUDA if supported, fp32 elsewhere."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device: 'cpu', 'cuda', 'mps'. Auto-detected if not set."
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0,
        help="Label smoothing factor (0.0 = off, 0.05-0.1 recommended)"
    )
    parser.add_argument(
        "--debug_text", action="store_true",
        help="Log first training example's constructed text for debugging"
    )

    args = parser.parse_args()

    # Override with config file if provided
    if args.config:
        cfg = load_config(args.config)
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
            else:
                logger.warning(f"Unknown config key: {key}")

    # Legacy fp16 flag → precision
    if args.fp16 and args.precision == "auto":
        args.precision = "fp16"

    return args


# ─── Training loop ─────────────────────────────────────────────────────────────


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    grad_accum: int = 1,
    precision: str = "fp32",
    task: str = "evasion",
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    autocast_dtype = _get_autocast_dtype(precision)
    use_amp = autocast_dtype is not None and device.type == "cuda"
    # GradScaler only for fp16 (bf16 doesn't need it)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" and device.type == "cuda" else None

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "evasion_label" in batch:
            kwargs["evasion_label"] = batch["evasion_label"].to(device)
        if "clarity_label" in batch:
            kwargs["clarity_label"] = batch["clarity_label"].to(device)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                outputs = model(**kwargs)
                loss = outputs["loss"] / grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()  # bf16 doesn't need scaler
        else:
            outputs = model(**kwargs)
            loss = outputs["loss"] / grad_accum
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs["loss"].item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    return {"train_loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task: str = "evasion",
    annotator_labels: Optional[List[List[str]]] = None,
    precision: str = "fp32",
) -> Dict[str, Any]:
    """Evaluate on validation set, return metrics.
    
    If annotator_labels is provided, computes both strict and multi-annotator F1.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_ev_preds: List[str] = []
    all_cl_preds: List[str] = []
    all_ev_golds: List[str] = []
    all_cl_golds: List[str] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "evasion_label" in batch:
            kwargs["evasion_label"] = batch["evasion_label"].to(device)
        if "clarity_label" in batch:
            kwargs["clarity_label"] = batch["clarity_label"].to(device)

        # Autocast for eval too — prevents dtype mismatch
        autocast_dtype = _get_autocast_dtype(precision)
        use_amp = autocast_dtype is not None and device.type == "cuda"

        if use_amp:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                outputs = model(**kwargs)
        else:
            outputs = model(**kwargs)

        if "loss" in outputs:
            total_loss += outputs["loss"].float().item()  # fp32 for stability
        num_batches += 1

        # Get logits
        logits = outputs["logits"]

        if hasattr(model, "predict_evasion"):
            ev_pred_ids = model.predict_evasion(logits).cpu().numpy()
        else:
            ev_pred_ids = logits.argmax(dim=-1).cpu().numpy()

        for eid in ev_pred_ids:
            ev_label = EVASION_ID2LABEL[int(eid)]
            all_ev_preds.append(ev_label)
            all_cl_preds.append(EVASION_TO_CLARITY[ev_label])

        if "evasion_label" in batch:
            for g in batch["evasion_label"].numpy():
                if g >= 0:
                    all_ev_golds.append(EVASION_ID2LABEL[int(g)])
                    all_cl_golds.append(
                        EVASION_TO_CLARITY[EVASION_ID2LABEL[int(g)]]
                    )
                else:
                    all_ev_golds.append("")
                    all_cl_golds.append("")

    # Compute metrics (filter out empty golds)
    valid_pairs_ev = [
        (p, g) for p, g in zip(all_ev_preds, all_ev_golds) if g
    ]
    valid_pairs_cl = [
        (p, g) for p, g in zip(all_cl_preds, all_cl_golds) if g
    ]

    metrics: Dict[str, Any] = {
        "val_loss": total_loss / max(num_batches, 1),
    }

    if valid_pairs_ev:
        ev_p, ev_g = zip(*valid_pairs_ev)
        cl_p, cl_g = (zip(*valid_pairs_cl) if valid_pairs_cl
                       else ([], []))
        ev_results = evaluate_all(
            evasion_preds=list(ev_p),
            clarity_preds=list(cl_p),
            evasion_golds=list(ev_g),
            clarity_golds=list(cl_g),
            annotator_labels=annotator_labels,
        )
        if "task2_evasion_strict" in ev_results:
            metrics["evasion_macro_f1"] = ev_results["task2_evasion_strict"]["macro_f1"]
        # Multi-annotator F1 — this is what SemEval actually scores on
        if "task2_evasion_multiannotator" in ev_results:
            metrics["evasion_multi_f1"] = ev_results["task2_evasion_multiannotator"]["macro_f1"]
            metrics["evasion_multi_acc"] = ev_results["task2_evasion_multiannotator"]["accuracy"]
        if "task1_clarity" in ev_results:
            metrics["clarity_macro_f1"] = ev_results["task1_clarity"]["macro_f1"]
        metrics["full_results"] = ev_results

    return metrics


def main():
    args = parse_args()

    # Setup
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir))

    logger.info(f"Configuration: {vars(args)}")

    # Device selection
    if args.device:
        req = args.device.lower()
        if req == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        elif req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(req)
    else:
        device = get_device()
        # DeBERTa-large's disentangled attention crashes MPS Metal kernels;
        # base is OK with eager attention + float32 hardening
        if device.type == "mps" and "deberta" in args.model_name.lower():
            if "large" in args.model_name.lower():
                logger.warning(
                    "DeBERTa-large crashes on MPS (Metal kernel incompatibility). "
                    "Falling back to CPU. Use --device cuda on a GPU machine."
                )
                device = torch.device("cpu")
            else:
                logger.info(
                    "DeBERTa-base on MPS: using eager attention + float32 hardening."
                )
    logger.info(f"Using device: {device}")

    # ── Precision & device-specific optimizations ──
    precision = _resolve_precision(args.precision, device)

    if device.type == "cuda":
        _setup_cuda_optimizations(precision)
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        logger.info(f"GPU: {gpu_name} ({vram / 1e9:.1f} GB)")
    elif device.type == "mps":
        _setup_mps_guards()

    # Attention: request SDPA on CUDA (models.py will override to eager for DeBERTa)
    if device.type == "cuda":
        attn_impl = "sdpa"
    else:
        attn_impl = "eager"

    eff_batch = args.batch_size * args.grad_accum
    logger.info(
        f"Precision: {precision} | Attention: {attn_impl} "
        f"(DeBERTa auto-overrides to eager) | Effective batch: {eff_batch}"
    )

    # Tokenizer
    # DeBERTa-v3 needs use_fast=False to avoid spm.model conversion errors
    use_fast = "deberta" not in args.model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=use_fast)
    logger.info(f"Loaded tokenizer: {args.model_name} (fast={use_fast})")

    # Datasets
    train_ds, dev_ds = build_datasets(
        train_path=args.data,
        dev_path=args.dev,
        tokenizer=tokenizer,
        max_length=args.max_length,
        task=args.task,
    )

    logger.info(f"Train: {len(train_ds)} examples")
    if dev_ds:
        logger.info(f"Dev: {len(dev_ds)} examples")

    # Debug text — show first constructed example
    if args.debug_text and len(train_ds) > 0:
        sample = train_ds[0]
        q = train_ds.questions[0]
        a = train_ds.answers[0]
        text = f"Q: {q} {tokenizer.sep_token} A: {a}"
        logger.info(f"DEBUG TEXT (first example, {len(text)} chars): {text[:300]}")

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    dev_loader = None
    if dev_ds:
        dev_loader = DataLoader(
            dev_ds,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    # Class weights
    ev_weights = None
    cl_weights = None
    if args.use_class_weights and train_ds.evasion_labels is not None:
        ev_weights = get_class_weights(
            train_ds.evasion_labels, NUM_EVASION_CLASSES, device
        )
        logger.info(f"Evasion class weights: {ev_weights.tolist()}")
    if args.use_class_weights and train_ds.clarity_labels is not None:
        cl_weights = get_class_weights(
            train_ds.clarity_labels, NUM_CLARITY_CLASSES, device
        )
        logger.info(f"Clarity class weights: {cl_weights.tolist()}")

    # Model
    model = build_model(
        model_name=args.model_name,
        task=args.task,
        dropout=args.dropout,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        alpha=args.alpha,
        consistency_beta=args.consistency_beta,
        evasion_class_weights=ev_weights,
        clarity_class_weights=cl_weights,
        label_smoothing=getattr(args, "label_smoothing", 0.0),
        attn_implementation=attn_impl,
    )
    model.to(device)

    # Dtype normalization: ALWAYS keep weights in fp32.
    # Standard mixed precision = fp32 master weights + autocast for bf16/fp16 forward.
    # HF DeBERTa safetensors ships some params as fp16; normalize them.
    # autocast in train_epoch / evaluate_epoch handles the fast bf16/fp16 matmuls.
    model.float()
    param_dtypes = {p.dtype for p in model.parameters()}
    logger.info(
        f"Model param dtypes (CUDA, precision={precision}, params kept fp32): {param_dtypes}"
    )
    if param_dtypes != {torch.float32}:
        logger.warning(f"Mixed dtypes detected: {param_dtypes}. Forcing all to float32.")
        for p in model.parameters():
            p.data = p.data.float()

    param_info = count_parameters(model)
    logger.info(f"Parameters: {param_info}")

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_groups, lr=args.lr)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    # Training loop
    best_f1 = -1.0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_accum=args.grad_accum,
            precision=precision,
            task=args.task,
        )
        elapsed = time.time() - t0
        logger.info(f"Train loss: {train_metrics['train_loss']:.4f} ({elapsed:.1f}s)")

        epoch_metrics = {**train_metrics, "epoch": epoch}

        # Evaluate
        if dev_loader:
            # Pass annotator labels for multi-annotator scoring (official SemEval metric)
            # But only if they actually contain data (dev may not have them)
            annot_labels = None
            if dev_ds and dev_ds.annotator_labels:
                has_annotations = any(
                    any(g.strip() for g in golds if g)
                    for golds in dev_ds.annotator_labels
                    if golds
                )
                if has_annotations:
                    annot_labels = dev_ds.annotator_labels
                else:
                    logger.info("Dev set has no multi-annotator labels; using strict scoring only.")
            val_metrics = evaluate_epoch(
                model, dev_loader, device,
                task=args.task,
                annotator_labels=annot_labels,
                precision=precision,
            )
            epoch_metrics.update(val_metrics)

            ev_f1 = val_metrics.get("evasion_macro_f1", 0.0)
            ev_multi_f1 = val_metrics.get("evasion_multi_f1", 0.0)
            ev_multi_acc = val_metrics.get("evasion_multi_acc", 0.0)
            cl_f1 = val_metrics.get("clarity_macro_f1", 0.0)

            logger.info(f"Val loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Evasion strict F1: {ev_f1:.4f}")
            if ev_multi_f1 > 0:
                logger.info(f"Evasion multi-annotator F1: {ev_multi_f1:.4f} (acc: {ev_multi_acc:.4f})")
            logger.info(f"Clarity macro F1: {cl_f1:.4f}")

            # Use multi-annotator F1 for early stopping if available (official metric)
            if args.task in ("evasion", "multitask"):
                primary_f1 = ev_multi_f1 if ev_multi_f1 > 0 else ev_f1
            else:
                primary_f1 = cl_f1

            if primary_f1 > best_f1:
                best_f1 = primary_f1
                patience_counter = 0
                # Save best model
                ckpt_path = output_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_f1": best_f1,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                logger.info(f"New best model saved (F1={best_f1:.4f}) → {ckpt_path}")
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {patience_counter}/{args.patience}"
                )

            if patience_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break
        else:
            # No dev set — save every epoch
            ckpt_path = output_dir / f"model_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )

        history.append(epoch_metrics)

    # Save training history
    save_metrics(
        {"history": history, "best_f1": best_f1, "args": vars(args)},
        str(output_dir / "metrics.json"),
    )

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        },
        final_path,
    )
    logger.info(f"Final model saved → {final_path}")
    logger.info(f"Best dev F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
