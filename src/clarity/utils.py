"""
Utility helpers: reproducibility, logging, config loading.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging to console + optional file."""
    logger = logging.getLogger("clarity")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file with basic validation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg)}")
    return cfg


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Save metrics dict as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
