"""
Model architectures for CLARITY tasks.

Provides:
  1. HierarchicalClassifier — single 9-way head; clarity derived via mapping
  2. MultitaskClassifier   — shared encoder with two heads (9-way + 3-way)
  3. FocalLoss             — class-imbalance-aware loss function
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from clarity.labels import (
    EVASION_ID_TO_CLARITY_ID,
    NUM_CLARITY_CLASSES,
    NUM_EVASION_CLASSES,
)

logger = logging.getLogger("clarity")

# DeBERTa model types that don't support SDPA in transformers
_DEBERTA_MODEL_TYPES = {"deberta-v2", "deberta"}


def _resolve_attn_impl(config: AutoConfig, requested: str) -> str:
    """Override attention implementation for models that don't support SDPA."""
    model_type = getattr(config, "model_type", "")
    if model_type in _DEBERTA_MODEL_TYPES and requested != "eager":
        logger.info(
            f"Attention override: eager (DeBERTaV2 does not support '{requested}')"
        )
        return "eager"
    return requested


# ─── Focal Loss ───────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure logits are float32 to avoid MPS mixed-dtype crashes
        logits = logits.float()
        w = self.weight.float() if self.weight is not None else None
        ce_loss = F.cross_entropy(
            logits, targets, weight=w, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        gamma = torch.tensor(self.gamma, dtype=logits.dtype, device=logits.device)
        focal_loss = ((1.0 - pt) ** gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ─── Hierarchical Classifier (primary) ────────────────────────────────────────


class HierarchicalClassifier(nn.Module):
    """
    Single 9-way evasion classifier on top of a pretrained encoder.
    Clarity labels are derived deterministically via EVASION_TO_CLARITY mapping.

    Architecture:
      [CLS] → dropout → linear(hidden, 9)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # DeBERTaV2 doesn't support SDPA — force eager
        actual_attn = _resolve_attn_impl(self.config, attn_implementation)
        self.encoder = AutoModel.from_pretrained(
            model_name, attn_implementation=actual_attn
        )
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, NUM_EVASION_CLASSES)

        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.loss_fn = FocalLoss(weight=class_weights, gamma=focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        evasion_label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cls_output = self.dropout(cls_output)
        # Defensive: align dtype with classifier weights (prevents Half/Float mismatch)
        cls_output = cls_output.to(self.classifier.weight.dtype)
        logits = self.classifier(cls_output)

        result: Dict[str, torch.Tensor] = {"logits": logits}

        if evasion_label is not None:
            # Filter out invalid labels (-1)
            valid_mask = evasion_label >= 0
            if valid_mask.any():
                # Compute loss in fp32 for numerical stability
                loss = self.loss_fn(logits[valid_mask].float(), evasion_label[valid_mask])
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            result["loss"] = loss

        return result

    def predict_evasion(self, logits: torch.Tensor) -> torch.Tensor:
        """Return predicted evasion class ids."""
        return logits.argmax(dim=-1)

    def predict_clarity(self, logits: torch.Tensor) -> torch.Tensor:
        """Map evasion predictions to clarity via hierarchy."""
        ev_preds = self.predict_evasion(logits)
        mapping = torch.tensor(
            [EVASION_ID_TO_CLARITY_ID[i] for i in range(NUM_EVASION_CLASSES)],
            device=logits.device,
        )
        return mapping[ev_preds]


# ─── Multitask Classifier (secondary) ─────────────────────────────────────────


class MultitaskClassifier(nn.Module):
    """
    Shared encoder with two classification heads:
      - 9-way evasion head
      - 3-way clarity head

    Loss = alpha * CE(evasion) + (1 - alpha) * CE(clarity)
           + beta * consistency_penalty

    Consistency: KL divergence between mapped evasion distribution
    and clarity head distribution.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
        alpha: float = 0.7,
        consistency_beta: float = 0.1,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        evasion_class_weights: Optional[torch.Tensor] = None,
        clarity_class_weights: Optional[torch.Tensor] = None,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        self.alpha = alpha
        self.consistency_beta = consistency_beta

        self.config = AutoConfig.from_pretrained(model_name)
        actual_attn = _resolve_attn_impl(self.config, attn_implementation)
        self.encoder = AutoModel.from_pretrained(
            model_name, attn_implementation=actual_attn
        )
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.evasion_head = nn.Linear(hidden_size, NUM_EVASION_CLASSES)
        self.clarity_head = nn.Linear(hidden_size, NUM_CLARITY_CLASSES)

        if use_focal_loss:
            self.evasion_loss_fn = FocalLoss(
                weight=evasion_class_weights, gamma=focal_gamma
            )
            self.clarity_loss_fn = FocalLoss(
                weight=clarity_class_weights, gamma=focal_gamma
            )
        else:
            self.evasion_loss_fn = nn.CrossEntropyLoss(weight=evasion_class_weights)
            self.clarity_loss_fn = nn.CrossEntropyLoss(weight=clarity_class_weights)

        # Build the evasion→clarity mapping matrix for consistency
        # mapping_matrix[evasion_class] = clarity_class
        self._register_consistency_matrix()

    def _register_consistency_matrix(self):
        """Create a matrix that maps evasion logits to clarity space."""
        # Shape: (NUM_CLARITY_CLASSES, NUM_EVASION_CLASSES)
        # mapping[clarity_i, evasion_j] = 1.0 if evasion_j maps to clarity_i
        mat = torch.zeros(NUM_CLARITY_CLASSES, NUM_EVASION_CLASSES)
        for ev_id, cl_id in EVASION_ID_TO_CLARITY_ID.items():
            mat[cl_id, ev_id] = 1.0
        self.register_buffer("mapping_matrix", mat)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        evasion_label: Optional[torch.Tensor] = None,
        clarity_label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Defensive: align dtype with head weights
        cls_output = cls_output.to(self.evasion_head.weight.dtype)
        ev_logits = self.evasion_head(cls_output)
        cl_logits = self.clarity_head(cls_output)

        result: Dict[str, torch.Tensor] = {
            "logits": ev_logits,  # primary: evasion
            "evasion_logits": ev_logits,
            "clarity_logits": cl_logits,
        }

        if evasion_label is not None or clarity_label is not None:
            loss = torch.tensor(0.0, device=ev_logits.device, requires_grad=True)

            if evasion_label is not None:
                valid_ev = evasion_label >= 0
                if valid_ev.any():
                    ev_loss = self.evasion_loss_fn(
                        ev_logits[valid_ev].float(), evasion_label[valid_ev]
                    )
                    loss = loss + self.alpha * ev_loss
                    result["evasion_loss"] = ev_loss

            if clarity_label is not None:
                valid_cl = clarity_label >= 0
                if valid_cl.any():
                    cl_loss = self.clarity_loss_fn(
                        cl_logits[valid_cl].float(), clarity_label[valid_cl]
                    )
                    loss = loss + (1 - self.alpha) * cl_loss
                    result["clarity_loss"] = cl_loss

            # Consistency regularizer
            if self.consistency_beta > 0:
                ev_probs = F.softmax(ev_logits.float(), dim=-1)
                # Map evasion probs to clarity space
                mapped_cl_probs = torch.matmul(
                    self.mapping_matrix, ev_probs.t()
                ).t()  # (B, 3)
                cl_log_probs = F.log_softmax(cl_logits, dim=-1)
                consistency = F.kl_div(
                    cl_log_probs, mapped_cl_probs, reduction="batchmean"
                )
                loss = loss + self.consistency_beta * consistency
                result["consistency_loss"] = consistency

            result["loss"] = loss

        return result

    def predict_evasion(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def predict_clarity_from_evasion(self, ev_logits: torch.Tensor) -> torch.Tensor:
        """Map evasion predictions to clarity via hierarchy."""
        ev_preds = ev_logits.argmax(dim=-1)
        mapping = torch.tensor(
            [EVASION_ID_TO_CLARITY_ID[i] for i in range(NUM_EVASION_CLASSES)],
            device=ev_logits.device,
        )
        return mapping[ev_preds]

    def predict_clarity_direct(self, cl_logits: torch.Tensor) -> torch.Tensor:
        """Direct clarity prediction from clarity head."""
        return cl_logits.argmax(dim=-1)


# ─── Model factory ────────────────────────────────────────────────────────────


def build_model(
    model_name: str = "roberta-base",
    task: str = "evasion",
    dropout: float = 0.1,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    alpha: float = 0.7,
    consistency_beta: float = 0.1,
    evasion_class_weights: Optional[torch.Tensor] = None,
    clarity_class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    attn_implementation: str = "eager",
) -> nn.Module:
    """Factory: build model based on task type."""
    if task in ("evasion", "clarity"):
        weights = evasion_class_weights if task == "evasion" else clarity_class_weights
        model = HierarchicalClassifier(
            model_name=model_name,
            dropout=dropout,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            class_weights=weights,
            label_smoothing=label_smoothing,
            attn_implementation=attn_implementation,
        )
        # Read the effective attention from the encoder config (reflects DeBERTa override)
        effective_attn = getattr(
            model.encoder.config, "_attn_implementation_internal",
            getattr(model.encoder.config, "attn_implementation", attn_implementation),
        )
        logger.info(
            f"Built HierarchicalClassifier with {model_name} "
            f"({'focal' if use_focal_loss else 'CE'} loss, "
            f"label_smoothing={label_smoothing}, attn={effective_attn})"
        )
    elif task == "multitask":
        model = MultitaskClassifier(
            model_name=model_name,
            dropout=dropout,
            alpha=alpha,
            consistency_beta=consistency_beta,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            evasion_class_weights=evasion_class_weights,
            clarity_class_weights=clarity_class_weights,
            attn_implementation=attn_implementation,
        )
        effective_attn = getattr(
            model.encoder.config, "_attn_implementation_internal",
            getattr(model.encoder.config, "attn_implementation", attn_implementation),
        )
        logger.info(
            f"Built MultitaskClassifier with {model_name} "
            f"(alpha={alpha}, beta={consistency_beta}, attn={effective_attn})"
        )
    else:
        raise ValueError(f"Unknown task: {task}. Expected: evasion, clarity, multitask")

    return model
