# Architecture

> Model design, hierarchy, and loss functions for CLARITY classification.

## Overview

CLARITY uses a **hierarchical classification** approach: a single 9-way evasion classifier determines both Task 2 (evasion type) and Task 1 (clarity level) through a fixed taxonomy mapping.

## Taxonomy Hierarchy

```
Clear Reply          → Explicit
Ambivalent Reply     → Implicit, General, Partial/half-answer, Dodging, Deflection
Clear Non-Reply      → Declining to answer, Claims ignorance, Clarification
```

Task 1 predictions are deterministically derived from Task 2 via this mapping — no separate classifier needed.

## Model Variants

### HierarchicalClassifier (Primary)

Single 9-way classifier over `[CLS]` representation.

```
Encoder (DeBERTa/RoBERTa) → [CLS] → Dropout → Linear(hidden, 9) → Evasion logits
                                                                    ↓
                                                        Taxonomy map → Clarity pred
```

**Advantages:**
- Task 1 performance comes "for free" from Task 2
- Fewer parameters (one head vs two)
- Enforces taxonomy consistency by construction

### MultitaskClassifier (Comparison)

Shared encoder with separate 9-way and 3-way classification heads.

**Loss:** `α × CE(evasion) + (1−α) × CE(clarity) + β × KL(consistency)`

The KL consistency term encourages the clarity head to agree with the evasion head's taxonomy-mapped predictions. Default `α=0.7`, `β=0.1`.

## Loss Functions

### Cross-Entropy with Label Smoothing

Default loss. `nn.CrossEntropyLoss(label_smoothing=0.05)` with optional inverse-frequency class weights.

Label smoothing prevents the model from becoming overconfident on majority classes, which is critical for 9-way imbalanced classification.

### Focal Loss (Optional)

`FocalLoss(gamma=2.0)` down-weights easy examples. All tensors are explicitly cast to float32 to prevent MPS dtype crashes.

Disabled by default (`use_focal_loss: false`) — label smoothing + class weights generally performs better for this task.

## Input Format

```
Q: {question} [SEP] A: {answer}
```

The `[SEP]` token is the model-specific separator (e.g., `</s>` for DeBERTa, `</s>` for RoBERTa).

## Attention Implementation

DeBERTa uses **disentangled attention** which requires special handling:
- `attn_implementation="eager"` is forced (avoids SDPA/flash attention crashes on MPS)
- All parameters are explicitly cast to `float32` after loading

See [Apple Silicon Guide](apple_silicon.md) for details.
