"""
Label definitions and hierarchy mappings for the CLARITY taxonomy.

Two-level taxonomy:
  Level 1 (Clarity): Clear Reply, Ambivalent Reply, Clear Non-Reply
  Level 2 (Evasion): 9 fine-grained evasion techniques
"""

from typing import Dict, List, Set

# ─── Canonical evasion (leaf) labels ──────────────────────────────────────────
EVASION_LABELS: List[str] = [
    "Explicit",
    "Implicit",
    "General",
    "Partial/half-answer",
    "Dodging",
    "Deflection",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]

# ─── Canonical clarity (parent) labels ────────────────────────────────────────
CLARITY_LABELS: List[str] = [
    "Clear Reply",
    "Ambivalent Reply",
    "Clear Non-Reply",
]

# ─── Hierarchy: evasion → clarity ─────────────────────────────────────────────
EVASION_TO_CLARITY: Dict[str, str] = {
    "Explicit":            "Clear Reply",
    "Implicit":            "Ambivalent Reply",
    "General":             "Ambivalent Reply",
    "Partial/half-answer": "Ambivalent Reply",
    "Dodging":             "Ambivalent Reply",
    "Deflection":          "Ambivalent Reply",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance":    "Clear Non-Reply",
    "Clarification":       "Clear Non-Reply",
}

# ─── Reverse hierarchy: clarity → evasion set ─────────────────────────────────
CLARITY_TO_EVASION: Dict[str, Set[str]] = {}
for ev, cl in EVASION_TO_CLARITY.items():
    CLARITY_TO_EVASION.setdefault(cl, set()).add(ev)

# ─── Integer mappings ─────────────────────────────────────────────────────────
EVASION_LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(EVASION_LABELS)}
EVASION_ID2LABEL: Dict[int, str] = {i: l for l, i in EVASION_LABEL2ID.items()}

CLARITY_LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(CLARITY_LABELS)}
CLARITY_ID2LABEL: Dict[int, str] = {i: l for l, i in CLARITY_LABEL2ID.items()}

NUM_EVASION_CLASSES = len(EVASION_LABELS)  # 9
NUM_CLARITY_CLASSES = len(CLARITY_LABELS)  # 3

# ─── Mapping evasion id → clarity id ──────────────────────────────────────────
EVASION_ID_TO_CLARITY_ID: Dict[int, int] = {
    EVASION_LABEL2ID[ev]: CLARITY_LABEL2ID[cl]
    for ev, cl in EVASION_TO_CLARITY.items()
}

# ─── Alternative label name normalization ─────────────────────────────────────
# Case-insensitive lookup table. Keys MUST be lowercase.
_ALIASES: Dict[str, str] = {
    # Evasion labels
    "explicit":              "Explicit",
    "implicit":              "Implicit",
    "general":               "General",
    "partial":               "Partial/half-answer",
    "partial/half-answer":   "Partial/half-answer",
    "half-answer":           "Partial/half-answer",
    "dodging":               "Dodging",
    "deflection":            "Deflection",
    "declining to answer":   "Declining to answer",
    "declining":             "Declining to answer",
    "claims ignorance":      "Claims ignorance",
    "clarification":         "Clarification",
    # Clarity labels — all known variants
    "clear reply":           "Clear Reply",
    "ambivalent reply":      "Ambivalent Reply",
    "ambivalent":            "Ambivalent Reply",      # standalone
    "ambiguous":             "Ambivalent Reply",       # website variant
    "ambiguous reply":       "Ambivalent Reply",       # combined variant
    "clear non-reply":       "Clear Non-Reply",
    "clear non reply":       "Clear Non-Reply",
    "clear nonreply":        "Clear Non-Reply",
    "non-reply":             "Clear Non-Reply",
}

# Also build entries for canonical labels themselves (lowercase versions)
for _label in EVASION_LABELS + CLARITY_LABELS:
    _ALIASES.setdefault(_label.lower(), _label)


def normalize_label(label) -> str:
    """Normalize a label string to its canonical form (case-insensitive)."""
    if label is None:
        return ""
    label = str(label).strip()
    if not label:
        return ""
    # Direct match against canonical labels first (fast path)
    if label in EVASION_LABELS or label in CLARITY_LABELS:
        return label
    # Case-insensitive lookup
    key = label.lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # Nothing matched
    return label


def evasion_to_clarity(evasion_label: str) -> str:
    """Map an evasion label to its parent clarity label."""
    norm = normalize_label(evasion_label)
    if norm not in EVASION_TO_CLARITY:
        raise ValueError(
            f"Unknown evasion label: '{evasion_label}' (normalized: '{norm}'). "
            f"Expected one of: {EVASION_LABELS}"
        )
    return EVASION_TO_CLARITY[norm]


def evasion_id_to_clarity_id(evasion_id: int) -> int:
    """Map an evasion label integer to its parent clarity integer."""
    return EVASION_ID_TO_CLARITY_ID[evasion_id]


def validate_labels():
    """Sanity-check all mappings are consistent."""
    for ev in EVASION_LABELS:
        assert ev in EVASION_TO_CLARITY, f"Missing mapping for evasion label: {ev}"
        cl = EVASION_TO_CLARITY[ev]
        assert cl in CLARITY_LABELS, f"Invalid clarity label '{cl}' for evasion '{ev}'"

    assert len(EVASION_LABELS) == NUM_EVASION_CLASSES == 9
    assert len(CLARITY_LABELS) == NUM_CLARITY_CLASSES == 3
    # Verify coverage: all clarity classes covered
    covered = set(EVASION_TO_CLARITY.values())
    assert covered == set(CLARITY_LABELS), f"Not all clarity labels covered: {covered}"


# Run on import to fail fast on broken label config
validate_labels()
