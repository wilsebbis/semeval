"""Tests for label mapping consistency."""

import pytest

from clarity.labels import (
    CLARITY_LABELS,
    CLARITY_LABEL2ID,
    CLARITY_TO_EVASION,
    EVASION_ID2LABEL,
    EVASION_ID_TO_CLARITY_ID,
    EVASION_LABEL2ID,
    EVASION_LABELS,
    EVASION_TO_CLARITY,
    NUM_CLARITY_CLASSES,
    NUM_EVASION_CLASSES,
    evasion_id_to_clarity_id,
    evasion_to_clarity,
    normalize_label,
)


class TestLabelConstants:
    def test_evasion_count(self):
        assert NUM_EVASION_CLASSES == 9

    def test_clarity_count(self):
        assert NUM_CLARITY_CLASSES == 3

    def test_evasion_labels_complete(self):
        expected = {
            "Explicit", "Implicit", "General", "Partial/half-answer",
            "Dodging", "Deflection", "Declining to answer",
            "Claims ignorance", "Clarification",
        }
        assert set(EVASION_LABELS) == expected

    def test_clarity_labels_complete(self):
        expected = {"Clear Reply", "Ambivalent Reply", "Clear Non-Reply"}
        assert set(CLARITY_LABELS) == expected


class TestHierarchy:
    def test_clear_reply_mapping(self):
        assert evasion_to_clarity("Explicit") == "Clear Reply"

    def test_ambivalent_reply_mappings(self):
        ambivalent_evasions = [
            "Implicit", "General", "Partial/half-answer",
            "Dodging", "Deflection",
        ]
        for ev in ambivalent_evasions:
            assert evasion_to_clarity(ev) == "Ambivalent Reply", (
                f"Expected 'Ambivalent Reply' for '{ev}', got '{evasion_to_clarity(ev)}'"
            )

    def test_clear_non_reply_mappings(self):
        non_reply_evasions = [
            "Declining to answer", "Claims ignorance", "Clarification",
        ]
        for ev in non_reply_evasions:
            assert evasion_to_clarity(ev) == "Clear Non-Reply"

    def test_all_evasions_mapped(self):
        for ev in EVASION_LABELS:
            cl = evasion_to_clarity(ev)
            assert cl in CLARITY_LABELS

    def test_all_clarity_covered(self):
        covered = set(EVASION_TO_CLARITY.values())
        assert covered == set(CLARITY_LABELS)

    def test_reverse_mapping_sizes(self):
        assert len(CLARITY_TO_EVASION["Clear Reply"]) == 1
        assert len(CLARITY_TO_EVASION["Ambivalent Reply"]) == 5
        assert len(CLARITY_TO_EVASION["Clear Non-Reply"]) == 3


class TestIntegerMappings:
    def test_label2id_roundtrip(self):
        for i, label in enumerate(EVASION_LABELS):
            assert EVASION_LABEL2ID[label] == i
            assert EVASION_ID2LABEL[i] == label

    def test_id_to_clarity_id_consistency(self):
        for ev_id in range(NUM_EVASION_CLASSES):
            cl_id = evasion_id_to_clarity_id(ev_id)
            assert 0 <= cl_id < NUM_CLARITY_CLASSES
            # Verify via string mapping
            ev_str = EVASION_ID2LABEL[ev_id]
            cl_str = EVASION_TO_CLARITY[ev_str]
            assert CLARITY_LABEL2ID[cl_str] == cl_id


class TestNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("Partial", "Partial/half-answer"),
        ("Partial/Half-Answer", "Partial/half-answer"),
        ("partial/half-answer", "Partial/half-answer"),
        ("explicit", "Explicit"),
        ("Declining", "Declining to answer"),
        ("Claims Ignorance", "Claims ignorance"),
        ("Clear reply", "Clear Reply"),
        ("Ambiguous", "Ambivalent Reply"),
        ("Clear non-reply", "Clear Non-Reply"),
    ])
    def test_alias_normalization(self, raw, expected):
        assert normalize_label(raw) == expected

    def test_canonical_label_unchanged(self):
        for label in EVASION_LABELS:
            assert normalize_label(label) == label
        for label in CLARITY_LABELS:
            assert normalize_label(label) == label

    def test_none_label(self):
        assert normalize_label(None) == ""

    def test_whitespace_stripping(self):
        assert normalize_label("  Explicit  ") == "Explicit"

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError):
            evasion_to_clarity("NotALabel")
