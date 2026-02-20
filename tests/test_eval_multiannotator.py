"""Tests for multi-annotator evaluation logic."""

import pytest

from clarity.eval import (
    compute_clarity_f1,
    compute_evasion_f1_multiannotator,
    compute_evasion_f1_strict,
    majority_vote_gold,
)


class TestClarityF1:
    def test_perfect_score(self):
        preds = ["Clear Reply", "Ambivalent Reply", "Clear Non-Reply"]
        golds = ["Clear Reply", "Ambivalent Reply", "Clear Non-Reply"]
        result = compute_clarity_f1(preds, golds)
        assert result["macro_f1"] == 1.0

    def test_all_wrong(self):
        preds = ["Clear Reply", "Clear Reply", "Clear Reply"]
        golds = ["Ambivalent Reply", "Clear Non-Reply", "Clear Non-Reply"]
        result = compute_clarity_f1(preds, golds)
        assert result["macro_f1"] == 0.0

    def test_partial_correct(self):
        preds = ["Clear Reply", "Clear Reply", "Clear Non-Reply"]
        golds = ["Clear Reply", "Ambivalent Reply", "Clear Non-Reply"]
        result = compute_clarity_f1(preds, golds)
        assert 0.0 < result["macro_f1"] < 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_clarity_f1(["Clear Reply"], ["Clear Reply", "Clear Non-Reply"])


class TestEvasionMultiAnnotator:
    def test_all_correct_single_annotator(self):
        """Prediction matches the only annotator label."""
        preds = ["Explicit", "Dodging", "Clarification"]
        annots = [["Explicit"], ["Dodging"], ["Clarification"]]
        result = compute_evasion_f1_multiannotator(preds, annots)
        # macro F1 averages over all 9 classes; only 3 have support, so
        # macro = (3 perfect classes * 1.0 + 6 absent classes * 0.0) / 9 = 1/3
        assert result["accuracy"] == 1.0
        # Per-class F1 for present classes should be 1.0
        for label in ["Explicit", "Dodging", "Clarification"]:
            assert result["per_class"][label]["f1"] == 1.0

    def test_correct_if_matches_any(self):
        """Prediction matches annotator2 but not annotator1."""
        preds = ["Dodging"]
        annots = [["Deflection", "Dodging", "Implicit"]]
        result = compute_evasion_f1_multiannotator(preds, annots)
        assert result["accuracy"] == 1.0

    def test_wrong_prediction(self):
        """Prediction matches none of the annotators."""
        preds = ["Explicit"]
        annots = [["Dodging", "Deflection"]]
        result = compute_evasion_f1_multiannotator(preds, annots)
        assert result["accuracy"] == 0.0

    def test_mixed_correctness(self):
        """Mix of correct and incorrect predictions."""
        preds = ["Explicit", "Dodging", "General"]
        annots = [
            ["Explicit", "Implicit"],       # correct (matches Explicit)
            ["Deflection", "Clarification"], # wrong (Dodging not in set)
            ["General"],                     # correct
        ]
        result = compute_evasion_f1_multiannotator(preds, annots)
        assert result["accuracy"] == pytest.approx(2.0 / 3.0)

    def test_empty_gold_skipped(self):
        """Empty gold set is skipped (not counted as correct)."""
        preds = ["Explicit"]
        annots = [[]]
        result = compute_evasion_f1_multiannotator(preds, annots)
        # No annotated examples → 0.0 accuracy
        assert result["accuracy"] == 0.0
        assert result["macro_f1"] == 0.0

    def test_normalization_in_gold(self):
        """Annotator labels get normalized before comparison."""
        preds = ["Partial/half-answer"]
        annots = [["Partial"]]
        result = compute_evasion_f1_multiannotator(preds, annots)
        assert result["accuracy"] == 1.0

    def test_deterministic_fallback(self):
        """When wrong, effective gold uses first sorted annotator label."""
        preds = ["Explicit", "Explicit"]
        annots = [
            ["Dodging", "Deflection"],  # wrong → effective_gold = "Deflection" (sorted)
            ["Implicit", "General"],    # wrong → effective_gold = "General" (sorted)
        ]
        result = compute_evasion_f1_multiannotator(preds, annots)
        assert result["accuracy"] == 0.0
        # Verify it's deterministic (run again)
        result2 = compute_evasion_f1_multiannotator(preds, annots)
        assert result["macro_f1"] == result2["macro_f1"]


class TestEvasionStrict:
    def test_perfect_score(self):
        preds = ["Explicit", "Dodging"]
        golds = ["Explicit", "Dodging"]
        result = compute_evasion_f1_strict(preds, golds)
        # macro F1 averages over all 9 classes; only 2 present → 2/9 macro
        for label in ["Explicit", "Dodging"]:
            assert result["per_class"][label]["f1"] == 1.0

    def test_all_wrong(self):
        preds = ["Explicit", "Explicit"]
        golds = ["Dodging", "Dodging"]
        result = compute_evasion_f1_strict(preds, golds)
        assert result["macro_f1"] == 0.0


class TestMajorityVote:
    def test_unanimous(self):
        annots = [["Dodging", "Dodging", "Dodging"]]
        result = majority_vote_gold(annots)
        assert result == ["Dodging"]

    def test_majority_2_1(self):
        annots = [["Dodging", "Dodging", "Deflection"]]
        result = majority_vote_gold(annots)
        assert result == ["Dodging"]

    def test_three_way_tie(self):
        """Tie broken alphabetically."""
        annots = [["Dodging", "Deflection", "Explicit"]]
        result = majority_vote_gold(annots)
        assert result == ["Deflection"]  # alphabetically first

    def test_empty(self):
        annots = [[]]
        result = majority_vote_gold(annots)
        assert result == [""]

    def test_normalization(self):
        annots = [["Partial", "Partial/half-answer", "Partial/Half-Answer"]]
        result = majority_vote_gold(annots)
        assert result == ["Partial/half-answer"]
