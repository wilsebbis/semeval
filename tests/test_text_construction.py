"""
Tests for text construction in ClarityDataset.

Verifies that:
  1. interview_question is preferred over question
  2. Text format is "Question: {q}\\nAnswer: {a}"
  3. Fallback to question works when interview_question is missing
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock


def _make_tokenizer():
    """Create a mock tokenizer for testing."""
    tok = MagicMock()
    tok.sep_token = "[SEP]"
    tok.return_value = {
        "input_ids": MagicMock(squeeze=lambda dim: MagicMock()),
        "attention_mask": MagicMock(squeeze=lambda dim: MagicMock()),
    }
    return tok


class TestTextConstruction:
    """Test that ClarityDataset constructs text correctly."""

    def test_uses_interview_question_over_question(self):
        """interview_question should be preferred when both columns exist."""
        from clarity.data import ClarityDataset

        df = pd.DataFrame({
            "interview_question": ["What is your policy on climate change?"],
            "question": ["climate"],
            "interview_answer": ["We believe in renewable energy solutions."],
            "evasion_label": ["Explicit"],
            "clarity_label": ["Clear Reply"],
        })

        tok = _make_tokenizer()
        ds = ClarityDataset(df, tok, max_length=128, task="evasion")

        assert ds.q_col == "interview_question"
        assert ds.questions[0] == "What is your policy on climate change?"

    def test_text_format(self):
        """Text should be formatted as 'Question: {q}\\nAnswer: {a}'."""
        from clarity.data import ClarityDataset

        df = pd.DataFrame({
            "interview_question": ["What do you think about taxes?"],
            "interview_answer": ["Taxes are necessary for public services."],
            "evasion_label": ["Explicit"],
            "clarity_label": ["Clear Reply"],
        })

        tok = _make_tokenizer()
        ds = ClarityDataset(df, tok, max_length=128, task="evasion")

        # The text should contain both question and answer
        q = ds.questions[0]
        a = ds.answers[0]
        expected = f"Question: {q}\nAnswer: {a}"

        # Verify by checking what was passed to tokenizer
        ds[0]  # trigger __getitem__
        call_args = tok.call_args
        actual_text = call_args[0][0]
        assert actual_text == expected

    def test_fallback_to_question_column(self):
        """Should use 'question' column when interview_question is missing."""
        from clarity.data import ClarityDataset

        df = pd.DataFrame({
            "question": ["What about the economy?"],
            "interview_answer": ["The economy is growing."],
            "evasion_label": ["Explicit"],
        })

        tok = _make_tokenizer()
        ds = ClarityDataset(df, tok, max_length=128, task="evasion")

        assert ds.q_col == "question"
        assert ds.questions[0] == "What about the economy?"

    def test_constructed_text_contains_both_fields(self):
        """The constructed text must contain BOTH question AND answer."""
        from clarity.data import ClarityDataset

        df = pd.DataFrame({
            "interview_question": ["Should we raise the minimum wage?"],
            "interview_answer": [
                "We need to consider the impact on small businesses "
                "while ensuring workers can afford basic needs."
            ],
            "evasion_label": ["Dodging"],
            "clarity_label": ["Ambivalent Reply"],
        })

        tok = _make_tokenizer()
        ds = ClarityDataset(df, tok, max_length=128, task="evasion")

        ds[0]
        actual_text = tok.call_args[0][0]

        assert "Should we raise the minimum wage?" in actual_text
        assert "consider the impact on small businesses" in actual_text
        assert actual_text.startswith("Question:")
        assert "\nAnswer:" in actual_text
