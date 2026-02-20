"""
Tests for ensemble logit averaging.
"""

import torch
import pytest


class TestLogitAveraging:
    """Test that logit averaging produces correct results."""

    def test_average_logits_two_models(self):
        """Averaging two logit tensors should match manual mean."""
        from clarity.ensemble import average_logits

        logits_a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        logits_b = torch.tensor([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])

        result = average_logits([logits_a, logits_b])
        expected = torch.tensor([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])

        assert torch.allclose(result, expected)

    def test_average_logits_three_models(self):
        """Averaging three logit tensors should match manual mean."""
        from clarity.ensemble import average_logits

        logits_a = torch.tensor([[1.0, 0.0, 0.0]])
        logits_b = torch.tensor([[0.0, 1.0, 0.0]])
        logits_c = torch.tensor([[0.0, 0.0, 1.0]])

        result = average_logits([logits_a, logits_b, logits_c])
        expected = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])

        assert torch.allclose(result, expected, atol=1e-6)

    def test_average_preserves_argmax_when_unanimous(self):
        """When all models agree, ensemble should agree too."""
        from clarity.ensemble import average_logits

        # All models predict class 0 (highest logit at index 0)
        logits = [torch.tensor([[5.0, 1.0, 1.0]])] * 3
        result = average_logits(logits)

        assert result.argmax(dim=-1).item() == 0

    def test_average_resolves_disagreement(self):
        """When 2/3 models agree, majority should win."""
        from clarity.ensemble import average_logits

        # 2 models predict class 1, 1 predicts class 2
        logits = [
            torch.tensor([[0.0, 5.0, 0.0]]),
            torch.tensor([[0.0, 4.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 5.0]]),
        ]
        result = average_logits(logits)

        # Average: [0.0, 3.0, 1.67] → class 1 wins
        assert result.argmax(dim=-1).item() == 1
