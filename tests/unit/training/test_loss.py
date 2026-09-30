import math

import pytest
import torch
import torch.nn.functional as F

from llm.training.loss import CausalLanguageModelingLoss


def test_matches_cross_entropy_on_flattened_inputs():
    logits = torch.randn(2, 3, 5)
    labels = torch.randint(0, 5, (2, 3))

    loss = CausalLanguageModelingLoss()(logits, labels)

    expected = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))
    assert torch.allclose(loss, expected)


def test_returns_scalar():
    loss = CausalLanguageModelingLoss()(
        torch.randn(2, 3, 5), torch.randint(0, 5, (2, 3))
    )
    assert loss.dim() == 0


def test_uniform_logits_give_log_vocab_size():
    loss = CausalLanguageModelingLoss()(
        torch.zeros(1, 4, 8), torch.zeros(1, 4, dtype=torch.long)
    )
    assert loss.item() == pytest.approx(math.log(8))


def test_confident_correct_prediction_has_near_zero_loss():
    labels = torch.tensor([[1, 2]])
    logits = torch.full((1, 2, 4), -50.0)
    logits[0, 0, 1] = 50.0
    logits[0, 1, 2] = 50.0

    assert CausalLanguageModelingLoss()(logits, labels).item() < 1e-6


def test_confident_wrong_prediction_has_large_loss():
    labels = torch.tensor([[0]])
    logits = torch.full((1, 1, 4), -50.0)
    logits[0, 0, 3] = 50.0

    assert CausalLanguageModelingLoss()(logits, labels).item() > 50


def test_gradient_flows_to_logits():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    CausalLanguageModelingLoss()(logits, torch.randint(0, 5, (1, 3))).backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_label_out_of_vocab_raises():
    with pytest.raises(IndexError):
        CausalLanguageModelingLoss()(
            torch.randn(1, 2, 4), torch.tensor([[0, 4]])
        )
