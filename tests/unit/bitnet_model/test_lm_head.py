import torch

from llm.bitnet_model.bitlinear import BitLinear
from llm.bitnet_model.lm_head import LMHead


def test_projects_hidden_states_to_vocabulary_logits():
    assert LMHead(8, 20)(torch.randn(2, 5, 8)).shape == (2, 5, 20)


def test_uses_bitlinear_projection():
    head = LMHead(8, 20)

    assert isinstance(head.projection, BitLinear)
    assert (head.projection.in_features, head.projection.out_features) == (8, 20)


def test_stores_dimensions():
    head = LMHead(8, 20)

    assert (head.d_model, head.vocab_size) == (8, 20)


def test_bias_option_is_forwarded():
    assert LMHead(8, 20, bias=True).projection.bias is not None
    assert LMHead(8, 20).projection.bias is None


def test_gradient_reaches_projection_weights():
    head = LMHead(8, 20).train()

    head(torch.randn(2, 3, 8)).sum().backward()

    assert head.projection.weight.grad.abs().sum() > 0
