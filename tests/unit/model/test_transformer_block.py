import pytest
import torch

from llm.model.transformer_block import TransformerBlock


def make(dim=8, heads=2, hidden=16, max_len=6):
    return TransformerBlock(dim, heads, hidden, max_len).eval()


def test_shape_preserved():
    x = torch.randn(2, 5, 8)
    assert make()(x).shape == x.shape


def test_invalid_heads_raises():
    with pytest.raises(ValueError):
        TransformerBlock(10, 3, 16, 4)


def test_residual_when_sublayers_zeroed():
    block = make()
    for module in (block.attention.output, block.feed_forward.network[2]):
        torch.nn.init.zeros_(module.weight)
        torch.nn.init.zeros_(module.bias)
    x = torch.randn(1, 4, 8)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_causality():
    block = make()
    x = torch.randn(1, 6, 8)
    x2 = x.clone()
    x2[:, 5] = torch.randn(8)
    assert torch.allclose(block(x)[:, :5], block(x2)[:, :5], atol=1e-6)


def test_sequence_longer_than_max_raises():
    with pytest.raises(RuntimeError):
        make(max_len=3)(torch.randn(1, 4, 8))


def test_gradients_flow():
    block = make().train()
    block(torch.randn(2, 4, 8)).sum().backward()
    assert all(p.grad is not None for p in block.parameters())
