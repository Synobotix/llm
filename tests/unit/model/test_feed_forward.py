import torch

from llm.model.feed_forward import FeedForward


def test_shape_preserved():
    x = torch.randn(2, 5, 8)
    assert FeedForward(8, 32)(x).shape == x.shape


def test_hidden_dim_smaller_than_embedding():
    assert FeedForward(8, 2)(torch.randn(1, 3, 8)).shape == (1, 3, 8)


def test_position_independent():
    ff = FeedForward(8, 16)
    x = torch.randn(1, 4, 8)
    full = ff(x)
    single = ff(x[:, 2:3])
    assert torch.allclose(full[:, 2:3], single, atol=1e-6)


def test_empty_batch_and_sequence():
    ff = FeedForward(4, 8)
    assert ff(torch.zeros(0, 3, 4)).shape == (0, 3, 4)
    assert ff(torch.zeros(2, 0, 4)).shape == (2, 0, 4)


def test_wrong_last_dim_raises():
    try:
        FeedForward(8, 16)(torch.randn(1, 2, 7))
    except RuntimeError:
        return
    raise AssertionError("RuntimeError attendu")


def test_gradients_flow():
    ff = FeedForward(4, 8)
    ff(torch.randn(2, 3, 4)).sum().backward()
    assert all(p.grad is not None for p in ff.parameters())
