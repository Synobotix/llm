import pytest
import torch

from llm.model.attention import MultiHeadCausalSelfAttention


def make(dim=8, heads=2, max_len=6):
    return MultiHeadCausalSelfAttention(dim, heads, max_len).eval()


def test_indivisible_dim_raises():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadCausalSelfAttention(10, 3, 4)


def test_zero_heads_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        MultiHeadCausalSelfAttention(8, 0, 4)


def test_head_dim():
    assert make(12, 3).head_dim == 4


def test_mask_is_lower_triangular_and_buffer():
    att = make(max_len=4)
    assert torch.equal(att.causal_mask, torch.tril(torch.ones(4, 4)))
    assert "causal_mask" in att.state_dict()


def test_output_shape():
    assert make()(torch.randn(3, 5, 8)).shape == (3, 5, 8)


def test_single_head_and_single_token():
    att = make(8, 1, 4)
    assert att(torch.randn(1, 1, 8)).shape == (1, 1, 8)


def test_sequence_equal_to_max_ok():
    assert make(max_len=6)(torch.randn(1, 6, 8)).shape == (1, 6, 8)


def test_sequence_longer_than_mask_raises():
    with pytest.raises(RuntimeError):
        make(max_len=4)(torch.randn(1, 5, 8))


def test_wrong_embedding_dim_raises():
    with pytest.raises(RuntimeError):
        make()(torch.randn(1, 3, 7))


def test_causality_future_tokens_do_not_change_past_outputs():
    att = make()
    x = torch.randn(1, 6, 8)
    y1 = att(x)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 8)
    y2 = att(x2)
    assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-6)
    assert not torch.allclose(y1[:, 4:], y2[:, 4:])


def test_first_token_depends_only_on_itself():
    att = make()
    x = torch.randn(1, 4, 8)
    alone = att(x[:, :1])
    assert torch.allclose(att(x)[:, :1], alone, atol=1e-6)


def test_batch_independence():
    att = make()
    x = torch.randn(2, 4, 8)
    assert torch.allclose(att(x)[0:1], att(x[0:1]), atol=1e-6)


def test_gradients_flow_to_all_projections():
    att = make().train()
    att(torch.randn(2, 4, 8)).sum().backward()
    for name in ("query", "key", "value", "output"):
        assert getattr(att, name).weight.grad is not None


def test_no_nan_for_large_inputs():
    out = make()(torch.randn(1, 6, 8) * 1e3)
    assert torch.isfinite(out).all()


def test_split_heads_shape():
    att = make(8, 2)
    assert att._split_heads(torch.randn(3, 5, 8)).shape == (3, 2, 5, 4)
