import pytest
import torch

from llm.bitnet_model.bit_attention import BitAttention
from llm.bitnet_model.bitlinear import BitLinear

XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "BitLinear.quantize_activations utilise un absmax unique pour tout "
        "le tenseur : changer un token futur modifie l'échelle de "
        "quantification des tokens précédents (fuite causale ~1e-2)."
    ),
)


def make(d_model=8, heads=2):
    return BitAttention(d_model, heads)


def test_indivisible_d_model_raises():
    with pytest.raises(ValueError):
        BitAttention(10, 3)


def test_uses_bitlinear_projections_with_fused_qkv():
    attention = make()

    assert isinstance(attention.qkv_proj, BitLinear)
    assert isinstance(attention.out_proj, BitLinear)
    assert attention.qkv_proj.out_features == 3 * 8
    assert attention.head_dim == 4


def test_output_shape_matches_input():
    assert make()(torch.randn(2, 5, 8)).shape == (2, 5, 8)


def test_split_then_merge_heads_is_identity():
    attention = make()
    x = torch.randn(2, 5, 8)

    split = attention.split_heads(x)

    assert split.shape == (2, 2, 5, 4)
    assert torch.equal(attention.merge_heads(split), x)


@XFAIL
def test_attention_is_causal():
    attention = make().eval()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[0, 5] += 10

    assert torch.allclose(
        attention(x)[:, :5], attention(changed)[:, :5], atol=1e-5
    )


def test_last_position_depends_on_earlier_tokens():
    attention = make().eval()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[0, 0] += 10

    assert not torch.allclose(attention(x)[:, 5], attention(changed)[:, 5])


def test_gradients_reach_both_projections():
    attention = make().train()

    attention(torch.randn(2, 4, 8)).sum().backward()

    assert attention.qkv_proj.weight.grad.abs().sum() > 0
    assert attention.out_proj.weight.grad.abs().sum() > 0


def test_bias_option_is_forwarded():
    attention = BitAttention(8, 2, bias=True)

    assert attention.qkv_proj.bias is not None
    assert attention.out_proj.bias is not None
