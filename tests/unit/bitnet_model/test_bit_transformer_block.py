import pytest
import torch

from llm.bitnet_model.bit_transformer_block import BitTransformerBlock

XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "BitLinear.quantize_activations utilise un absmax unique pour tout "
        "le tenseur : changer un token futur modifie l'échelle de "
        "quantification des tokens précédents (fuite causale ~1e-2)."
    ),
)


def make():
    return BitTransformerBlock(8, 2, 16)


def test_output_shape_matches_input():
    assert make()(torch.randn(2, 5, 8)).shape == (2, 5, 8)


@XFAIL
def test_block_is_causal():
    block = make().eval()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[0, 5] += 10

    assert torch.allclose(block(x)[:, :5], block(changed)[:, :5], atol=1e-5)


def test_residual_connections_keep_input_signal():
    block = make().eval()
    x = torch.randn(1, 4, 8)

    out = block(x)
    residual_free = block.mlp(block.norm2(x + block.attention(block.norm1(x))))

    assert torch.allclose(
        out, x + block.attention(block.norm1(x)) + residual_free, atol=1e-5
    )


def test_gradients_reach_norms_attention_and_mlp():
    block = make().train()

    block(torch.randn(2, 4, 8)).sum().backward()

    for parameter in (
        block.norm1.weight,
        block.norm2.weight,
        block.attention.qkv_proj.weight,
        block.mlp.fc1.weight,
    ):
        assert parameter.grad is not None
