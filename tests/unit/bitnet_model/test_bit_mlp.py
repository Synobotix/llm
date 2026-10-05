import pytest
import torch

from llm.bitnet_model.bit_mlp import BitMLP
from llm.bitnet_model.bitlinear import BitLinear

XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "BitLinear.quantize_activations utilise un absmax unique pour tout "
        "le tenseur : changer un token futur modifie l'échelle de "
        "quantification des tokens précédents (fuite causale ~1e-2)."
    ),
)


def test_output_shape_matches_input():
    assert BitMLP(8, 16)(torch.randn(2, 5, 8)).shape == (2, 5, 8)


def test_dimensions_go_through_hidden_layer():
    mlp = BitMLP(8, 16)

    assert isinstance(mlp.fc1, BitLinear) and isinstance(mlp.fc2, BitLinear)
    assert (mlp.fc1.in_features, mlp.fc1.out_features) == (8, 16)
    assert (mlp.fc2.in_features, mlp.fc2.out_features) == (16, 8)


@XFAIL
def test_positions_are_processed_independently():
    mlp = BitMLP(8, 16).eval()
    x = torch.randn(1, 4, 8)
    changed = x.clone()
    changed[0, 3] += 5

    assert torch.allclose(mlp(x)[:, :3], mlp(changed)[:, :3], atol=1e-5)


def test_gradients_reach_both_layers():
    mlp = BitMLP(8, 16).train()

    mlp(torch.randn(2, 3, 8)).sum().backward()

    assert mlp.fc1.weight.grad.abs().sum() > 0
    assert mlp.fc2.weight.grad.abs().sum() > 0


def test_bias_option_is_forwarded():
    mlp = BitMLP(4, 6, bias=True)

    assert mlp.fc1.bias is not None and mlp.fc2.bias is not None
