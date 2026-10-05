import pytest
import torch

from llm.bitnet_model.bitlinear import BitLinear


# ------------------------------------------------------------
# construction
# ------------------------------------------------------------

def test_weight_shape_and_dtype():
    layer = BitLinear(5, 3)

    assert layer.weight.shape == (3, 5)
    assert layer.weight.dtype == torch.float32


def test_no_bias_by_default():
    assert BitLinear(4, 2).bias is None


def test_bias_is_zero_initialised():
    layer = BitLinear(4, 2, bias=True)

    assert torch.equal(layer.bias, torch.zeros(2))


def test_inference_cache_is_not_saved_in_state_dict():
    keys = set(BitLinear(4, 2).state_dict())

    assert keys == {"weight"}


# ------------------------------------------------------------
# quantification des poids
# ------------------------------------------------------------

def test_ternarize_values_are_in_minus_one_zero_one():
    ternary, _ = BitLinear.ternarize_weights(torch.randn(8, 8))

    assert set(ternary.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_ternarize_scale_is_mean_absolute_value():
    weight = torch.tensor([[1.0, -3.0], [2.0, -2.0]])

    _, scale = BitLinear.ternarize_weights(weight)

    assert scale.item() == pytest.approx(2.0)


def test_ternarize_known_example():
    weight = torch.tensor([[4.0, -4.0, 0.1, 0.0]])  # scale = 2.025

    ternary, _ = BitLinear.ternarize_weights(weight)

    assert ternary.tolist() == [[1.0, -1.0, 0.0, 0.0]]


def test_ternarize_zero_weights_use_min_scale_and_stay_zero():
    ternary, scale = BitLinear.ternarize_weights(torch.zeros(2, 2))

    assert scale.item() == pytest.approx(BitLinear.MIN_SCALE)
    assert torch.equal(ternary, torch.zeros(2, 2))


def test_quantize_weights_returns_ternary_times_scale():
    weight = torch.randn(4, 4)

    quantized, scale = BitLinear.quantize_weights(weight)
    ternary, _ = BitLinear.ternarize_weights(weight)

    assert torch.equal(quantized, ternary * scale)


# ------------------------------------------------------------
# quantification des activations
# ------------------------------------------------------------

def test_activation_quantization_preserves_shape_and_is_close():
    x = torch.randn(2, 3, 16)

    quantized = BitLinear.quantize_activations(x)

    assert quantized.shape == x.shape
    assert (quantized - x).abs().max() <= x.abs().max() / 127 + 1e-6


def test_activation_quantization_has_at_most_255_levels():
    x = torch.linspace(-1, 1, 5000)

    assert BitLinear.quantize_activations(x).unique().numel() <= 255


def test_activation_quantization_of_zero_is_zero():
    assert torch.equal(
        BitLinear.quantize_activations(torch.zeros(3)), torch.zeros(3)
    )


def test_activation_quantization_fewer_bits_is_coarser():
    x = torch.linspace(-1, 1, 1000)

    levels_4 = BitLinear.quantize_activations(x, bits=4).unique().numel()
    levels_8 = BitLinear.quantize_activations(x, bits=8).unique().numel()

    assert levels_4 < levels_8


# ------------------------------------------------------------
# STE
# ------------------------------------------------------------

def test_ste_forward_value_is_quantized():
    original = torch.tensor([1.0, 2.0])
    quantized = torch.tensor([0.0, 5.0])

    assert torch.equal(BitLinear.ste(original, quantized), quantized)


def test_ste_gradient_flows_to_original_only():
    original = torch.tensor([1.0, 2.0], requires_grad=True)
    quantized = torch.tensor([0.0, 5.0], requires_grad=True)

    BitLinear.ste(original, quantized).sum().backward()

    assert torch.equal(original.grad, torch.ones(2))
    assert quantized.grad is None


# ------------------------------------------------------------
# forward
# ------------------------------------------------------------

def test_forward_shape_in_train_and_eval():
    layer = BitLinear(6, 4)
    x = torch.randn(2, 3, 6)

    assert layer.train()(x).shape == (2, 3, 4)
    assert layer.eval()(x).shape == (2, 3, 4)


def test_training_forward_matches_manual_computation():
    layer = BitLinear(6, 4, bias=True).train()
    x = torch.randn(2, 6)

    weight, _ = BitLinear.quantize_weights(layer.weight)
    expected = torch.nn.functional.linear(
        BitLinear.quantize_activations(x), weight, layer.bias
    )

    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_eval_matches_train_output():
    layer = BitLinear(8, 4)
    x = torch.randn(3, 8)

    train_out = layer.train()(x)
    eval_out = layer.eval()(x)

    assert torch.allclose(train_out, eval_out, atol=1e-5)


def test_training_gradient_reaches_master_weights_and_input():
    layer = BitLinear(4, 3).train()
    x = torch.randn(2, 4, requires_grad=True)

    layer(x).sum().backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0
    assert x.grad is not None


def test_bias_is_added_after_projection():
    layer = BitLinear(4, 2, bias=True).eval()
    with torch.no_grad():
        layer.bias.copy_(torch.tensor([1.0, -1.0]))
    with_bias = layer(torch.randn(1, 4))
    layer_no_bias = BitLinear(4, 2).eval()
    layer_no_bias.load_state_dict({"weight": layer.weight}, strict=True)

    x = torch.randn(1, 4)
    assert torch.allclose(
        layer(x) - layer_no_bias(x), torch.tensor([[1.0, -1.0]]), atol=1e-5
    )
    assert with_bias.shape == (1, 2)


# ------------------------------------------------------------
# inférence
# ------------------------------------------------------------

def test_prepare_for_inference_stores_int8_ternary_weights():
    layer = BitLinear(5, 3)

    layer.prepare_for_inference()

    assert layer.weight_ternary.dtype == torch.int8
    assert layer.weight_ternary.shape == (3, 5)
    assert set(layer.weight_ternary.unique().tolist()) <= {-1, 0, 1}


def test_eval_forward_prepares_weights_lazily():
    layer = BitLinear(5, 3).eval()
    assert layer.weight_ternary.numel() == 0

    layer(torch.randn(1, 5))

    assert layer.weight_ternary.numel() == 15


def test_inference_weight_is_ternary_times_scale():
    layer = BitLinear(5, 3)
    layer.prepare_for_inference()

    weight = layer._inference_weight(torch.float32)

    assert torch.allclose(
        weight, layer.weight_ternary.float() * layer.weight_scale
    )


def test_prepare_for_inference_does_not_track_gradients():
    layer = BitLinear(5, 3)

    layer.prepare_for_inference()

    assert not layer.weight_ternary.requires_grad
    assert not layer.weight_scale.requires_grad
