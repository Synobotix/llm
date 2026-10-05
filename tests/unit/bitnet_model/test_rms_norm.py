import pytest
import torch

from llm.bitnet_model.rms_norm import RMSNorm


def test_output_has_unit_rms_with_default_scale():
    out = RMSNorm(8)(torch.randn(2, 3, 8) * 5)

    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_preserves_shape():
    assert RMSNorm(4)(torch.randn(2, 5, 4)).shape == (2, 5, 4)


def test_weight_scales_output_per_feature():
    norm = RMSNorm(4)
    with torch.no_grad():
        norm.weight.mul_(2.0)
    x = torch.randn(1, 2, 4)

    assert torch.allclose(norm(x), 2 * RMSNorm(4)(x), atol=1e-6)


def test_invariant_to_input_scale():
    norm = RMSNorm(6)
    x = torch.randn(2, 3, 6)

    assert torch.allclose(norm(x), norm(x * 10), atol=1e-4)


def test_zero_input_stays_finite_thanks_to_eps():
    out = RMSNorm(4)(torch.zeros(1, 1, 4))

    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros(1, 1, 4))


def test_weight_is_learnable_and_receives_gradient():
    norm = RMSNorm(4)
    norm(torch.randn(2, 3, 4)).sum().backward()

    assert norm.weight.requires_grad
    assert norm.weight.grad is not None


def test_stores_hyperparameters():
    norm = RMSNorm(4, eps=1e-3)

    assert (norm.d_model, norm.eps) == (4, 1e-3)
    assert norm.weight.shape == (4,)
