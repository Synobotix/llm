import math

import pytest
import torch

from llm.model.positional_encoding import PositionalEncoding


def test_buffer_shape_and_not_a_parameter():
    pe = PositionalEncoding(8, 16)
    assert pe.positional_encoding.shape == (1, 16, 8)
    assert list(pe.parameters()) == []
    assert "positional_encoding" in pe.state_dict()


def test_position_zero():
    pe = PositionalEncoding(8, 4).positional_encoding[0, 0]
    assert torch.allclose(pe[0::2], torch.zeros(4))
    assert torch.allclose(pe[1::2], torch.ones(4))


def test_known_value_position_one_first_pair():
    pe = PositionalEncoding(4, 3).positional_encoding[0, 1]
    assert pe[0].item() == pytest.approx(math.sin(1.0), abs=1e-6)
    assert pe[1].item() == pytest.approx(math.cos(1.0), abs=1e-6)


def test_values_bounded():
    pe = PositionalEncoding(16, 100).positional_encoding
    assert pe.min() >= -1.0 and pe.max() <= 1.0


def test_adds_encoding_to_input():
    pe = PositionalEncoding(8, 10)
    x = torch.zeros(2, 5, 8)
    assert torch.allclose(pe(x), pe.positional_encoding[:, :5].expand(2, -1, -1))


def test_sequence_longer_than_max_raises():
    pe = PositionalEncoding(8, 4)
    with pytest.raises(RuntimeError):
        pe(torch.zeros(1, 5, 8))


def test_sequence_equal_to_max_ok():
    assert PositionalEncoding(8, 4)(torch.zeros(1, 4, 8)).shape == (1, 4, 8)


def test_odd_embedding_dim_fails():
    # sin/cos remplissent 0::2 et 1::2 : taille incompatible si dim impaire.
    with pytest.raises(RuntimeError):
        PositionalEncoding(7, 4)


def test_embedding_dim_two():
    assert PositionalEncoding(2, 3).positional_encoding.shape == (1, 3, 2)


def test_max_length_one():
    assert PositionalEncoding(4, 1)(torch.zeros(1, 1, 4)).shape == (1, 1, 4)
