import math

import torch

from llm.model.positional_encoding import PositionalEncoding


def reference_encoding(sequence_length, embedding_dim):
    encoding = torch.zeros(sequence_length, embedding_dim)

    for position in range(sequence_length):
        for pair in range(embedding_dim // 2):
            angle = position / (10000.0 ** (2 * pair / embedding_dim))
            encoding[position, 2 * pair] = math.sin(angle)
            encoding[position, 2 * pair + 1] = math.cos(angle)

    return encoding


def test_encoding_matches_sinusoidal_reference():
    module = PositionalEncoding(embedding_dim=8, max_sequence_length=6)

    output = module(torch.zeros(1, 6, 8))

    assert torch.allclose(output[0], reference_encoding(6, 8), atol=1e-6)


def test_encoding_uses_true_position_index():
    module = PositionalEncoding(embedding_dim=4, max_sequence_length=5)

    output = module(torch.zeros(1, 5, 4))[0]

    assert torch.allclose(output[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.allclose(
        output[3, :2],
        torch.tensor([math.sin(3.0), math.cos(3.0)]),
        atol=1e-6,
    )


def test_encoding_is_added_to_input():
    module = PositionalEncoding(embedding_dim=8, max_sequence_length=6)
    x = torch.full((2, 4, 8), 2.0)

    output = module(x)

    expected = x + reference_encoding(4, 8)
    assert torch.allclose(output, expected, atol=1e-6)
