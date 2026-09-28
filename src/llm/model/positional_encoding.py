import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding added to the token embeddings.
    """

    def __init__(self, embedding_dim: int, max_sequence_length: int):
        super().__init__()

        position = torch.arange(
            max_sequence_length, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )

        encoding = torch.zeros(max_sequence_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("positional_encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)

        return x + self.positional_encoding[:, :sequence_length]
