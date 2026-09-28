import math

import torch
import torch.nn as nn


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Each token only attends to the tokens before it (and itself),
    as required to predict the next token.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_sequence_length: int,
    ):
        """
        Args:
            embedding_dim: Dimension of the token vectors.
                Must be divisible by num_heads.
            num_heads: Number of attention heads.
            max_sequence_length: Maximum length covered by the causal mask.

        Raises:
            ValueError: If embedding_dim is not divisible by num_heads.
        """
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim doit être divisible par num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)

        causal_mask = torch.tril(
            torch.ones(max_sequence_length, max_sequence_length)
        )
        self.register_buffer("causal_mask", causal_mask)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch, sequence, embedding_dim)
        -> (batch, num_heads, sequence, head_dim)
        """
        batch_size, sequence_length, _ = x.shape

        return x.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch, num_heads, sequence, head_dim)
        -> (batch, sequence, embedding_dim)
        """
        batch_size, _, sequence_length, _ = x.shape

        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, sequence, embedding_dim).

        Returns:
            Tensor of shape (batch, sequence, embedding_dim).
        """
        sequence_length = x.size(1)

        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        attention_output = self._merge_heads(
            torch.matmul(attention_weights, v)
        )

        return self.output(attention_output)
