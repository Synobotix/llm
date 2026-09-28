import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Convert token IDs into learned vectors.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            embedding_dim: Dimension of the output vectors.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Integer tensor of shape (batch, sequence).

        Returns:
            Tensor of shape (batch, sequence, embedding_dim).
        """
        return self.embedding(input_ids)
