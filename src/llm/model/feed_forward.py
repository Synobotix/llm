import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed-forward network applied independently at each position:
    Linear -> GELU -> Linear.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Args:
            embedding_dim: Input and output dimension.
            hidden_dim: Dimension of the hidden layer.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, sequence, embedding_dim).

        Returns:
            Tensor with the same shape as x.
        """
        return self.network(x)
