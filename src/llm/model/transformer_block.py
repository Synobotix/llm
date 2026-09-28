import torch
import torch.nn as nn

from llm.model.attention import MultiHeadCausalSelfAttention
from llm.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    LayerNorm -> causal attention -> residual connection,
    then LayerNorm -> feed-forward -> residual connection.
    """

    def __init__(self, embedding_dim: int, num_heads: int,
                 ffn_hidden_dim: int, max_sequence_length: int):
        """
        Args:
            embedding_dim: Dimension of the token vectors.
            num_heads: Number of attention heads.
            ffn_hidden_dim: Hidden dimension of the feed-forward.
            max_sequence_length: Maximum sequence length.
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim, num_heads, max_sequence_length)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, sequence, embedding_dim).

        Returns:
            Tensor with the same shape as x.
        """

        # 1. Attention + connexion résiduelle
        attention_input = self.layer_norm_1(x)
        attention_output = self.attention(attention_input)
        x = x + attention_output

        # 2. Feed Forward + connexion résiduelle
        feed_forward_input = self.layer_norm_2(x)
        feed_forward_output = self.feed_forward(feed_forward_input)
        x = x + feed_forward_output

        return x
