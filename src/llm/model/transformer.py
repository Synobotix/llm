import torch
import torch.nn as nn

from llm.model.embedding import TokenEmbedding
from llm.model.positional_encoding import PositionalEncoding
from llm.model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    """
    GPT-style causal language model.

    Token embedding, positional encoding, stack of Transformer blocks,
    final LayerNorm, then projection to the vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        num_layers: int,
        max_sequence_length: int,
    ):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            embedding_dim: Dimension of the token vectors.
            num_heads: Number of attention heads per block.
            ffn_hidden_dim: Hidden dimension of the feed-forward.
            num_layers: Number of Transformer blocks.
            max_sequence_length: Maximum sequence length.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_sequence_length)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads,
                                 ffn_hidden_dim, max_sequence_length)
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.output_projection_to_vocab = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Integer tensor of shape (batch, sequence).

        Returns:
            Logits of shape (batch, sequence, vocab_size),
            one score per vocabulary token and per position.
        """

        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        logits = self.output_projection_to_vocab(x)

        return logits
