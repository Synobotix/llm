import torch
import torch.nn as nn

from llm.bitnet_model.bit_transformer_block import BitTransformerBlock
from llm.bitnet_model.rms_norm import RMSNorm


class BitTransformer(nn.Module):
    """
    BitNet Transformer backbone: token embedding, stack of
    BitTransformerBlock, final RMSNorm. Returns hidden states;
    the projection to the vocabulary is done by LMHead.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        num_blocks: int,
        max_sequence_length: int,
        bias: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.max_sequence_length = max_sequence_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [
                BitTransformerBlock(d_model, num_heads, hidden_dim, bias)
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"La séquence ({sequence_length}) dépasse "
                f"max_sequence_length ({self.max_sequence_length})."
            )

        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)
