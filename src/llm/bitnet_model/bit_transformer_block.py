import torch
import torch.nn as nn

from llm.bitnet_model.bit_attention import BitAttention
from llm.bitnet_model.bit_mlp import BitMLP
from llm.bitnet_model.rms_norm import RMSNorm


class BitTransformerBlock(nn.Module):
    """
    Pre-norm block: RMSNorm -> attention -> residual,
    then RMSNorm -> MLP -> residual.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attention = BitAttention(d_model, num_heads, bias)

        self.norm2 = RMSNorm(d_model)
        self.mlp = BitMLP(d_model, hidden_dim, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
