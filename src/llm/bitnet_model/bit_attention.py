import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.bitnet_model.bitlinear import BitLinear


class BitAttention(nn.Module):
    """
    Causal multi-head self-attention with BitLinear projections.

    - Q/K/V are computed by one fused BitLinear (d_model -> 3 * d_model);
    - causality and scaling are handled by
      F.scaled_dot_product_attention(is_causal=True);
    - the output projection is a BitLinear.
    """

    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model doit être divisible par num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = BitLinear(d_model, 3 * d_model, bias=bias)
        self.out_proj = BitLinear(d_model, d_model, bias=bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, sequence, d_model] -> [batch, heads, sequence, head_dim]"""
        batch_size, sequence_length, _ = x.shape

        return x.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, heads, sequence, head_dim] -> [batch, sequence, d_model]"""
        batch_size, _, sequence_length, _ = x.shape

        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, sequence, d_model] -> [batch, sequence, d_model]"""
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        attention_output = F.scaled_dot_product_attention(
            self.split_heads(q),
            self.split_heads(k),
            self.split_heads(v),
            is_causal=True,
        )

        return self.out_proj(self.merge_heads(attention_output))
