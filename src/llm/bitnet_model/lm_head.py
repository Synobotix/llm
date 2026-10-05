import torch
import torch.nn as nn

from llm.bitnet_model.bitlinear import BitLinear


class LMHead(nn.Module):
    """
    Projects hidden states to vocabulary logits with a BitLinear layer.
    """

    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.projection = BitLinear(
            in_features=d_model,
            out_features=vocab_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
