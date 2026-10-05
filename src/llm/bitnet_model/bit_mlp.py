import torch
import torch.nn as nn

from llm.bitnet_model.bitlinear import BitLinear


class BitMLP(nn.Module):
    """
    BitLinear -> GELU -> BitLinear, applied at each position.

    Dimensions: d_model -> hidden_dim -> d_model.
    """

    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.fc1 = BitLinear(d_model, hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.fc2 = BitLinear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, sequence, d_model) -> (batch, sequence, d_model)"""
        return self.fc2(self.activation(self.fc1(x)))
