import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square normalization with a learned per-feature scale.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        return self.weight * (x / rms)
