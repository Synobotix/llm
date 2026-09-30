import torch
import torch.nn as nn


class CausalLanguageModelingLoss(nn.Module):
    """
    Cross-entropy between the predicted logits and the next-token labels.
    """

    def __init__(self):
        super().__init__()

        self.loss_function = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (batch, sequence, vocab_size).
            labels: Integer tensor of shape (batch, sequence).

        Returns:
            Scalar loss averaged over all positions.
        """
        vocab_size = logits.size(-1)

        return self.loss_function(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )
