import torch


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer
