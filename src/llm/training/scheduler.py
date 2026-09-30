import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    min_learning_rate: float = 1e-5,
):
    """
    Crée un scheduler Cosine Annealing.

    Le learning rate diminue progressivement
    entre le learning rate initial et
    min_learning_rate.
    """

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=min_learning_rate,
    )

    return scheduler
