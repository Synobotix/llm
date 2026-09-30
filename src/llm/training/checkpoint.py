from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    checkpoint_path: str | Path,
) -> None:
    """
    Sauvegarde l'état complet de l'entraînement :
    modèle, optimizer, scheduler (s'il existe) et global_step.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint sauvegardé : {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> int:
    """
    Recharge l'état complet de l'entraînement (modèle, optimizer,
    scheduler s'il est présent des deux côtés) et retourne global_step.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    global_step = checkpoint.get("global_step", 0)

    print(f"Checkpoint chargé : {checkpoint_path}")
    print(f"Global step restauré : {global_step}")

    return global_step
