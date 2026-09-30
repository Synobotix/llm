from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from llm.training.checkpoint import save_checkpoint
from llm.training.loss import CausalLanguageModelingLoss

FINAL_CHECKPOINT_NAME = "student_v1_final.pt"
SEPARATOR = "=" * 60


class Trainer:
    """
    Gestionnaire de l'entraînement du Student v1.

    Pipeline d'un batch :
        forward -> loss -> backward -> gradient clipping
        -> optimizer.step() -> scheduler.step()
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cpu",
        epochs: int = 1,
        gradient_clip: float = 1.0,
        eval_every: int = 20,
        save_every: int = 50,
        checkpoint_dir: str = "checkpoints/student_v1",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.epochs = epochs
        self.gradient_clip = gradient_clip
        self.eval_every = eval_every
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.loss_function = CausalLanguageModelingLoss()

        self.model.to(self.device)

    # ------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------

    def train(self) -> None:
        self._print_header()

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)

            print()
            print(f"Epoch {epoch}/{self.epochs} - Loss : {train_loss:.4f}")

            if self.validation_dataloader is not None:
                print(f"Validation Loss : {self.evaluate():.4f}")

        self.save_checkpoint(filename=FINAL_CHECKPOINT_NAME)

        print(SEPARATOR)
        print("ENTRAÎNEMENT TERMINÉ")
        print(SEPARATOR)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch in self.train_dataloader:
            loss = self._train_step(batch)

            self.global_step += 1
            total_loss += loss
            num_batches += 1

            self._log_step(epoch, loss)
            self._save_periodic_checkpoint()

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: dict) -> float:
        """Exécute un pas d'optimisation et retourne la loss du batch."""
        input_ids, labels = self._unpack_batch(batch)

        self.optimizer.zero_grad(set_to_none=True)

        loss = self.loss_function(self.model(input_ids), labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip
        )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    # ------------------------------------------------------------
    # Évaluation
    # ------------------------------------------------------------

    def evaluate(self) -> float:
        if self.validation_dataloader is None:
            raise ValueError("Validation DataLoader absent.")

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.validation_dataloader:
                input_ids, labels = self._unpack_batch(batch)

                total_loss += self.loss_function(
                    self.model(input_ids), labels
                ).item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            checkpoint_path=self.checkpoint_dir / filename,
        )

    def _save_periodic_checkpoint(self) -> None:
        if self.save_every > 0 and self.global_step % self.save_every == 0:
            self.save_checkpoint(
                filename=f"checkpoint_step_{self.global_step}.pt"
            )

    # ------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------

    def _unpack_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        return input_ids, labels

    def _log_step(self, epoch: int, loss: float) -> None:
        learning_rate = self.optimizer.param_groups[0]["lr"]

        print(
            f"Step {self.global_step:04d} "
            f"| Epoch {epoch} "
            f"| Loss {loss:.4f} "
            f"| LR {learning_rate:.8f}"
        )

    def _print_header(self) -> None:
        print(SEPARATOR)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print(SEPARATOR)
        print(f"Device      : {self.device}")
        print(f"Epochs      : {self.epochs}")
        print(f"Train size  : {len(self.train_dataloader.dataset)}")

        if self.validation_dataloader is not None:
            print(f"Validation  : {len(self.validation_dataloader.dataset)}")

        print(SEPARATOR)
