from pathlib import Path

import torch

from llm.data.dataloader import create_validation_dataloader
from llm.tokenizer.tokenizer import get_vocab_size
from llm.model.transformer import Transformer
from llm.config.parameters import MAX_SEQUENCE_LENGTH
from llm.training.loss import CausalLanguageModelingLoss


# ============================================================
# Configuration du modèle
# ============================================================

EMBEDDING_DIM = 120
NUM_HEADS = 3
FFN_HIDDEN_DIM = 480
NUM_LAYERS = 4

CHECKPOINT_PATH = Path("checkpoints/student_v1/student_v1_final.pt")

DEVICE = "cpu"


# ============================================================
# Évaluation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)

        loss = loss_function(logits, labels)

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Le DataLoader de validation est vide.")

    return total_loss / total_batches


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("ÉVALUATION STUDENT V1")
    print("=" * 60)

    device = torch.device(DEVICE)

    # --------------------------------------------------------
    # Vérification du checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {CHECKPOINT_PATH}")

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    validation_dataloader = (create_validation_dataloader())

    print("\nDataset de validation :")
    print(f"Nombre de séquences : " f"{len(validation_dataloader.dataset)}")

    print(f"Nombre de batches : " f"{len(validation_dataloader)}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    vocab_size = get_vocab_size()

    print(f"\nVocabulary size : {vocab_size}")

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Paramètres : {parameter_count:,}")

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    print(f"\nCheckpoint chargé : " f"{CHECKPOINT_PATH}")

    print(f"Global step : " f"{checkpoint['global_step']}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_function = CausalLanguageModelingLoss()

    # --------------------------------------------------------
    # Évaluation
    # --------------------------------------------------------

    print("\nÉvaluation en cours...")

    validation_loss = evaluate(
        model=model,
        dataloader=validation_dataloader,
        loss_function=loss_function,
        device=device,
    )

    # --------------------------------------------------------
    # Perplexité
    # --------------------------------------------------------

    perplexity = torch.exp(torch.tensor(validation_loss)).item()

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RÉSULTATS")
    print("=" * 60)

    print(f"\nValidation Loss : " f"{validation_loss:.4f}")

    print(f"Perplexité       : " f"{perplexity:.4f}")

    print("\n" + "=" * 60)
    print("✓ ÉVALUATION TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    main()
