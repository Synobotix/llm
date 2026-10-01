import sys
from pathlib import Path


# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from llm.config.parameters import (
    D_MODEL,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_BLOCKS,
    MAX_SEQUENCE_LENGTH,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    RANDOM_SEED,
    TOKENIZER_CONFIG_FILE,
)

from scripts.duration import TrainingTimer

from scripts.gpt_classic.checkpoint_path_gpt_classic import (
    get_checkpoint_paths,
)

from scripts.gpt_classic.save_result_gpt_classic import (save_training_result)

from llm.data.dataloader import (
    create_train_dataloader,
    create_validation_dataloader,
)

from llm.model.transformer import Transformer

from llm.training.checkpoint import load_checkpoint


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# RÉCUPÉRATION DU VOCABULAIRE RÉEL
# ============================================================

def get_vocab_size() -> int:
    """
    Récupère la taille réelle du vocabulaire
    depuis le config.json généré par le tokenizer.
    """

    if not TOKENIZER_CONFIG_FILE.exists():

        raise FileNotFoundError(
            "Configuration du tokenizer introuvable : "
            f"{TOKENIZER_CONFIG_FILE}"
        )

    with TOKENIZER_CONFIG_FILE.open("r", encoding="utf-8") as file:

        config = json.load(file)

    if "actual_vocab_size" not in config:

        raise KeyError(
            "La clé 'actual_vocab_size' est absente de "
            f"{TOKENIZER_CONFIG_FILE}"
        )

    vocab_size = config["actual_vocab_size"]

    if not isinstance(vocab_size, int):

        raise TypeError("'actual_vocab_size' doit être un entier.")

    if vocab_size <= 0:

        raise ValueError("'actual_vocab_size' doit être supérieur à 0.")

    return vocab_size


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# NOMBRE DE PARAMÈTRES
# ============================================================

def count_parameters(model):

    return sum(parameter.numel() for parameter in model.parameters())


# ============================================================
# PARAMÈTRES ENTRAÎNABLES
# ============================================================

def print_trainable_parameters(model):

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    frozen_parameters = [p for p in model.parameters() if not p.requires_grad]

    total_trainable = sum(p.numel() for p in trainable_parameters)

    total_frozen = sum(p.numel() for p in frozen_parameters)

    total_parameters = (total_trainable + total_frozen)

    trainable_ratio = (
        (total_trainable / total_parameters * 100.0)
        if total_parameters > 0
        else 0.0
    )

    modules_info = []

    for name, module in model.named_children():

        module_total = sum(p.numel() for p in module.parameters())

        module_trainable = sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )

        modules_info.append(
            {
                "name": name,
                "total": module_total,
                "trainable": module_trainable,
            }
        )

    # ---------- Affichage terminal ----------

    print("\n" + "=" * 70)

    print("PARAMÈTRES ENTRAÎNABLES")

    print("=" * 70)

    print(f"\nTenseurs entraînables  : " f"{len(trainable_parameters)}")

    print(f"Tenseurs gelés         : " f"{len(frozen_parameters)}")

    print(f"\nParamètres entraînables : " f"{total_trainable:,}")

    print(f"Paramètres gelés        : " f"{total_frozen:,}")

    print(f"Paramètres totaux       : " f"{total_parameters:,}")

    if total_parameters > 0:

        print(f"Ratio entraînable       : " f"{trainable_ratio:.4f} %")

    print("\nDétail par sous-module :")

    print("-" * 70)

    for module in modules_info:

        print(
            f"  {module['name']:<30} "
            f"total = {module['total']:>12,}  "
            f"entraînable = {module['trainable']:>12,}"
        )

    print("-" * 70)

    # ---------- Retour pour le rapport Markdown ----------

    return {
        "trainable_tensors": len(trainable_parameters),
        "frozen_tensors": len(frozen_parameters),
        "trainable_parameters": total_trainable,
        "frozen_parameters": total_frozen,
        "total_parameters": total_parameters,
        "trainable_ratio": f"{trainable_ratio:.4f} %",
        "modules": modules_info,
    }


# ============================================================
# PERPLEXITÉ
# ============================================================

def calculate_perplexity(loss):

    try:

        return math.exp(loss)

    except OverflowError:

        return float("inf")


# ============================================================
# NORME DU GRADIENT
# ============================================================

def calculate_gradient_norm(model):

    total_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            parameter_norm = (parameter.grad.detach() .data.norm(2) .item())

            total_norm += (parameter_norm ** 2)

    return math.sqrt(total_norm)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    gradient_clip,
    global_step,
):

    model.train()

    total_loss = 0.0

    total_batches = 0

    total_gradient_norm = 0.0

    for batch in dataloader:

        input_ids = (batch["input_ids"] .to(device))

        labels = (batch["labels"] .to(device))

        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            labels.reshape(-1),
        )

        loss.backward()

        gradient_norm = (calculate_gradient_norm(model))

        total_gradient_norm += (gradient_norm)

        if gradient_clip is not None:

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        if scheduler is not None:

            scheduler.step()

        total_loss += loss.item()

        total_batches += 1

        global_step += 1

    if total_batches == 0:

        raise RuntimeError("Le DataLoader d'entraînement est vide.")

    average_loss = (total_loss / total_batches)

    average_gradient_norm = (total_gradient_norm / total_batches)

    perplexity = (calculate_perplexity(average_loss))

    return (average_loss, perplexity, average_gradient_norm, global_step)


# ============================================================
# VALIDATION
# ============================================================

def validate(model, dataloader, device):

    model.eval()

    total_loss = 0.0

    total_batches = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = (batch["input_ids"] .to(device))

            labels = (batch["labels"] .to(device))

            logits = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
            )

            total_loss += (loss.item())

            total_batches += 1

    if total_batches == 0:

        raise RuntimeError("Le DataLoader de validation est vide.")

    average_loss = (total_loss / total_batches)

    perplexity = (calculate_perplexity(average_loss))

    return (average_loss, perplexity)


# ============================================================
# SAUVEGARDE DU CHECKPOINT
# ============================================================

def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    training_results,
    configuration,
):

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {

        "epoch": epoch,

        "global_step": global_step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "training_results":
            training_results,

        "configuration":
            configuration,
    }

    torch.save(checkpoint, checkpoint_path)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("ENTRAÎNEMENT GPT CLASSIC V1")
    print("=" * 70)

    print(f"\nDevice : {DEVICE}")

    if DEVICE.type == "cuda":

        print(f"GPU : " f"{torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # CHECKPOINTS
    # --------------------------------------------------------

    (checkpoint_to_load, checkpoint_to_save) = get_checkpoint_paths()

    print("\nCheckpoint à charger : " f"{checkpoint_to_load}")

    print("Checkpoint à sauvegarder : " f"{checkpoint_to_save}")

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nChargement du tokenizer...")

    VOCAB_SIZE = (get_vocab_size())

    print(f"Vocabulaire réel : " f"{VOCAB_SIZE}")

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    print("\nCréation des DataLoaders...")

    train_dataloader = (create_train_dataloader())

    validation_dataloader = (create_validation_dataloader())

    print(f"Train batches      : " f"{len(train_dataloader)}")

    print(f"Validation batches : " f"{len(validation_dataloader)}")

    print(f"Batch size         : " f"{train_dataloader.batch_size}")

    print(f"Sequence length    : " f"{MAX_SEQUENCE_LENGTH}")

    # --------------------------------------------------------
    # TRANSFORMER
    # --------------------------------------------------------

    print("\nCréation du Transformer...")

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=HIDDEN_DIM,
        num_layers=NUM_BLOCKS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    ).to(DEVICE)

    total_parameters = (count_parameters(model))

    print(f"\nParamètres Transformer : " f"{total_parameters:,}")

    print(f"Paramètres totaux     : " f"{total_parameters:,}")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    total_training_steps = (EPOCHS * len(train_dataloader))

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                total_training_steps,
            ),
        )
    )

    # --------------------------------------------------------
    # CHECKPOINT PRÉCÉDENT
    # --------------------------------------------------------

    print("\nChargement du checkpoint précédent...")

    global_step = 0

    start_epoch = 1

    if checkpoint_to_load is not None:

        global_step = (
            load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_to_load,
                device=str(DEVICE),
            )
        )

        print(f"Checkpoint chargé : " f"{checkpoint_to_load}")

        print(f"Global step restauré : " f"{global_step}")

    else:

        print("Aucun checkpoint précédent.")

        print("Début d'un nouvel entraînement.")

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    print("\nConfiguration :")

    print(f"  Vocabulaire         = " f"{VOCAB_SIZE}")

    print(f"  D model             = " f"{D_MODEL}")

    print(f"  Num heads           = " f"{NUM_HEADS}")

    print(f"  Hidden dim          = " f"{HIDDEN_DIM}")

    print(f"  Num blocks          = " f"{NUM_BLOCKS}")

    print(f"  Max sequence        = " f"{MAX_SEQUENCE_LENGTH}")

    print(f"  Batch size          = " f"{train_dataloader.batch_size}")

    print(f"  Nouvelles époques   = " f"{EPOCHS}")

    print(f"  Learning rate       = " f"{LEARNING_RATE}")

    print(f"  Weight decay        = " f"{WEIGHT_DECAY}")

    print(f"  Gradient clip       = " f"{GRADIENT_CLIP}")

    print(f"  Ancien checkpoint   = " f"{checkpoint_to_load}")

    print(f"  Nouveau checkpoint  = " f"{checkpoint_to_save}")

    print("\n" + "=" * 70)

    print("ENTRAÎNEMENT")

    print("=" * 70)

    print(f"\nGlobal step initial : " f"{global_step}")

    print(f"Époques à effectuer : " f"{EPOCHS}")

    # --------------------------------------------------------
    # CHRONOMÉTRAGE
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("CHRONOMÉTRAGE DE L'ENTRAÎNEMENT")

    print("=" * 70)

    training_timer = (TrainingTimer())

    training_timer.start()

    # --------------------------------------------------------
    # INITIALISATION DES DERNIERS RÉSULTATS
    # --------------------------------------------------------

    last_train_loss = None

    last_validation_loss = None

    last_train_ppl = None

    last_validation_ppl = None

    last_gradient_norm = None

    last_learning_rate = None

    # --------------------------------------------------------
    # ENTRAÎNEMENT
    # --------------------------------------------------------

    for epoch in range(start_epoch, EPOCHS + 1):

        print("\n" + "=" * 70)

        print(f"ÉPOQUE {epoch}/{EPOCHS}")

        print("=" * 70)

        (
            train_loss,
            train_ppl,
            gradient_norm,
            global_step,
        ) = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            gradient_clip=GRADIENT_CLIP,
            global_step=global_step,
        )

        (
            validation_loss,
            validation_ppl,
        ) = validate(
            model=model,
            dataloader=validation_dataloader,
            device=DEVICE,
        )

        print(f"\nTrain loss        : " f"{train_loss:.6f}")

        print(f"Validation loss   : " f"{validation_loss:.6f}")

        print(f"Train PPL         : " f"{train_ppl:.6f}")

        print(f"Validation PPL    : " f"{validation_ppl:.6f}")

        print(f"Gradient norm     : " f"{gradient_norm:.6f}")

        print(f"Global step       : " f"{global_step}")

        print(f"Learning rate     : " f"{scheduler.get_last_lr()[0]:.10f}")

        # ----------------------------------------------------
        # MÉMORISATION DES DERNIERS RÉSULTATS
        # ----------------------------------------------------

        last_train_loss = train_loss

        last_validation_loss = validation_loss

        last_train_ppl = train_ppl

        last_validation_ppl = validation_ppl

        last_gradient_norm = gradient_norm

        last_learning_rate = (scheduler.get_last_lr()[0])

    # --------------------------------------------------------
    # FIN DU CHRONOMÉTRAGE
    # --------------------------------------------------------

    training_duration = (training_timer.stop())

    print("\n" + "=" * 70)

    print("DURÉE TOTALE DE L'ENTRAÎNEMENT")

    print("=" * 70)

    print(
        f"\nDurée totale de l'entraînement : "
        f"{training_timer.format_duration(training_duration)}"
    )

    print(f"Durée totale en secondes : " f"{training_duration:.2f}s")

    # --------------------------------------------------------
    # RÉSULTATS DE L'ENTRAÎNEMENT
    # --------------------------------------------------------

    training_results = {

        "train_loss":
            last_train_loss,

        "validation_loss":
            last_validation_loss,

        "train_ppl":
            last_train_ppl,

        "validation_ppl":
            last_validation_ppl,

        "gradient_norm":
            last_gradient_norm,

        "learning_rate":
            last_learning_rate,

        "duration":
            training_timer.format_duration(
                training_duration
            ),

        "duration_seconds":
            training_duration,
    }

    # --------------------------------------------------------
    # CONFIGURATION DU MODÈLE
    # --------------------------------------------------------

    configuration = {

        "vocab_size":
            VOCAB_SIZE,

        "d_model":
            D_MODEL,

        "num_heads":
            NUM_HEADS,

        "hidden_dim":
            HIDDEN_DIM,

        "num_blocks":
            NUM_BLOCKS,

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "batch_size":
            train_dataloader.batch_size,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRADIENT_CLIP,
    }

    # --------------------------------------------------------
    # SAUVEGARDE DU CHECKPOINT
    # --------------------------------------------------------

    save_checkpoint(
        checkpoint_path=checkpoint_to_save,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=EPOCHS,
        global_step=global_step,
        training_results=training_results,
        configuration=configuration,
    )

    print(f"\nNouveau checkpoint sauvegardé : " f"{checkpoint_to_save}")

    # --------------------------------------------------------
    # PARAMÈTRES ENTRAÎNABLES
    # --------------------------------------------------------

    trainable_info = (print_trainable_parameters(model))

    # --------------------------------------------------------
    # GÉNÉRATION DU RAPPORT MARKDOWN
    # --------------------------------------------------------

    gpu_name = (
        torch.cuda.get_device_name(0)
        if DEVICE.type == "cuda"
        else None
    )

    save_training_result(
        checkpoint_path=checkpoint_to_save,
        trainable_parameters=trainable_info,
        device=str(DEVICE),
        gpu_name=gpu_name,
    )

    # --------------------------------------------------------
    # FIN
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("ENTRAÎNEMENT GPT CLASSIC TERMINÉ")

    print("=" * 70)

    print(f"\nDernier global step : " f"{global_step}")

    print(f"Ancien checkpoint : " f"{checkpoint_to_load}")

    print(f"Nouveau checkpoint : " f"{checkpoint_to_save}")


if __name__ == "__main__":

    main()
