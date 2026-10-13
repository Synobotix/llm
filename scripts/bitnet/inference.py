from pathlib import Path
from scripts.duration import TrainingTimer

import torch

from llm.inference.generate_bitnet import (
    load_tokenizer,
    load_checkpoint,
    create_bitnet_model,
    generate,
)

from scripts.prompt import PROMPT


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_FILE = Path("checkpoints/bitnet/entrainement_bitnet_2.pt")


# ============================================================
# PARAMÈTRES DE GÉNÉRATION
# ============================================================

MAX_NEW_TOKENS = 50

TEMPERATURE = 1.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print("INFERENCE BITNET")

    print("=" * 60)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice : {device}")

    if device.type == "cuda":

        print(f"GPU : " f"{torch.cuda.get_device_name(0)}")

    # ========================================================
    # TOKENIZER
    # ========================================================

    print("\nChargement du tokenizer...")

    tokenizer = load_tokenizer()

    print("Tokenizer chargé.")

    print(f"Vocabulaire tokenizer : " f"{tokenizer.get_vocab_size()}")

    # ========================================================
    # CHECKPOINT
    # ========================================================

    print("\nChargement du checkpoint...")

    checkpoint = load_checkpoint(CHECKPOINT_FILE)

    print(f"Checkpoint chargé : " f"{CHECKPOINT_FILE}")

    # ========================================================
    # MODÈLE
    # ========================================================

    print("\nCréation du modèle BitNet...")

    (
        transformer,
        lm_head,
    ) = create_bitnet_model(
        checkpoint=checkpoint,
        device=device,
    )

    print("Modèle BitNet chargé.")

    # ========================================================
    # CONFIGURATION DU MODÈLE
    # ========================================================

    config = checkpoint["config"]

    print("\nConfiguration :")

    print(f"  vocab_size       : " f"{config['vocab_size']}")

    print(f"  d_model          : " f"{config['d_model']}")

    print(f"  num_heads        : " f"{config['num_heads']}")

    print(f"  hidden_dim       : " f"{config['hidden_dim']}")

    print(f"  num_blocks       : " f"{config['num_blocks']}")

    print(f"  context length   : " f"{config['max_sequence_length']}")

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = PROMPT

    # ========================================================
    # AFFICHAGE DU PROMPT
    # ========================================================

    print("\n" + "-" * 60)

    print("PROMPT")

    print("-" * 60)

    print(prompt)

    # ========================================================
    # TOKENIZATION DU PROMPT
    # ========================================================

    encoding = tokenizer.encode(prompt)

    print("\n" + "-" * 60)

    print("TOKENIZATION")

    print("-" * 60)

    print(f"Tokens : " f"{encoding.tokens}")

    print(f"IDs : " f"{encoding.ids}")

    print(f"Nombre de tokens : " f"{len(encoding.ids)}")

    # ========================================================
    # GÉNÉRATION
    # ========================================================

    print("\n" + "-" * 60)

    print("GÉNÉRATION")

    print("-" * 60)

    timer = TrainingTimer()

    timer.start()

    # génération
    generated_text = generate(
        transformer=transformer,
        lm_head=lm_head,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    duration = timer.stop()

    print(f"Durée de l'inférence : " f"{timer.format_duration(duration)}")

    # ========================================================
    # RÉSULTAT
    # ========================================================

    print("\n" + "=" * 60)

    print("TEXTE GÉNÉRÉ")

    print("=" * 60)

    print(generated_text)

    print("\n" + "=" * 60)

    print("FIN DE L'INFERENCE")

    print("=" * 60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":

    main()
