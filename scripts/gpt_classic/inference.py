import sys
from pathlib import Path

import torch

from scripts.prompt import PROMPT
from scripts.duration import TrainingTimer

# ============================================================
# PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from llm.config.parameters import (
    D_MODEL,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_BLOCKS,
    MAX_SEQUENCE_LENGTH,
    TOKENIZER_CONFIG_FILE,
)

from llm.model.transformer import Transformer
from llm.tokenizer.tokenizer import load_tokenizer


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = Path("checkpoints/gpt_classic/test1.pt")

MAX_NEW_TOKENS = 50


# ============================================================
# VOCABULAIRE
# ============================================================

def get_vocab_size():

    import json

    with TOKENIZER_CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return config["actual_vocab_size"]


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {CHECKPOINT_PATH}")

    vocab_size = get_vocab_size()

    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=D_MODEL,
        num_heads=NUM_HEADS,
        ffn_hidden_dim=HIDDEN_DIM,
        num_layers=NUM_BLOCKS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# GÉNÉRATION
# ============================================================

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens):

    input_ids = tokenizer.encode(prompt).ids

    if not input_ids:
        raise ValueError("Le prompt ne contient aucun token.")

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):

        context = input_ids[:, -MAX_SEQUENCE_LENGTH:]

        logits = model(context)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INFERENCE - GPT CLASSIC V1")
    print("=" * 70)

    tokenizer = load_tokenizer()
    model = load_model()

    print()
    print("Prompt :")
    print(PROMPT)

    timer = TrainingTimer()
    timer.start()

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    duration = timer.stop()

    print()
    print("Génération :")
    print(generated_text)

    print()
    print(f"Durée de l'inférence : " f"{timer.format_duration(duration)}")

    print("=" * 70)
    print("TEST TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
