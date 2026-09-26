import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# DATASET
# ============================================================

MAX_DOCUMENT = int(os.getenv("MAX_DOCUMENT", 70000))
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", 0.80))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


# ============================================================
# TOKENIZER
# ============================================================

# Taille maximale demandée au tokenizer (≠ taille réelle du vocabulaire).
MAX_VOCAB_SIZE = int(os.getenv("MAX_VOCAB_SIZE", 20000))
MIN_FREQUENCY = int(os.getenv("MIN_FREQUENCY", 2))
SPECIAL_TOKENS = os.getenv(
    "SPECIAL_TOKENS", "<pad>,<unk>,<bos>,<eos>"
).split(",")


# ============================================================
# TOKENISATION
# ============================================================

BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", 512))


# ============================================================
# BITNET / TRANSFORMER
# ============================================================

D_MODEL = int(os.getenv("D_MODEL", 640))
NUM_HEADS = int(os.getenv("NUM_HEADS", 4))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", 500))
NUM_BLOCKS = int(os.getenv("NUM_BLOCKS", 6))
MAX_SEQUENCE_LENGTH = BLOCK_SIZE


# ============================================================
# ENTRAÎNEMENT
# ============================================================

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 6))
EPOCHS = int(os.getenv("EPOCHS", 2))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-5))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
GRADIENT_CLIP = float(os.getenv("GRADIENT_CLIP", 1.0))


# ============================================================
# SAUVEGARDE PÉRIODIQUE
# ============================================================

SAVE_EVERY_N_STEPS = int(os.getenv("SAVE_EVERY_N_STEPS", 1000))
AUTO_CLEAN_STEP_CHECKPOINTS = (
    os.getenv("AUTO_CLEAN_STEP_CHECKPOINTS", "true").lower() == "true"
)


# ============================================================
# CHEMINS
# ============================================================

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", DATA_DIR / "processed"))
TOKENIZED_DIR = Path(os.getenv("TOKENIZED_DIR", DATA_DIR / "tokenized"))
TOKENIZER_DIR = Path(os.getenv("TOKENIZER_DIR", "tokenizer"))

CLEAN_DATASET_FILE = PROCESSED_DIR / "c4_clean.jsonl"
TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VALIDATION_FILE = PROCESSED_DIR / "validation.jsonl"

TRAIN_OUTPUT_FILE = TOKENIZED_DIR / "train.jsonl"
VALIDATION_OUTPUT_FILE = TOKENIZED_DIR / "validation.jsonl"

TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"
TOKENIZER_VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
TOKENIZER_CONFIG_FILE = TOKENIZER_DIR / "config.json"


# ============================================================
# AFFICHAGE
# ============================================================

def print_parameters():
    """
    Affiche les paramètres principaux du projet.
    """

    print("=" * 70)
    print("PARAMÈTRES DU PROJET BITNET")
    print("=" * 70)

    print()
    print("DATASET")
    print(f"  Documents maximum : {MAX_DOCUMENT}")
    print(f"  Train ratio       : {TRAIN_RATIO}")
    print(f"  Random seed       : {RANDOM_SEED}")

    print()
    print("TOKENIZER")
    print(f"  Max vocab size    : {MAX_VOCAB_SIZE}")
    print(f"  Min frequency     : {MIN_FREQUENCY}")

    print()
    print("TOKENISATION")
    print(f"  Block size        : {BLOCK_SIZE}")

    print()
    print("BITNET")
    print(f"  D model           : {D_MODEL}")
    print(f"  Num heads         : {NUM_HEADS}")
    print(f"  Hidden dim        : {HIDDEN_DIM}")
    print(f"  Num blocks        : {NUM_BLOCKS}")
    print(f"  Max sequence      : {MAX_SEQUENCE_LENGTH}")

    print()
    print("ENTRAÎNEMENT")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Epochs            : {EPOCHS}")
    print(f"  Learning rate     : {LEARNING_RATE}")
    print(f"  Weight decay      : {WEIGHT_DECAY}")
    print(f"  Gradient clip     : {GRADIENT_CLIP}")

    print("=" * 70)
