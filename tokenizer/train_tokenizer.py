import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = Path("data/processed/train.jsonl")
TOKENIZER_DIR = Path("tokenizer")

TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"
VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
CONFIG_FILE = TOKENIZER_DIR / "config.json"

VOCAB_SIZE = 16_000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


# ============================================================
# Lecture du dataset
# ============================================================

def text_iterator(file_path: Path):
    """
    Lit train.jsonl et retourne le champ 'text'
    de chaque document.
    """

    with file_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"[WARNING] Ligne {line_number} ignorée : {error}")
                continue
            text = document.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if text:
                yield text


# ============================================================
# Création du tokenizer
# ============================================================

def create_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer


# ============================================================
# Entraînement
# ============================================================

def train_tokenizer(tokenizer):

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(text_iterator(TRAIN_FILE), trainer=trainer)


# ============================================================
# Sauvegarde
# ============================================================

def save_tokenizer(tokenizer):

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # tokenizer.json
    # --------------------------------------------------------

    tokenizer.save(str(TOKENIZER_FILE))

    # --------------------------------------------------------
    # vocab.json
    # --------------------------------------------------------

    vocab = tokenizer.get_vocab()

    sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))

    with VOCAB_FILE.open("w", encoding="utf-8") as file:
        json.dump(sorted_vocab, file, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # config.json
    # --------------------------------------------------------

    config = {
        "type": "BPE",
        "vocab_size": VOCAB_SIZE,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pre_tokenizer": "ByteLevel",
    }

    with CONFIG_FILE.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


# ============================================================
# Test
# ============================================================

def test_tokenizer():

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    text = "Bonjour, je suis un modèle de langage."

    encoding = tokenizer.encode(text)

    print("\n" + "=" * 60)
    print("TEST DU TOKENIZER")
    print("=" * 60)

    print("\nTexte :")
    print(text)

    print("\nTokens :")
    print(encoding.tokens)

    print("\nIDs :")
    print(encoding.ids)

    print("\nNombre de tokens :")
    print(len(encoding.ids))

    print("\nDécodage :")
    print(tokenizer.decode(encoding.ids))


# ============================================================
# Programme principal
# ============================================================

def main():

    print("=" * 60)
    print("ENTRAÎNEMENT DU TOKENIZER BPE")
    print("=" * 60)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Dataset introuvable : {TRAIN_FILE}")

    print(f"\nDataset : {TRAIN_FILE}")
    print(f"Vocabulaire cible : {VOCAB_SIZE}")
    print(f"Fréquence minimale : {MIN_FREQUENCY}")

    print("\nCréation du tokenizer...")

    tokenizer = create_tokenizer()

    print("Entraînement...\n")

    train_tokenizer(tokenizer)

    print("\nSauvegarde...")

    save_tokenizer(tokenizer)

    print("\nTokenizer entraîné avec succès.")

    print(f"Nombre de tokens : " f"{tokenizer.get_vocab_size()}")

    print("\nFichiers créés :")
    print(f"  → {TOKENIZER_FILE}")
    print(f"  → {VOCAB_FILE}")
    print(f"  → {CONFIG_FILE}")

    test_tokenizer()


if __name__ == "__main__":
    main()
