import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

MAX_DOCUMENTS = 1000

OUTPUT_FILE = Path("data/raw/c4_1000.jsonl")


# ============================================================
# CONVERSION TIMESTAMP
# ============================================================

def serialize_timestamp(timestamp):
    """
    Convertit le timestamp C4 en valeur compatible JSON.
    """

    if timestamp is None:
        return None

    if hasattr(timestamp, "isoformat"):
        return timestamp.isoformat()

    return str(timestamp)


# ============================================================
# RÉCUPÉRATION C4
# ============================================================

def load_c4():

    return load_dataset(
        "allenai/c4",
        "fr",
        split="train",
        streaming=True,
    )


# ============================================================
# PRÉPARATION
# ============================================================

def prepare_c4():

    dataset = load_c4()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents_saved = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        progress = tqdm(
            total=MAX_DOCUMENTS,
            desc="Récupération C4",
            unit="doc",
        )

        for example in dataset:

            document = {
                "text": example.get("text", ""),
                "url": example.get("url", ""),
                "timestamp": serialize_timestamp(
                    example.get("timestamp")
                ),
            }

            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

            documents_saved += 1

            progress.update(1)

            if documents_saved >= MAX_DOCUMENTS:
                break

        progress.close()

    # ========================================================
    # STATISTIQUES
    # ========================================================

    print()
    print("=" * 60)
    print("RÉCUPÉRATION TERMINÉE")
    print("=" * 60)

    print(f"Documents récupérés : {documents_saved}")
    print(f"Fichier             : {OUTPUT_FILE}")
    print("=" * 60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    prepare_c4()