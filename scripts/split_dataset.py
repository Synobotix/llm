import json
import random
from pathlib import Path

from llm.config.parameters import (
    CLEAN_DATASET_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
    MAX_DOCUMENT,
    TRAIN_RATIO,
    RANDOM_SEED,
)


def read_documents(file_path):

    documents = []

    with file_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)

            except json.JSONDecodeError as error:

                print(f"[WARNING] Ligne {line_number} ignorée : " f"{error}")

                continue

            text = document.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            documents.append(document)

    return documents


def main():

    print("=" * 70)
    print("SPLIT DU DATASET")
    print("=" * 70)

    print()
    print(f"Dataset source : {CLEAN_DATASET_FILE}")
    print(f"Documents demandés : {MAX_DOCUMENT}")
    print(f"Train ratio : {TRAIN_RATIO}")
    print(f"Seed : {RANDOM_SEED}")

    if not CLEAN_DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset introuvable : "
            f"{CLEAN_DATASET_FILE}"
        )

    documents = read_documents(CLEAN_DATASET_FILE)

    print()
    print(f"Documents disponibles : " f"{len(documents)}")

    if MAX_DOCUMENT > len(documents):

        print(
            f"\n⚠️ MAX_DOCUMENT={MAX_DOCUMENT} "
            f"mais seulement "
            f"{len(documents)} documents "
            f"sont disponibles."
        )
        print(f"→ Utilisation de {len(documents)} documents.\n")

        effective_max = len(documents)

    else:

        effective_max = MAX_DOCUMENT

    random.seed(RANDOM_SEED)

    random.shuffle(documents)

    documents = documents[:effective_max]

    train_size = int(len(documents) * TRAIN_RATIO)

    validation_size = (len(documents) - train_size)

    train_documents = documents[:train_size]

    validation_documents = documents[train_size:]

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with TRAIN_FILE.open("w", encoding="utf-8") as file:

        for document in train_documents:

            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    with VALIDATION_FILE.open("w", encoding="utf-8") as file:

        for document in validation_documents:

            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    print()
    print("=" * 70)
    print("SPLIT TERMINÉ")
    print("=" * 70)

    print()
    print(f"Total      : {len(documents)}")
    print(f"Train      : {len(train_documents)}")
    print(f"Validation : {len(validation_documents)}")

    print()
    print(f"→ {TRAIN_FILE}")
    print(f"→ {VALIDATION_FILE}")

    # ========================================================
    # UPLOAD VERS HUGGING FACE
    # ========================================================

    try:

        from scripts.hub.hub_sync import (upload_file_to_hub)

        print()
        print("=" * 70)
        print("UPLOAD VERS HUGGING FACE")
        print("=" * 70)

        upload_file_to_hub(
            TRAIN_FILE,
            remote_path=(
                f"data/processed/{TRAIN_FILE.name}"
            ),
        )

        upload_file_to_hub(
            VALIDATION_FILE,
            remote_path=(
                f"data/processed/{VALIDATION_FILE.name}"
            ),
        )

    except Exception as error:

        print(f"\n⚠️ Upload split échoué : {error}")


if __name__ == "__main__":
    main()
