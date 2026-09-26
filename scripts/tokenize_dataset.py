import json
from pathlib import Path

from llm.tokenizer.tokenizer import load_tokenizer

from llm.config.parameters import (
    TRAIN_FILE,
    VALIDATION_FILE,
    TRAIN_OUTPUT_FILE,
    VALIDATION_OUTPUT_FILE,
    BLOCK_SIZE,
)


# ============================================================
# LECTURE DU DATASET
# ============================================================

def read_documents(
    file_path: Path,
):
    """
    Lit les documents JSONL et retourne
    uniquement les textes valides.
    """

    documents = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)

            except json.JSONDecodeError as error:

                print(
                    f"[WARNING] "
                    f"Ligne {line_number} ignorée : "
                    f"{error}"
                )

                continue

            text = document.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            documents.append(text)

    return documents


# ============================================================
# TOKENISATION
# ============================================================

def tokenize_text(
    tokenizer,
    text: str,
):
    """
    Transforme un texte en IDs de tokens.
    """

    encoding = tokenizer.encode(text)

    return encoding.ids


# ============================================================
# CRÉATION DES SÉQUENCES
# ============================================================

def create_sequences(
    token_ids,
):
    """
    Découpe les tokens en séquences de :

        BLOCK_SIZE + 1

    Exemple avec BLOCK_SIZE = 128 :

        [token0 ... token128]

    Puis :

        input_ids = [token0 ... token127]
        labels    = [token1 ... token128]

    Les séquences incomplètes sont supprimées.
    """

    sequence_length = BLOCK_SIZE + 1

    sequences = []

    for start in range(
        0,
        len(token_ids) - sequence_length + 1,
        BLOCK_SIZE,
    ):

        sequence = token_ids[
            start:start + sequence_length
        ]

        # ----------------------------------------------------
        # Sécurité supplémentaire
        # ----------------------------------------------------

        if len(sequence) != sequence_length:
            continue

        input_ids = sequence[:-1]

        labels = sequence[1:]

        # ----------------------------------------------------
        # Vérification
        # ----------------------------------------------------

        if len(input_ids) != BLOCK_SIZE:
            continue

        if len(labels) != BLOCK_SIZE:
            continue

        sequences.append(
            {
                "input_ids": input_ids,
                "labels": labels,
            }
        )

    return sequences


# ============================================================
# TRAITEMENT D'UN FICHIER
# ============================================================

def process_dataset(
    tokenizer,
    input_file: Path,
    output_file: Path,
):
    """
    Tokenise un fichier JSONL et génère
    un fichier JSONL contenant les séquences.
    """

    print("\n" + "=" * 60)

    print(
        f"TRAITEMENT : {input_file}"
    )

    print("=" * 60)

    documents = read_documents(
        input_file
    )

    print(
        f"Documents lus : {len(documents)}"
    )

    total_tokens = 0
    total_sequences = 0

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as output:

        for document_index, text in enumerate(
            documents,
            start=1,
        ):

            # ----------------------------------------------
            # Tokenisation
            # ----------------------------------------------

            token_ids = tokenize_text(
                tokenizer,
                text,
            )

            total_tokens += len(token_ids)

            # ----------------------------------------------
            # Création des séquences
            # ----------------------------------------------

            sequences = create_sequences(
                token_ids
            )

            # ----------------------------------------------
            # Écriture
            # ----------------------------------------------

            for sequence in sequences:

                output.write(
                    json.dumps(
                        sequence,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_sequences += 1

    print(
        f"Tokens totaux      : "
        f"{total_tokens:,}"
    )

    print(
        f"Séquences créées   : "
        f"{total_sequences:,}"
    )

    print(
        f"Block size          : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Longueur input_ids  : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Longueur labels     : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Fichier             : "
        f"{output_file}"
    )

    return total_sequences


# ============================================================
# VÉRIFICATION DU FICHIER
# ============================================================

def verify_dataset(
    file_path: Path,
):
    """
    Vérifie que toutes les séquences ont
    exactement BLOCK_SIZE tokens.
    """

    print("\n" + "=" * 60)

    print(
        f"VÉRIFICATION : {file_path}"
    )

    print("=" * 60)

    number_of_sequences = 0

    invalid_sequences = 0

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)

            except json.JSONDecodeError:

                print(
                    f"[ERROR] "
                    f"Ligne {line_number} invalide."
                )

                invalid_sequences += 1

                continue

            input_ids = sample.get(
                "input_ids"
            )

            labels = sample.get(
                "labels"
            )

            number_of_sequences += 1

            # ----------------------------------------------
            # Vérification input_ids
            # ----------------------------------------------

            if (
                not isinstance(
                    input_ids,
                    list,
                )
                or len(input_ids) != BLOCK_SIZE
            ):

                print(
                    f"[ERROR] "
                    f"Ligne {line_number} : "
                    f"input_ids = "
                    f"{len(input_ids) if isinstance(input_ids, list) else 'INVALID'}"
                )

                invalid_sequences += 1

            # ----------------------------------------------
            # Vérification labels
            # ----------------------------------------------

            if (
                not isinstance(
                    labels,
                    list,
                )
                or len(labels) != BLOCK_SIZE
            ):

                print(
                    f"[ERROR] "
                    f"Ligne {line_number} : "
                    f"labels = "
                    f"{len(labels) if isinstance(labels, list) else 'INVALID'}"
                )

                invalid_sequences += 1

            # ----------------------------------------------
            # Vérification du décalage
            # ----------------------------------------------

            if (
                isinstance(input_ids, list)
                and isinstance(labels, list)
                and len(input_ids) == BLOCK_SIZE
                and len(labels) == BLOCK_SIZE
            ):

                if input_ids[1:] != labels[:-1]:

                    print(
                        f"[ERROR] "
                        f"Ligne {line_number} : "
                        f"input_ids / labels "
                        f"mal décalés."
                    )

                    invalid_sequences += 1

    print(
        f"\nSéquences vérifiées : "
        f"{number_of_sequences:,}"
    )

    print(
        f"Erreurs              : "
        f"{invalid_sequences:,}"
    )

    if invalid_sequences == 0:

        print(
            "✓ Dataset valide"
        )

    else:

        raise ValueError(
            "Le dataset contient "
            "des séquences invalides."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TOKENISATION DU DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Vérification des fichiers sources
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Dataset train introuvable : "
            f"{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Dataset validation introuvable : "
            f"{VALIDATION_FILE}"
        )

    # --------------------------------------------------------
    # Chargement du tokenizer
    # --------------------------------------------------------

    print(
        "\nChargement du tokenizer..."
    )

    tokenizer = load_tokenizer()

    vocab_size = tokenizer.get_vocab_size()

    print(
        f"Vocabulaire : "
        f"{vocab_size}"
    )

    print(
        f"Block size : "
        f"{BLOCK_SIZE}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_sequences = process_dataset(
        tokenizer=tokenizer,
        input_file=TRAIN_FILE,
        output_file=TRAIN_OUTPUT_FILE,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_sequences = process_dataset(
        tokenizer=tokenizer,
        input_file=VALIDATION_FILE,
        output_file=VALIDATION_OUTPUT_FILE,
    )

    # --------------------------------------------------------
    # Vérification Train
    # --------------------------------------------------------

    verify_dataset(
        TRAIN_OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Vérification Validation
    # --------------------------------------------------------

    verify_dataset(
        VALIDATION_OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOKENISATION TERMINÉE")
    print("=" * 60)

    print(
        f"\nVocabulaire : "
        f"{vocab_size}"
    )

    print(
        f"Train      : "
        f"{train_sequences:,} séquences"
    )

    print(
        f"Validation : "
        f"{validation_sequences:,} séquences"
    )

    print(
        f"\nFichiers générés :"
    )

    print(
        f"  → {TRAIN_OUTPUT_FILE}"
    )

    print(
        f"  → {VALIDATION_OUTPUT_FILE}"
    )

    print(
        "\n✓ Toutes les séquences ont "
        f"{BLOCK_SIZE} input_ids et "
        f"{BLOCK_SIZE} labels."
    )

    # ========================================================
    # UPLOAD VERS HUGGING FACE
    # ========================================================

    try:

        from scripts.hub.hub_sync import (
            upload_file_to_hub,
        )

        print()
        print("=" * 70)
        print("UPLOAD VERS HUGGING FACE")
        print("=" * 70)

        upload_file_to_hub(
            TRAIN_OUTPUT_FILE,
            remote_path=(
                f"data/tokenized/{TRAIN_OUTPUT_FILE.name}"
            ),
        )

        upload_file_to_hub(
            VALIDATION_OUTPUT_FILE,
            remote_path=(
                f"data/tokenized/{VALIDATION_OUTPUT_FILE.name}"
            ),
        )

    except Exception as error:

        print(
            f"\n⚠️ Upload dataset tokenisé échoué : "
            f"{error}"
        )


if __name__ == "__main__":
    main()