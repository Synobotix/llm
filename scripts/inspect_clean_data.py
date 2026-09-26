
import json
import re
from collections import Counter
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/c4_clean.jsonl")

# Nombre de documents dont on affiche un extrait
SAMPLE_COUNT = 5

# Nombre de mots les plus fréquents à afficher
TOP_WORDS = 20

# Taille maximale des extraits affichés
PREVIEW_LENGTH = 300


# ============================================================
# OUTILS
# ============================================================

def load_documents():
    """Charge les documents du fichier JSONL."""

    documents = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            try:
                document = json.loads(line)
                documents.append(document)

            except json.JSONDecodeError:
                print(
                    f"⚠️ JSON invalide à la ligne {line_number}"
                )

    return documents


def clean_for_word_count(text):
    """Prépare le texte pour compter les mots."""

    return re.findall(
        r"\b[\wÀ-ÿ'-]+\b",
        text.lower(),
    )


def contains_html(text):
    """Détecte la présence éventuelle de balises HTML."""

    return bool(
        re.search(
            r"<[^>]+>",
            text,
        )
    )


def contains_url(text):
    """Détecte la présence éventuelle d'URL."""

    return bool(
        re.search(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE,
        )
    )


def contains_email(text):
    """Détecte la présence éventuelle d'adresse email."""

    return bool(
        re.search(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text,
        )
    )


# ============================================================
# INSPECTION
# ============================================================

def inspect_c4():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    documents = load_documents()

    if not documents:

        print("Aucun document trouvé.")
        return

    # --------------------------------------------------------
    # Statistiques générales
    # --------------------------------------------------------

    lengths = []
    word_counts = []

    html_count = 0
    url_count = 0
    email_count = 0

    all_words = Counter()

    for document in documents:

        text = document.get(
            "text",
            "",
        )

        lengths.append(
            len(text)
        )

        words = clean_for_word_count(
            text
        )

        word_counts.append(
            len(words)
        )

        all_words.update(
            words
        )

        if contains_html(text):
            html_count += 1

        if contains_url(text):
            url_count += 1

        if contains_email(text):
            email_count += 1

    # --------------------------------------------------------
    # Calculs
    # --------------------------------------------------------

    total_documents = len(documents)

    total_characters = sum(
        lengths
    )

    total_words = sum(
        word_counts
    )

    average_length = (
        total_characters / total_documents
    )

    average_words = (
        total_words / total_documents
    )

    min_length = min(lengths)
    max_length = max(lengths)

    min_words = min(word_counts)
    max_words = max(word_counts)

    # --------------------------------------------------------
    # Affichage général
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("INSPECTION DU CORPUS C4 NETTOYÉ")
    print("=" * 60)

    print(
        f"Documents                 : {total_documents}"
    )

    print(
        f"Caractères totaux         : {total_characters:,}"
    )

    print(
        f"Mots totaux               : {total_words:,}"
    )

    print()

    print("LONGUEUR DES DOCUMENTS")
    print("-" * 60)

    print(
        f"Minimum                   : {min_length:,} caractères"
    )

    print(
        f"Moyenne                   : {average_length:,.0f} caractères"
    )

    print(
        f"Maximum                   : {max_length:,} caractères"
    )

    print()

    print("NOMBRE DE MOTS")
    print("-" * 60)

    print(
        f"Minimum                   : {min_words:,} mots"
    )

    print(
        f"Moyenne                   : {average_words:,.0f} mots"
    )

    print(
        f"Maximum                   : {max_words:,} mots"
    )

    # --------------------------------------------------------
    # Vérification du nettoyage
    # --------------------------------------------------------

    print()
    print("VÉRIFICATION DU NETTOYAGE")
    print("-" * 60)

    print(
        f"Documents avec HTML      : {html_count}"
    )

    print(
        f"Documents avec URL       : {url_count}"
    )

    print(
        f"Documents avec email     : {email_count}"
    )

    # --------------------------------------------------------
    # Mots fréquents
    # --------------------------------------------------------

    print()
    print("20 MOTS LES PLUS FRÉQUENTS")
    print("-" * 60)

    for word, count in all_words.most_common(
        TOP_WORDS
    ):

        print(
            f"{word:<20} : {count}"
        )

    # --------------------------------------------------------
    # Extraits
    # --------------------------------------------------------

    print()
    print("EXTRAITS DE DOCUMENTS")
    print("=" * 60)

    samples = documents[
        :SAMPLE_COUNT
    ]

    for index, document in enumerate(
        samples,
        start=1,
    ):

        text = document.get(
            "text",
            "",
        )

        print()
        print(
            f"DOCUMENT {index}"
        )
        print("-" * 60)

        preview = text[
            :PREVIEW_LENGTH
        ]

        if len(text) > PREVIEW_LENGTH:
            preview += "..."

        print(preview)

    # --------------------------------------------------------
    # Documents les plus courts
    # --------------------------------------------------------

    print()
    print("DOCUMENTS LES PLUS COURTS")
    print("=" * 60)

    shortest = sorted(
        documents,
        key=lambda document: len(
            document.get("text", "")
        ),
    )[:5]

    for index, document in enumerate(
        shortest,
        start=1,
    ):

        text = document.get(
            "text",
            "",
        )

        print()
        print(
            f"DOCUMENT COURT {index}"
        )
        print(
            f"Longueur : {len(text)} caractères"
        )

        preview = text[
            :PREVIEW_LENGTH
        ]

        if len(text) > PREVIEW_LENGTH:
            preview += "..."

        print(preview)

    print()
    print("=" * 60)
    print("INSPECTION TERMINÉE")
    print("=" * 60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    inspect_c4()
