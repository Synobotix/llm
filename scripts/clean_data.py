
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/raw/c4_1000.jsonl")
OUTPUT_FILE = Path("data/processed/c4_clean.jsonl")

# Taille minimale d'un document après nettoyage
MIN_TEXT_LENGTH = 200

# Nombre minimum de répétitions consécutives d'un même mot
MAX_CONSECUTIVE_WORD_REPETITIONS = 5

# Nombre minimum de répétitions consécutives d'une même phrase
MAX_REPEATED_PHRASE = 3

# Longueur minimale d'une phrase répétée
MIN_REPEATED_PHRASE_WORDS = 4

# Ratio minimum de caractères alphabétiques
MIN_ALPHA_RATIO = 0.40


# ============================================================
# NETTOYAGE DU TEXTE
# ============================================================

def normalize_unicode(text: str) -> str:
    """Normalise les caractères Unicode."""
    return unicodedata.normalize("NFKC", text)


def remove_html(text: str) -> str:
    """Supprime les éléments HTML."""

    text = html.unescape(text)

    # Supprime les scripts
    text = re.sub(
        r"<(script|style).*?>.*?</\1>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Supprime les commentaires HTML
    text = re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.DOTALL,
    )

    # Supprime les balises HTML restantes
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    return text


def remove_urls(text: str) -> str:
    """Supprime les URLs."""

    return re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )


def remove_emails(text: str) -> str:
    """Supprime les adresses email."""

    return re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text,
    )


def normalize_spaces(text: str) -> str:
    """Normalise les espaces et retours à la ligne."""

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def clean_text(text: str) -> str:
    """Pipeline complet de nettoyage."""

    text = normalize_unicode(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = normalize_spaces(text)

    return text


# ============================================================
# FILTRES DE QUALITÉ
# ============================================================

def is_too_short(text: str) -> bool:
    """Rejette les documents trop courts."""

    return len(text) < MIN_TEXT_LENGTH


def alphabetic_ratio(text: str) -> float:
    """Calcule le ratio de caractères alphabétiques."""

    if not text:
        return 0.0

    alphabetic = sum(
        char.isalpha()
        for char in text
    )

    return alphabetic / len(text)


def has_low_alphabetic_ratio(text: str) -> bool:
    """Détecte les textes contenant trop peu de lettres."""

    return alphabetic_ratio(text) < MIN_ALPHA_RATIO


# ============================================================
# DÉTECTION DES RÉPÉTITIONS ARTIFICIELLES
# ============================================================

# Nombre maximum de répétitions consécutives du même mot
MAX_CONSECUTIVE_WORD_REPETITIONS = 5

# Nombre de répétitions d'un même groupe de mots
MAX_REPEATED_NGRAM = 6

# Taille des groupes de mots analysés
NGRAM_SIZES = (3, 4, 5)

# Nombre minimum de mots dans un document pour appliquer
# la détection de répétition par n-grammes
MIN_WORDS_FOR_NGRAM_CHECK = 50

# Nombre minimum de répétitions d'une même phrase
MAX_REPEATED_PHRASE = 3

# Longueur minimale d'une phrase répétée
MIN_REPEATED_PHRASE_WORDS = 4


def tokenize_words(text: str) -> list[str]:
    """
    Transforme le texte en liste de mots.

    Exemple :

        "Costume gonflable de clown"

    devient :

        ["costume", "gonflable", "de", "clown"]
    """

    return re.findall(
        r"\b[\wÀ-ÿ'-]+\b",
        text.lower(),
    )


def has_consecutive_word_repetition(text: str) -> bool:
    """
    Détecte un même mot répété plusieurs fois consécutivement.

    Exemple détecté :

        acheter acheter acheter acheter acheter

    Exemple normal :

        "Le développement de la plateforme..."
    """

    words = tokenize_words(text)

    if not words:
        return False

    consecutive_count = 1
    previous_word = words[0]

    for word in words[1:]:

        if word == previous_word:
            consecutive_count += 1

            if consecutive_count >= MAX_CONSECUTIVE_WORD_REPETITIONS:
                return True

        else:
            consecutive_count = 1

        previous_word = word

    return False


def normalize_ngram(ngram: tuple[str, ...]) -> tuple[str, ...]:
    """
    Normalise un groupe de mots pour faciliter la comparaison.
    """

    return tuple(
        word.lower().strip()
        for word in ngram
    )


def has_repeated_ngrams(text: str) -> bool:
    """
    Détecte les répétitions artificielles de groupes de mots.

    Exemple suspect :

        costume gonflable clown
        costume gonflable clown
        costume gonflable clown
        costume gonflable clown
        costume gonflable clown
        costume gonflable clown

    Cette méthode est plus efficace que de chercher uniquement
    des phrases identiques.
    """

    words = tokenize_words(text)

    if len(words) < MIN_WORDS_FOR_NGRAM_CHECK:
        return False

    for ngram_size in NGRAM_SIZES:

        ngram_counts = {}

        for i in range(
            len(words) - ngram_size + 1
        ):

            ngram = normalize_ngram(
                tuple(
                    words[
                        i:i + ngram_size
                    ]
                )
            )

            ngram_counts[ngram] = (
                ngram_counts.get(ngram, 0) + 1
            )

            if (
                ngram_counts[ngram]
                >= MAX_REPEATED_NGRAM
            ):
                return True

    return False


def normalize_phrase(phrase: str) -> str:
    """
    Normalise une phrase pour comparer
    les répétitions.
    """

    phrase = phrase.lower()

    phrase = re.sub(
        r"\s+",
        " ",
        phrase,
    )

    return phrase.strip()


def has_repeated_phrases(text: str) -> bool:
    """
    Détecte une phrase complète répétée plusieurs fois.

    Exemple :

        "Achetez notre produit maintenant.
         Achetez notre produit maintenant.
         Achetez notre produit maintenant."
    """

    sentences = re.split(
        r"[.!?]+",
        text,
    )

    normalized_sentences = []

    for sentence in sentences:

        sentence = normalize_phrase(
            sentence
        )

        words = sentence.split()

        if len(words) < MIN_REPEATED_PHRASE_WORDS:
            continue

        normalized_sentences.append(
            sentence
        )

    if not normalized_sentences:
        return False

    counts = {}

    for sentence in normalized_sentences:

        counts[sentence] = (
            counts.get(sentence, 0) + 1
        )

        if (
            counts[sentence]
            >= MAX_REPEATED_PHRASE
        ):
            return True

    return False


def has_excessive_repetition(text: str) -> bool:
    """
    Détecte uniquement les répétitions manifestement
    artificielles.

    Trois niveaux :

    1. Mot répété consécutivement
    2. Groupe de mots répété
    3. Phrase répétée
    """

    # --------------------------------------------------------
    # Niveau 1 : même mot répété
    # --------------------------------------------------------

    if has_consecutive_word_repetition(text):
        return True

    # --------------------------------------------------------
    # Niveau 2 : même groupe de mots répété
    # --------------------------------------------------------

    if has_repeated_ngrams(text):
        return True

    # --------------------------------------------------------
    # Niveau 3 : même phrase répétée
    # --------------------------------------------------------

    if has_repeated_phrases(text):
        return True

    return False



# ============================================================
# DÉDUPLICATION
# ============================================================

def text_hash(text: str) -> str:
    """Calcule une empreinte SHA-256 du texte."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# TRAITEMENT PRINCIPAL
# ============================================================

def clean_c4():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    saved = 0

    removed_empty = 0
    removed_short = 0
    removed_alpha = 0
    removed_repetition = 0
    """ removed_spam = 0 """
    removed_duplicate = 0

    seen_hashes = set()

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as input_file, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:

        for line in input_file:

            total += 1

            # ------------------------------------------------
            # Lecture JSON
            # ------------------------------------------------

            try:

                document = json.loads(line)

            except json.JSONDecodeError:

                continue

            original_text = document.get(
                "text",
                "",
            )

            if not original_text:

                removed_empty += 1
                continue

            # ------------------------------------------------
            # Nettoyage
            # ------------------------------------------------

            cleaned = clean_text(
                original_text
            )

            # ------------------------------------------------
            # Filtre texte vide
            # ------------------------------------------------

            if not cleaned:

                removed_empty += 1
                continue

            # ------------------------------------------------
            # Filtre longueur
            # ------------------------------------------------

            if is_too_short(cleaned):

                removed_short += 1
                continue

            # ------------------------------------------------
            # Ratio alphabétique
            # ------------------------------------------------

            if has_low_alphabetic_ratio(cleaned):

                removed_alpha += 1
                continue

            # ------------------------------------------------
            # Répétitions anormales
            # ------------------------------------------------

            if has_excessive_repetition(cleaned):

                removed_repetition += 1
                continue


            # ------------------------------------------------
            # Déduplication
            # ------------------------------------------------

            document_hash = text_hash(
                cleaned
            )

            if document_hash in seen_hashes:

                removed_duplicate += 1
                continue

            seen_hashes.add(
                document_hash
            )

            # ------------------------------------------------
            # Document final
            # ------------------------------------------------

            cleaned_document = {
                "text": cleaned,
                "url": document.get(
                    "url",
                    "",
                ),
                "timestamp": document.get(
                    "timestamp"
                ),
            }

            output_file.write(
                json.dumps(
                    cleaned_document,
                    ensure_ascii=False,
                )
                + "\n"
            )

            saved += 1

    # ========================================================
    # STATISTIQUES
    # ========================================================

    removed_total = total - saved

    print()
    print("=" * 60)
    print("NETTOYAGE C4 TERMINÉ")
    print("=" * 60)

    print(
        f"Documents analysés       : {total}"
    )

    print(
        f"Documents conservés      : {saved}"
    )

    print(
        f"Documents supprimés      : {removed_total}"
    )

    print()
    print("DÉTAIL DES SUPPRESSIONS")
    print("-" * 60)

    print(
        f"Vides                    : {removed_empty}"
    )

    print(
        f"Trop courts              : {removed_short}"
    )

    print(
        f"Ratio alphabétique faible: {removed_alpha}"
    )

    print(
        f"Répétition excessive     : {removed_repetition}"
    )

    """ print(
        f"Spam / SEO               : {removed_spam}"
    ) """

    print(
        f"Doublons                 : {removed_duplicate}"
    )

    print()
    print(
        f"Fichier de sortie        : {OUTPUT_FILE}"
    )

    print("=" * 60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    clean_c4()
