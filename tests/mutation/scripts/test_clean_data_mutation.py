import builtins
import itertools
import json

import pytest

from scripts import clean_data


def unique_words(count, offset=0):
    words = (
        "".join(letters)
        for letters in itertools.product("abcdefghij", repeat=3)
    )
    return list(itertools.islice(words, offset, offset + count))


def make_text(count=60, offset=0):
    return " ".join(unique_words(count, offset))


def write_input(path, documents):
    lines = [
        document if isinstance(document, str) else json.dumps(document)
        for document in documents
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def words_ending_with_six_trigrams(prefix_count):
    words = unique_words(prefix_count, offset=300)

    for index in range(6):
        separator = unique_words(1, offset=400 + index)
        words += separator + ["xray", "yankee", "zulu"]

    return words


@pytest.fixture
def files(tmp_path, monkeypatch):
    input_file = tmp_path / "raw" / "in.jsonl"
    output_file = tmp_path / "level_one" / "level_two" / "out.jsonl"
    input_file.parent.mkdir()
    monkeypatch.setattr(clean_data, "INPUT_FILE", input_file)
    monkeypatch.setattr(clean_data, "OUTPUT_FILE", output_file)
    return input_file, output_file


# ------------------------------------------------------------
# Fonctions de nettoyage
# ------------------------------------------------------------

def test_html_comment_containing_a_closing_bracket_is_removed_whole():
    assert clean_data.remove_html("a<!-- x >\n y -->b") == "a b"


def test_uppercase_email_is_removed():
    assert clean_data.remove_emails("write JOHN.DOE@MAIL.EXAMPLE.COM now") == (
        "write   now"
    )


# ------------------------------------------------------------
# Répétitions
# ------------------------------------------------------------

def test_consecutive_repetition_starts_counting_from_first_word():
    assert clean_data.has_consecutive_word_repetition("a b b b b") is False


def test_ngram_check_applies_to_a_document_of_exactly_fifty_words():
    words = words_ending_with_six_trigrams(prefix_count=26)

    assert len(words) == 50
    assert clean_data.has_repeated_ngrams(" ".join(words)) is True


def test_ngram_check_counts_the_last_position_of_the_document():
    words = words_ending_with_six_trigrams(prefix_count=40)

    assert clean_data.has_repeated_ngrams(" ".join(words)) is True


def test_phrase_check_continues_after_a_short_sentence():
    sentence = "Achetez notre produit maintenant"
    text = f"Un deux. {sentence}. {sentence}. {sentence}."

    assert clean_data.has_repeated_phrases(text) is True


def test_phrase_check_does_not_merge_different_sentences():
    text = (
        "Achetez notre produit maintenant. "
        "Visitez notre magasin aujourd'hui. "
        "Découvrez nos offres spéciales ici."
    )

    assert clean_data.has_repeated_phrases(text) is False


# ------------------------------------------------------------
# clean_c4
# ------------------------------------------------------------

def test_output_directories_are_created_even_when_nested(files):
    input_file, output_file = files
    input_file.write_text("", encoding="utf-8")

    clean_data.clean_c4()

    assert output_file.exists()


def test_input_and_output_are_opened_as_utf8_text(files, monkeypatch):
    input_file, output_file = files
    input_file.write_text("", encoding="utf-8")
    calls = []
    original_open = builtins.open

    def spy_open(*args, **kwargs):
        calls.append((args, kwargs))
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)

    clean_data.clean_c4()

    assert calls == [
        ((input_file, "r"), {"encoding": "utf-8"}),
        ((output_file, "w"), {"encoding": "utf-8"}),
    ]


def test_exact_statistics_output(files, capsys):
    input_file, output_file = files
    good = make_text()
    write_input(input_file, [
        {"text": good},
        {"text": ""},
        {"text": "court"},
        {"text": "bref"},
        {"text": "1234567890 " * 30},
        {"text": "0987654321 " * 30},
        {"text": "a a a a a " + make_text(60, offset=100)},
        {"text": "b b b b b " + make_text(60, offset=200)},
        {"text": good},
        {"text": good},
    ])

    clean_data.clean_c4()

    assert capsys.readouterr().out == (
        "\n"
        + "=" * 60 + "\n"
        + "NETTOYAGE C4 TERMINÉ\n"
        + "=" * 60 + "\n"
        + "Documents analysés       : 10\n"
        + "Documents conservés      : 1\n"
        + "Documents supprimés      : 9\n"
        + "\n"
        + "DÉTAIL DES SUPPRESSIONS\n"
        + "-" * 60 + "\n"
        + "Vides                    : 1\n"
        + "Trop courts              : 2\n"
        + "Ratio alphabétique faible: 2\n"
        + "Répétition excessive     : 2\n"
        + "Doublons                 : 2\n"
        + "\n"
        + f"Fichier de sortie        : {output_file}\n"
        + "=" * 60 + "\n"
    )


def test_existing_output_directory_is_accepted(files):
    input_file, output_file = files
    input_file.write_text("", encoding="utf-8")
    output_file.parent.mkdir(parents=True)

    clean_data.clean_c4()

    assert output_file.exists()
