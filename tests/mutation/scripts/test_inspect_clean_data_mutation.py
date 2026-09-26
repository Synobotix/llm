import builtins
import json
import re

import pytest

from scripts import inspect_clean_data

SEPARATOR = "=" * 60
DASHES = "-" * 60

GOLDEN_OUTPUT = f"""
{SEPARATOR}
INSPECTION DU CORPUS C4 NETTOYÉ
{SEPARATOR}
Documents                 : 3
Caractères totaux         : 72
Mots totaux               : 15

LONGUEUR DES DOCUMENTS
{DASHES}
Minimum                   : 5 caractères
Moyenne                   : 24 caractères
Maximum                   : 36 caractères

NOMBRE DE MOTS
{DASHES}
Minimum                   : 1 mots
Moyenne                   : 5 mots
Maximum                   : 8 mots

VÉRIFICATION DU NETTOYAGE
{DASHES}
Documents avec HTML      : 1
Documents avec URL       : 1
Documents avec email     : 1

20 MOTS LES PLUS FRÉQUENTS
{DASHES}
alpha                : 3
b                    : 2
beta                 : 1
gamma                : 1
https                : 1
x                    : 1
io                   : 1
contact              : 1
me                   : 1
mail                 : 1
com                  : 1
short                : 1

EXTRAITS DE DOCUMENTS
{SEPARATOR}

DOCUMENT 1
{DASHES}
Alpha beta <b>gamma</b> https://x.io

DOCUMENT 2
{DASHES}
Contact me@mail.com alpha alpha

DOCUMENT 3
{DASHES}
short

DOCUMENTS LES PLUS COURTS
{SEPARATOR}

DOCUMENT COURT 1
Longueur : 5 caractères
short

DOCUMENT COURT 2
Longueur : 31 caractères
Contact me@mail.com alpha alpha

DOCUMENT COURT 3
Longueur : 36 caractères
Alpha beta <b>gamma</b> https://x.io

{SEPARATOR}
INSPECTION TERMINÉE
{SEPARATOR}
"""

CORPUS = [
    {"text": "Alpha beta <b>gamma</b> https://x.io"},
    {"text": "Contact me@mail.com alpha alpha"},
    {"text": "short"},
]


def write_documents(path, documents):
    lines = [
        item if isinstance(item, str) else json.dumps(item)
        for item in documents
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def input_file(tmp_path, monkeypatch):
    path = tmp_path / "clean.jsonl"
    monkeypatch.setattr(inspect_clean_data, "INPUT_FILE", path)
    return path


def test_full_report_matches_reference_output(input_file, capsys):
    write_documents(input_file, CORPUS)

    inspect_clean_data.inspect_c4()

    assert capsys.readouterr().out == GOLDEN_OUTPUT


def test_input_file_is_opened_as_utf8_text(input_file, monkeypatch):
    write_documents(input_file, CORPUS)
    calls = []
    original_open = builtins.open

    def spy_open(*args, **kwargs):
        calls.append((args, kwargs))
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)

    inspect_clean_data.load_documents()

    assert calls == [((input_file, "r"), {"encoding": "utf-8"})]


def test_invalid_line_message_is_exact(input_file, capsys):
    write_documents(input_file, ["{broken"])

    inspect_clean_data.load_documents()

    assert capsys.readouterr().out == "⚠️ JSON invalide à la ligne 1\n"


def test_invalid_line_numbers_start_at_one(input_file, capsys):
    write_documents(input_file, [{"text": "a"}, {"text": "b"}, "{broken"])

    inspect_clean_data.load_documents()

    assert capsys.readouterr().out == "⚠️ JSON invalide à la ligne 3\n"


def test_missing_file_error_names_the_path(input_file):
    with pytest.raises(FileNotFoundError) as error:
        inspect_clean_data.inspect_c4()

    assert str(error.value) == f"Fichier introuvable : {input_file}"


def test_preview_boundary_is_exactly_three_hundred_characters(
    input_file, capsys
):
    write_documents(input_file, [{"text": "x" * 300}])

    inspect_clean_data.inspect_c4()

    lines = capsys.readouterr().out.splitlines()
    assert "x" * 300 in lines
    assert "x" * 300 + "..." not in lines


def test_preview_of_long_text_is_cut_then_ellipsis_is_appended(
    input_file, capsys
):
    write_documents(input_file, [{"text": "y" * 301}])

    inspect_clean_data.inspect_c4()

    lines = capsys.readouterr().out.splitlines()
    assert lines.count("y" * 300 + "...") == 2


def test_exactly_five_samples_and_five_shortest_are_shown(
    input_file, capsys
):
    write_documents(input_file, [{"text": f"doc{i}"} for i in range(6)])

    inspect_clean_data.inspect_c4()

    lines = capsys.readouterr().out.splitlines()
    assert len([l for l in lines if re.fullmatch(r"DOCUMENT \d+", l)]) == 5
    assert len([l for l in lines if l.startswith("DOCUMENT COURT ")]) == 5


def test_exactly_twenty_most_common_words_are_listed(input_file, capsys):
    words = " ".join(f"mot{i:02d}" for i in range(21))
    write_documents(input_file, [{"text": words}])

    inspect_clean_data.inspect_c4()

    lines = capsys.readouterr().out.splitlines()
    assert len([l for l in lines if re.fullmatch(r"mot\d+ +: 1", l)]) == 20


def test_shortest_documents_are_sorted_by_length(input_file, capsys):
    write_documents(input_file, [
        {"text": "cccccc"},
        {"text": "a"},
        {"text": "bbb"},
    ])

    inspect_clean_data.inspect_c4()

    out = capsys.readouterr().out
    shortest = out.split("DOCUMENTS LES PLUS COURTS")[1]
    assert shortest.index("Longueur : 1 ") < shortest.index("Longueur : 3 ")
    assert shortest.index("Longueur : 3 ") < shortest.index("Longueur : 6 ")


def test_thousands_separators_are_used_for_large_totals(input_file, capsys):
    write_documents(input_file, [{"text": "x" * 1234}])

    inspect_clean_data.inspect_c4()

    out = capsys.readouterr().out
    assert "Caractères totaux         : 1,234" in out
    assert "Moyenne                   : 1,234 caractères" in out


def test_cleanliness_counters_accumulate_across_documents(
    input_file, capsys
):
    write_documents(input_file, [
        {"text": "<b>a</b> https://x.io me@mail.com"},
        {"text": "<i>b</i> www.y.io you@mail.com"},
    ])

    inspect_clean_data.inspect_c4()

    out = capsys.readouterr().out
    assert "Documents avec HTML      : 2" in out
    assert "Documents avec URL       : 2" in out
    assert "Documents avec email     : 2" in out


def test_sample_of_document_without_text_is_empty(input_file, capsys):
    write_documents(input_file, [{"url": "u"}])

    inspect_clean_data.inspect_c4()

    out = capsys.readouterr().out
    assert f"DOCUMENT 1\n{DASHES}\n\n" in out
