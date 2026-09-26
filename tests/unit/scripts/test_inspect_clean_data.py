import json

import pytest

from scripts import inspect_clean_data


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


# ------------------------------------------------------------
# Outils
# ------------------------------------------------------------

def test_load_documents_reads_every_valid_line(input_file):
    write_documents(input_file, [{"text": "a"}, {"text": "b"}])

    assert inspect_clean_data.load_documents() == [
        {"text": "a"}, {"text": "b"},
    ]


def test_load_documents_reports_invalid_lines(input_file, capsys):
    write_documents(input_file, [{"text": "a"}, "{broken", {"text": "b"}])

    documents = inspect_clean_data.load_documents()

    assert documents == [{"text": "a"}, {"text": "b"}]
    assert "⚠️ JSON invalide à la ligne 2" in capsys.readouterr().out


def test_load_documents_empty_file(input_file):
    input_file.write_text("", encoding="utf-8")

    assert inspect_clean_data.load_documents() == []


def test_clean_for_word_count():
    assert inspect_clean_data.clean_for_word_count("Été, l'ami-fidèle!") == [
        "été", "l'ami-fidèle",
    ]
    assert inspect_clean_data.clean_for_word_count("") == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<p>hi</p>", True),
        ("a < b", False),
        ("plain", False),
    ],
)
def test_contains_html(text, expected):
    assert inspect_clean_data.contains_html(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("see https://a.com", True),
        ("see www.a.com", True),
        ("see HTTP://A.COM", True),
        ("no link", False),
    ],
)
def test_contains_url(text, expected):
    assert inspect_clean_data.contains_url(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("me@mail.com", True),
        ("ME@MAIL.COM", True),
        ("me@mail", False),
        ("no email", False),
    ],
)
def test_contains_email(text, expected):
    assert inspect_clean_data.contains_email(text) is expected


# ------------------------------------------------------------
# inspect_c4
# ------------------------------------------------------------

def test_inspect_missing_file_raises(input_file):
    with pytest.raises(FileNotFoundError, match="Fichier introuvable"):
        inspect_clean_data.inspect_c4()


def test_inspect_without_documents(input_file, capsys):
    input_file.write_text("", encoding="utf-8")

    inspect_clean_data.inspect_c4()

    assert capsys.readouterr().out == "Aucun document trouvé.\n"


def test_inspect_reports_corpus_statistics(input_file, capsys):
    write_documents(input_file, [
        {"text": "alpha beta"},
        {"text": "gamma <b>delta</b> https://x.io me@mail.com"},
    ])

    inspect_clean_data.inspect_c4()

    out = capsys.readouterr().out
    assert "Documents                 : 2" in out
    assert "Documents avec HTML      : 1" in out
    assert "Documents avec URL       : 1" in out
    assert "Documents avec email     : 1" in out
    assert "alpha                : 1" in out


def test_inspect_truncates_long_previews(input_file, capsys):
    write_documents(input_file, [
        {"text": "x" * 300},
        {"text": "y" * 301},
    ])

    inspect_clean_data.inspect_c4()

    lines = capsys.readouterr().out.splitlines()
    assert "x" * 300 + "..." not in lines
    assert "y" * 300 + "..." in lines
    assert "y" * 301 not in lines


def test_inspect_limits_samples_and_shortest_documents(input_file, capsys):
    write_documents(input_file, [{"text": f"doc{i}"} for i in range(8)])

    inspect_clean_data.inspect_c4()

    out = capsys.readouterr().out
    assert "DOCUMENT 5" in out
    assert "DOCUMENT 6" not in out
    assert "DOCUMENT COURT 5" in out
    assert "DOCUMENT COURT 6" not in out


def test_inspect_limits_top_words(input_file, capsys):
    words = " ".join(f"mot{i:02d}" for i in range(25))
    write_documents(input_file, [{"text": words}])

    inspect_clean_data.inspect_c4()

    out = capsys.readouterr().out
    assert "mot19" in out
    assert "mot20                :" not in out


def test_inspect_handles_document_without_text(input_file, capsys):
    write_documents(input_file, [{"url": "u"}, {"text": "abc"}])

    inspect_clean_data.inspect_c4()

    out = capsys.readouterr().out
    assert "Longueur : 0 caractères" in out
    assert "Minimum                   : 0 caractères" in out
