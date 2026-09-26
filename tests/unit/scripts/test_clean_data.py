import hashlib
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


def read_output(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.fixture
def files(tmp_path, monkeypatch):
    input_file = tmp_path / "raw" / "in.jsonl"
    output_file = tmp_path / "processed" / "out.jsonl"
    input_file.parent.mkdir()
    monkeypatch.setattr(clean_data, "INPUT_FILE", input_file)
    monkeypatch.setattr(clean_data, "OUTPUT_FILE", output_file)
    return input_file, output_file


# ------------------------------------------------------------
# Nettoyage
# ------------------------------------------------------------

def test_normalize_unicode_applies_nfkc():
    assert clean_data.normalize_unicode("ﬁ ① Ａ") == "fi 1 A"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a<script>evil()</script>b", "a b"),
        ("a<SCRIPT type='x'>evil()</Script>b", "a b"),
        ("a<style>p {}</style>b", "a b"),
        ("a<script>\nline\n</script>b", "a b"),
        ("a<!-- hidden\ncomment -->b", "a b"),
        ("<p>hi</p>", " hi "),
        ("&lt;p&gt;hi", " hi"),
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("a < b but never closed", "a < b but never closed"),
        ("1 < 2 and 3 > 2", "1   2"),
        ("plain text", "plain text"),
        ("", ""),
    ],
)
def test_remove_html(text, expected):
    assert clean_data.remove_html(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("see https://a.com/x?y=1 now", "see   now"),
        ("see http://a.com now", "see   now"),
        ("see www.a.com now", "see   now"),
        ("see HTTPS://A.COM now", "see   now"),
        ("no link", "no link"),
    ],
)
def test_remove_urls(text, expected):
    assert clean_data.remove_urls(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("write to john.doe+x@mail.example.com now", "write to   now"),
        ("not an email@ here", "not an email@ here"),
        ("a@b.c is too short", "a@b.c is too short"),
    ],
)
def test_remove_emails(text, expected):
    assert clean_data.remove_emails(text) == expected


def test_normalize_spaces_collapses_and_strips():
    assert clean_data.normalize_spaces("  a \n\t b   c  ") == "a b c"
    assert clean_data.normalize_spaces("   ") == ""


def test_clean_text_runs_the_whole_pipeline():
    raw = "  ﬁn <b>bold</b> https://x.io mail@x.io\n\nend  "

    assert clean_data.clean_text(raw) == "fin bold end"


# ------------------------------------------------------------
# Filtres de qualité
# ------------------------------------------------------------

def test_is_too_short_boundary():
    assert clean_data.is_too_short("a" * 199) is True
    assert clean_data.is_too_short("a" * 200) is False


def test_alphabetic_ratio():
    assert clean_data.alphabetic_ratio("") == 0.0
    assert clean_data.alphabetic_ratio("ab12") == 0.5
    assert clean_data.alphabetic_ratio("éàü") == 1.0
    assert clean_data.alphabetic_ratio("1234") == 0.0


def test_low_alphabetic_ratio_boundary():
    assert clean_data.has_low_alphabetic_ratio("abcd123456") is False
    assert clean_data.has_low_alphabetic_ratio("abc1234567") is True
    assert clean_data.has_low_alphabetic_ratio("") is True


# ------------------------------------------------------------
# Répétitions
# ------------------------------------------------------------

def test_tokenize_words():
    assert clean_data.tokenize_words("Costume GONFLABLE, de clown!") == [
        "costume", "gonflable", "de", "clown",
    ]
    assert clean_data.tokenize_words("L'été-chaud à Paris") == [
        "l'été-chaud", "à", "paris",
    ]
    assert clean_data.tokenize_words("") == []


def test_consecutive_word_repetition_threshold():
    assert clean_data.has_consecutive_word_repetition("") is False
    assert clean_data.has_consecutive_word_repetition("a a a a") is False
    assert clean_data.has_consecutive_word_repetition("a a a a a") is True
    assert clean_data.has_consecutive_word_repetition("A a A a A") is True


def test_consecutive_word_repetition_resets_on_other_word():
    assert clean_data.has_consecutive_word_repetition(
        "a a a a b a a a a"
    ) is False


def test_normalize_ngram():
    assert clean_data.normalize_ngram((" Ab ", "CD")) == ("ab", "cd")


def ngram_text(repeats, filler_count=50):
    parts = []
    fillers = unique_words(filler_count, offset=100)

    for index in range(repeats):
        parts.append("xray yankee zulu")
        parts.append(fillers[index])

    parts.extend(fillers[repeats:])
    return " ".join(parts)


def test_repeated_ngrams_detects_six_repetitions():
    assert clean_data.has_repeated_ngrams(ngram_text(6)) is True


def test_repeated_ngrams_tolerates_five_repetitions():
    assert clean_data.has_repeated_ngrams(ngram_text(5)) is False


def test_repeated_ngrams_skips_short_documents():
    assert clean_data.has_repeated_ngrams("xray yankee zulu " * 6) is False


def test_repeated_ngrams_is_case_insensitive():
    text = ngram_text(6).replace("xray", "XRAY", 3)

    assert clean_data.has_repeated_ngrams(text) is True


def test_normalize_phrase():
    assert clean_data.normalize_phrase("  Hello   WORLD \n") == "hello world"


def test_repeated_phrases_threshold():
    sentence = "Achetez notre produit maintenant"

    assert clean_data.has_repeated_phrases(f"{sentence}. {sentence}.") is False
    assert clean_data.has_repeated_phrases(
        f"{sentence}. {sentence}! {sentence}?"
    ) is True


def test_repeated_phrases_ignores_short_sentences():
    assert clean_data.has_repeated_phrases("Un deux trois. " * 5) is False
    assert clean_data.has_repeated_phrases("") is False


def test_repeated_phrases_normalizes_case_and_spaces():
    text = (
        "Achetez  notre produit maintenant. "
        "achetez notre PRODUIT maintenant. "
        "achetez notre produit  maintenant"
    )

    assert clean_data.has_repeated_phrases(text) is True


def test_excessive_repetition_each_level():
    assert clean_data.has_excessive_repetition("a a a a a") is True
    assert clean_data.has_excessive_repetition(ngram_text(6)) is True
    assert clean_data.has_excessive_repetition(
        "Achetez notre produit maintenant. " * 3
    ) is True
    assert clean_data.has_excessive_repetition(make_text()) is False


def test_text_hash_is_sha256_hex():
    assert clean_data.text_hash("abc") == hashlib.sha256(b"abc").hexdigest()
    assert clean_data.text_hash("é") == hashlib.sha256("é".encode()).hexdigest()


# ------------------------------------------------------------
# clean_c4
# ------------------------------------------------------------

def test_clean_c4_missing_input_raises(files):
    with pytest.raises(FileNotFoundError, match="Fichier introuvable"):
        clean_data.clean_c4()


def test_clean_c4_keeps_valid_document_and_cleans_it(files):
    input_file, output_file = files
    text = "<p>" + make_text() + "</p> https://x.io"
    write_input(input_file, [
        {"text": text, "url": "u", "timestamp": "2024-01-01T00:00:00"},
    ])

    clean_data.clean_c4()

    assert read_output(output_file) == [
        {
            "text": make_text(),
            "url": "u",
            "timestamp": "2024-01-01T00:00:00",
        }
    ]


def test_clean_c4_fills_missing_url_and_timestamp(files):
    input_file, output_file = files
    write_input(input_file, [{"text": make_text()}])

    clean_data.clean_c4()

    assert read_output(output_file) == [
        {"text": make_text(), "url": "", "timestamp": None}
    ]


def test_clean_c4_applies_every_filter(files, capsys):
    input_file, output_file = files
    good = make_text()
    write_input(input_file, [
        {"text": good},
        "not json",
        {"text": ""},
        {"url": "no text"},
        {"text": "<p></p>"},
        {"text": "court"},
        {"text": "1234567890 " * 30},
        {"text": "a a a a a " + make_text()},
        {"text": good},
        {"text": make_text(60, offset=100)},
    ])

    clean_data.clean_c4()

    assert [doc["text"] for doc in read_output(output_file)] == [
        good, make_text(60, offset=100),
    ]
    out = capsys.readouterr().out
    assert "Documents analysés       : 10" in out
    assert "Documents conservés      : 2" in out
    assert "Documents supprimés      : 8" in out
    assert "Vides                    : 3" in out
    assert "Trop courts              : 1" in out
    assert "Ratio alphabétique faible: 1" in out
    assert "Répétition excessive     : 1" in out
    assert "Doublons                 : 1" in out


def test_clean_c4_deduplicates_after_cleaning(files):
    input_file, output_file = files
    write_input(input_file, [
        {"text": make_text()},
        {"text": "<b>" + make_text() + "</b>"},
    ])

    clean_data.clean_c4()

    assert len(read_output(output_file)) == 1


def test_clean_c4_keeps_non_ascii_characters(files):
    input_file, output_file = files
    text = make_text() + " été"
    write_input(input_file, [{"text": text}])

    clean_data.clean_c4()

    assert "été" in output_file.read_text(encoding="utf-8")


def test_clean_c4_empty_input_produces_empty_output(files, capsys):
    input_file, output_file = files
    input_file.write_text("", encoding="utf-8")

    clean_data.clean_c4()

    assert output_file.read_text(encoding="utf-8") == ""
    assert "Documents analysés       : 0" in capsys.readouterr().out
