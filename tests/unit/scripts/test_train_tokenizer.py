import json

import pytest
from tokenizers import Tokenizer

from tokenizer import train_tokenizer

CORPUS = [
    {"text": "Bonjour le monde, bonjour le modèle."},
    {"text": "Le modèle de langage apprend le français."},
    {"text": "Un petit texte pour entraîner le tokenizer."},
] * 5


def write_lines(path, items):
    lines = [
        item if isinstance(item, str) else json.dumps(item)
        for item in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    train_file = tmp_path / "train.jsonl"
    tokenizer_dir = tmp_path / "tokenizer_out"
    write_lines(train_file, CORPUS)
    monkeypatch.setattr(train_tokenizer, "TRAIN_FILE", train_file)
    monkeypatch.setattr(train_tokenizer, "TOKENIZER_DIR", tokenizer_dir)
    monkeypatch.setattr(
        train_tokenizer, "TOKENIZER_FILE", tokenizer_dir / "tokenizer.json"
    )
    monkeypatch.setattr(
        train_tokenizer, "VOCAB_FILE", tokenizer_dir / "vocab.json"
    )
    monkeypatch.setattr(
        train_tokenizer, "CONFIG_FILE", tokenizer_dir / "config.json"
    )
    monkeypatch.setattr(train_tokenizer, "VOCAB_SIZE", 300)
    monkeypatch.setattr(train_tokenizer, "MIN_FREQUENCY", 1)
    return tmp_path


# ------------------------------------------------------------
# text_iterator
# ------------------------------------------------------------

def test_text_iterator_yields_stripped_texts(tmp_path):
    path = tmp_path / "in.jsonl"
    write_lines(path, [{"text": "  un  "}, {"text": "deux"}])

    assert list(train_tokenizer.text_iterator(path)) == ["un", "deux"]


def test_text_iterator_skips_invalid_entries(tmp_path, capsys):
    path = tmp_path / "in.jsonl"
    write_lines(path, [
        "",
        "{broken",
        {"text": 3},
        {"text": "   "},
        {"other": 1},
        {"text": "ok"},
    ])

    assert list(train_tokenizer.text_iterator(path)) == ["ok"]
    assert "[WARNING] Ligne 2 ignorée :" in capsys.readouterr().out


# ------------------------------------------------------------
# create_tokenizer / train_tokenizer
# ------------------------------------------------------------

def test_create_tokenizer_uses_byte_level_bpe():
    config = json.loads(train_tokenizer.create_tokenizer().to_str())

    assert config["model"]["type"] == "BPE"
    assert config["model"]["unk_token"] == "<unk>"
    assert config["pre_tokenizer"]["type"] == "ByteLevel"
    assert config["pre_tokenizer"]["add_prefix_space"] is False
    assert config["decoder"]["type"] == "ByteLevel"


def test_train_tokenizer_learns_vocabulary_with_special_tokens(workspace):
    tokenizer = train_tokenizer.create_tokenizer()

    train_tokenizer.train_tokenizer(tokenizer)

    vocab = tokenizer.get_vocab()
    assert [vocab[token] for token in train_tokenizer.SPECIAL_TOKENS] == [
        0, 1, 2, 3,
    ]
    assert 4 < tokenizer.get_vocab_size() <= 300


def test_trained_tokenizer_round_trips_text(workspace):
    tokenizer = train_tokenizer.create_tokenizer()
    train_tokenizer.train_tokenizer(tokenizer)

    encoding = tokenizer.encode("Bonjour le modèle")

    assert tokenizer.decode(encoding.ids) == "Bonjour le modèle"


# ------------------------------------------------------------
# save_tokenizer
# ------------------------------------------------------------

def test_save_tokenizer_writes_three_files(workspace):
    tokenizer = train_tokenizer.create_tokenizer()
    train_tokenizer.train_tokenizer(tokenizer)

    train_tokenizer.save_tokenizer(tokenizer)

    assert Tokenizer.from_file(str(train_tokenizer.TOKENIZER_FILE))
    vocab = json.loads(train_tokenizer.VOCAB_FILE.read_text(encoding="utf-8"))
    assert list(vocab.values()) == sorted(vocab.values())
    config = json.loads(
        train_tokenizer.CONFIG_FILE.read_text(encoding="utf-8")
    )
    assert config["type"] == "BPE"
    assert config["vocab_size"] == 300
    assert config["actual_vocab_size"] == tokenizer.get_vocab_size()
    assert config["pre_tokenizer"] == "ByteLevel"


def test_save_tokenizer_keeps_non_ascii_tokens_readable(workspace):
    tokenizer = train_tokenizer.create_tokenizer()
    train_tokenizer.train_tokenizer(tokenizer)

    train_tokenizer.save_tokenizer(tokenizer)

    text = train_tokenizer.VOCAB_FILE.read_text(encoding="utf-8")
    assert "Ġ" in text
    assert "\\u" not in text


# ------------------------------------------------------------
# test_tokenizer / main
# ------------------------------------------------------------

def test_test_tokenizer_prints_encoding_report(workspace, capsys):
    tokenizer = train_tokenizer.create_tokenizer()
    train_tokenizer.train_tokenizer(tokenizer)
    train_tokenizer.save_tokenizer(tokenizer)

    train_tokenizer.test_tokenizer()

    out = capsys.readouterr().out
    assert "TEST DU TOKENIZER" in out
    assert "Bonjour, je suis un modèle de langage." in out
    assert "Nombre de tokens :" in out


def test_main_trains_saves_and_tests(workspace, capsys):
    train_tokenizer.main()

    assert train_tokenizer.TOKENIZER_FILE.exists()
    assert train_tokenizer.VOCAB_FILE.exists()
    assert train_tokenizer.CONFIG_FILE.exists()
    out = capsys.readouterr().out
    assert "Tokenizer entraîné avec succès." in out
    assert "Vocabulaire cible : 300" in out


def test_main_requires_train_file(workspace):
    train_tokenizer.TRAIN_FILE.unlink()

    with pytest.raises(FileNotFoundError, match="Dataset introuvable"):
        train_tokenizer.main()
