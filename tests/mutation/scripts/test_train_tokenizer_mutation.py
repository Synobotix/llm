import json
import types
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from tokenizer import train_tokenizer

WORDS = ["Bonjour", ",", "je", "suis", "un", "modèle", "de", "langage", "."]


class FakeTokenizer:
    def __init__(self):
        self.saved_to = None
        self.training = None

    def save(self, path):
        self.saved_to = path
        Path(path).write_text("{}", encoding="utf-8")

    def get_vocab(self):
        return {"é": 2, "a": 0, "b": 1}

    def get_vocab_size(self):
        return 3

    def train_from_iterator(self, iterator, trainer):
        self.training = (list(iterator), trainer)


def write_lines(path, items):
    lines = [
        item if isinstance(item, str) else json.dumps(item)
        for item in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def spy_on_path_open(monkeypatch):
    calls = []
    original_open = Path.open

    def spy(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    return calls


@pytest.fixture
def paths(tmp_path, monkeypatch):
    directory = tmp_path / "level_one" / "tokenizer_out"
    values = types.SimpleNamespace(
        train=tmp_path / "train.jsonl",
        directory=directory,
        tokenizer=directory / "tokenizer.json",
        vocab=directory / "vocab.json",
        config=directory / "config.json",
    )
    monkeypatch.setattr(train_tokenizer, "TRAIN_FILE", values.train)
    monkeypatch.setattr(train_tokenizer, "TOKENIZER_DIR", values.directory)
    monkeypatch.setattr(train_tokenizer, "TOKENIZER_FILE", values.tokenizer)
    monkeypatch.setattr(train_tokenizer, "VOCAB_FILE", values.vocab)
    monkeypatch.setattr(train_tokenizer, "CONFIG_FILE", values.config)
    monkeypatch.setattr(train_tokenizer, "VOCAB_SIZE", 1234)
    monkeypatch.setattr(train_tokenizer, "MIN_FREQUENCY", 7)
    return values


# ------------------------------------------------------------
# text_iterator
# ------------------------------------------------------------

def test_text_iterator_opens_file_as_utf8_text(tmp_path, monkeypatch):
    path = tmp_path / "in.jsonl"
    write_lines(path, [{"text": "a"}])
    calls = spy_on_path_open(monkeypatch)

    list(train_tokenizer.text_iterator(path))

    assert calls == [(path, ("r",), {"encoding": "utf-8"})]


def test_text_iterator_warning_message_and_numbering(tmp_path, capsys):
    path = tmp_path / "in.jsonl"
    write_lines(path, ["{broken"])

    list(train_tokenizer.text_iterator(path))

    assert capsys.readouterr().out.startswith(
        "[WARNING] Ligne 1 ignorée : "
    )


def test_text_iterator_continues_after_invalid_json(tmp_path):
    path = tmp_path / "in.jsonl"
    write_lines(path, ["{broken", {"text": "ok"}, {"text": "encore"}])

    assert list(train_tokenizer.text_iterator(path)) == ["ok", "encore"]


def test_text_iterator_continues_after_blank_and_non_string(tmp_path):
    path = tmp_path / "in.jsonl"
    write_lines(path, ["", {"text": 4}, {"text": " "}, {"text": " a "}])

    assert list(train_tokenizer.text_iterator(path)) == ["a"]


# ------------------------------------------------------------
# train_tokenizer
# ------------------------------------------------------------

def test_trainer_receives_the_configured_arguments(paths, monkeypatch):
    write_lines(paths.train, [{"text": "un"}, {"text": "deux"}])
    created = []

    class RecordingTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(train_tokenizer, "BpeTrainer", RecordingTrainer)
    tokenizer = FakeTokenizer()

    train_tokenizer.train_tokenizer(tokenizer)

    assert created[0].kwargs == {
        "vocab_size": 1234,
        "min_frequency": 7,
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "show_progress": True,
    }
    assert tokenizer.training == (["un", "deux"], created[0])


# ------------------------------------------------------------
# save_tokenizer
# ------------------------------------------------------------

def test_save_tokenizer_creates_all_missing_directories(paths):
    train_tokenizer.save_tokenizer(FakeTokenizer())

    assert paths.directory.is_dir()


def test_save_tokenizer_accepts_existing_directory(paths):
    paths.directory.mkdir(parents=True)

    train_tokenizer.save_tokenizer(FakeTokenizer())

    assert paths.tokenizer.exists()


def test_save_tokenizer_writes_tokenizer_file_path(paths):
    tokenizer = FakeTokenizer()

    train_tokenizer.save_tokenizer(tokenizer)

    assert tokenizer.saved_to == str(paths.tokenizer)


def test_vocab_file_is_sorted_by_id_and_keeps_unicode(paths):
    train_tokenizer.save_tokenizer(FakeTokenizer())

    assert paths.vocab.read_text(encoding="utf-8") == (
        '{\n  "a": 0,\n  "b": 1,\n  "é": 2\n}'
    )


def test_config_file_content_is_exact(paths):
    train_tokenizer.save_tokenizer(FakeTokenizer())

    expected = {
        "type": "BPE",
        "vocab_size": 1234,
        "actual_vocab_size": 3,
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pre_tokenizer": "ByteLevel",
    }
    assert json.loads(paths.config.read_text(encoding="utf-8")) == expected
    assert paths.config.read_text(encoding="utf-8") == json.dumps(
        expected, ensure_ascii=False, indent=2
    )


def test_vocab_and_config_are_opened_as_utf8_for_writing(paths, monkeypatch):
    calls = spy_on_path_open(monkeypatch)

    train_tokenizer.save_tokenizer(FakeTokenizer())

    assert (paths.vocab, ("w",), {"encoding": "utf-8"}) in calls
    assert (paths.config, ("w",), {"encoding": "utf-8"}) in calls


# ------------------------------------------------------------
# test_tokenizer
# ------------------------------------------------------------

def test_test_tokenizer_exact_report(paths, capsys):
    paths.directory.mkdir(parents=True)
    vocab = {word: index for index, word in enumerate(WORDS)}
    vocab["<unk>"] = len(WORDS)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(paths.tokenizer))

    train_tokenizer.test_tokenizer()

    assert capsys.readouterr().out == (
        "\n" + "=" * 60 + "\n"
        + "TEST DU TOKENIZER\n"
        + "=" * 60 + "\n"
        + "\nTexte :\n"
        + "Bonjour, je suis un modèle de langage.\n"
        + "\nTokens :\n"
        + "['Bonjour', ',', 'je', 'suis', 'un', 'modèle', 'de', 'langage', '.']\n"
        + "\nIDs :\n"
        + "[0, 1, 2, 3, 4, 5, 6, 7, 8]\n"
        + "\nNombre de tokens :\n"
        + "9\n"
        + "\nDécodage :\n"
        + "Bonjour , je suis un modèle de langage .\n"
    )


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def test_main_exact_output_and_call_order(paths, monkeypatch, capsys):
    write_lines(paths.train, [{"text": "un"}])
    calls = []
    fake = FakeTokenizer()
    monkeypatch.setattr(
        train_tokenizer,
        "create_tokenizer",
        lambda: calls.append("create") or fake,
    )
    monkeypatch.setattr(
        train_tokenizer,
        "train_tokenizer",
        lambda tokenizer: calls.append(("train", tokenizer)),
    )
    monkeypatch.setattr(
        train_tokenizer,
        "save_tokenizer",
        lambda tokenizer: calls.append(("save", tokenizer)),
    )
    monkeypatch.setattr(
        train_tokenizer, "test_tokenizer", lambda: calls.append("test")
    )

    train_tokenizer.main()

    assert calls == ["create", ("train", fake), ("save", fake), "test"]
    assert capsys.readouterr().out == (
        "=" * 60 + "\n"
        + "ENTRAÎNEMENT DU TOKENIZER BPE\n"
        + "=" * 60 + "\n"
        + f"\nDataset : {paths.train}\n"
        + "Vocabulaire cible : 1234\n"
        + "Fréquence minimale : 7\n"
        + "\nCréation du tokenizer...\n"
        + "Entraînement...\n\n"
        + "\nSauvegarde...\n"
        + "\nTokenizer entraîné avec succès.\n"
        + "Nombre de tokens : 3\n"
        + "\nFichiers créés :\n"
        + f"  → {paths.tokenizer}\n"
        + f"  → {paths.vocab}\n"
        + f"  → {paths.config}\n"
    )


def test_main_missing_dataset_error_names_the_path(paths):
    with pytest.raises(FileNotFoundError) as error:
        train_tokenizer.main()

    assert str(error.value) == f"Dataset introuvable : {paths.train}"
