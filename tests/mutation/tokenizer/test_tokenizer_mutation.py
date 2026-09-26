import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import llm.tokenizer.tokenizer as tokenizer_module


def save_tokenizer(path, words):
    vocab = {"<unk>": 0, **{word: index + 1 for index, word in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


def test_explicit_file_has_priority_over_default(tmp_path, monkeypatch):
    default_file = tmp_path / "default.json"
    explicit_file = tmp_path / "explicit.json"
    save_tokenizer(default_file, ["a"])
    save_tokenizer(explicit_file, ["a", "b", "c"])
    monkeypatch.setattr(tokenizer_module, "TOKENIZER_FILE", default_file)

    tokenizer = tokenizer_module.load_tokenizer(explicit_file)

    assert tokenizer.get_vocab_size() == 4


def test_default_file_is_used_without_argument(tmp_path, monkeypatch):
    default_file = tmp_path / "default.json"
    save_tokenizer(default_file, ["a", "b"])
    monkeypatch.setattr(tokenizer_module, "TOKENIZER_FILE", default_file)

    assert tokenizer_module.load_tokenizer().get_vocab_size() == 3
    assert tokenizer_module.get_vocab_size() == 3


def test_missing_file_error_names_the_path(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError) as error:
        tokenizer_module.load_tokenizer(missing)

    assert str(error.value) == f"Tokenizer introuvable : {missing}"


def test_loaded_tokenizer_encodes_known_words(tmp_path):
    path = tmp_path / "tokenizer.json"
    save_tokenizer(path, ["hello", "world"])

    encoding = tokenizer_module.load_tokenizer(path).encode("hello world")

    assert encoding.ids == [1, 2]
