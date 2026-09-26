import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import llm.tokenizer.tokenizer as tk


@pytest.fixture
def tokenizer_file(tmp_path, monkeypatch):
    vocab = {"<unk>": 0, "hello": 1, "world": 2}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    monkeypatch.setattr(tk, "TOKENIZER_FILE", path)
    return path


def test_load_tokenizer(tokenizer_file):
    tokenizer = tk.load_tokenizer()
    assert tokenizer.encode("hello world").ids == [1, 2]


def test_unknown_word_maps_to_unk(tokenizer_file):
    assert tk.load_tokenizer().encode("zzz").ids == [0]


def test_empty_text(tokenizer_file):
    assert tk.load_tokenizer().encode("").ids == []


def test_get_vocab_size(tokenizer_file):
    assert tk.get_vocab_size() == 3


def test_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(tk, "TOKENIZER_FILE", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="Tokenizer introuvable"):
        tk.load_tokenizer()
    with pytest.raises(FileNotFoundError):
        tk.get_vocab_size()


def test_corrupted_file_raises(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    monkeypatch.setattr(tk, "TOKENIZER_FILE", path)
    with pytest.raises(Exception):
        tk.load_tokenizer()
