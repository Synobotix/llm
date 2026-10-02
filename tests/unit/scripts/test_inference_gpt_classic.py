import json
import types

import pytest
import torch

from llm.model.transformer import Transformer
from scripts.gpt_classic import inference

VOCAB = 8


class FixedModel(torch.nn.Module):
    def __init__(self, target=2):
        super().__init__()
        self.target = target
        self.context_lengths = []

    def forward(self, input_ids):
        self.context_lengths.append(input_ids.size(1))
        logits = torch.zeros(*input_ids.shape, VOCAB)
        logits[..., self.target] = 5.0
        return logits


class FakeTokenizer:
    def __init__(self, ids=None):
        self.ids = ids

    def encode(self, text):
        ids = [ord(c) - ord("a") for c in text] if self.ids is None else self.ids
        return types.SimpleNamespace(ids=ids)

    def decode(self, ids):
        return "".join(chr(ord("a") + i) for i in ids)


# ------------------------------------------------------------
# get_vocab_size
# ------------------------------------------------------------

def test_get_vocab_size_reads_config(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"actual_vocab_size": 77}))
    monkeypatch.setattr(inference, "TOKENIZER_CONFIG_FILE", config)

    assert inference.get_vocab_size() == 77


def test_get_vocab_size_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "TOKENIZER_CONFIG_FILE", tmp_path / "no.json")

    with pytest.raises(FileNotFoundError):
        inference.get_vocab_size()


# ------------------------------------------------------------
# generate
# ------------------------------------------------------------

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(inference, "DEVICE", torch.device("cpu"))


def test_generate_appends_greedy_tokens():
    text = inference.generate(FixedModel(2), FakeTokenizer(), "ab", 3)

    assert text == "abccc"


def test_generate_zero_tokens_returns_decoded_prompt():
    assert inference.generate(FixedModel(), FakeTokenizer(), "ab", 0) == "ab"


def test_generate_empty_prompt_raises():
    with pytest.raises(ValueError):
        inference.generate(FixedModel(), FakeTokenizer(ids=[]), "", 3)


def test_generate_crops_context_to_max_sequence_length(monkeypatch):
    monkeypatch.setattr(inference, "MAX_SEQUENCE_LENGTH", 3)
    model = FixedModel()

    inference.generate(model, FakeTokenizer(), "abcde", 2)

    assert model.context_lengths == [3, 3]


# ------------------------------------------------------------
# load_model
# ------------------------------------------------------------

def tiny_parameters(monkeypatch, config_path):
    monkeypatch.setattr(inference, "D_MODEL", 8)
    monkeypatch.setattr(inference, "NUM_HEADS", 2)
    monkeypatch.setattr(inference, "HIDDEN_DIM", 16)
    monkeypatch.setattr(inference, "NUM_BLOCKS", 1)
    monkeypatch.setattr(inference, "MAX_SEQUENCE_LENGTH", 6)
    config_path.write_text(json.dumps({"actual_vocab_size": VOCAB}))
    monkeypatch.setattr(inference, "TOKENIZER_CONFIG_FILE", config_path)


def test_load_model_missing_checkpoint_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "CHECKPOINT_PATH", tmp_path / "no.pt")

    with pytest.raises(FileNotFoundError):
        inference.load_model()


def test_load_model_restores_weights_in_eval_mode(tmp_path, monkeypatch):
    tiny_parameters(monkeypatch, tmp_path / "config.json")
    source = Transformer(VOCAB, 8, 2, 16, 1, 6)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    monkeypatch.setattr(inference, "CHECKPOINT_PATH", checkpoint)

    model = inference.load_model()

    assert not model.training
    assert all(
        torch.equal(a, b)
        for a, b in zip(source.parameters(), model.parameters())
    )
