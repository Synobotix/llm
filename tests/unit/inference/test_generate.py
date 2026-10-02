import types

import pytest
import torch

from llm.inference.generate import TextGenerator

VOCAB = 12


class FakeTokenizer:
    """Un token par caractère ; les ids sont décodés en lettres a, b, c..."""

    def encode(self, text):
        return types.SimpleNamespace(ids=[ord(c) - ord("a") for c in text])

    def decode(self, ids):
        return "".join(chr(ord("a") + i) for i in ids)


class FixedModel(torch.nn.Module):
    """Prédit toujours `target` ; mémorise la longueur du contexte reçu."""

    def __init__(self, target, vocab=VOCAB):
        super().__init__()
        self.target = target
        self.vocab = vocab
        self.context_lengths = []
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        self.context_lengths.append(input_ids.size(1))
        logits = torch.zeros(*input_ids.shape, self.vocab)
        logits[..., self.target] = 10.0
        return logits


class CountingModel(FixedModel):
    """Prédit (dernier token + 1) % vocab."""

    def forward(self, input_ids):
        self.context_lengths.append(input_ids.size(1))
        logits = torch.zeros(*input_ids.shape, self.vocab)
        next_ids = (input_ids + 1) % self.vocab
        logits.scatter_(-1, next_ids.unsqueeze(-1), 10.0)
        return logits


def make(model):
    return TextGenerator(model, FakeTokenizer())


def test_init_puts_model_in_eval_mode():
    model = FixedModel(1).train()

    make(model)

    assert not model.training


def test_generate_appends_greedy_tokens_to_prompt():
    assert make(FixedModel(3)).generate("ab", max_new_tokens=3) == "abddd"


def test_generate_follows_the_model_step_by_step():
    assert make(CountingModel(0)).generate("a", max_new_tokens=4) == "abcde"


def test_generate_zero_new_tokens_returns_prompt():
    assert make(FixedModel(3)).generate("abc", max_new_tokens=0) == "abc"


def test_generate_is_deterministic():
    generator = make(FixedModel(2))

    assert generator.generate("a", 5) == generator.generate("a", 5)


def test_generate_crops_context_to_context_size(monkeypatch):
    monkeypatch.setattr(TextGenerator, "CONTEXT_SIZE", 4)
    model = FixedModel(1)

    make(model).generate("abcdef", max_new_tokens=2)

    assert model.context_lengths == [4, 4]


def test_generate_keeps_whole_prompt_when_shorter_than_context():
    model = FixedModel(1)

    make(model).generate("ab", max_new_tokens=2)

    assert model.context_lengths == [2, 3]


def test_generate_does_not_build_autograd_graph():
    model = FixedModel(1)
    make(model).generate("a", max_new_tokens=1)

    assert model.anchor.grad is None


def test_decode_token_falls_back_when_tokenizer_fails():
    class Broken(FakeTokenizer):
        def decode(self, ids):
            raise ValueError("boom")

    generator = TextGenerator(FixedModel(1), Broken())

    assert generator._decode_token(1) == "<erreur>"


def test_next_token_logits_returns_last_position_logits():
    generator = make(FixedModel(4))

    logits = generator._next_token_logits(torch.tensor([[0, 1, 2]]), step=1)

    assert logits.shape == (1, VOCAB)
    assert logits.argmax(-1).item() == 4


def test_logits_debug_printed_only_on_first_step(capsys):
    generator = make(FixedModel(1))

    generator._next_token_logits(torch.tensor([[0]]), step=0)
    first = capsys.readouterr().out
    generator._next_token_logits(torch.tensor([[0]]), step=1)
    second = capsys.readouterr().out

    assert "DEBUG DES LOGITS" in first
    assert "DEBUG DES LOGITS" not in second


def test_logs_report_start_chosen_token_and_end(capsys):
    make(FixedModel(3)).generate("ab", max_new_tokens=1)

    output = capsys.readouterr().out
    assert "DÉBUT DE LA GÉNÉRATION" in output
    assert "Prompt : 'ab'" in output
    assert "TOKEN CHOISI" in output
    assert "Token texte : 'd'" in output
    assert "Texte généré : abd" in output
