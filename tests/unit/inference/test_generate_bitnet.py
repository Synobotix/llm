import types

import pytest
import torch

from llm.bitnet_model.bit_transformer import BitTransformer
from llm.bitnet_model.lm_head import LMHead
from llm.inference import generate_bitnet as module

VOCAB = 12
SEQUENCE = 6


class FakeTokenizer:
    """Un token par caractère ; les ids sont décodés en lettres a, b, c..."""

    def encode(self, text):
        return types.SimpleNamespace(ids=[ord(c) - ord("a") for c in text])

    def decode(self, ids):
        return "".join(chr(ord("a") + i) for i in ids)


def make_models():
    transformer = BitTransformer(VOCAB, 8, 2, 16, 1, SEQUENCE).eval()
    lm_head = LMHead(8, VOCAB).eval()
    return transformer, lm_head


def make_checkpoint(transformer, lm_head, **config_overrides):
    config = {
        "vocab_size": VOCAB,
        "d_model": 8,
        "num_heads": 2,
        "hidden_dim": 16,
        "num_blocks": 1,
        "max_sequence_length": SEQUENCE,
    }
    config.update(config_overrides)
    return {
        "config": config,
        "transformer_state_dict": transformer.state_dict(),
        "lm_head_state_dict": lm_head.state_dict(),
    }


# ------------------------------------------------------------
# chargement
# ------------------------------------------------------------

def test_load_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.load_checkpoint(tmp_path / "absent.pt")


def test_load_checkpoint_reads_file(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"config": {"vocab_size": 3}}, path)

    assert module.load_checkpoint(path)["config"]["vocab_size"] == 3


def test_load_tokenizer_uses_configured_file(monkeypatch):
    received = []
    monkeypatch.setattr(
        module.tokenizer_loader,
        "load_tokenizer",
        lambda path: received.append(path) or "tokenizer",
    )

    assert module.load_tokenizer() == "tokenizer"
    assert received == [module.TOKENIZER_FILE]


# ------------------------------------------------------------
# create_bitnet_model
# ------------------------------------------------------------

def test_create_model_rebuilds_architecture_from_config():
    transformer, lm_head = make_models()

    new_transformer, new_head = module.create_bitnet_model(
        make_checkpoint(transformer, lm_head), torch.device("cpu")
    )

    assert isinstance(new_transformer, BitTransformer)
    assert isinstance(new_head, LMHead)
    assert new_transformer.max_sequence_length == SEQUENCE
    assert new_transformer.num_blocks == 1
    assert not new_transformer.training and not new_head.training


def test_create_model_restores_weights():
    transformer, lm_head = make_models()

    new_transformer, new_head = module.create_bitnet_model(
        make_checkpoint(transformer, lm_head), torch.device("cpu")
    )

    assert all(
        torch.equal(a, b)
        for a, b in zip(transformer.parameters(), new_transformer.parameters())
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(lm_head.parameters(), new_head.parameters())
    )


def test_create_model_with_mismatching_config_fails_on_load():
    transformer, lm_head = make_models()
    checkpoint = make_checkpoint(transformer, lm_head, d_model=16)

    with pytest.raises((RuntimeError, ValueError)):
        module.create_bitnet_model(checkpoint, torch.device("cpu"))


# ------------------------------------------------------------
# encodage / décodage
# ------------------------------------------------------------

def test_encode_prompt_returns_batch_of_one_long_tensor():
    ids = module.encode_prompt(FakeTokenizer(), "abc", torch.device("cpu"))

    assert ids.shape == (1, 3)
    assert ids.dtype == torch.long
    assert ids.tolist() == [[0, 1, 2]]


def test_decode_tokens_delegates_to_tokenizer():
    assert module.decode_tokens(FakeTokenizer(), [0, 1]) == "ab"


# ------------------------------------------------------------
# generate
# ------------------------------------------------------------

def run(prompt="ab", **kwargs):
    transformer, lm_head = make_models()
    return module.generate(
        transformer, lm_head, FakeTokenizer(), prompt, torch.device("cpu"),
        **kwargs,
    )


def test_generate_keeps_prompt_and_adds_requested_tokens():
    text = run("ab", max_new_tokens=5)

    assert text.startswith("ab")
    assert len(text) == 2 + 5


def test_generate_zero_new_tokens_returns_prompt():
    assert run("abc", max_new_tokens=0) == "abc"


def test_generated_tokens_stay_inside_vocabulary():
    text = run("a", max_new_tokens=20)

    assert all(0 <= ord(c) - ord("a") < VOCAB for c in text)


def test_generate_is_reproducible_with_same_seed():
    torch.manual_seed(1)
    first = run("ab", max_new_tokens=8)
    torch.manual_seed(1)
    second = run("ab", max_new_tokens=8)

    assert first == second


def test_very_low_temperature_behaves_like_greedy_decoding():
    transformer, lm_head = make_models()
    cold = [
        module.generate(
            transformer, lm_head, FakeTokenizer(), "ab",
            torch.device("cpu"), max_new_tokens=4, temperature=1e-4,
        )
        for _ in range(3)
    ]

    assert len(set(cold)) == 1


def test_generate_crops_context_to_max_sequence_length():
    # Un prompt plus long que max_sequence_length ne doit pas lever d'erreur.
    text = run("a" * (SEQUENCE + 3), max_new_tokens=2)

    assert len(text) == SEQUENCE + 3 + 2


@pytest.mark.parametrize("temperature", [0, -1.0])
def test_non_positive_temperature_raises(temperature):
    with pytest.raises(ValueError, match="temperature"):
        run("ab", max_new_tokens=1, temperature=temperature)
