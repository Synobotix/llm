import types

import pytest
import torch

from scripts.bitnet import inference as module


class FakeTokenizer:
    def get_vocab_size(self):
        return 77

    def encode(self, prompt):
        return types.SimpleNamespace(
            tokens=list(prompt), ids=list(range(len(prompt)))
        )


@pytest.fixture
def pipeline(monkeypatch):
    calls = types.SimpleNamespace(
        checkpoint_path=None, model_args=None, generate_args=None
    )
    checkpoint = {
        "config": {
            "vocab_size": 77, "d_model": 8, "num_heads": 2, "hidden_dim": 16,
            "num_blocks": 1, "max_sequence_length": 6,
        }
    }

    def load_checkpoint(path):
        calls.checkpoint_path = path
        return checkpoint

    def create_bitnet_model(checkpoint, device):
        calls.model_args = (checkpoint, device)
        return "transformer", "lm_head"

    def generate(**kwargs):
        calls.generate_args = kwargs
        return "texte généré"

    monkeypatch.setattr(module, "load_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(module, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(module, "create_bitnet_model", create_bitnet_model)
    monkeypatch.setattr(module, "generate", generate)
    monkeypatch.setattr(module, "PROMPT", "abc")
    calls.checkpoint = checkpoint
    return calls


def test_main_loads_configured_checkpoint(pipeline):
    module.main()

    assert pipeline.checkpoint_path == module.CHECKPOINT_FILE


def test_main_builds_model_from_checkpoint(pipeline):
    module.main()

    checkpoint, device = pipeline.model_args
    assert checkpoint is pipeline.checkpoint
    assert isinstance(device, torch.device)


def test_main_generates_with_configured_parameters(pipeline):
    module.main()

    arguments = pipeline.generate_args
    assert arguments["transformer"] == "transformer"
    assert arguments["lm_head"] == "lm_head"
    assert arguments["prompt"] == "abc"
    assert arguments["max_new_tokens"] == module.MAX_NEW_TOKENS
    assert arguments["temperature"] == module.TEMPERATURE


def test_main_prints_tokenization_configuration_and_generated_text(
    pipeline, capsys
):
    module.main()

    output = capsys.readouterr().out
    assert "INFERENCE BITNET" in output
    assert "Vocabulaire tokenizer : 77" in output
    assert "Nombre de tokens : 3" in output
    assert "context length   : 6" in output
    assert "Durée de l'inférence" in output
    assert "texte généré" in output
    assert "FIN DE L'INFERENCE" in output


def test_main_propagates_missing_checkpoint(monkeypatch, pipeline):
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(module, "load_checkpoint", missing)

    with pytest.raises(FileNotFoundError):
        module.main()
