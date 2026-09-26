import importlib
from pathlib import Path

import pytest

import llm.config.parameters as parameters

ENV_KEYS = [
    "MAX_DOCUMENT", "TRAIN_RATIO", "RANDOM_SEED", "MAX_VOCAB_SIZE",
    "MIN_FREQUENCY", "SPECIAL_TOKENS", "BLOCK_SIZE", "D_MODEL", "NUM_HEADS",
    "HIDDEN_DIM", "NUM_BLOCKS", "BATCH_SIZE", "EPOCHS", "LEARNING_RATE",
    "WEIGHT_DECAY", "GRADIENT_CLIP", "SAVE_EVERY_N_STEPS",
    "AUTO_CLEAN_STEP_CHECKPOINTS", "DATA_DIR", "PROCESSED_DIR",
    "TOKENIZED_DIR", "TOKENIZER_DIR",
]


@pytest.fixture
def reload_params(monkeypatch):
    """Recharge le module sans .env et avec un environnement vide."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    def _reload():
        return importlib.reload(parameters)

    yield _reload

    monkeypatch.undo()
    importlib.reload(parameters)


def test_defaults(reload_params):
    p = reload_params()
    assert p.MAX_DOCUMENT == 70000
    assert p.TRAIN_RATIO == 0.80
    assert p.RANDOM_SEED == 42
    assert p.BLOCK_SIZE == 512
    assert p.MAX_SEQUENCE_LENGTH == p.BLOCK_SIZE
    assert p.SPECIAL_TOKENS == ["<pad>", "<unk>", "<bos>", "<eos>"]
    assert p.AUTO_CLEAN_STEP_CHECKPOINTS is True
    assert p.D_MODEL % p.NUM_HEADS == 0


def test_env_overrides_int_and_float(reload_params, monkeypatch):
    monkeypatch.setenv("BATCH_SIZE", "32")
    monkeypatch.setenv("LEARNING_RATE", "1e-3")
    p = reload_params()
    assert p.BATCH_SIZE == 32
    assert p.LEARNING_RATE == pytest.approx(1e-3)


def test_max_sequence_length_follows_block_size(reload_params, monkeypatch):
    monkeypatch.setenv("BLOCK_SIZE", "64")
    assert reload_params().MAX_SEQUENCE_LENGTH == 64


def test_invalid_int_raises(reload_params, monkeypatch):
    monkeypatch.setenv("BATCH_SIZE", "abc")
    with pytest.raises(ValueError):
        reload_params()


def test_invalid_float_raises(reload_params, monkeypatch):
    monkeypatch.setenv("TRAIN_RATIO", "oops")
    with pytest.raises(ValueError):
        reload_params()


def test_empty_int_raises(reload_params, monkeypatch):
    monkeypatch.setenv("EPOCHS", "")
    with pytest.raises(ValueError):
        reload_params()


@pytest.mark.parametrize(
    "value, expected",
    [("true", True), ("TRUE", True), ("True", True),
     ("false", False), ("0", False), ("yes", False), ("", False)],
)
def test_auto_clean_bool_parsing(reload_params, monkeypatch, value, expected):
    monkeypatch.setenv("AUTO_CLEAN_STEP_CHECKPOINTS", value)
    assert reload_params().AUTO_CLEAN_STEP_CHECKPOINTS is expected


def test_special_tokens_split(reload_params, monkeypatch):
    monkeypatch.setenv("SPECIAL_TOKENS", "<a>,<b>")
    assert reload_params().SPECIAL_TOKENS == ["<a>", "<b>"]


def test_special_tokens_empty_string_gives_single_empty(
        reload_params, monkeypatch):
    monkeypatch.setenv("SPECIAL_TOKENS", "")
    assert reload_params().SPECIAL_TOKENS == [""]


def test_paths_derived_from_data_dir(reload_params, monkeypatch):
    monkeypatch.setenv("DATA_DIR", "mydata")
    p = reload_params()
    assert p.DATA_DIR == Path("mydata")
    assert p.PROCESSED_DIR == Path("mydata/processed")
    assert p.TOKENIZED_DIR == Path("mydata/tokenized")
    assert p.TRAIN_FILE == Path("mydata/processed/train.jsonl")
    assert p.TRAIN_OUTPUT_FILE == Path("mydata/tokenized/train.jsonl")


def test_explicit_dirs_override_data_dir(reload_params, monkeypatch):
    monkeypatch.setenv("DATA_DIR", "mydata")
    monkeypatch.setenv("PROCESSED_DIR", "elsewhere")
    p = reload_params()
    assert p.PROCESSED_DIR == Path("elsewhere")
    assert p.CLEAN_DATASET_FILE == Path("elsewhere/c4_clean.jsonl")
    assert p.TOKENIZED_DIR == Path("mydata/tokenized")


def test_tokenizer_files(reload_params, monkeypatch):
    monkeypatch.setenv("TOKENIZER_DIR", "tok")
    p = reload_params()
    assert p.TOKENIZER_FILE == Path("tok/tokenizer.json")
    assert p.TOKENIZER_VOCAB_FILE == Path("tok/vocab.json")
    assert p.TOKENIZER_CONFIG_FILE == Path("tok/config.json")


def test_print_parameters(capsys):
    parameters.print_parameters()
    out = capsys.readouterr().out
    for label in ("DATASET", "TOKENIZER", "BITNET", "ENTRAÎNEMENT",
                  "Block size", "Gradient clip"):
        assert label in out
    assert out.count("=" * 70) == 3
