import pytest

import llm.config.parameters as parameters

PATCHED_VALUES = {
    "MAX_DOCUMENT": 11,
    "TRAIN_RATIO": 0.5,
    "RANDOM_SEED": 12,
    "MAX_VOCAB_SIZE": 13,
    "MIN_FREQUENCY": 14,
    "BLOCK_SIZE": 15,
    "D_MODEL": 16,
    "NUM_HEADS": 17,
    "HIDDEN_DIM": 18,
    "NUM_BLOCKS": 19,
    "MAX_SEQUENCE_LENGTH": 20,
    "BATCH_SIZE": 21,
    "EPOCHS": 22,
    "LEARNING_RATE": 0.25,
    "WEIGHT_DECAY": 0.75,
    "GRADIENT_CLIP": 2.5,
}

EXPECTED_OUTPUT = f"""{"=" * 70}
PARAMÈTRES DU PROJET BITNET
{"=" * 70}

DATASET
  Documents maximum : 11
  Train ratio       : 0.5
  Random seed       : 12

TOKENIZER
  Max vocab size    : 13
  Min frequency     : 14

TOKENISATION
  Block size        : 15

BITNET
  D model           : 16
  Num heads         : 17
  Hidden dim        : 18
  Num blocks        : 19
  Max sequence      : 20

ENTRAÎNEMENT
  Batch size        : 21
  Epochs            : 22
  Learning rate     : 0.25
  Weight decay      : 0.75
  Gradient clip     : 2.5
{"=" * 70}
"""


@pytest.fixture
def patched_parameters(monkeypatch):
    for name, value in PATCHED_VALUES.items():
        monkeypatch.setattr(parameters, name, value)


def test_print_parameters_exact_output(patched_parameters, capsys):
    parameters.print_parameters()

    assert capsys.readouterr().out == EXPECTED_OUTPUT
