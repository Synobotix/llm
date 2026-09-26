import json
import sys
import types

import pytest

from scripts import tokenize_dataset


class CharTokenizer:
    """Tokenizer factice : un token par caractère."""

    def encode(self, text):
        return types.SimpleNamespace(ids=[ord(char) for char in text])

    def get_vocab_size(self):
        return 99


def write_lines(path, items):
    lines = [
        item if isinstance(item, str) else json.dumps(item)
        for item in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_lines(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def sequence(start, block=4):
    ids = list(range(start, start + block + 1))
    return {"input_ids": ids[:-1], "labels": ids[1:]}


@pytest.fixture(autouse=True)
def small_block_size(monkeypatch):
    monkeypatch.setattr(tokenize_dataset, "BLOCK_SIZE", 4)


# ------------------------------------------------------------
# read_documents
# ------------------------------------------------------------

def test_read_documents_keeps_stripped_valid_texts(tmp_path):
    path = tmp_path / "in.jsonl"
    write_lines(path, [{"text": "  un  "}, {"text": "deux"}])

    assert tokenize_dataset.read_documents(path) == ["un", "deux"]


def test_read_documents_skips_blank_empty_and_non_string(tmp_path, capsys):
    path = tmp_path / "in.jsonl"
    write_lines(path, [
        "",
        {"text": "   "},
        {"text": 12},
        {"text": None},
        {"other": "x"},
        "{broken",
        {"text": "ok"},
    ])

    assert tokenize_dataset.read_documents(path) == ["ok"]
    assert "[WARNING] Ligne 6 ignorée :" in capsys.readouterr().out


def test_read_documents_empty_file(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text("", encoding="utf-8")

    assert tokenize_dataset.read_documents(path) == []


# ------------------------------------------------------------
# tokenize_text / create_sequences
# ------------------------------------------------------------

def test_tokenize_text_returns_ids():
    assert tokenize_dataset.tokenize_text(CharTokenizer(), "ab") == [97, 98]


@pytest.mark.parametrize(
    "token_count, expected_starts",
    [
        (0, []),
        (4, []),
        (5, [0]),
        (8, [0]),
        (9, [0, 4]),
        (13, [0, 4, 8]),
    ],
)
def test_create_sequences(token_count, expected_starts):
    sequences = tokenize_dataset.create_sequences(list(range(token_count)))

    assert sequences == [sequence(start) for start in expected_starts]


# ------------------------------------------------------------
# process_dataset
# ------------------------------------------------------------

def test_process_dataset_writes_sequences_and_returns_count(
    tmp_path, capsys
):
    source = tmp_path / "in.jsonl"
    target = tmp_path / "nested" / "out.jsonl"
    write_lines(
        source, [{"text": "abcde"}, {"text": "ab"}, {"text": "abcdefghi"}]
    )

    count = tokenize_dataset.process_dataset(CharTokenizer(), source, target)

    assert count == 3
    sequences = read_lines(target)
    assert len(sequences) == 3
    assert sequences[0]["input_ids"] == [97, 98, 99, 100]
    assert sequences[0]["labels"] == [98, 99, 100, 101]
    out = capsys.readouterr().out
    assert "Documents lus : 3" in out
    assert "Tokens totaux      : 16" in out
    assert "Séquences créées   : 3" in out


def test_process_dataset_with_no_documents(tmp_path):
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    source.write_text("", encoding="utf-8")

    assert tokenize_dataset.process_dataset(
        CharTokenizer(), source, target
    ) == 0
    assert target.read_text(encoding="utf-8") == ""


# ------------------------------------------------------------
# verify_dataset
# ------------------------------------------------------------

def test_verify_dataset_accepts_valid_file(tmp_path, capsys):
    path = tmp_path / "ok.jsonl"
    write_lines(path, [sequence(0), "", sequence(10)])

    tokenize_dataset.verify_dataset(path)

    out = capsys.readouterr().out
    assert "Séquences vérifiées : 2" in out
    assert "✓ Dataset valide" in out


@pytest.mark.parametrize(
    "bad_line, message",
    [
        ("{broken", "[ERROR] Ligne 1 invalide."),
        (
            {"input_ids": [1, 2, 3], "labels": [2, 3, 4, 5]},
            "[ERROR] Ligne 1 : input_ids = 3",
        ),
        (
            {"input_ids": [1, 2, 3, 4], "labels": [2, 3]},
            "[ERROR] Ligne 1 : labels = 2",
        ),
        (
            {"input_ids": "x", "labels": [2, 3, 4, 5]},
            "[ERROR] Ligne 1 : input_ids = INVALID",
        ),
        (
            {"input_ids": [1, 2, 3, 4]},
            "[ERROR] Ligne 1 : labels = INVALID",
        ),
        (
            {"input_ids": [1, 2, 3, 4], "labels": [9, 9, 9, 9]},
            "[ERROR] Ligne 1 : input_ids / labels mal décalés.",
        ),
    ],
)
def test_verify_dataset_rejects_invalid_lines(
    tmp_path, capsys, bad_line, message
):
    path = tmp_path / "bad.jsonl"
    write_lines(path, [bad_line])

    with pytest.raises(ValueError, match="séquences invalides"):
        tokenize_dataset.verify_dataset(path)

    assert message in capsys.readouterr().out


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    train = tmp_path / "processed" / "train.jsonl"
    validation = tmp_path / "processed" / "validation.jsonl"
    train.parent.mkdir()
    write_lines(train, [{"text": "abcdefghi"}])
    write_lines(validation, [{"text": "abcde"}])
    monkeypatch.setattr(tokenize_dataset, "TRAIN_FILE", train)
    monkeypatch.setattr(tokenize_dataset, "VALIDATION_FILE", validation)
    monkeypatch.setattr(
        tokenize_dataset, "TRAIN_OUTPUT_FILE", tmp_path / "tok" / "train.jsonl"
    )
    monkeypatch.setattr(
        tokenize_dataset,
        "VALIDATION_OUTPUT_FILE",
        tmp_path / "tok" / "validation.jsonl",
    )
    monkeypatch.setattr(tokenize_dataset, "load_tokenizer", CharTokenizer)
    return train, validation


def install_hub(monkeypatch, uploads):
    module = types.ModuleType("scripts.hub.hub_sync")
    module.upload_file_to_hub = lambda path, remote_path: uploads.append(
        (path, remote_path)
    )
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", module)


def test_main_runs_the_whole_pipeline_and_uploads(
    pipeline, monkeypatch, capsys
):
    uploads = []
    install_hub(monkeypatch, uploads)

    tokenize_dataset.main()

    assert len(read_lines(tokenize_dataset.TRAIN_OUTPUT_FILE)) == 2
    assert len(read_lines(tokenize_dataset.VALIDATION_OUTPUT_FILE)) == 1
    assert uploads == [
        (tokenize_dataset.TRAIN_OUTPUT_FILE, "data/tokenized/train.jsonl"),
        (
            tokenize_dataset.VALIDATION_OUTPUT_FILE,
            "data/tokenized/validation.jsonl",
        ),
    ]
    out = capsys.readouterr().out
    assert "Vocabulaire : 99" in out
    assert "Train      : 2 séquences" in out
    assert "Validation : 1 séquences" in out


def test_main_reports_upload_failure_without_raising(
    pipeline, monkeypatch, capsys
):
    module = types.ModuleType("scripts.hub.hub_sync")

    def failing_upload(path, remote_path):
        raise RuntimeError("hub down")

    module.upload_file_to_hub = failing_upload
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", module)

    tokenize_dataset.main()

    assert "⚠️ Upload dataset tokenisé échoué : hub down" in (
        capsys.readouterr().out
    )


def test_main_survives_a_missing_hub_module(pipeline, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", None)

    tokenize_dataset.main()

    assert "Upload dataset tokenisé échoué" in capsys.readouterr().out


def test_main_requires_train_file(pipeline):
    train, _ = pipeline
    train.unlink()

    with pytest.raises(FileNotFoundError, match="Dataset train introuvable"):
        tokenize_dataset.main()


def test_main_requires_validation_file(pipeline):
    _, validation = pipeline
    validation.unlink()

    with pytest.raises(
        FileNotFoundError, match="Dataset validation introuvable"
    ):
        tokenize_dataset.main()
