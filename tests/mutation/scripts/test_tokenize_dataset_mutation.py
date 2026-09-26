import json
import sys
import types
from pathlib import Path

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


def sequence(start, block=4):
    ids = list(range(start, start + block + 1))
    return {"input_ids": ids[:-1], "labels": ids[1:]}


def spy_on_path_open(monkeypatch):
    calls = []
    original_open = Path.open

    def spy(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    return calls


def processing_block(source, target, documents, tokens, sequences):
    return (
        "\n" + "=" * 60 + "\n"
        + f"TRAITEMENT : {source}\n"
        + "=" * 60 + "\n"
        + f"Documents lus : {documents}\n"
        + f"Tokens totaux      : {tokens}\n"
        + f"Séquences créées   : {sequences}\n"
        + "Block size          : 4\n"
        + "Longueur input_ids  : 4\n"
        + "Longueur labels     : 4\n"
        + f"Fichier             : {target}\n"
    )


def verification_block(path, verified, errors):
    return (
        "\n" + "=" * 60 + "\n"
        + f"VÉRIFICATION : {path}\n"
        + "=" * 60 + "\n"
        + f"\nSéquences vérifiées : {verified}\n"
        + f"Erreurs              : {errors}\n"
    )


@pytest.fixture(autouse=True)
def small_block_size(monkeypatch):
    monkeypatch.setattr(tokenize_dataset, "BLOCK_SIZE", 4)


# ------------------------------------------------------------
# read_documents
# ------------------------------------------------------------

def test_read_documents_opens_file_as_utf8_text(tmp_path, monkeypatch):
    path = tmp_path / "in.jsonl"
    write_lines(path, [{"text": "a"}])
    calls = spy_on_path_open(monkeypatch)

    tokenize_dataset.read_documents(path)

    assert calls == [(path, ("r",), {"encoding": "utf-8"})]


def test_read_documents_warning_message_is_exact(tmp_path, capsys):
    path = tmp_path / "in.jsonl"
    write_lines(path, ["{broken"])

    tokenize_dataset.read_documents(path)

    out = capsys.readouterr().out
    assert out.startswith("[WARNING] Ligne 1 ignorée : ")


# ------------------------------------------------------------
# process_dataset
# ------------------------------------------------------------

def test_process_dataset_exact_output(tmp_path, capsys):
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    write_lines(
        source, [{"text": "abcde"}, {"text": "ab"}, {"text": "abcdefghi"}]
    )

    tokenize_dataset.process_dataset(CharTokenizer(), source, target)

    assert capsys.readouterr().out == processing_block(
        source, target, documents=3, tokens=16, sequences=3
    )


def test_process_dataset_creates_all_missing_parent_directories(tmp_path):
    source = tmp_path / "in.jsonl"
    target = tmp_path / "level_one" / "level_two" / "out.jsonl"
    source.write_text("", encoding="utf-8")

    tokenize_dataset.process_dataset(CharTokenizer(), source, target)

    assert target.exists()


def test_process_dataset_accepts_existing_output_directory(tmp_path):
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out" / "out.jsonl"
    target.parent.mkdir()
    source.write_text("", encoding="utf-8")

    tokenize_dataset.process_dataset(CharTokenizer(), source, target)

    assert target.exists()


def test_process_dataset_opens_files_as_utf8_text(tmp_path, monkeypatch):
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    write_lines(source, [{"text": "abcde"}])
    calls = spy_on_path_open(monkeypatch)

    tokenize_dataset.process_dataset(CharTokenizer(), source, target)

    assert calls == [
        (source, ("r",), {"encoding": "utf-8"}),
        (target, ("w",), {"encoding": "utf-8"}),
    ]


def test_process_dataset_writes_one_compact_json_object_per_line(tmp_path):
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    write_lines(source, [{"text": "abcde"}])

    tokenize_dataset.process_dataset(CharTokenizer(), source, target)

    assert target.read_text(encoding="utf-8") == (
        '{"input_ids": [97, 98, 99, 100], "labels": [98, 99, 100, 101]}\n'
    )


# ------------------------------------------------------------
# verify_dataset
# ------------------------------------------------------------

def test_verify_dataset_exact_output_for_valid_file(tmp_path, capsys):
    path = tmp_path / "ok.jsonl"
    write_lines(path, [sequence(0), sequence(10)])

    tokenize_dataset.verify_dataset(path)

    assert capsys.readouterr().out == (
        verification_block(path, verified=2, errors=0) + "✓ Dataset valide\n"
    )


def test_verify_dataset_counts_every_kind_of_error(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    short_ids = {"input_ids": [1, 2, 3], "labels": [2, 3, 4, 5]}
    short_labels = {"input_ids": [1, 2, 3, 4], "labels": [2, 3]}
    shifted = {"input_ids": [1, 2, 3, 4], "labels": [9, 9, 9, 9]}
    write_lines(path, [
        "{b1", "{b2",
        short_ids, short_ids,
        short_labels, short_labels,
        shifted, shifted,
    ])

    with pytest.raises(ValueError) as error:
        tokenize_dataset.verify_dataset(path)

    assert str(error.value) == "Le dataset contient des séquences invalides."
    assert capsys.readouterr().out == (
        "\n" + "=" * 60 + "\n"
        + f"VÉRIFICATION : {path}\n"
        + "=" * 60 + "\n"
        + "[ERROR] Ligne 1 invalide.\n"
        + "[ERROR] Ligne 2 invalide.\n"
        + "[ERROR] Ligne 3 : input_ids = 3\n"
        + "[ERROR] Ligne 4 : input_ids = 3\n"
        + "[ERROR] Ligne 5 : labels = 2\n"
        + "[ERROR] Ligne 6 : labels = 2\n"
        + "[ERROR] Ligne 7 : input_ids / labels mal décalés.\n"
        + "[ERROR] Ligne 8 : input_ids / labels mal décalés.\n"
        + "\nSéquences vérifiées : 6\n"
        + "Erreurs              : 8\n"
    )


def test_verify_dataset_opens_file_as_utf8_text(tmp_path, monkeypatch):
    path = tmp_path / "ok.jsonl"
    write_lines(path, [sequence(0)])
    calls = spy_on_path_open(monkeypatch)

    tokenize_dataset.verify_dataset(path)

    assert calls == [(path, ("r",), {"encoding": "utf-8"})]


def test_verify_dataset_error_on_both_fields_counts_two_errors(
    tmp_path, capsys
):
    path = tmp_path / "bad.jsonl"
    write_lines(path, [{"input_ids": "x", "labels": "y"}])

    with pytest.raises(ValueError):
        tokenize_dataset.verify_dataset(path)

    out = capsys.readouterr().out
    assert "input_ids = INVALID" in out
    assert "labels = INVALID" in out
    assert "Erreurs              : 2" in out


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
    train_out = tmp_path / "tok" / "train.jsonl"
    validation_out = tmp_path / "tok" / "validation.jsonl"
    monkeypatch.setattr(tokenize_dataset, "TRAIN_FILE", train)
    monkeypatch.setattr(tokenize_dataset, "VALIDATION_FILE", validation)
    monkeypatch.setattr(tokenize_dataset, "TRAIN_OUTPUT_FILE", train_out)
    monkeypatch.setattr(
        tokenize_dataset, "VALIDATION_OUTPUT_FILE", validation_out
    )
    monkeypatch.setattr(tokenize_dataset, "load_tokenizer", CharTokenizer)
    return train, validation, train_out, validation_out


def test_main_exact_output(pipeline, monkeypatch, capsys):
    train, validation, train_out, validation_out = pipeline
    module = types.ModuleType("scripts.hub.hub_sync")
    module.upload_file_to_hub = lambda path, remote_path: None
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", module)

    tokenize_dataset.main()

    assert capsys.readouterr().out == (
        "=" * 60 + "\n"
        + "TOKENISATION DU DATASET\n"
        + "=" * 60 + "\n"
        + "\nChargement du tokenizer...\n"
        + "Vocabulaire : 99\n"
        + "Block size : 4\n"
        + processing_block(train, train_out, 1, 9, 2)
        + processing_block(validation, validation_out, 1, 5, 1)
        + verification_block(train_out, 2, 0) + "✓ Dataset valide\n"
        + verification_block(validation_out, 1, 0) + "✓ Dataset valide\n"
        + "\n" + "=" * 60 + "\n"
        + "TOKENISATION TERMINÉE\n"
        + "=" * 60 + "\n"
        + "\nVocabulaire : 99\n"
        + "Train      : 2 séquences\n"
        + "Validation : 1 séquences\n"
        + "\nFichiers générés :\n"
        + f"  → {train_out}\n"
        + f"  → {validation_out}\n"
        + "\n✓ Toutes les séquences ont 4 input_ids et 4 labels.\n"
        + "\n" + "=" * 70 + "\n"
        + "UPLOAD VERS HUGGING FACE\n"
        + "=" * 70 + "\n"
    )


def test_main_failure_message_is_exact(pipeline, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", None)

    tokenize_dataset.main()

    out = capsys.readouterr().out
    assert out.rstrip("\n").splitlines()[-1].startswith(
        "⚠️ Upload dataset tokenisé échoué : "
    )


def test_main_missing_file_messages_name_the_path(pipeline):
    train, validation, _, _ = pipeline
    train.unlink()

    with pytest.raises(FileNotFoundError) as error:
        tokenize_dataset.main()

    assert str(error.value) == f"Dataset train introuvable : {train}"
