import json

import pytest

from scripts import split_dataset


def write_lines(path, items):
    lines = [
        item if isinstance(item, str) else json.dumps(item)
        for item in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_texts(path):
    return [
        json.loads(line)["text"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    source = tmp_path / "clean.jsonl"
    train = tmp_path / "out" / "train.jsonl"
    validation = tmp_path / "out" / "validation.jsonl"
    monkeypatch.setattr(split_dataset, "CLEAN_DATASET_FILE", source)
    monkeypatch.setattr(split_dataset, "TRAIN_FILE", train)
    monkeypatch.setattr(split_dataset, "VALIDATION_FILE", validation)
    monkeypatch.setattr(split_dataset, "RANDOM_SEED", 1)
    monkeypatch.setattr(split_dataset, "TRAIN_RATIO", 0.8)
    monkeypatch.setattr(split_dataset, "MAX_DOCUMENT", 100)
    # Pas d'upload réseau : le module hub est remplacé par un échec contrôlé.
    monkeypatch.setitem(__import__("sys").modules, "scripts.hub.hub_sync", None)
    return source, train, validation


# ------------------------------------------------------------
# read_documents
# ------------------------------------------------------------

def test_read_documents_keeps_valid_documents(tmp_path):
    path = tmp_path / "in.jsonl"
    write_lines(path, [{"text": "a", "id": 1}, {"text": "b"}])

    assert split_dataset.read_documents(path) == [
        {"text": "a", "id": 1},
        {"text": "b"},
    ]


def test_read_documents_skips_blank_invalid_and_non_text(tmp_path, capsys):
    path = tmp_path / "in.jsonl"
    write_lines(
        path,
        [
            "",
            "{not json",
            {"other": "x"},
            {"text": 5},
            {"text": "   "},
            {"text": " keep "},
        ],
    )

    documents = split_dataset.read_documents(path)

    assert documents == [{"text": " keep "}]
    assert "Ligne 2 ignorée" in capsys.readouterr().out


def test_read_documents_empty_file(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text("", encoding="utf-8")

    assert split_dataset.read_documents(path) == []


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def test_main_missing_source_raises(paths):
    with pytest.raises(FileNotFoundError):
        split_dataset.main()


def test_main_splits_with_train_ratio_and_no_overlap(paths):
    source, train, validation = paths
    write_lines(source, [{"text": f"doc{i}"} for i in range(10)])

    split_dataset.main()

    train_texts = read_texts(train)
    validation_texts = read_texts(validation)
    assert len(train_texts) == 8
    assert len(validation_texts) == 2
    assert not set(train_texts) & set(validation_texts)
    assert set(train_texts) | set(validation_texts) == {
        f"doc{i}" for i in range(10)
    }


def test_main_is_reproducible_with_same_seed(paths):
    source, train, _ = paths
    write_lines(source, [{"text": f"doc{i}"} for i in range(20)])

    split_dataset.main()
    first = read_texts(train)
    split_dataset.main()

    assert read_texts(train) == first


def test_main_respects_max_document(paths, monkeypatch):
    source, train, validation = paths
    monkeypatch.setattr(split_dataset, "MAX_DOCUMENT", 5)
    write_lines(source, [{"text": f"doc{i}"} for i in range(10)])

    split_dataset.main()

    assert len(read_texts(train)) + len(read_texts(validation)) == 5


def test_main_uses_all_documents_when_max_is_too_large(paths, capsys):
    source, train, validation = paths
    write_lines(source, [{"text": f"doc{i}"} for i in range(4)])

    split_dataset.main()

    assert len(read_texts(train)) + len(read_texts(validation)) == 4
    assert "Utilisation de 4 documents" in capsys.readouterr().out


def test_main_preserves_non_ascii_text(paths):
    source, train, validation = paths
    write_lines(source, [{"text": "é à ü"}])
    split_dataset.TRAIN_RATIO = 1.0

    split_dataset.main()

    assert "é à ü" in train.read_text(encoding="utf-8")


def test_main_survives_upload_failure(paths, capsys):
    source, _, _ = paths
    write_lines(source, [{"text": "a"}, {"text": "b"}])

    split_dataset.main()

    assert "Upload split échoué" in capsys.readouterr().out
