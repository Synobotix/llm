import json
from datetime import datetime

import pytest

from scripts import prepare_data


class FakeProgress:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = 0
        self.closed = False
        FakeProgress.instances.append(self)

    def update(self, amount):
        self.updates += amount

    def close(self):
        self.closed = True


@pytest.fixture
def output_file(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "c4.jsonl"
    monkeypatch.setattr(prepare_data, "OUTPUT_FILE", path)
    monkeypatch.setattr(prepare_data, "tqdm", FakeProgress)
    FakeProgress.instances = []
    return path


def read_documents(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (datetime(2024, 5, 6, 7, 8, 9), "2024-05-06T07:08:09"),
        ("2024-05-06", "2024-05-06"),
        (12345, "12345"),
    ],
)
def test_serialize_timestamp(value, expected):
    assert prepare_data.serialize_timestamp(value) == expected


def test_load_c4_requests_french_streaming_train_split(monkeypatch):
    calls = []
    monkeypatch.setattr(
        prepare_data,
        "load_dataset",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "dataset",
    )

    assert prepare_data.load_c4() == "dataset"
    assert calls == [
        (("allenai/c4", "fr"), {"split": "train", "streaming": True})
    ]


def test_prepare_c4_writes_one_json_line_per_document(
    output_file, monkeypatch
):
    examples = [
        {"text": "é à", "url": "u1", "timestamp": datetime(2024, 1, 2)},
        {"text": "deux", "url": "u2", "timestamp": None},
    ]
    monkeypatch.setattr(prepare_data, "load_c4", lambda: iter(examples))

    prepare_data.prepare_c4()

    assert read_documents(output_file) == [
        {"text": "é à", "url": "u1", "timestamp": "2024-01-02T00:00:00"},
        {"text": "deux", "url": "u2", "timestamp": None},
    ]
    assert "é à" in output_file.read_text(encoding="utf-8")


def test_prepare_c4_fills_missing_fields(output_file, monkeypatch):
    monkeypatch.setattr(prepare_data, "load_c4", lambda: iter([{}]))

    prepare_data.prepare_c4()

    assert read_documents(output_file) == [
        {"text": "", "url": "", "timestamp": None}
    ]


def test_prepare_c4_stops_at_max_documents(output_file, monkeypatch):
    monkeypatch.setattr(prepare_data, "MAX_DOCUMENTS", 3)
    consumed = []

    def source():
        for index in range(10):
            consumed.append(index)
            yield {"text": str(index)}

    monkeypatch.setattr(prepare_data, "load_c4", source)

    prepare_data.prepare_c4()

    assert [doc["text"] for doc in read_documents(output_file)] == [
        "0", "1", "2",
    ]
    assert consumed == [0, 1, 2]


def test_prepare_c4_with_empty_dataset_writes_empty_file(
    output_file, monkeypatch, capsys
):
    monkeypatch.setattr(prepare_data, "load_c4", lambda: iter([]))

    prepare_data.prepare_c4()

    assert output_file.read_text(encoding="utf-8") == ""
    assert "Documents récupérés : 0" in capsys.readouterr().out


def test_prepare_c4_overwrites_existing_file(output_file, monkeypatch):
    output_file.parent.mkdir(parents=True)
    output_file.write_text("ancien contenu\n", encoding="utf-8")
    monkeypatch.setattr(
        prepare_data, "load_c4", lambda: iter([{"text": "neuf"}])
    )

    prepare_data.prepare_c4()

    assert [doc["text"] for doc in read_documents(output_file)] == ["neuf"]


def test_prepare_c4_updates_and_closes_progress_bar(
    output_file, monkeypatch
):
    monkeypatch.setattr(
        prepare_data, "load_c4", lambda: iter([{"text": "a"}, {"text": "b"}])
    )

    prepare_data.prepare_c4()

    progress = FakeProgress.instances[0]
    assert progress.updates == 2
    assert progress.closed is True
