import builtins
from datetime import datetime

import pytest

from scripts import prepare_data


class FakeProgress:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeProgress.instances.append(self)

    def update(self, amount):
        pass

    def close(self):
        pass


@pytest.fixture
def output_file(tmp_path, monkeypatch):
    path = tmp_path / "out" / "c4.jsonl"
    monkeypatch.setattr(prepare_data, "OUTPUT_FILE", path)
    monkeypatch.setattr(prepare_data, "tqdm", FakeProgress)
    FakeProgress.instances = []
    return path


def test_progress_bar_arguments(output_file, monkeypatch):
    monkeypatch.setattr(prepare_data, "MAX_DOCUMENTS", 7)
    monkeypatch.setattr(prepare_data, "load_c4", lambda: iter([]))

    prepare_data.prepare_c4()

    assert FakeProgress.instances[0].kwargs == {
        "total": 7,
        "desc": "Récupération C4",
        "unit": "doc",
    }


def test_output_file_is_opened_for_utf8_writing(output_file, monkeypatch):
    monkeypatch.setattr(prepare_data, "load_c4", lambda: iter([]))
    calls = []
    original_open = builtins.open

    def spy_open(*args, **kwargs):
        calls.append((args, kwargs))
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)

    prepare_data.prepare_c4()

    assert calls == [((output_file, "w"), {"encoding": "utf-8"})]


def test_output_directory_is_created_even_if_it_exists(
    output_file, monkeypatch
):
    output_file.parent.mkdir(parents=True)
    monkeypatch.setattr(prepare_data, "load_c4", lambda: iter([]))

    prepare_data.prepare_c4()

    assert output_file.exists()


def test_exact_summary_output(output_file, monkeypatch, capsys):
    monkeypatch.setattr(
        prepare_data,
        "load_c4",
        lambda: iter([{"text": "a"}, {"text": "b"}]),
    )

    prepare_data.prepare_c4()

    assert capsys.readouterr().out == (
        "\n"
        + "=" * 60 + "\n"
        + "RÉCUPÉRATION TERMINÉE\n"
        + "=" * 60 + "\n"
        + "Documents récupérés : 2\n"
        + f"Fichier             : {output_file}\n"
        + "=" * 60 + "\n"
    )


def test_timestamp_object_without_isoformat_uses_str():
    class Stamp:
        def __str__(self):
            return "stamp"

    assert prepare_data.serialize_timestamp(Stamp()) == "stamp"


def test_timestamp_with_isoformat_is_not_stringified():
    assert prepare_data.serialize_timestamp(datetime(2020, 1, 1)) == (
        "2020-01-01T00:00:00"
    )


def test_stops_exactly_at_limit_even_with_more_documents(
    output_file, monkeypatch
):
    monkeypatch.setattr(prepare_data, "MAX_DOCUMENTS", 1)
    monkeypatch.setattr(
        prepare_data,
        "load_c4",
        lambda: iter([{"text": "a"}, {"text": "b"}]),
    )

    prepare_data.prepare_c4()

    assert output_file.read_text(encoding="utf-8").count("\n") == 1


def test_missing_parent_directories_are_all_created(tmp_path, monkeypatch):
    path = tmp_path / "level_one" / "level_two" / "c4.jsonl"
    monkeypatch.setattr(prepare_data, "OUTPUT_FILE", path)
    monkeypatch.setattr(prepare_data, "tqdm", FakeProgress)
    monkeypatch.setattr(prepare_data, "load_c4", lambda: iter([]))

    prepare_data.prepare_c4()

    assert path.exists()


def test_summary_is_printed_when_the_limit_stops_the_loop(
    output_file, monkeypatch, capsys
):
    monkeypatch.setattr(prepare_data, "MAX_DOCUMENTS", 1)
    monkeypatch.setattr(
        prepare_data,
        "load_c4",
        lambda: iter([{"text": "a"}, {"text": "b"}]),
    )

    prepare_data.prepare_c4()

    assert "Documents récupérés : 1" in capsys.readouterr().out
