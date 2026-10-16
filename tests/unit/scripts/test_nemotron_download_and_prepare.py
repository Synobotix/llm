import json
import sys
import types

import pytest

from scripts.nemotron import download_and_prepare as module


def read_texts(path):
    return [
        json.loads(line)["text"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Dataset en streaming simulé, sortie temporaire, upload simulé."""
    state = types.SimpleNamespace(
        output=tmp_path / "processed" / "nemotron.jsonl",
        examples=[],
        load_args=None,
        uploads=[],
    )
    monkeypatch.setattr(module, "CLEAN_DATASET_FILE", state.output)
    monkeypatch.setattr(module, "MAX_DOCUMENT", 100)

    def load_dataset(dataset_id, split, streaming):
        state.load_args = (dataset_id, split, streaming)
        return iter(state.examples)

    monkeypatch.setattr(module, "load_dataset", load_dataset)

    hub = types.ModuleType("hub")
    hub.upload_file_to_hub = lambda path, remote_path: state.uploads.append(
        (path.name, remote_path)
    )
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", hub)
    return state


def conversation(*pairs):
    return {"messages": [{"role": r, "content": c} for r, c in pairs]}


# ------------------------------------------------------------
# format_messages
# ------------------------------------------------------------

def test_format_messages_prefixes_each_message_with_its_role():
    text = module.format_messages(
        [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Salut"},
        ]
    )

    assert text == "<|user|>\nBonjour\n\n<|assistant|>\nSalut"


def test_format_messages_skips_empty_content():
    text = module.format_messages(
        [
            {"role": "user", "content": ""},
            {"role": "assistant"},
            {"role": "user", "content": "ok"},
        ]
    )

    assert text == "<|user|>\nok"


def test_format_messages_without_role_uses_empty_role():
    assert module.format_messages([{"content": "x"}]) == "<||>\nx"


def test_format_messages_empty_list_gives_empty_text():
    assert module.format_messages([]) == ""


# ------------------------------------------------------------
# download_and_prepare
# ------------------------------------------------------------

def test_streams_the_configured_dataset_and_split(env):
    module.download_and_prepare()

    assert env.load_args == (module.HF_DATASET_ID, module.TARGET_SPLIT, True)


def test_writes_one_text_record_per_conversation(env):
    env.examples = [
        conversation(("user", "q1"), ("assistant", "a1")),
        conversation(("user", "q2")),
    ]

    path = module.download_and_prepare()

    assert path == env.output
    assert read_texts(env.output) == [
        "<|user|>\nq1\n\n<|assistant|>\na1",
        "<|user|>\nq2",
    ]


def test_records_only_contain_the_text_field(env):
    env.examples = [conversation(("user", "q"))]

    module.download_and_prepare()

    assert list(json.loads(env.output.read_text())) == ["text"]


def test_skips_examples_without_usable_messages(env, capsys):
    env.examples = [
        {},
        {"messages": []},
        conversation(("user", "")),
        conversation(("user", "valide")),
    ]

    module.download_and_prepare()

    assert read_texts(env.output) == ["<|user|>\nvalide"]
    assert "Ignorés   : 3" in capsys.readouterr().out


def test_stops_at_max_document(env, monkeypatch):
    monkeypatch.setattr(module, "MAX_DOCUMENT", 2)
    env.examples = [conversation(("user", f"q{i}")) for i in range(10)]

    module.download_and_prepare()

    assert len(read_texts(env.output)) == 2


def test_non_ascii_text_is_kept_readable(env):
    env.examples = [conversation(("user", "é à ü — 日本語"))]

    module.download_and_prepare()

    assert "é à ü — 日本語" in env.output.read_text(encoding="utf-8")


def test_creates_output_directory(env):
    module.download_and_prepare()

    assert env.output.parent.is_dir()


def test_overwrites_previous_output(env):
    env.output.parent.mkdir(parents=True)
    env.output.write_text('{"text": "old"}\n')
    env.examples = [conversation(("user", "new"))]

    module.download_and_prepare()

    assert read_texts(env.output) == ["<|user|>\nnew"]


def test_uploads_prepared_file_to_hub(env):
    env.examples = [conversation(("user", "q"))]

    module.download_and_prepare()

    assert env.uploads == [("nemotron.jsonl", "data/processed/nemotron.jsonl")]


def test_upload_failure_does_not_fail_the_preparation(env, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", None)
    env.examples = [conversation(("user", "q"))]

    path = module.download_and_prepare()

    assert path.exists()
    assert "Upload Nemotron brut échoué" in capsys.readouterr().out
