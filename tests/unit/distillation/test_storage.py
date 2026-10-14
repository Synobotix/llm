import json

from llm.distillation.storage import save_distilled_example


def read(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def save(path, **overrides):
    arguments = dict(
        output_path=path,
        source="texte source",
        prompt="prompt",
        response="réponse",
        teacher="teacher/model",
    )
    arguments.update(overrides)
    save_distilled_example(**arguments)


def test_writes_one_json_line_with_all_fields(tmp_path):
    path = tmp_path / "out.jsonl"

    save(path)

    assert read(path) == [
        {
            "source": "texte source",
            "prompt": "prompt",
            "response": "réponse",
            "teacher": "teacher/model",
        }
    ]


def test_each_call_appends_a_new_line(tmp_path):
    path = tmp_path / "out.jsonl"

    save(path, response="un")
    save(path, response="deux")

    assert [e["response"] for e in read(path)] == ["un", "deux"]


def test_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "out.jsonl"

    save(path)

    assert path.is_file()


def test_accepts_string_path(tmp_path):
    path = tmp_path / "out.jsonl"

    save(str(path))

    assert len(read(path)) == 1


def test_non_ascii_text_is_kept_readable(tmp_path):
    path = tmp_path / "out.jsonl"

    save(path, response="é à ü — 日本語")

    assert "é à ü — 日本語" in path.read_text(encoding="utf-8")


def test_newlines_in_text_do_not_break_the_jsonl_format(tmp_path):
    path = tmp_path / "out.jsonl"

    save(path, source="ligne 1\nligne 2")

    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert read(path)[0]["source"] == "ligne 1\nligne 2"


def test_existing_content_is_preserved(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"source": "old"}\n', encoding="utf-8")

    save(path)

    assert [e["source"] for e in read(path)] == ["old", "texte source"]
