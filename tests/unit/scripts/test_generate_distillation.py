import json
import types

import pytest

from scripts import generate_distillation as module


def write_documents(path, items):
    path.write_text(
        "\n".join(i if isinstance(i, str) else json.dumps(i) for i in items)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_output(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Chemins temporaires et teacher simulé (aucun appel réseau)."""
    state = types.SimpleNamespace(
        input=tmp_path / "c4.jsonl",
        output=tmp_path / "out" / "distilled.jsonl",
        teacher_calls=[],
        response="r" * 80,
        model="vendor/model",
        failing_prompts=set(),
    )
    monkeypatch.setattr(module, "INPUT_PATH", state.input)
    monkeypatch.setattr(module, "OUTPUT_PATH", state.output)
    monkeypatch.setattr(module, "MAX_DOCUMENTS", 3)

    def fake_teacher(prompt, temperature, max_tokens):
        state.teacher_calls.append(
            {"prompt": prompt, "temperature": temperature,
             "max_tokens": max_tokens}
        )
        if any(marker in prompt for marker in state.failing_prompts):
            raise RuntimeError("teacher failed")
        return state.response, state.model

    monkeypatch.setattr(module, "generate_with_teachers", fake_teacher)
    return state


# ------------------------------------------------------------
# load_c4_documents
# ------------------------------------------------------------

def test_loader_yields_texts_in_file_order(tmp_path):
    path = write_documents(tmp_path / "in.jsonl", [{"text": "a"}, {"text": "b"}])

    assert list(module.load_c4_documents(path)) == ["a", "b"]


def test_loader_skips_blank_lines_and_documents_without_text(tmp_path):
    path = write_documents(
        tmp_path / "in.jsonl",
        ["", {"other": 1}, {"text": ""}, {"text": None}, {"text": "ok"}],
    )

    assert list(module.load_c4_documents(path)) == ["ok"]


def test_loader_is_lazy(tmp_path):
    path = write_documents(tmp_path / "in.jsonl", [{"text": "a"}, "{broken"])

    texts = module.load_c4_documents(path)

    assert next(texts) == "a"  # la ligne invalide n'est pas encore lue


def test_loader_raises_on_invalid_json(tmp_path):
    path = write_documents(tmp_path / "in.jsonl", ["{broken"])

    with pytest.raises(json.JSONDecodeError):
        list(module.load_c4_documents(path))


# ------------------------------------------------------------
# process_document
# ------------------------------------------------------------

def test_process_document_saves_valid_example(env):
    assert module.process_document("Texte source du document.", 0) is True

    (example,) = read_output(env.output)
    assert example["source"] == "Texte source du document."
    assert "Texte source du document." in example["prompt"]
    assert example["response"] == env.response
    assert example["teacher"] == "vendor/model"


def test_process_document_sends_built_prompt_with_fixed_sampling(env):
    module.process_document("Mon texte", 0)

    (call,) = env.teacher_calls
    assert "Mon texte" in call["prompt"]
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 8200


def test_process_document_rejects_invalid_response_without_saving(env, capsys):
    env.response = "trop court"

    assert module.process_document("Mon texte", 0) is False

    assert not env.output.exists()
    assert "Réponse rejetée" in capsys.readouterr().out


def test_process_document_propagates_teacher_errors(env):
    env.failing_prompts.add("Mon texte")

    with pytest.raises(RuntimeError, match="teacher failed"):
        module.process_document("Mon texte", 0)

    assert not env.output.exists()


def test_process_document_with_blank_text_raises_before_calling_teacher(env):
    with pytest.raises(ValueError):
        module.process_document("   ", 0)

    assert env.teacher_calls == []


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def test_main_missing_input_raises(env):
    with pytest.raises(FileNotFoundError):
        module.main()


def test_main_processes_documents_and_reports_totals(env, capsys):
    write_documents(env.input, [{"text": f"doc {i}"} for i in range(2)])

    module.main()

    assert len(read_output(env.output)) == 2
    output = capsys.readouterr().out
    assert "Documents traités : 2" in output
    assert "Réponses validées : 2" in output
    assert "Réponses rejetées : 0" in output


def test_main_stops_after_max_documents(env):
    write_documents(env.input, [{"text": f"doc {i}"} for i in range(10)])

    module.main()

    assert len(env.teacher_calls) == 3
    assert len(read_output(env.output)) == 3


def test_main_counts_rejected_responses(env, capsys):
    env.response = "court"
    write_documents(env.input, [{"text": "a"}, {"text": "b"}])

    module.main()

    output = capsys.readouterr().out
    assert "Réponses validées : 0" in output
    assert "Réponses rejetées : 2" in output


def test_main_continues_after_a_failing_document(env, capsys):
    env.failing_prompts.add("document-en-erreur")
    write_documents(
        env.input,
        [{"text": "document-en-erreur"}, {"text": "document valide"}],
    )

    module.main()

    (example,) = read_output(env.output)
    assert example["source"] == "document valide"
    output = capsys.readouterr().out
    assert "Erreur : teacher failed" in output
    assert "Réponses validées : 1" in output
    assert "Réponses rejetées : 1" in output


def test_main_without_documents_reports_zero(env, capsys):
    write_documents(env.input, [""])

    module.main()

    assert "Documents traités : 0" in capsys.readouterr().out
    assert not env.output.exists()
