import types

import pytest

from scripts.bitnet.reset_data import delete_all_data_json_jsonl_pt_hf as hf_module
from scripts.bitnet.reset_data import delete_all_data_json_jsonl_pt_local as local_module


# ------------------------------------------------------------
# suppression locale (toujours dans tmp_path : chemins relatifs !)
# ------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_delete_target_absent_path_returns_zero(workspace, capsys):
    assert local_module.delete_target("nothing") == 0
    assert "Absent" in capsys.readouterr().out


def test_delete_target_removes_a_file(workspace):
    (workspace / "a.json").write_text("{}")

    assert local_module.delete_target("a.json") == 1
    assert not (workspace / "a.json").exists()


def test_delete_target_removes_a_directory_recursively(workspace):
    (workspace / "dir" / "sub").mkdir(parents=True)
    (workspace / "dir" / "sub" / "f.pt").write_bytes(b"x" * 100)

    assert local_module.delete_target("dir") == 1
    assert not (workspace / "dir").exists()


def test_recreate_empty_dirs_adds_gitkeep(workspace):
    local_module.recreate_empty_dirs()

    for directory in (
        "data/processed",
        "data/tokenized",
        "checkpoints/bitnet",
        "documentation/result/bitnet",
    ):
        assert (workspace / directory / ".gitkeep").is_file()


def test_main_deletes_targets_and_recreates_directories(workspace, capsys):
    (workspace / "data/processed").mkdir(parents=True)
    (workspace / "data/processed/train.jsonl").write_text("x")
    (workspace / "tokenizer").mkdir()
    (workspace / "tokenizer/tokenizer.json").write_text("{}")
    (workspace / "tokenizer/keep.txt").write_text("keep")
    (workspace / "checkpoints/bitnet").mkdir(parents=True)
    (workspace / "checkpoints/bitnet/c.pt").write_bytes(b"x")

    local_module.main()

    assert not (workspace / "data/processed/train.jsonl").exists()
    assert (workspace / "data/processed/.gitkeep").exists()
    assert not (workspace / "tokenizer/tokenizer.json").exists()
    assert (workspace / "tokenizer/keep.txt").exists()
    assert not (workspace / "checkpoints/bitnet/c.pt").exists()
    assert "élément(s) supprimé(s)" in capsys.readouterr().out


def test_targets_never_include_dataset_sources_or_code():
    forbidden = ("src", "scripts", "tests", ".git")

    assert not any(
        target.split("/")[0] in forbidden for target in local_module.TARGETS
    )


# ------------------------------------------------------------
# suppression sur Hugging Face
# ------------------------------------------------------------

class FakeApi:
    def __init__(self, files=None, list_error=None, failing=()):
        self.files = files or []
        self.list_error = list_error
        self.failing = set(failing)
        self.deleted = []

    def list_repo_files(self, repo_id, repo_type):
        if self.list_error:
            raise self.list_error
        return self.files

    def delete_file(self, path_in_repo, repo_id, repo_type):
        if path_in_repo in self.failing:
            raise RuntimeError("HF down")
        self.deleted.append(path_in_repo)


@pytest.fixture
def hf(monkeypatch):
    holder = types.SimpleNamespace(api=None)
    monkeypatch.setattr(hf_module, "HfApi", lambda: holder.api)
    return holder


def answer(monkeypatch, text):
    monkeypatch.setattr("builtins.input", lambda prompt="": text)


def test_delete_hf_file_returns_one_on_success_and_zero_on_failure():
    api = FakeApi(failing={"bad"})

    assert hf_module.delete_hf_file(api, "good") == 1
    assert hf_module.delete_hf_file(api, "bad") == 0
    assert api.deleted == ["good"]


def test_hf_main_deletes_only_targeted_prefixes_after_confirmation(hf, monkeypatch):
    hf.api = FakeApi(
        files=[
            "data/processed/train.jsonl",
            "tokenizer/vocab.json",
            "checkpoints/bitnet/c.pt",
            "documentation/result/r.md",
            "README.md",
            ".gitattributes",
        ]
    )
    answer(monkeypatch, "o")

    assert hf_module.main() == 0

    assert hf.api.deleted == [
        "data/processed/train.jsonl",
        "tokenizer/vocab.json",
        "checkpoints/bitnet/c.pt",
        "documentation/result/r.md",
    ]


@pytest.mark.parametrize("reply", ["", "n", "N", "oui", "y"])
def test_hf_main_deletes_nothing_without_exact_confirmation(hf, monkeypatch, reply):
    hf.api = FakeApi(files=["data/a.json"])
    answer(monkeypatch, reply)

    assert hf_module.main() == 0
    assert hf.api.deleted == []


def test_hf_main_confirmation_is_case_insensitive_and_trimmed(hf, monkeypatch):
    hf.api = FakeApi(files=["data/a.json"])
    answer(monkeypatch, "  O ")

    hf_module.main()

    assert hf.api.deleted == ["data/a.json"]


def test_hf_main_without_matching_files_does_not_ask(hf, monkeypatch, capsys):
    hf.api = FakeApi(files=["README.md"])
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": pytest.fail("aucune confirmation attendue"),
    )

    assert hf_module.main() == 0
    assert "Aucun fichier à supprimer" in capsys.readouterr().out


def test_hf_main_listing_failure_returns_error_code(hf, capsys):
    hf.api = FakeApi(list_error=RuntimeError("offline"))

    assert hf_module.main() == 1
    assert "Impossible de lister" in capsys.readouterr().out


def test_hf_main_continues_after_a_failed_deletion(hf, monkeypatch, capsys):
    hf.api = FakeApi(files=["data/a", "data/b"], failing={"data/a"})
    answer(monkeypatch, "o")

    hf_module.main()

    assert hf.api.deleted == ["data/b"]
    assert "1 fichier(s) supprimé(s)" in capsys.readouterr().out
