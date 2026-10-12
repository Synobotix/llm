import types

import pytest

from scripts.bitnet.reset_data import download_data_json_jsonl_md as module


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ------------------------------------------------------------
# authentification
# ------------------------------------------------------------

def test_token_from_environment_is_used(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "abc")

    module.ensure_authenticated()

    assert "HF_TOKEN trouvé" in capsys.readouterr().out


def test_token_is_read_from_home_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    (tmp_path / ".hf_token").write_text("  secret-token \n")

    module.ensure_authenticated()

    import os
    assert os.environ["HF_TOKEN"] == "secret-token"
    monkeypatch.delenv("HF_TOKEN")


def test_missing_token_raises_with_instructions(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        module.ensure_authenticated()


# ------------------------------------------------------------
# filtrage
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        "data/processed/train.jsonl",
        "data/tokenized/validation.jsonl",
        "tokenizer/vocab.json",
        "documentation/result/bitnet/result1.md",
        "checkpoints/bitnet/entrainement_bitnet_1.pt",
        "checkpoints/bitnet/UPPER.PT",
    ],
)
def test_should_download_accepts_targeted_files(path):
    assert module.should_download(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/file.jsonl",
        "checkpoints/gpt_classic/model.pt",
        "tokenizer/notes.txt",
        "README.md",
        "data/processed/archive.zip",
    ],
)
def test_should_download_rejects_other_prefixes_and_extensions(path):
    assert module.should_download(path) is False


# ------------------------------------------------------------
# download_file
# ------------------------------------------------------------

@pytest.fixture
def hub(monkeypatch, tmp_path):
    state = types.SimpleNamespace(downloaded=[], files=[], list_error=None,
                                  failing=set())
    cache = tmp_path / "_cache"
    cache.mkdir()

    def hf_hub_download(repo_id, filename, repo_type):
        if filename in state.failing:
            raise RuntimeError("download failed")
        state.downloaded.append(filename)
        cached = cache / filename.replace("/", "_")
        cached.write_text("remote")
        return str(cached)

    def list_repo_files(**kwargs):
        if state.list_error:
            raise state.list_error
        return state.files

    monkeypatch.setattr(module, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(module, "list_repo_files", list_repo_files)
    monkeypatch.setenv("HF_TOKEN", "x")
    return state


def test_download_file_copies_into_relative_path(workspace, hub):
    assert module.download_file("data/processed/train.jsonl") is True

    assert (workspace / "data/processed/train.jsonl").read_text() == "remote"


def test_download_file_skips_existing_file(workspace, hub):
    target = workspace / "data/processed/train.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("local")

    assert module.download_file("data/processed/train.jsonl") is False

    assert target.read_text() == "local"
    assert hub.downloaded == []


def test_download_file_failure_returns_false(workspace, hub, capsys):
    hub.failing.add("data/processed/a.jsonl")

    assert module.download_file("data/processed/a.jsonl") is False
    assert "Échec" in capsys.readouterr().out


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def test_main_without_token_returns_error(workspace, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path / "nohome")

    assert module.main() == 1


def test_main_listing_failure_returns_error(workspace, hub):
    hub.list_error = RuntimeError("offline")

    assert module.main() == 1


def test_main_nothing_to_download_returns_zero(workspace, hub, capsys):
    hub.files = ["README.md"]

    assert module.main() == 0
    assert "Aucun fichier à télécharger" in capsys.readouterr().out


def test_main_downloads_only_targeted_files(workspace, hub):
    hub.files = [
        "data/processed/train.jsonl",
        "checkpoints/bitnet/entrainement_bitnet_1.pt",
        "README.md",
        "data/raw/ignored.jsonl",
    ]

    assert module.main() == 0

    assert sorted(hub.downloaded) == [
        "checkpoints/bitnet/entrainement_bitnet_1.pt",
        "data/processed/train.jsonl",
    ]
    assert not (workspace / "data/raw").exists()


def test_main_does_not_redownload_existing_files(workspace, hub, capsys):
    existing = workspace / "data/processed/train.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text("local")
    hub.files = ["data/processed/train.jsonl"]

    assert module.main() == 0

    assert hub.downloaded == []
    assert "Déjà présents   : 1" in capsys.readouterr().out


@pytest.mark.xfail(
    strict=True,
    reason=(
        "download_file absorbe les exceptions et retourne False : main() "
        "compte l'échec comme « déjà présent » et retourne 0 au lieu de 1."
    ),
)
def test_main_returns_error_code_when_a_download_fails(workspace, hub):
    hub.files = ["data/processed/a.jsonl"]
    hub.failing.add("data/processed/a.jsonl")

    assert module.main() == 1
