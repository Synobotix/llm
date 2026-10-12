import types

import pytest

from scripts.bitnet import verify_tokenizer_in_hugging_face as module

FILES = ["tokenizer/tokenizer.json", "tokenizer/vocab.json",
         "tokenizer/config.json"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    local_dir = tmp_path / "tokenizer"
    files = [str(tmp_path / name) for name in FILES]
    monkeypatch.setattr(module, "LOCAL_TOKENIZER_DIR", local_dir)
    monkeypatch.setattr(module, "TOKENIZER_FILES", files)

    state = types.SimpleNamespace(
        local_dir=local_dir, files=files, remote_files=list(files),
        list_error=None, downloaded=[],
    )
    cache = tmp_path / "cache"
    cache.mkdir()

    def list_repo_files(**kwargs):
        if state.list_error:
            raise state.list_error
        return state.remote_files

    def hf_hub_download(repo_id, filename, repo_type):
        state.downloaded.append(filename)
        cached = cache / filename.replace("/", "_").replace(":", "_")
        cached.write_text("remote")
        return str(cached)

    monkeypatch.setattr(module, "list_repo_files", list_repo_files)
    monkeypatch.setattr(module, "hf_hub_download", hf_hub_download)
    return state


def create_local(env):
    from pathlib import Path

    for file in env.files:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local")


def test_tokenizer_exists_locally_requires_all_three_files(env):
    from pathlib import Path

    assert module.tokenizer_exists_locally() is False

    create_local(env)
    assert module.tokenizer_exists_locally() is True

    Path(env.files[1]).unlink()
    assert module.tokenizer_exists_locally() is False


def test_download_copies_every_file_into_local_directory(env):
    assert module.download_tokenizer() is True

    names = sorted(p.name for p in env.local_dir.iterdir())
    assert names == ["config.json", "tokenizer.json", "vocab.json"]
    assert len(env.downloaded) == 3


def test_download_fails_when_a_file_is_missing_on_hub(env, capsys):
    env.remote_files = env.files[:2]

    assert module.download_tokenizer() is False

    assert env.downloaded == []
    assert "config.json" in capsys.readouterr().out


def test_download_fails_when_hub_listing_fails(env, capsys):
    env.list_error = RuntimeError("offline")

    assert module.download_tokenizer() is False
    assert "Impossible de lister le repo HF" in capsys.readouterr().out


def test_verify_does_not_download_when_tokenizer_is_present(env):
    create_local(env)

    assert module.verify_and_sync_tokenizer() is True
    assert env.downloaded == []


def test_verify_downloads_when_tokenizer_is_absent(env):
    assert module.verify_and_sync_tokenizer() is True
    assert len(env.downloaded) == 3


def test_verify_reports_failure_and_suggests_training(env, capsys):
    env.remote_files = []

    assert module.verify_and_sync_tokenizer() is False
    assert "train_tokenizer" in capsys.readouterr().out
