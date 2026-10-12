import types

import pytest

from scripts.bitnet import verify_checkpoint_in_hugging_face as module


@pytest.fixture
def env(tmp_path, monkeypatch):
    local = tmp_path / "local"
    monkeypatch.setattr(module, "LOCAL_CHECKPOINT_FOLDER", local)
    monkeypatch.setattr(module, "HF_CHECKPOINT_FOLDER", "remote/ckpt")
    monkeypatch.setattr(module, "HF_REPO_ID", "owner/repo")
    monkeypatch.setattr(module, "HF_REPO_TYPE", "dataset")

    state = types.SimpleNamespace(
        local=local, remote_files=[], list_error=None, downloaded=[],
        cache=tmp_path / "cache",
    )
    state.cache.mkdir()

    def list_repo_files(**kwargs):
        if state.list_error:
            raise state.list_error
        return state.remote_files

    def hf_hub_download(repo_id, filename, repo_type):
        state.downloaded.append(filename)
        cached = state.cache / filename.replace("/", "_")
        cached.write_bytes(b"remote")
        return str(cached)

    monkeypatch.setattr(module, "list_repo_files", list_repo_files)
    monkeypatch.setattr(module, "hf_hub_download", hf_hub_download)
    return state


def touch_local(env, *numbers):
    env.local.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        (env.local / f"entrainement_bitnet_{number}.pt").write_bytes(b"local")


# ------------------------------------------------------------
# dernier checkpoint local
# ------------------------------------------------------------

def test_latest_local_without_folder(env):
    assert module.get_latest_local_checkpoint() == (None, None)


def test_latest_local_is_highest_number(env):
    touch_local(env, 2, 10, 9)

    number, path = module.get_latest_local_checkpoint()

    assert number == 10
    assert path.name == "entrainement_bitnet_10.pt"


def test_latest_local_ignores_other_files_and_directories(env):
    touch_local(env, 1)
    (env.local / "checkpoint_step5.pt").write_text("x")
    (env.local / "entrainement_bitnet_7.pt").mkdir()

    assert module.get_latest_local_checkpoint()[0] == 1


# ------------------------------------------------------------
# dernier checkpoint distant
# ------------------------------------------------------------

def test_latest_remote_is_highest_number_in_checkpoint_folder(env):
    env.remote_files = [
        "remote/ckpt/entrainement_bitnet_2.pt",
        "remote/ckpt/entrainement_bitnet_11.pt",
        "remote/ckpt/checkpoint_step9.pt",
        "other/entrainement_bitnet_99.pt",
        "README.md",
    ]

    assert module.get_latest_remote_checkpoint() == (
        11, "remote/ckpt/entrainement_bitnet_11.pt",
    )


def test_latest_remote_without_matching_files(env):
    env.remote_files = ["remote/ckpt/notes.txt"]

    assert module.get_latest_remote_checkpoint() == (None, None)


def test_latest_remote_listing_failure_is_reported(env, capsys):
    env.list_error = RuntimeError("offline")

    assert module.get_latest_remote_checkpoint() == (None, None)
    assert "Impossible de lister le repo HF" in capsys.readouterr().out


# ------------------------------------------------------------
# téléchargement
# ------------------------------------------------------------

def test_download_copies_checkpoint_into_local_folder(env):
    target = module.download_checkpoint("remote/ckpt/entrainement_bitnet_3.pt")

    assert target == env.local / "entrainement_bitnet_3.pt"
    assert target.read_bytes() == b"remote"


# ------------------------------------------------------------
# verify_and_sync
# ------------------------------------------------------------

def test_nothing_anywhere_means_training_from_scratch(env):
    assert module.verify_and_sync() is None


def test_local_only_uses_local_checkpoint(env):
    touch_local(env, 4)

    path = module.verify_and_sync()

    assert path.name == "entrainement_bitnet_4.pt"
    assert env.downloaded == []


def test_remote_only_downloads_checkpoint(env):
    env.remote_files = ["remote/ckpt/entrainement_bitnet_2.pt"]

    path = module.verify_and_sync()

    assert path == env.local / "entrainement_bitnet_2.pt"
    assert path.exists()


def test_newer_remote_checkpoint_is_downloaded(env):
    touch_local(env, 1)
    env.remote_files = ["remote/ckpt/entrainement_bitnet_3.pt"]

    path = module.verify_and_sync()

    assert path.name == "entrainement_bitnet_3.pt"
    assert env.downloaded == ["remote/ckpt/entrainement_bitnet_3.pt"]


def test_same_number_keeps_local_checkpoint(env):
    touch_local(env, 3)
    env.remote_files = ["remote/ckpt/entrainement_bitnet_3.pt"]

    path = module.verify_and_sync()

    assert path.read_bytes() == b"local"
    assert env.downloaded == []


def test_newer_local_checkpoint_is_kept(env):
    touch_local(env, 5)
    env.remote_files = ["remote/ckpt/entrainement_bitnet_3.pt"]

    assert module.verify_and_sync().name == "entrainement_bitnet_5.pt"
    assert env.downloaded == []


def test_unreachable_hub_falls_back_to_local(env):
    touch_local(env, 2)
    env.list_error = RuntimeError("offline")

    assert module.verify_and_sync().name == "entrainement_bitnet_2.pt"
