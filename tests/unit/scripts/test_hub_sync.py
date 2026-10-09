import types

import pytest

from scripts.hub import hub_sync


@pytest.fixture
def hub(monkeypatch):
    """Remplace les appels réseau par des enregistreurs."""
    calls = types.SimpleNamespace(files=[], folders=[], repos=[])

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.repos.append(kwargs)

    monkeypatch.setattr(hub_sync, "HfApi", FakeApi)
    monkeypatch.setattr(
        hub_sync, "upload_file", lambda **kw: calls.files.append(kw)
    )
    monkeypatch.setattr(
        hub_sync, "upload_folder", lambda **kw: calls.folders.append(kw)
    )
    monkeypatch.setattr(hub_sync, "HF_REPO_ID", "owner/repo")
    monkeypatch.setattr(hub_sync, "HF_REPO_TYPE", "dataset")
    return calls


# ------------------------------------------------------------
# upload_file_to_hub
# ------------------------------------------------------------

def test_upload_file_returns_resolve_url_and_uploads(tmp_path, hub):
    path = tmp_path / "model.pt"
    path.write_text("x")

    url = hub_sync.upload_file_to_hub(path, remote_path="ckpt/model.pt")

    assert url == (
        "https://huggingface.co/datasets/owner/repo/resolve/main/ckpt/model.pt"
    )
    assert hub.files == [
        {
            "path_or_fileobj": str(path),
            "path_in_repo": "ckpt/model.pt",
            "repo_id": "owner/repo",
            "repo_type": "dataset",
        }
    ]


def test_upload_file_defaults_remote_path_to_file_name(tmp_path, hub):
    path = tmp_path / "model.pt"
    path.write_text("x")

    hub_sync.upload_file_to_hub(path)

    assert hub.files[0]["path_in_repo"] == "model.pt"


def test_upload_file_creates_repo_if_missing(tmp_path, hub):
    path = tmp_path / "a.txt"
    path.write_text("x")

    hub_sync.upload_file_to_hub(path)

    assert hub.repos == [
        {"repo_id": "owner/repo", "repo_type": "dataset", "exist_ok": True}
    ]


def test_upload_missing_file_is_skipped(tmp_path, hub, capsys):
    assert hub_sync.upload_file_to_hub(tmp_path / "absent") is None

    assert hub.files == [] and hub.repos == []
    assert "Fichier introuvable" in capsys.readouterr().out


def test_upload_is_skipped_without_huggingface_hub(tmp_path, hub, monkeypatch, capsys):
    path = tmp_path / "a.txt"
    path.write_text("x")
    monkeypatch.setattr(hub_sync, "_is_hf_available", lambda: False)

    assert hub_sync.upload_file_to_hub(path) is None

    assert hub.files == []
    assert "non installé" in capsys.readouterr().out


# ------------------------------------------------------------
# upload_folder_to_hub
# ------------------------------------------------------------

def test_upload_folder_uploads_with_remote_folder(tmp_path, hub):
    assert hub_sync.upload_folder_to_hub(tmp_path, "remote/dir") is True

    assert hub.folders == [
        {
            "folder_path": str(tmp_path),
            "repo_id": "owner/repo",
            "repo_type": "dataset",
            "path_in_repo": "remote/dir",
        }
    ]


def test_upload_folder_defaults_remote_to_local_path(tmp_path, hub):
    hub_sync.upload_folder_to_hub(tmp_path)

    assert hub.folders[0]["path_in_repo"] == str(tmp_path)


def test_upload_missing_folder_is_skipped(tmp_path, hub, capsys):
    assert hub_sync.upload_folder_to_hub(tmp_path / "absent") is None

    assert hub.folders == []
    assert "Dossier introuvable" in capsys.readouterr().out


def test_upload_folder_is_skipped_without_huggingface_hub(
    tmp_path, hub, monkeypatch
):
    monkeypatch.setattr(hub_sync, "_is_hf_available", lambda: False)

    assert hub_sync.upload_folder_to_hub(tmp_path) is None
    assert hub.folders == []


# ------------------------------------------------------------
# sync_all_to_hub
# ------------------------------------------------------------

def test_sync_uploads_only_existing_folders(tmp_path, hub, monkeypatch, capsys):
    present = tmp_path / "present"
    present.mkdir()
    monkeypatch.setattr(
        hub_sync,
        "SYNC_FOLDERS",
        [(str(present), "remote/present"), (str(tmp_path / "gone"), "remote/gone")],
    )

    hub_sync.sync_all_to_hub()

    assert [f["path_in_repo"] for f in hub.folders] == ["remote/present"]
    assert "Ignoré (absent)" in capsys.readouterr().out


def test_sync_is_skipped_without_huggingface_hub(hub, monkeypatch, capsys):
    monkeypatch.setattr(hub_sync, "_is_hf_available", lambda: False)

    hub_sync.sync_all_to_hub()

    assert hub.folders == []
    assert "sync ignorée" in capsys.readouterr().out
