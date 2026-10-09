import pytest

from scripts.hub import hub_download


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(hub_download, "HF_REPO_ID", "owner/repo")
    monkeypatch.setattr(hub_download, "HF_REPO_TYPE", "dataset")
    monkeypatch.setattr(
        hub_download, "SYNC_FOLDERS", [("a", "data/a"), ("b", "data/b")]
    )


def fake_downloads(monkeypatch, files, fail_on=None):
    downloads = []
    monkeypatch.setattr(hub_download, "list_repo_files", lambda **kw: files)

    def snapshot_download(**kwargs):
        pattern = kwargs["allow_patterns"][0]
        if fail_on and fail_on in pattern:
            raise RuntimeError("network error")
        downloads.append(kwargs)
        return "."

    monkeypatch.setattr(hub_download, "snapshot_download", snapshot_download)
    return downloads


def test_empty_repository_downloads_nothing(configured, monkeypatch, capsys):
    downloads = fake_downloads(monkeypatch, [])

    hub_download.download_all_from_hub()

    assert downloads == []
    assert "Aucun fichier disponible" in capsys.readouterr().out


def test_no_file_in_target_folders_downloads_nothing(
    configured, monkeypatch, capsys
):
    downloads = fake_downloads(monkeypatch, ["other/file.txt", "README.md"])

    hub_download.download_all_from_hub()

    assert downloads == []
    assert "Aucun fichier dans les dossiers cibles" in capsys.readouterr().out


def test_downloads_each_synced_folder_into_current_directory(
    configured, monkeypatch
):
    downloads = fake_downloads(monkeypatch, ["data/a/x.txt", "other/y.txt"])

    hub_download.download_all_from_hub()

    assert [d["allow_patterns"] for d in downloads] == [
        ["data/a/*"], ["data/b/*"],
    ]
    assert all(d["local_dir"] == "." for d in downloads)
    assert all(d["repo_id"] == "owner/repo" for d in downloads)


def test_folder_prefix_must_match_exactly(configured, monkeypatch, capsys):
    downloads = fake_downloads(monkeypatch, ["data/ab/x.txt"])

    hub_download.download_all_from_hub()

    assert downloads == []


def test_one_failing_folder_does_not_stop_the_others(
    configured, monkeypatch, capsys
):
    downloads = fake_downloads(
        monkeypatch, ["data/a/x.txt"], fail_on="data/a"
    )

    hub_download.download_all_from_hub()

    assert [d["allow_patterns"] for d in downloads] == [["data/b/*"]]
    assert "Ignoré : network error" in capsys.readouterr().out
