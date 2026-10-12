import types

import pytest

from scripts.bitnet import upload_to_hub as module


@pytest.fixture
def hub(monkeypatch):
    calls = types.SimpleNamespace(uploads=[], repos=[])

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.repos.append(kwargs)

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setattr(
        module, "upload_file", lambda **kw: calls.uploads.append(kw)
    )
    return calls


def test_upload_file_missing_raises(tmp_path, hub):
    with pytest.raises(FileNotFoundError):
        module.upload_file_to_hub(tmp_path / "absent.pt", "folder")

    assert hub.uploads == []


def test_upload_file_targets_remote_folder_and_returns_url(tmp_path, hub):
    path = tmp_path / "model.pt"
    path.write_text("x")

    url = module.upload_file_to_hub(
        path, "checkpoints/bitnet", repo_id="owner/repo", repo_type="dataset"
    )

    assert url == (
        "https://huggingface.co/datasets/owner/repo/resolve/main/"
        "checkpoints/bitnet/model.pt"
    )
    assert hub.uploads == [
        {
            "path_or_fileobj": str(path),
            "path_in_repo": "checkpoints/bitnet/model.pt",
            "repo_id": "owner/repo",
            "repo_type": "dataset",
        }
    ]


def test_upload_artifacts_creates_repo_and_uploads_both_files(tmp_path, hub):
    checkpoint = tmp_path / "entrainement_bitnet_1.pt"
    result = tmp_path / "result_entrainement_bitnet1.md"
    checkpoint.write_text("x")
    result.write_text("y")

    urls = module.upload_training_artifacts(
        checkpoint, result, repo_id="owner/repo", repo_type="dataset"
    )

    assert hub.repos == [
        {"repo_id": "owner/repo", "repo_type": "dataset", "exist_ok": True}
    ]
    assert [u["path_in_repo"] for u in hub.uploads] == [
        f"{module.HF_CHECKPOINT_FOLDER}/entrainement_bitnet_1.pt",
        f"{module.HF_RESULT_FOLDER}/result_entrainement_bitnet1.md",
    ]
    assert urls["checkpoint_url"].endswith("entrainement_bitnet_1.pt")
    assert urls["result_url"].endswith("result_entrainement_bitnet1.md")


def test_upload_artifacts_fails_before_result_upload_if_checkpoint_missing(
    tmp_path, hub
):
    result = tmp_path / "r.md"
    result.write_text("y")

    with pytest.raises(FileNotFoundError):
        module.upload_training_artifacts(tmp_path / "absent.pt", result)

    assert hub.uploads == []
