import json
import types

import pytest

from scripts.bitnet.delete_data import delete_checkpoint as module


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "JSON_FILE", tmp_path / "state.json")
    monkeypatch.setattr(module, "HF_CHECKPOINT_FOLDER", "remote/ckpt")
    monkeypatch.setattr(module, "HF_REPO_ID", "owner/repo")
    monkeypatch.setattr(module, "HF_REPO_TYPE", "dataset")

    deleted = []
    failing = set()

    class FakeApi:
        def delete_file(self, path_in_repo, repo_id, repo_type):
            if path_in_repo in failing:
                raise RuntimeError("HF down")
            deleted.append((path_in_repo, repo_id, repo_type))

    monkeypatch.setattr(module, "HfApi", FakeApi)
    return types.SimpleNamespace(
        dir=tmp_path, deleted=deleted, failing=failing
    )


def make_checkpoint(env, name, step, kind="step", create_file=True):
    path = env.dir / name
    if create_file:
        path.write_bytes(b"x" * 10)
    return {"name": name, "type": kind, "step": step, "path": str(path)}


def write_state(checkpoints):
    module.JSON_FILE.write_text(json.dumps({"checkpoints": checkpoints}))


# ------------------------------------------------------------
# état JSON
# ------------------------------------------------------------

def test_load_state_missing_or_corrupted_returns_none(env):
    assert module.load_state() is None

    module.JSON_FILE.write_text("{broken")

    assert module.load_state() is None


def test_save_state_creates_parent_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "JSON_FILE", tmp_path / "a" / "b" / "s.json")

    module.save_state({"checkpoints": []})

    assert json.loads(module.JSON_FILE.read_text()) == {"checkpoints": []}


# ------------------------------------------------------------
# sélection
# ------------------------------------------------------------

def test_step_checkpoints_are_sorted_and_filtered():
    state = {
        "checkpoints": [
            {"name": "e", "type": "epoch", "step": 1},
            {"name": "s30", "type": "step", "step": 30},
            {"name": "s10", "type": "step", "step": 10},
        ]
    }

    assert [c["name"] for c in module.get_step_checkpoints(state)] == [
        "s10", "s30",
    ]
    assert module.get_step_checkpoints(None) == []


def test_old_checkpoints_exclude_the_latest_one():
    state = {
        "checkpoints": [
            {"name": f"s{n}", "type": "step", "step": n} for n in (10, 20, 30)
        ]
    }

    assert [c["name"] for c in module.get_old_step_checkpoints(state)] == [
        "s10", "s20",
    ]


def test_single_or_no_step_checkpoint_has_nothing_to_delete():
    one = {"checkpoints": [{"name": "s", "type": "step", "step": 1}]}

    assert module.get_old_step_checkpoints(one) == []
    assert module.get_old_step_checkpoints({"checkpoints": []}) == []


# ------------------------------------------------------------
# suppression locale / HF
# ------------------------------------------------------------

def test_delete_local_removes_existing_file(env):
    checkpoint = make_checkpoint(env, "a.pt", 1)

    assert module.delete_local_checkpoint(checkpoint) is True
    assert not (env.dir / "a.pt").exists()


def test_delete_local_reports_missing_file(env):
    checkpoint = make_checkpoint(env, "a.pt", 1, create_file=False)

    assert module.delete_local_checkpoint(checkpoint) is False


def test_delete_hf_targets_remote_checkpoint_folder(env):
    assert module.delete_hf_checkpoint({"name": "a.pt"}) is True

    assert env.deleted == [("remote/ckpt/a.pt", "owner/repo", "dataset")]


def test_delete_hf_failure_returns_false(env):
    env.failing.add("remote/ckpt/a.pt")

    assert module.delete_hf_checkpoint({"name": "a.pt"}) is False


def test_remove_checkpoints_from_state_saves_remaining():
    state = {"checkpoints": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}

    module.remove_checkpoints_from_state(state, [{"name": "a"}, {"name": "c"}])

    assert json.loads(module.JSON_FILE.read_text())["checkpoints"] == [
        {"name": "b"}
    ]


# ------------------------------------------------------------
# clean_old_step_checkpoints
# ------------------------------------------------------------

def test_clean_without_state_does_nothing(env, capsys):
    assert module.clean_old_step_checkpoints() == 0
    assert "JSON introuvable" in capsys.readouterr().out


def test_clean_keeps_only_latest_step_checkpoint(env):
    checkpoints = [
        make_checkpoint(env, "s10.pt", 10),
        make_checkpoint(env, "s20.pt", 20),
        make_checkpoint(env, "s30.pt", 30),
        make_checkpoint(env, "epoch1.pt", 5, kind="epoch"),
    ]
    write_state(checkpoints)

    total = module.clean_old_step_checkpoints()

    assert total == 4  # 2 locaux + 2 HF
    assert not (env.dir / "s10.pt").exists()
    assert not (env.dir / "s20.pt").exists()
    assert (env.dir / "s30.pt").exists()
    assert (env.dir / "epoch1.pt").exists()
    assert [d[0] for d in env.deleted] == ["remote/ckpt/s10.pt", "remote/ckpt/s20.pt"]
    names = [c["name"] for c in json.loads(module.JSON_FILE.read_text())["checkpoints"]]
    assert names == ["s30.pt", "epoch1.pt"]


def test_clean_with_single_step_checkpoint_deletes_nothing(env, capsys):
    write_state([make_checkpoint(env, "s10.pt", 10)])

    assert module.clean_old_step_checkpoints() == 0
    assert (env.dir / "s10.pt").exists()
    assert env.deleted == []


def test_clean_counts_local_and_hf_deletions_independently(env):
    write_state(
        [
            make_checkpoint(env, "s10.pt", 10, create_file=False),
            make_checkpoint(env, "s20.pt", 20),
        ]
    )

    assert module.clean_old_step_checkpoints() == 1  # HF seulement

    names = [c["name"] for c in json.loads(module.JSON_FILE.read_text())["checkpoints"]]
    assert names == ["s20.pt"]
