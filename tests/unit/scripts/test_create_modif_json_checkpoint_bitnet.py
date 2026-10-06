import json
import sys

import pytest

from scripts.bitnet.verif_save import create_modif_json_checkpoint as module


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """JSON local dans tmp_path ; l'upload HF échoue (module indisponible)."""
    monkeypatch.setattr(module, "JSON_FILE", tmp_path / "ckpt" / "state.json")
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", None)


def read_state():
    return json.loads(module.JSON_FILE.read_text(encoding="utf-8"))


def register(name="checkpoint_step10.pt", kind="step", epoch=1, step=10, **kw):
    return module.register_checkpoint(
        checkpoint_path=f"checkpoints/bitnet/{name}",
        checkpoint_type=kind,
        epoch=epoch,
        step=step,
        **kw,
    )


# ------------------------------------------------------------
# load_state / save_state
# ------------------------------------------------------------

def test_load_state_returns_empty_state_when_file_missing():
    assert module.load_state() == {
        "training_finished": False,
        "last_update": None,
        "checkpoints": [],
    }


def test_load_state_resets_corrupted_json(capsys):
    module.JSON_FILE.parent.mkdir(parents=True)
    module.JSON_FILE.write_text("{broken")

    state = module.load_state()

    assert state["checkpoints"] == []
    assert "Erreur lecture JSON" in capsys.readouterr().out


def test_save_state_writes_json_and_timestamp(capsys):
    module.save_state({"training_finished": False, "checkpoints": []})

    saved = read_state()
    assert saved["last_update"] is not None
    assert "Upload JSON échoué" in capsys.readouterr().out


def test_save_state_then_load_state_round_trip():
    module.save_state({"training_finished": True, "checkpoints": [{"name": "a"}]})

    state = module.load_state()

    assert state["training_finished"] is True
    assert state["checkpoints"] == [{"name": "a"}]


# ------------------------------------------------------------
# register_checkpoint
# ------------------------------------------------------------

def test_register_creates_entry_with_all_fields():
    entry = register(
        duration_seconds=12.5,
        train_loss=1.0,
        validation_loss=1.2,
        train_ppl=2.7,
        validation_ppl=3.3,
        gradient_norm=0.4,
        gpu_metrics={"devices": []},
    )

    assert entry["name"] == "checkpoint_step10.pt"
    assert entry["type"] == "step"
    assert entry["path"].endswith("checkpoint_step10.pt")
    assert (entry["epoch"], entry["step"]) == (1, 10)
    assert entry["duration_seconds"] == 12.5
    assert entry["gradient_norm"] == 0.4
    assert read_state()["checkpoints"][0]["name"] == "checkpoint_step10.pt"


def test_register_defaults_gpu_metrics_to_empty_dict():
    assert register()["gpu_metrics"] == {}


def test_register_same_name_updates_instead_of_duplicating():
    register(train_loss=2.0)
    register(train_loss=1.0)

    checkpoints = read_state()["checkpoints"]
    assert len(checkpoints) == 1
    assert checkpoints[0]["train_loss"] == 1.0


def test_register_sorts_by_epoch_then_step():
    register("c.pt", epoch=2, step=5)
    register("a.pt", epoch=1, step=20)
    register("b.pt", epoch=1, step=10)

    assert [c["name"] for c in read_state()["checkpoints"]] == [
        "b.pt", "a.pt", "c.pt",
    ]


def test_register_prints_gpu_line_when_devices_present(capsys):
    register(
        gpu_metrics={
            "devices": [
                {
                    "memory_used_mb": 1,
                    "memory_total_mb": 2,
                    "memory_percent": 50.0,
                    "utilization_gpu_percent": 3,
                    "temperature_c": 4,
                }
            ]
        }
    )

    assert "GPU : 1/2 Mo (50.0%)" in capsys.readouterr().out


# ------------------------------------------------------------
# cycle de vie de l'entraînement
# ------------------------------------------------------------

def test_mark_training_finished_sets_flag_and_keeps_checkpoints():
    register()

    module.mark_training_finished()

    state = read_state()
    assert state["training_finished"] is True
    assert len(state["checkpoints"]) == 1


def test_start_training_clears_finished_flag_and_keeps_checkpoints():
    register()
    module.mark_training_finished()

    module.start_training()

    state = read_state()
    assert state["training_finished"] is False
    assert len(state["checkpoints"]) == 1


def test_reset_state_removes_everything():
    register()
    module.mark_training_finished()

    module.reset_state()

    state = read_state()
    assert state["checkpoints"] == []
    assert state["training_finished"] is False


def test_print_state_lists_checkpoints(capsys):
    register(train_loss=1.5)

    module.print_state()

    output = capsys.readouterr().out
    assert "Nombre de checkpoints : 1" in output
    assert "checkpoint_step10.pt" in output
    assert "loss=1.5" in output
