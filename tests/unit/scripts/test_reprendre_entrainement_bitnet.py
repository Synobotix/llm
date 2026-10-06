import json
import sys

import pytest

from scripts.bitnet.reprendre_entrainement import reprendre_entrainement as module


@pytest.fixture
def json_file(tmp_path, monkeypatch):
    path = tmp_path / "ckpt" / "state.json"
    monkeypatch.setattr(module, "JSON_FILE", path)
    return path


def write_state(path, checkpoints, finished=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"training_finished": finished, "checkpoints": checkpoints})
    )


def epoch(name, epoch, step, duration=None):
    return {
        "name": name, "type": "epoch", "epoch": epoch, "step": step,
        "duration_seconds": duration,
    }


def step(name, step_number, epoch_number=1):
    return {
        "name": name, "type": "step", "epoch": epoch_number,
        "step": step_number,
    }


# ------------------------------------------------------------
# load_state
# ------------------------------------------------------------

def test_load_state_missing_file_returns_none(json_file):
    assert module.load_state() is None


def test_load_state_corrupted_file_returns_none(json_file):
    json_file.parent.mkdir(parents=True)
    json_file.write_text("{broken")

    assert module.load_state() is None


def test_load_state_reads_json(json_file):
    write_state(json_file, [])

    assert module.load_state()["checkpoints"] == []


# ------------------------------------------------------------
# download_json_from_hf
# ------------------------------------------------------------

def test_download_skips_when_json_already_local(json_file, capsys):
    write_state(json_file, [])

    assert module.download_json_from_hf() is True
    assert "JSON présent en local" in capsys.readouterr().out


def test_download_returns_false_when_hub_unavailable(
    json_file, monkeypatch, capsys
):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert module.download_json_from_hf() is False
    assert "JSON introuvable sur HF" in capsys.readouterr().out


# ------------------------------------------------------------
# durée cumulée
# ------------------------------------------------------------

def test_cumulative_duration_is_zero_without_state(json_file):
    assert module.get_cumulative_duration() == 0.0


def test_cumulative_duration_takes_largest_epoch_duration(json_file):
    write_state(
        json_file,
        [epoch("e1", 1, 10, 100.0), epoch("e2", 2, 20, 250.5),
         epoch("e0", 0, 5, 50)],
    )

    assert module.get_cumulative_duration() == 250.5


def test_cumulative_duration_ignores_step_checkpoints_and_invalid_values(
    json_file,
):
    write_state(
        json_file,
        [
            {"type": "step", "duration_seconds": 9999},
            epoch("e1", 1, 10, None),
            epoch("e2", 2, 20, "n/a"),
            epoch("e3", 3, 30, 40),
        ],
    )

    assert module.get_cumulative_duration() == 40


# ------------------------------------------------------------
# derniers checkpoints
# ------------------------------------------------------------

def test_last_checkpoints_are_none_without_state_or_matching_type(json_file):
    assert module.get_last_epoch_checkpoint() is None
    assert module.get_last_step_checkpoint() is None

    write_state(json_file, [])

    assert module.get_last_epoch_checkpoint() is None
    assert module.get_last_step_checkpoint() is None


def test_last_epoch_checkpoint_has_highest_epoch(json_file):
    write_state(
        json_file, [epoch("e2", 2, 20), epoch("e1", 1, 10), step("s", 99)]
    )

    assert module.get_last_epoch_checkpoint()["name"] == "e2"


def test_last_step_checkpoint_has_highest_step(json_file):
    write_state(
        json_file, [step("s10", 10), step("s30", 30), step("s20", 20),
                    epoch("e", 1, 99)]
    )

    assert module.get_last_step_checkpoint()["name"] == "s30"


# ------------------------------------------------------------
# état de reprise
# ------------------------------------------------------------

def test_resume_state_without_json_is_a_fresh_start(json_file):
    assert module.get_resume_state() == {
        "cumulative_duration_seconds": 0.0,
        "cumulative_duration_human": "00h 00min 00s",
        "last_epoch": 0,
        "last_step": 0,
        "training_finished": False,
        "total_epochs_done": 0,
        "last_checkpoint": None,
    }


def test_resume_state_summarises_progress(json_file):
    write_state(
        json_file,
        [epoch("e1", 1, 100, 3600.0), epoch("e2", 2, 200, 7325.5),
         step("s250", 250, 3)],
        finished=True,
    )

    state = module.get_resume_state()

    assert state["last_epoch"] == 2
    assert state["last_step"] == 250
    assert state["total_epochs_done"] == 2
    assert state["training_finished"] is True
    assert state["cumulative_duration_seconds"] == 7325.5
    assert state["cumulative_duration_human"] == "02h 02min 05.50s"


def test_last_checkpoint_is_the_one_with_highest_step(json_file):
    write_state(json_file, [epoch("e1", 1, 300), step("s250", 250)])
    assert module.get_resume_state()["last_checkpoint"] == "e1"

    write_state(json_file, [epoch("e1", 1, 100), step("s250", 250)])
    assert module.get_resume_state()["last_checkpoint"] == "s250"


def test_last_checkpoint_with_only_one_kind(json_file):
    write_state(json_file, [epoch("e1", 1, 100)])
    assert module.get_resume_state()["last_checkpoint"] == "e1"

    write_state(json_file, [step("s10", 10)])
    assert module.get_resume_state()["last_checkpoint"] == "s10"


def test_resume_state_with_empty_checkpoint_list(json_file):
    write_state(json_file, [])

    state = module.get_resume_state()

    assert state["last_checkpoint"] is None
    assert state["total_epochs_done"] == 0


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00h 00min 00.00s"),
        (65.25, "00h 01min 05.25s"),
        (3725, "01h 02min 05.00s"),
    ],
)
def test_format_duration(seconds, expected):
    assert module.format_duration(seconds) == expected


def test_print_resume_state_returns_state_and_prints_summary(json_file, capsys):
    write_state(json_file, [epoch("e1", 1, 10, 5.0)])

    state = module.print_resume_state()

    output = capsys.readouterr().out
    assert state["last_epoch"] == 1
    assert "ÉTAT DE REPRISE" in output
    assert "Dernier checkpoint    : e1" in output
