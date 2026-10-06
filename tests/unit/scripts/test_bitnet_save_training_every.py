import math
import sys
import types

import pytest

from scripts.bitnet import bitnet_save_training_every as module
from scripts.bitnet.bitnet_save_training_every import (
    StepCheckpointSaver,
    safe_perplexity,
)


# ------------------------------------------------------------
# safe_perplexity
# ------------------------------------------------------------

def test_perplexity_none_stays_none():
    assert safe_perplexity(None) is None


def test_perplexity_is_exp_of_loss():
    assert safe_perplexity(0.0) == 1.0
    assert safe_perplexity(2.0) == pytest.approx(math.exp(2))


def test_perplexity_overflow_is_infinite():
    assert safe_perplexity(1e6) == float("inf")


# ------------------------------------------------------------
# should_save / chemin
# ------------------------------------------------------------

def make_saver(tmp_path, **kwargs):
    kwargs.setdefault("save_every_n_steps", 10)
    kwargs.setdefault("auto_clean", False)
    return StepCheckpointSaver(local_dir=tmp_path / "ckpt", **kwargs)


def test_init_creates_local_directory(tmp_path):
    make_saver(tmp_path)

    assert (tmp_path / "ckpt").is_dir()


def test_should_save_only_from_first_milestone(tmp_path):
    saver = make_saver(tmp_path)

    assert not saver.should_save(9)
    assert saver.should_save(10)
    assert saver.should_save(15)


def test_should_save_waits_for_next_milestone_after_a_save(tmp_path):
    saver = make_saver(tmp_path)
    saver.last_saved_step = 10

    assert not saver.should_save(19)
    assert saver.should_save(20)


def test_should_save_catches_up_after_skipped_steps(tmp_path):
    saver = make_saver(tmp_path)
    saver.last_saved_step = 10

    assert saver.should_save(57)


def test_disabled_saver_never_saves(tmp_path):
    assert not make_saver(tmp_path, enabled=False).should_save(1000)


def test_checkpoint_path_uses_step_in_name(tmp_path):
    saver = make_saver(tmp_path)

    assert saver.get_checkpoint_path(30) == \
        tmp_path / "ckpt" / "checkpoint_step30.pt"


# ------------------------------------------------------------
# save
# ------------------------------------------------------------

class Recorder:
    """Faux save_checkpoint_fn : écrit un fichier et mémorise les kwargs."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, checkpoint_path, **kwargs):
        self.calls.append((checkpoint_path, kwargs))
        if self.fail:
            raise OSError("disk full")
        checkpoint_path.write_bytes(b"x" * 1024)


@pytest.fixture
def external(monkeypatch):
    """Isole GPU, hub, registre JSON et nettoyage."""
    monkeypatch.setattr(module, "get_gpu_metrics", lambda: {"available": False})
    registered = []
    cleaned = []

    verif = types.ModuleType("verif")
    verif.register_checkpoint = lambda **kw: registered.append(kw)
    delete = types.ModuleType("delete")
    delete.clean_old_step_checkpoints = lambda: cleaned.append(True)

    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", None)
    monkeypatch.setitem(
        sys.modules,
        "scripts.bitnet.verif_save.create_modif_json_checkpoint",
        verif,
    )
    monkeypatch.setitem(
        sys.modules, "scripts.bitnet.delete_data.delete_checkpoint", delete
    )
    return types.SimpleNamespace(registered=registered, cleaned=cleaned)


def test_save_before_milestone_does_nothing(tmp_path, external):
    recorder = Recorder()

    assert make_saver(tmp_path).save(5, recorder) is None
    assert recorder.calls == []


def test_save_writes_checkpoint_and_updates_counter(tmp_path, external):
    saver = make_saver(tmp_path)
    recorder = Recorder()

    path = saver.save(10, recorder, epoch=1, train_loss=2.0)

    assert path == tmp_path / "ckpt" / "checkpoint_step10.pt"
    assert path.exists()
    assert saver.last_saved_step == 10
    assert not saver.should_save(15)


def test_save_injects_gpu_metrics_and_computed_perplexities(tmp_path, external):
    recorder = Recorder()

    make_saver(tmp_path).save(
        10, recorder, train_loss=0.0, validation_loss=1.0
    )

    kwargs = recorder.calls[0][1]
    assert kwargs["gpu_metrics"] == {"available": False}
    assert kwargs["train_ppl"] == 1.0
    assert kwargs["validation_ppl"] == pytest.approx(math.e)


def test_save_keeps_perplexities_given_by_caller(tmp_path, external):
    recorder = Recorder()

    make_saver(tmp_path).save(
        10, recorder, train_loss=0.0, train_ppl=42.0
    )

    assert recorder.calls[0][1]["train_ppl"] == 42.0


def test_save_registers_checkpoint_in_json(tmp_path, external):
    make_saver(tmp_path).save(
        10, Recorder(), epoch=2, duration_seconds=5.0, train_loss=1.0
    )

    (entry,) = external.registered
    assert entry["checkpoint_type"] == "step"
    assert entry["step"] == 10 and entry["epoch"] == 2
    assert entry["duration_seconds"] == 5.0
    assert entry["checkpoint_path"].name == "checkpoint_step10.pt"


def test_local_failure_aborts_without_counter_update(tmp_path, external, capsys):
    saver = make_saver(tmp_path)

    assert saver.save(10, Recorder(fail=True)) is None

    assert saver.last_saved_step == 0
    assert external.registered == []
    assert "Échec sauvegarde locale" in capsys.readouterr().out


def test_hub_failure_keeps_local_checkpoint(tmp_path, external, capsys):
    path = make_saver(tmp_path).save(10, Recorder())

    assert path.exists()
    assert "Upload HF échoué" in capsys.readouterr().out


def test_hub_upload_uses_remote_folder(tmp_path, external, monkeypatch):
    uploads = []
    hub = types.ModuleType("hub")
    hub.upload_file_to_hub = lambda path, remote_path: uploads.append(
        (path.name, remote_path)
    )
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", hub)

    make_saver(tmp_path, hf_folder="remote/dir").save(10, Recorder())

    assert uploads == [("checkpoint_step10.pt", "remote/dir/checkpoint_step10.pt")]


def test_auto_clean_removes_old_step_checkpoints(tmp_path, external):
    make_saver(tmp_path, auto_clean=True).save(10, Recorder())

    assert external.cleaned == [True]


def test_no_cleaning_when_auto_clean_disabled(tmp_path, external):
    make_saver(tmp_path, auto_clean=False).save(10, Recorder())

    assert external.cleaned == []


def test_json_registration_failure_does_not_abort(tmp_path, external, monkeypatch, capsys):
    def boom(**kwargs):
        raise RuntimeError("json broken")

    sys.modules[
        "scripts.bitnet.verif_save.create_modif_json_checkpoint"
    ].register_checkpoint = boom

    saver = make_saver(tmp_path)

    assert saver.save(10, Recorder()) is not None
    assert saver.last_saved_step == 10
    assert "Enregistrement JSON échoué" in capsys.readouterr().out


def test_gpu_line_is_printed_when_a_device_is_available(
    tmp_path, external, monkeypatch, capsys
):
    monkeypatch.setattr(
        module,
        "get_gpu_metrics",
        lambda: {
            "available": True,
            "devices": [
                {
                    "memory_used_mb": 10,
                    "memory_total_mb": 20,
                    "memory_percent": 50.0,
                    "utilization_gpu_percent": 7,
                    "temperature_c": 55,
                }
            ],
        },
    )

    make_saver(tmp_path).save(10, Recorder())

    assert "GPU : 10/20 Mo (50.0%)" in capsys.readouterr().out


def test_print_status_reports_next_step(tmp_path, capsys):
    saver = make_saver(tmp_path)
    saver.last_saved_step = 10

    saver.print_status()

    assert "Prochain : step 20" in capsys.readouterr().out
