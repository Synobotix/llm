import pytest

from scripts.bitnet.heartbeat import heartbeat as heartbeat_module
from scripts.bitnet.heartbeat.heartbeat import Heartbeat


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(heartbeat_module.time, "time", lambda: now[0])
    return now


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(
        heartbeat_module, "get_gpu_metrics", lambda: {"available": False}
    )


def test_defaults():
    beat = Heartbeat()

    assert (beat.interval, beat.enabled, beat.prefix) == (30.0, True, "💓")


def test_no_message_before_interval(clock, capsys):
    beat = Heartbeat(interval=30)
    clock[0] += 29

    beat.beat(step=1)

    assert capsys.readouterr().out == ""


def test_message_emitted_once_interval_elapsed(clock, capsys):
    beat = Heartbeat(interval=30)
    clock[0] += 30

    beat.beat(step=7, loss=1.23456)

    output = capsys.readouterr().out
    assert "Step 7" in output
    assert "Loss: 1.2346" in output
    assert output.startswith("💓 [")
    assert beat._beat_count == 1


def test_interval_restarts_after_each_beat(clock, capsys):
    beat = Heartbeat(interval=10)
    clock[0] += 10
    beat.beat()
    capsys.readouterr()

    clock[0] += 5
    beat.beat()

    assert capsys.readouterr().out == ""


def test_force_emits_before_interval(clock, capsys):
    beat = Heartbeat(interval=1000)

    beat.beat(step=1, force=True)

    assert "Step 1" in capsys.readouterr().out


def test_disabled_never_emits_even_when_forced(clock, capsys):
    beat = Heartbeat(interval=0, enabled=False)

    beat.beat(force=True)

    assert capsys.readouterr().out == ""
    assert beat._beat_count == 0


def test_extra_text_and_custom_prefix_are_included(clock, capsys):
    beat = Heartbeat(prefix="HB")

    beat.beat(extra="validation", force=True)

    output = capsys.readouterr().out
    assert output.startswith("HB [")
    assert output.rstrip().endswith("validation")


def test_message_omits_step_and_loss_when_not_given(clock, capsys):
    Heartbeat().beat(force=True)

    output = capsys.readouterr().out
    assert "Step" not in output
    assert "Loss" not in output


def test_gpu_metrics_are_appended_when_available(clock, capsys, monkeypatch):
    monkeypatch.setattr(
        heartbeat_module,
        "get_gpu_metrics",
        lambda: {
            "available": True,
            "devices": [
                {
                    "memory_used_mb": 512,
                    "memory_total_mb": 1024,
                    "memory_percent": 50.0,
                    "utilization_gpu_percent": 80,
                    "temperature_c": 65,
                }
            ],
        },
    )

    Heartbeat().beat(force=True)

    output = capsys.readouterr().out
    assert "GPU 512/1024 Mo (50.0%)" in output
    assert "Util 80%" in output
    assert "65°C" in output


def test_gpu_monitoring_failure_does_not_break_heartbeat(
    clock, capsys, monkeypatch
):
    def boom():
        raise RuntimeError("nvidia-smi down")

    monkeypatch.setattr(heartbeat_module, "get_gpu_metrics", boom)

    Heartbeat().beat(step=3, force=True)

    assert "Step 3" in capsys.readouterr().out


def test_elapsed_and_human_format(clock):
    beat = Heartbeat()
    clock[0] += 3725

    assert beat.elapsed() == 3725
    assert beat.elapsed_human() == "1:02:05"


def test_reset_restarts_timer_and_counter(clock, capsys):
    beat = Heartbeat(interval=0)
    beat.beat()
    clock[0] += 100

    beat.reset()

    assert beat.elapsed() == 0
    assert beat._beat_count == 0


def test_instance_is_callable_with_keyword_arguments(clock, capsys):
    Heartbeat()(step=9, force=True)

    assert "Step 9" in capsys.readouterr().out
