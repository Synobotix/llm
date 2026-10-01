import pytest

from scripts import duration
from scripts.duration import TrainingTimer


@pytest.fixture
def clock(monkeypatch):
    """Horloge contrôlée : retourne la valeur courante de `now[0]`."""
    now = [100.0]
    monkeypatch.setattr(duration.time, "perf_counter", lambda: now[0])
    return now


def test_stop_returns_elapsed_time(clock):
    timer = TrainingTimer()
    timer.start()
    clock[0] = 112.5

    assert timer.stop() == 12.5


def test_stop_adds_initial_duration(clock):
    timer = TrainingTimer(initial_duration=60)
    timer.start()
    clock[0] = 105.0

    assert timer.stop() == 65.0


def test_initial_duration_is_converted_to_float():
    assert isinstance(TrainingTimer(initial_duration=3).initial_duration, float)


def test_stop_before_start_raises():
    with pytest.raises(RuntimeError):
        TrainingTimer().stop()


def test_elapsed_before_start_returns_initial_duration():
    assert TrainingTimer(initial_duration=7).elapsed() == 7.0


def test_elapsed_does_not_stop_the_timer(clock):
    timer = TrainingTimer()
    timer.start()
    clock[0] = 103.0
    assert timer.elapsed() == 3.0

    clock[0] = 110.0
    assert timer.elapsed() == 10.0
    assert timer.end_time is None


def test_start_resets_end_time(clock):
    timer = TrainingTimer()
    timer.start()
    timer.stop()
    assert timer.end_time is not None

    timer.start()

    assert timer.end_time is None


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00h 00min 00.00s"),
        (5.5, "00h 00min 05.50s"),
        (65, "00h 01min 05.00s"),
        (3600, "01h 00min 00.00s"),
        (3725.25, "01h 02min 05.25s"),
        (90000, "25h 00min 00.00s"),
    ],
)
def test_format_duration(seconds, expected):
    assert TrainingTimer.format_duration(seconds) == expected
