import json

import pytest

from scripts.bitnet.heartbeat import analyse_monitoring


@pytest.fixture
def history(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    monkeypatch.setattr(analyse_monitoring, "HISTORY_PATH", path)
    return path


def entry(step, rss=1.0, loss=1.0, alloc=1.0, **extra):
    data = {
        "step": step,
        "loss": loss,
        "process_ram": {"rss_go": rss},
        "ram": {"total_go": 16.0, "available_go": 8.0},
        "gpu_torch": {"allocated_go": alloc, "reserved_go": alloc + 1},
        "python": {},
    }
    data.update(extra)
    return data


def write(path, entries):
    path.write_text(
        "\n".join(
            e if isinstance(e, str) else json.dumps(e) for e in entries
        )
        + "\n"
    )


def run(capsys):
    analyse_monitoring.analyse()
    return capsys.readouterr().out


def test_missing_file_is_reported(history, capsys):
    assert "Fichier introuvable" in run(capsys)


def test_empty_history_is_reported(history, capsys):
    history.write_text("\n\n")

    assert "Aucune entrée" in run(capsys)


def test_reports_entry_count_and_step_range(history, capsys):
    write(history, [entry(10), entry(20), entry(30)])

    output = run(capsys)

    assert "3 entrées analysées" in output
    assert "Premier step : 10" in output
    assert "Dernier step : 30" in output


def test_invalid_json_lines_are_skipped(history, capsys):
    write(history, [entry(1), "{broken", entry(2)])

    assert "2 entrées analysées" in run(capsys)


def test_memory_leak_warning_when_rss_grows_by_more_than_50_percent(
    history, capsys
):
    write(history, [entry(1, rss=1.0), entry(2, rss=1.6)])

    output = run(capsys)

    assert "FUITE POSSIBLE" in output
    assert "Delta   : +0.600 Go" in output


def test_no_leak_warning_when_rss_is_stable(history, capsys):
    write(history, [entry(1, rss=1.0), entry(2, rss=1.2)])

    assert "FUITE POSSIBLE" not in run(capsys)


def test_ram_jumps_above_100_mo_are_listed(history, capsys):
    write(history, [entry(1, rss=1.0), entry(2, rss=1.05), entry(3, rss=1.5)])

    output = run(capsys)

    jumps = output.split("SAUTS ANORMAUX DANS LA RAM")[1]
    assert "step=    3 : +0.450 Go" in jumps
    assert "step=    2" not in jumps


def test_gpu_memory_section_reports_allocated_max(history, capsys):
    write(history, [entry(1, alloc=1.0), entry(2, alloc=3.0), entry(3, alloc=2.0)])

    output = run(capsys)

    assert "allocated max     : 3.000 Go" in output
    assert "reserved max      : 4.000 Go" in output


def test_tensor_counts_are_reported_when_present(history, capsys):
    write(
        history,
        [
            entry(1, python={"torch_tensors": {"total": 10, "cuda": 2}}),
            entry(2, python={"torch_tensors": {"total": 15, "cuda": 4}}),
        ],
    )

    output = run(capsys)

    assert "Tenseurs total dernier : 15" in output
    assert "Delta                  : +5" in output


def test_only_last_ten_entries_are_listed(history, capsys):
    write(history, [entry(step) for step in range(1, 16)])

    output = run(capsys).split("DERNIÈRES 10 ENTRÉES")[1].split("SAUTS")[0]

    assert output.count("step=") == 10
    assert "step=    6" in output and "step=    5" not in output


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Le résumé des dernières entrées formate loss/RSS/GPU avec :.4f/:.3f "
        "sans gérer None : une entrée sans ces champs fait planter l'analyse."
    ),
)
def test_entry_without_loss_does_not_crash_the_report(history, capsys):
    write(history, [entry(1, loss=None)])

    run(capsys)
