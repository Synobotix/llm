import json
import subprocess
import types

import pytest
import torch

from scripts.bitnet.heartbeat import gpu_monitor


# ------------------------------------------------------------
# get_gpu_metrics
# ------------------------------------------------------------

def test_no_cuda_returns_unavailable_defaults(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert gpu_monitor.get_gpu_metrics() == {
        "available": False,
        "device_count": 0,
        "devices": [],
        "torch": {},
    }


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2e9)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 3e9)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 4e9)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 5e9)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(total_memory=16e9),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {"num_alloc_retries": 3, "active_bytes.all.current": 1e9},
    )


def test_torch_memory_metrics_are_reported_in_go(fake_cuda):
    metrics = gpu_monitor.get_gpu_metrics()

    assert metrics["available"] is True
    assert metrics["device_count"] == 2
    assert metrics["torch"]["allocated_go"] == 2.0
    assert metrics["torch"]["reserved_go"] == 3.0
    assert metrics["torch"]["max_allocated_go"] == 4.0
    assert metrics["torch"]["total_go"] == 16.0
    assert metrics["torch"]["num_alloc_retries"] == 3
    assert metrics["torch"]["num_ooms"] == 0
    assert metrics["torch"]["active_bytes"] == 1.0


def test_nvidia_smi_not_called_by_default(fake_cuda, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("nvidia-smi ne doit pas être appelé")

    monkeypatch.setattr(subprocess, "run", forbidden)

    assert gpu_monitor.get_gpu_metrics()["devices"] == []


def test_nvidia_smi_output_is_parsed(fake_cuda, monkeypatch):
    stdout = (
        "0, NVIDIA T4, 1024, 16384, 55, 60, 70.5\n"
        "1, NVIDIA T4, 0, 16384, 0, 40, 10.0\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout=stdout),
    )

    devices = gpu_monitor.get_gpu_metrics(use_nvidia_smi=True)["devices"]

    assert len(devices) == 2
    assert devices[0] == {
        "index": 0,
        "name": "NVIDIA T4",
        "memory_used_mb": 1024,
        "memory_total_mb": 16384,
        "memory_percent": 6.25,
        "utilization_gpu_percent": 55,
        "temperature_c": 60,
        "power_w": 70.5,
    }


def test_nvidia_smi_skips_malformed_lines(fake_cuda, monkeypatch):
    stdout = "garbage\n0, T4, x, 16384, 1, 1, 1\n0, T4, 1, 0, 1, 1, 1\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout=stdout),
    )

    devices = gpu_monitor.get_gpu_metrics(use_nvidia_smi=True)["devices"]

    assert len(devices) == 1
    assert devices[0]["memory_percent"] == 0.0


def test_nvidia_smi_failure_is_reported_not_raised(fake_cuda, monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)

    metrics = gpu_monitor.get_gpu_metrics(use_nvidia_smi=True)

    assert "nvidia-smi" in metrics["nvidia_smi_error"]
    assert metrics["torch"]["allocated_go"] == 2.0


def test_torch_failure_is_reported_in_metrics(fake_cuda, monkeypatch):
    def boom():
        raise RuntimeError("cuda broken")

    monkeypatch.setattr(torch.cuda, "memory_allocated", boom)

    metrics = gpu_monitor.get_gpu_metrics()

    assert metrics["torch"]["error"] == "cuda broken"


# ------------------------------------------------------------
# CPU / Python / système
# ------------------------------------------------------------

def test_cpu_metrics_structure_and_process_ram():
    metrics = gpu_monitor.get_cpu_metrics()

    assert set(metrics) == {
        "available", "cpu", "ram", "process_ram", "disk", "load_avg",
    }
    assert metrics["process_ram"]["rss_go"] > 0
    assert metrics["ram"]["total_go"] > 0
    assert 0 <= metrics["ram"]["percent"] <= 100


def test_cpu_metrics_without_psutil_leave_cpu_disk_load_empty(monkeypatch):
    monkeypatch.setattr(gpu_monitor, "PSUTIL_AVAILABLE", False)

    metrics = gpu_monitor.get_cpu_metrics()

    assert metrics["available"] is False
    assert metrics["cpu"] == {} and metrics["disk"] == {}
    assert metrics["load_avg"] == {}


def test_python_metrics_report_gc_counters():
    metrics = gpu_monitor.get_python_metrics()

    assert set(metrics["gc_counts"]) == {"gen0", "gen1", "gen2"}
    assert len(metrics["gc_stats"]) == 3
    assert set(metrics["gc_stats"][0]) == {
        "collections", "collected", "uncollectable",
    }


def test_system_metrics_combine_all_sections(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    metrics = gpu_monitor.get_system_metrics()

    assert set(metrics) == {"timestamp", "gpu", "cpu", "python"}
    assert metrics["gpu"]["available"] is False


# ------------------------------------------------------------
# write_monitoring_json
# ------------------------------------------------------------

@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gpu_monitor, "TRAINING_STATE_PATH",
                        tmp_path / "training_state.json")
    return types.SimpleNamespace(
        snapshot=tmp_path / "out" / "monitoring.json",
        history=tmp_path / "out" / "history.jsonl",
        state=tmp_path / "training_state.json",
    )


def write(outputs, **kwargs):
    return gpu_monitor.write_monitoring_json(
        path=outputs.snapshot, history_path=outputs.history, **kwargs
    )


def test_snapshot_contains_training_values_and_system_metrics(outputs):
    payload = write(outputs, step=5, loss=1.5, gradient_norm=0.3, epoch=2)

    saved = json.loads(outputs.snapshot.read_text(encoding="utf-8"))
    assert saved["step"] == 5 and saved["epoch"] == 2
    assert saved["loss"] == 1.5 and saved["gradient_norm"] == 0.3
    assert "gpu" in saved["system"]
    assert payload["step"] == 5


def test_extra_fields_are_merged_into_snapshot(outputs):
    write(outputs, step=1, extra={"note": "ok"})

    assert json.loads(outputs.snapshot.read_text())["note"] == "ok"


def test_snapshot_is_overwritten_and_no_tmp_file_is_left(outputs):
    write(outputs, step=1)
    write(outputs, step=2)

    assert json.loads(outputs.snapshot.read_text())["step"] == 2
    assert not outputs.snapshot.with_suffix(".tmp").exists()
    assert not outputs.state.with_suffix(".tmp").exists()


def test_history_is_appended_one_line_per_call(outputs):
    write(outputs, step=1, loss=2.0)
    write(outputs, step=2, loss=1.0)

    lines = outputs.history.read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]
    entry = json.loads(lines[0])
    assert {"gpu_torch", "gpu_devices", "ram", "process_ram", "python"} \
        <= set(entry)


def test_training_state_allows_resuming(outputs):
    write(outputs, step=9, loss=0.7, gradient_norm=0.1, epoch=3)

    state = json.loads(outputs.state.read_text())
    assert state["last_step"] == 9
    assert state["last_epoch"] == 3
    assert state["last_train_loss"] == 0.7
    assert state["last_gradient_norm"] == 0.1
    assert state["can_resume"] is True


def test_history_failure_does_not_prevent_snapshot_and_state(
    outputs, tmp_path, capsys
):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    outputs.history = blocker / "history.jsonl"

    write(outputs, step=4)

    assert outputs.snapshot.exists()
    assert json.loads(outputs.state.read_text())["last_step"] == 4
    assert "Écriture historique échouée" in capsys.readouterr().out
