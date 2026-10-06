# scripts/bitnet/heartbeat/gpu_monitor.py

"""
Collecte des métriques GPU + CPU + RAM + Python.

Écrit :
- monitoring.json          : snapshot courant (écrasé)
- monitoring_history.jsonl : historique complet (append)
- training_state.json      : état de reprise

⚠️ Optimisations :
- Pas de gc.get_objects() (fuite RAM évitée)
- nvidia-smi appelé uniquement si demandé (use_nvidia_smi=True)
- torch.cuda.memory_stats() est O(1) et toujours disponible
"""

import gc
import json
import os
import subprocess
import time
from pathlib import Path

import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================
# CHEMINS
# ============================================================

MONITORING_JSON_PATH = Path("monitoring.json")
MONITORING_HISTORY_PATH = Path("monitoring_history.jsonl")
TRAINING_STATE_PATH = Path("training_state.json")


# ============================================================
# MÉTRIQUES GPU (torch toujours, nvidia-smi si demandé)
# ============================================================

def get_gpu_metrics(use_nvidia_smi: bool = False):
    """
    Retourne un dictionnaire des métriques GPU.

    Args:
        use_nvidia_smi: si True, appelle nvidia-smi pour les
            métriques matérielles (temperature, power, etc.).
            Si False, utilise uniquement torch.cuda (O(1),
            pas de sous-processus, plus rapide et plus léger).
    """

    metrics = {
        "available": False,
        "device_count": 0,
        "devices": [],
        "torch": {},
    }

    if not torch.cuda.is_available():
        return metrics

    metrics["available"] = True
    metrics["device_count"] = torch.cuda.device_count()

    # --------------------------------------------------------
    # Métriques torch (toujours, O(1))
    # --------------------------------------------------------

    try:

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        max_reserved = torch.cuda.max_memory_reserved() / 1e9

        total_mem = (torch.cuda.get_device_properties(0).total_memory / 1e9)

        metrics["torch"] = {
            "allocated_go": round(allocated, 3),
            "reserved_go": round(reserved, 3),
            "max_allocated_go": round(max_allocated, 3),
            "max_reserved_go": round(max_reserved, 3),
            "total_go": round(total_mem, 3),
        }

        # Stats détaillées (O(1))
        try:
            stats = torch.cuda.memory_stats()
            metrics["torch"]["num_alloc_retries"] = stats.get(
                "num_alloc_retries", 0
            )
            metrics["torch"]["num_ooms"] = stats.get("num_ooms", 0)
            metrics["torch"]["active_bytes"] = round(
                stats.get("active_bytes.all.current", 0) / 1e9, 3
            )
            metrics["torch"]["inactive_split_bytes"] = round(
                stats.get("inactive_split_bytes.all.current", 0) / 1e9, 3
            )
        except Exception:
            pass

    except Exception as error:
        metrics["torch"]["error"] = str(error)

    # --------------------------------------------------------
    # Métriques nvidia-smi (uniquement si demandé)
    # --------------------------------------------------------

    if use_nvidia_smi:

        try:

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,"
                    "utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            for line in result.stdout.strip().splitlines():

                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 7:
                    continue

                (index, name, mem_used, mem_total, util, temp, power) = parts

                try:
                    mem_used = int(float(mem_used))
                    mem_total = int(float(mem_total))
                    util = int(float(util))
                    temp = int(float(temp))
                    power_val = float(power)
                except ValueError:
                    continue

                metrics["devices"].append({
                    "index": int(index),
                    "name": name,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                    "memory_percent": round(
                        mem_used / mem_total * 100, 2
                    ) if mem_total > 0 else 0.0,
                    "utilization_gpu_percent": util,
                    "temperature_c": temp,
                    "power_w": power_val,
                })

        except Exception as error:
            metrics["nvidia_smi_error"] = str(error)

    return metrics


# ============================================================
# MÉTRIQUES CPU + RAM (via /proc, sans psutil obligatoire)
# ============================================================

def get_cpu_metrics():
    """
    Retourne un dictionnaire des métriques CPU + RAM + disque.

    Utilise /proc quand psutil n'est pas disponible.
    """

    metrics = {
        "available": PSUTIL_AVAILABLE,
        "cpu": {},
        "ram": {},
        "process_ram": {},
        "disk": {},
        "load_avg": {},
    }

    # --------------------------------------------------------
    # Process RAM (toujours dispo via /proc)
    # --------------------------------------------------------

    try:

        pid = os.getpid()
        proc_path = Path(f"/proc/{pid}/status")

        if proc_path.exists():

            with proc_path.open() as f:

                for line in f:

                    if line.startswith("VmRSS:"):
                        value_kb = int(line.split()[1])
                        metrics["process_ram"]["rss_go"] = round(
                            value_kb / 1024 / 1024, 3
                        )

                    elif line.startswith("VmSize:"):
                        value_kb = int(line.split()[1])
                        metrics["process_ram"]["vms_go"] = round(
                            value_kb / 1024 / 1024, 3
                        )

                    elif line.startswith("VmPeak:"):
                        value_kb = int(line.split()[1])
                        metrics["process_ram"]["peak_go"] = round(
                            value_kb / 1024 / 1024, 3
                        )

    except Exception as error:
        metrics["process_ram"]["error"] = str(error)

    # --------------------------------------------------------
    # RAM système (via /proc/meminfo)
    # --------------------------------------------------------

    try:

        meminfo = Path("/proc/meminfo")

        if meminfo.exists():

            with meminfo.open() as f:

                info = {}

                for line in f:
                    key, val = line.split(":", 1)
                    info[key.strip()] = val.strip()

            def to_go(value_str):
                kb = int(value_str.split()[0])
                return round(kb / 1024 / 1024, 3)

            metrics["ram"] = {
                "total_go": to_go(info.get("MemTotal", "0 kB")),
                "free_go": to_go(info.get("MemFree", "0 kB")),
                "available_go": to_go(info.get("MemAvailable", "0 kB")),
                "buffers_go": to_go(info.get("Buffers", "0 kB")),
                "cached_go": to_go(info.get("Cached", "0 kB")),
                "swap_total_go": to_go(info.get("SwapTotal", "0 kB")),
                "swap_free_go": to_go(info.get("SwapFree", "0 kB")),
            }

            total = metrics["ram"]["total_go"]
            available = metrics["ram"]["available_go"]

            if total > 0:
                metrics["ram"]["percent"] = round(
                    (1 - available / total) * 100, 2
                )

    except Exception as error:
        metrics["ram"]["error"] = str(error)

    # --------------------------------------------------------
    # CPU + disque + load (psutil si dispo)
    # --------------------------------------------------------

    if PSUTIL_AVAILABLE:

        try:
            metrics["cpu"] = {
                "percent": psutil.cpu_percent(interval=None),
                "count_logical": psutil.cpu_count(logical=True),
                "count_physical": psutil.cpu_count(logical=False),
            }
        except Exception:
            pass

        try:
            disk = psutil.disk_usage("/")
            metrics["disk"] = {
                "total_go": round(disk.total / 1e9, 3),
                "used_go": round(disk.used / 1e9, 3),
                "free_go": round(disk.free / 1e9, 3),
                "percent": disk.percent,
            }
        except Exception:
            pass

        try:
            load1, load5, load15 = psutil.getloadavg()
            metrics["load_avg"] = {
                "1min": round(load1, 2),
                "5min": round(load5, 2),
                "15min": round(load15, 2),
            }
        except Exception:
            pass

    return metrics


# ============================================================
# MÉTRIQUES PYTHON (LÉGER — SANS gc.get_objects)
# ============================================================

def get_python_metrics():
    """
    Retourne des métriques légères sur l'état Python.

    ⚠️ NE PAS utiliser gc.get_objects() : trop coûteux,
    peut générer des millions de warnings et saturer la mémoire.
    """

    metrics = {}

    try:

        # Compteurs GC (O(1))
        gc_counts = gc.get_count()
        metrics["gc_counts"] = {
            "gen0": gc_counts[0],
            "gen1": gc_counts[1],
            "gen2": gc_counts[2],
        }

        # Stats GC (O(1))
        stats = gc.get_stats()
        metrics["gc_stats"] = [
            {
                "collections": s.get("collections", 0),
                "collected": s.get("collected", 0),
                "uncollectable": s.get("uncollectable", 0),
            }
            for s in stats
        ]

    except Exception as error:
        metrics["error"] = str(error)

    return metrics


# ============================================================
# MÉTRIQUES COMBINÉES
# ============================================================

def get_system_metrics(use_nvidia_smi: bool = False):
    """
    Retourne toutes les métriques : GPU + CPU + RAM + Python.
    """

    return {
        "timestamp": time.time(),
        "gpu": get_gpu_metrics(use_nvidia_smi=use_nvidia_smi),
        "cpu": get_cpu_metrics(),
        "python": get_python_metrics(),
    }


# ============================================================
# ÉCRITURE DE MONITORING.JSON + HISTORIQUE + STATE
# ============================================================

def write_monitoring_json(
    step: int,
    loss: float = None,
    gradient_norm: float = None,
    epoch: int = None,
    extra: dict = None,
    path: Path = MONITORING_JSON_PATH,
    history_path: Path = MONITORING_HISTORY_PATH,
    use_nvidia_smi: bool = False,
):
    """
    Écrit :
    - monitoring.json          : snapshot courant (écrasé)
    - monitoring_history.jsonl : historique complet (append)
    - training_state.json      : état de reprise

    Args:
        use_nvidia_smi: si True, appelle nvidia-smi pour les
            métriques matérielles. À utiliser tous les N steps
            (par exemple 20) pour éviter de créer trop de
            sous-processus.
    """

    payload = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": get_system_metrics(use_nvidia_smi=use_nvidia_smi),
    }

    if extra:
        payload.update(extra)

    # --------------------------------------------------------
    # Écriture atomique de monitoring.json
    # --------------------------------------------------------

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    tmp_path.replace(path)

    # --------------------------------------------------------
    # Append dans monitoring_history.jsonl
    # --------------------------------------------------------

    try:

        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        history_entry = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "updated_at": payload["updated_at"],
            "gpu_torch": payload["system"]["gpu"].get("torch", {}),
            "gpu_devices": [
                {
                    "index": d["index"],
                    "memory_used_mb": d["memory_used_mb"],
                    "utilization_gpu_percent": d["utilization_gpu_percent"],
                }
                for d in payload["system"]["gpu"].get("devices", [])
            ],
            "ram": payload["system"]["cpu"].get("ram", {}),
            "process_ram": payload["system"]["cpu"].get("process_ram", {}),
            "python": payload["system"]["python"],
        }

        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")

    except Exception as error:
        print(f"\n⚠️ Écriture historique échouée : {error}")

    # --------------------------------------------------------
    # Écriture training_state.json
    # --------------------------------------------------------

    state_payload = {
        "last_step": step,
        "last_epoch": epoch,
        "last_train_loss": loss,
        "last_gradient_norm": gradient_norm,
        "last_update": payload["updated_at"],
        "can_resume": True,
    }

    state_path = TRAINING_STATE_PATH
    tmp_state = state_path.with_suffix(".tmp")

    with tmp_state.open("w", encoding="utf-8") as f:
        json.dump(state_payload, f, indent=2, ensure_ascii=False)

    tmp_state.replace(state_path)

    return payload
