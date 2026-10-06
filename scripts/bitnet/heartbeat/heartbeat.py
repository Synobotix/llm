# scripts/bitnet/heartbeat/heartbeat.py

"""
Module de heartbeat pour l'entraînement BitNet.

Objectif :
    Émettre un message périodique pendant l'entraînement afin de :
    - prouver au client Jupyter (nbclient) que la cellule est active ;
    - éviter le timeout de cellule silencieuse ;
    - surveiller l'usage mémoire GPU en temps réel ;
    - fournir une trace horodatée pour diagnostiquer les blocages.

Usage :
    from scripts.bitnet.heartbeat.heartbeat import Heartbeat

    heartbeat = Heartbeat(interval=30)

    for step, batch in enumerate(dataloader):
        ...
        heartbeat.beat(step=step, loss=loss.item())
"""

import sys
import time
from datetime import datetime, timedelta

import torch

from scripts.bitnet.heartbeat.gpu_monitor import get_gpu_metrics


class Heartbeat:
    """
    Émet un message à intervalle régulier pour maintenir la session active
    et surveiller la mémoire GPU.
    """

    def __init__(
        self,
        interval: float = 30.0,
        enabled: bool = True,
        prefix: str = "💓",
    ):
        """
        Args:
            interval: intervalle en secondes entre deux battements.
            enabled: active ou désactive le heartbeat.
            prefix: préfixe visuel des messages.
        """

        self.interval = float(interval)
        self.enabled = bool(enabled)
        self.prefix = prefix

        self._last_beat_time = time.time()
        self._start_time = time.time()
        self._beat_count = 0

    # ------------------------------------------------------------------
    # CONTRÔLE
    # ------------------------------------------------------------------

    def reset(self):
        """Réinitialise le compteur et le timer."""
        self._last_beat_time = time.time()
        self._start_time = time.time()
        self._beat_count = 0

    def elapsed(self) -> float:
        """Temps écoulé depuis le début."""
        return time.time() - self._start_time

    def elapsed_human(self) -> str:
        """Temps écoulé formaté."""
        return str(timedelta(seconds=int(self.elapsed())))

    # ------------------------------------------------------------------
    # HEARTBEAT
    # ------------------------------------------------------------------

    def beat(
        self,
        step: int | None = None,
        loss: float | None = None,
        extra: str = "",
        force: bool = False,
    ):
        """
        Émet un heartbeat si l'intervalle est écoulé.

        Args:
            step: numéro de step actuel.
            loss: loss actuelle.
            extra: information supplémentaire libre.
            force: force l'émission même si l'intervalle n'est pas écoulé.
        """

        if not self.enabled:
            return

        now = time.time()

        if not force and (now - self._last_beat_time) < self.interval:
            return

        self._last_beat_time = now
        self._beat_count += 1

        # -------- Construction du message --------

        timestamp = datetime.now().strftime("%H:%M:%S")

        parts = [f"{self.prefix} [{timestamp}]"]

        if step is not None:
            parts.append(f"Step {step}")

        if loss is not None:
            parts.append(f"Loss: {loss:.4f}")

        parts.append(f"| {self.elapsed_human()}")

        # -------- Métriques GPU --------

        try:
            gpu = get_gpu_metrics()

            if gpu.get("available") and gpu.get("devices"):
                d = gpu["devices"][0]
                parts.append(
                    f"| GPU {d['memory_used_mb']}/{d['memory_total_mb']} Mo "
                    f"({d['memory_percent']}%) "
                    f"| Util {d['utilization_gpu_percent']}% "
                    f"| {d['temperature_c']}°C"
                )
        except Exception:
            pass

        if extra:
            parts.append(extra)

        message = " ".join(parts)

        # -------- Émission --------

        print(message, flush=True)
        sys.stdout.flush()

    def __call__(self, **kwargs):
        """Permet d'appeler l'instance directement."""
        self.beat(**kwargs)
