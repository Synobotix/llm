# scripts/bitnet/analyse_monitoring.py

"""
Analyse du fichier monitoring_history.jsonl.

Détecte :
- Les fuites mémoire (RAM qui monte en flèche)
- Les fuites GPU (mémoire allouée qui monte)
- Les tenseurs Python qui s'accumulent
- Le step exact du dernier enregistrement avant crash
"""

import json
import sys
from pathlib import Path


HISTORY_PATH = Path("monitoring_history.jsonl")


def analyse():
    if not HISTORY_PATH.exists():
        print(f"❌ Fichier introuvable : {HISTORY_PATH}")
        return

    entries = []
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        print("❌ Aucune entrée dans l'historique")
        return

    print(f"📊 {len(entries)} entrées analysées")
    print(f"   Premier step : {entries[0].get('step')}")
    print(f"   Dernier step : {entries[-1].get('step')}")
    print()

    # --------------------------------------------------------
    # Évolution de la RAM processus
    # --------------------------------------------------------

    print("=" * 70)
    print("ÉVOLUTION DE LA RAM PROCESSUS (RSS)")
    print("=" * 70)

    rss_values = [
        e.get("process_ram", {}).get("rss_go")
        for e in entries
        if e.get("process_ram", {}).get("rss_go") is not None
    ]

    if rss_values:
        print(f"   Premier : {rss_values[0]:.3f} Go")
        print(f"   Dernier : {rss_values[-1]:.3f} Go")
        print(f"   Min     : {min(rss_values):.3f} Go")
        print(f"   Max     : {max(rss_values):.3f} Go")
        print(f"   Delta   : {rss_values[-1] - rss_values[0]:+.3f} Go")

        if rss_values[-1] > rss_values[0] * 1.5:
            print("   ⚠️ FUITE POSSIBLE : la RAM a augmenté de plus de 50%")
    else:
        print("   Pas de données RSS")

    # --------------------------------------------------------
    # Évolution de la RAM système
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ÉVOLUTION DE LA RAM SYSTÈME")
    print("=" * 70)

    ram_used = [
        e.get("ram", {}).get("total_go", 0) - e.get("ram", {}).get("available_go", 0)
        for e in entries
        if e.get("ram", {}).get("total_go") is not None
    ]

    if ram_used:
        print(f"   Premier : {ram_used[0]:.3f} Go utilisés")
        print(f"   Dernier : {ram_used[-1]:.3f} Go utilisés")
        print(f"   Delta   : {ram_used[-1] - ram_used[0]:+.3f} Go")

    # --------------------------------------------------------
    # Évolution de la mémoire GPU
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ÉVOLUTION DE LA MÉMOIRE GPU (torch)")
    print("=" * 70)

    gpu_allocated = [
        e.get("gpu_torch", {}).get("allocated_go")
        for e in entries
        if e.get("gpu_torch", {}).get("allocated_go") is not None
    ]

    gpu_reserved = [
        e.get("gpu_torch", {}).get("reserved_go")
        for e in entries
        if e.get("gpu_torch", {}).get("reserved_go") is not None
    ]

    if gpu_allocated:
        print(f"   allocated premier : {gpu_allocated[0]:.3f} Go")
        print(f"   allocated dernier : {gpu_allocated[-1]:.3f} Go")
        print(f"   allocated max     : {max(gpu_allocated):.3f} Go")

    if gpu_reserved:
        print(f"   reserved premier  : {gpu_reserved[0]:.3f} Go")
        print(f"   reserved dernier  : {gpu_reserved[-1]:.3f} Go")
        print(f"   reserved max      : {max(gpu_reserved):.3f} Go")

    # --------------------------------------------------------
    # Évolution des tenseurs Python
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ÉVOLUTION DES TENSEURS PYTHON")
    print("=" * 70)

    tensor_counts = [
        e.get("python", {}).get("torch_tensors", {}).get("total")
        for e in entries
        if e.get("python", {}).get("torch_tensors", {}).get("total") is not None
    ]

    cuda_tensor_counts = [
        e.get("python", {}).get("torch_tensors", {}).get("cuda")
        for e in entries
        if e.get("python", {}).get("torch_tensors", {}).get("cuda") is not None
    ]

    if tensor_counts:
        print(f"   Tenseurs total premier : {tensor_counts[0]}")
        print(f"   Tenseurs total dernier : {tensor_counts[-1]}")
        print(f"   Delta                  : {tensor_counts[-1] - tensor_counts[0]:+d}")

    if cuda_tensor_counts:
        print(f"   Tenseurs CUDA premier  : {cuda_tensor_counts[0]}")
        print(f"   Tenseurs CUDA dernier  : {cuda_tensor_counts[-1]}")
        print(f"   Delta                  : {cuda_tensor_counts[-1] - cuda_tensor_counts[0]:+d}")

    # --------------------------------------------------------
    # Dernières entrées
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DERNIÈRES 10 ENTRÉES")
    print("=" * 70)

    for e in entries[-10:]:
        step = e.get("step")
        loss = e.get("loss")
        rss = e.get("process_ram", {}).get("rss_go")
        alloc = e.get("gpu_torch", {}).get("allocated_go")

        print(
            f"   step={step:>5} "
            f"loss={loss:.4f} "
            f"RSS={rss:.3f} Go "
            f"GPU_alloc={alloc:.3f} Go"
        )

    # --------------------------------------------------------
    # Détection de sauts anormaux
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAUTS ANORMAUX DANS LA RAM")
    print("=" * 70)

    if len(rss_values) > 1:
        for i in range(1, len(rss_values)):
            delta = rss_values[i] - rss_values[i - 1]
            if abs(delta) > 0.1:  # saut de 100 Mo
                step = entries[i].get("step")
                print(f"   step={step:>5} : {delta:+.3f} Go")


if __name__ == "__main__":
    analyse()
