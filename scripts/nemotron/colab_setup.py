"""
Pipeline complet Nemotron → BitNet (préparation)
pour Colab ou pour d'autre serveur GPU ou CPU
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STEPS = [

    (
        "1. Téléchargement Nemotron",
        "scripts.nemotron.download_and_prepare",
    ),

    (
        "2. Split train/validation",
        "scripts.split_dataset",
    ),

    (
        "3. Entraînement du tokenizer",
        "src.tokenizer.train_tokenizer",
    ),

    (
        "4. Tokenisation",
        "scripts.tokenize_dataset",
    ),
]


def run_step(name, module):

    print("\n" + "=" * 70)
    print(f"ÉTAPE {name}")
    print("=" * 70)

    result = subprocess.run([sys.executable, "-m", module], text=True)

    if result.returncode != 0:
        raise RuntimeError(f"❌ Échec : {name}")

    print(f"\n✅ {name} — OK")


def main():

    print("=" * 70)
    print("PIPELINE NEMOTRON — PRÉPARATION")
    print("=" * 70)

    for name, module in STEPS:
        run_step(name, module)

    print("\n" + "=" * 70)
    print("PRÉPARATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
