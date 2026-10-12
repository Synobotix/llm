"""
Supprime TOUS les fichiers locaux de données :
- data/ (JSON, JSONL)
- tokenizer/ (JSON)
- checkpoints/ (PT)
- documentation/result/ (MD)
- training_state.json

⚠️ À utiliser pour repartir FROM SCRATCH.

Utilisation :
    poetry run python -m scripts.bitnet.reset_data.delete_all_data_json_jsonl_pt_local
"""

import shutil
import sys
from pathlib import Path


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

TARGETS = [
    # Datasets
    "data/processed",
    "data/tokenized",

    # Tokenizer
    "tokenizer/tokenizer.json",
    "tokenizer/vocab.json",
    "tokenizer/config.json",

    # Checkpoints
    "checkpoints/bitnet",

    # Résultats
    "documentation/result/bitnet",
]


# ============================================================
# SUPPRESSION
# ============================================================

def delete_target(path):
    """
    Supprime un fichier ou dossier.
    """

    path = Path(path)

    if not path.exists():

        print(f"   ⏭️ Absent : {path}")

        return 0

    # Fichier simple
    if path.is_file():

        size_mb = path.stat().st_size / (1024 * 1024)

        path.unlink()

        print(f"   🗑️ {path} ({size_mb:.2f} Mo)")

        return 1

    # Dossier
    if path.is_dir():

        size_mb = sum(
            f.stat().st_size
            for f in path.rglob("*")
            if f.is_file()
        ) / (1024 * 1024)

        shutil.rmtree(path)

        print(f"   🗑️ {path}/ ({size_mb:.2f} Mo)")

        return 1

    return 0


def recreate_empty_dirs():
    """
    Recrée les dossiers vides pour le pipeline.
    """

    empty_dirs = [
        "data/processed",
        "data/tokenized",
        "checkpoints/bitnet",
        "documentation/result/bitnet",
    ]

    for d in empty_dirs:

        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)

        # Fichier .gitkeep pour Git
        gitkeep = path / ".gitkeep"
        gitkeep.touch()

        print(f"   📁 {path}/ (recréé)")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SUPPRESSION DES DONNÉES LOCALES")
    print("=" * 70)

    print(f"\n📁 Racine : {PROJECT_ROOT}")

    print("\n🗑️ Suppression...")

    total = 0

    for target in TARGETS:

        print(f"\n▶ {target}")

        total += delete_target(target)

    print("\n" + "=" * 70)
    print("RECRÉATION DES DOSSIERS VIDES")
    print("=" * 70)

    recreate_empty_dirs()

    print("\n" + "=" * 70)
    print(f"✅ {total} élément(s) supprimé(s)")
    print("=" * 70)

    print("\n💡 Le pipeline peut maintenant repartir from scratch :")
    print("   1. poetry run python -m scripts.nemotron.download_and_prepare")
    print("   2. poetry run python -m scripts.split_dataset")
    print("   3. poetry run python -m tokenizer.train_tokenizer")
    print("   4. poetry run python -m scripts.tokenize_dataset")
    print("   5. poetry run python -m scripts.bitnet.train_bitnet")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
