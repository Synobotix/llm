"""
Supprime TOUS les fichiers de données sur Hugging Face :
- data/ (JSON, JSONL)
- tokenizer/ (JSON)
- checkpoints/ (PT)
- documentation/result/ (MD)
- training_state.json

⚠️ À utiliser pour repartir FROM SCRATCH.

Utilisation :
    poetry run python -m scripts.bitnet.reset_data.delete_all_data_json_jsonl_pt_hf
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.bitnet.hub_config import (HF_REPO_ID, HF_REPO_TYPE)


# ============================================================
# CONFIGURATION
# ============================================================

# Préfixes à supprimer sur HF
DELETE_PREFIXES = ["data/", "tokenizer/", "checkpoints/", "documentation/"]


# ============================================================
# SUPPRESSION
# ============================================================

def delete_hf_file(api, path_in_repo):
    """
    Supprime un fichier sur HF.
    """

    try:

        api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )

        print(f"   🗑️ {path_in_repo}")

        return 1

    except Exception as error:

        print(f"   ⚠️ Échec : {path_in_repo} ({error})")

        return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SUPPRESSION DES DONNÉES SUR HUGGING FACE")
    print("=" * 70)

    print(f"\n📁 Repo : {HF_REPO_ID}")
    print(f"📁 Type : {HF_REPO_TYPE}")

    api = HfApi()

    # --------------------------------------------------------
    # Lister les fichiers
    # --------------------------------------------------------

    print("\n🔍 Récupération de la liste des fichiers...")

    try:

        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)

    except Exception as error:

        print(f"\n❌ Impossible de lister les fichiers : {error}")
        return 1

    # --------------------------------------------------------
    # Filtrer
    # --------------------------------------------------------

    to_delete = [
        f for f in files
        if any(f.startswith(prefix) for prefix in DELETE_PREFIXES)
    ]

    if not to_delete:

        print("\n✅ Aucun fichier à supprimer")
        return 0

    print(f"\n📋 {len(to_delete)} fichier(s) à supprimer :\n")

    for f in to_delete:
        print(f"   - {f}")

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    confirmation = input(
        f"\n⚠️ Supprimer ces {len(to_delete)} fichiers ? (o/N) : "
    ).strip().lower()

    if confirmation != "o":

        print("\n⏭️ Annulé")

        return 0

    # --------------------------------------------------------
    # Suppression
    # --------------------------------------------------------

    print("\n🗑️ Suppression...\n")

    total = 0

    for f in to_delete:

        total += delete_hf_file(api, f)

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"✅ {total} fichier(s) supprimé(s) sur HF")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
