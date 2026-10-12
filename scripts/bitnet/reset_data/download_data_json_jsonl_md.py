"""
Télécharge depuis Hugging Face :
- data/processed/ (JSON, JSONL)
- data/tokenized/ (JSON, JSONL)
- tokenizer/ (JSON)
- documentation/result/bitnet/ (MD)
- checkpoints/bitnet/ (PT)

⚠️ Les checkpoints (.pt) sont volumineux (plusieurs centaines de Mo).
   Le téléchargement peut prendre du temps.

Utilisation :
    poetry run python -m scripts.bitnet.reset_data.download_data_json_jsonl_md
"""

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import list_repo_files, hf_hub_download


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.bitnet.hub_config import (HF_REPO_ID, HF_REPO_TYPE)


# ============================================================
# AUTHENTIFICATION
# ============================================================

def ensure_authenticated():
    """
    Vérifie que l'utilisateur est authentifié.
    Charge le token depuis ~/.hf_token si nécessaire.
    """

    # Déjà dans l'environnement ?
    if os.environ.get("HF_TOKEN"):
        print("✅ HF_TOKEN trouvé dans l'environnement")
        return

    # Lire ~/.hf_token
    hf_token_file = Path.home() / ".hf_token"

    if hf_token_file.exists():

        token = hf_token_file.read_text().strip()

        os.environ["HF_TOKEN"] = token

        print(f"✅ Token chargé depuis {hf_token_file}")

        return

    # Dernier recours
    raise RuntimeError(
        "\n❌ Aucun HF_TOKEN trouvé.\n"
        "\nSolutions :\n"
        "  1. export HF_TOKEN=$(cat ~/.hf_token)\n"
        "  2. Créer ~/.hf_token avec votre token\n"
        "  3. poetry run huggingface-cli login"
    )


# ============================================================
# CONFIGURATION
# ============================================================

# Dossiers à télécharger
DOWNLOAD_PREFIXES = [
    "data/processed/",
    "data/tokenized/",
    "tokenizer/",
    "documentation/result/bitnet/",
    "checkpoints/bitnet/",
]

# Extensions autorisées
ALLOWED_EXTENSIONS = [".json", ".jsonl", ".md", ".pt"]


# ============================================================
# FILTRAGE
# ============================================================

def should_download(file_path):
    """
    Vérifie si un fichier doit être téléchargé.
    """

    # Vérifier le préfixe
    if not any(file_path.startswith(p) for p in DOWNLOAD_PREFIXES):
        return False

    # Vérifier l'extension
    ext = Path(file_path).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        return False

    return True


# ============================================================
# TÉLÉCHARGEMENT
# ============================================================

def download_file(file_path):
    """
    Télécharge un fichier depuis HF.
    """

    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Déjà présent ?
    if target.exists():

        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"   ⏭️ Déjà présent : {file_path} ({size_mb:.2f} Mo)")
        return False

    # Télécharger
    try:

        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=file_path,
            repo_type=HF_REPO_TYPE,
        )

        shutil.copy(cached, target)

        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"   ✅ {file_path} ({size_mb:.2f} Mo)")

        return True

    except Exception as error:

        print(f"   ⚠️ Échec : {file_path} ({error})")

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TÉLÉCHARGEMENT DES DONNÉES DEPUIS HUGGING FACE")
    print("=" * 70)

    # --------------------------------------------------------
    # Authentification
    # --------------------------------------------------------

    print("\n🔐 Vérification de l'authentification...")

    try:
        ensure_authenticated()
    except RuntimeError as error:
        print(error)
        return 1

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print(f"\n📁 Repo : {HF_REPO_ID}")
    print(f"📁 Type : {HF_REPO_TYPE}")

    print("\n📋 Dossiers ciblés :")
    for p in DOWNLOAD_PREFIXES:
        print(f"   - {p}")

    print("\n📋 Extensions autorisées :")
    for e in ALLOWED_EXTENSIONS:
        print(f"   - {e}")

    # --------------------------------------------------------
    # Lister
    # --------------------------------------------------------

    print("\n🔍 Récupération de la liste des fichiers...")

    try:

        all_files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)

    except Exception as error:

        print(f"\n❌ Impossible de lister : {error}")
        return 1

    # --------------------------------------------------------
    # Filtrer
    # --------------------------------------------------------

    to_download = [f for f in all_files if should_download(f)]

    if not to_download:

        print("\n✅ Aucun fichier à télécharger")
        return 0

    # --------------------------------------------------------
    # Calculer la taille totale (approximatif)
    # --------------------------------------------------------

    print(f"\n📋 {len(to_download)} fichier(s) à télécharger :\n")

    for f in sorted(to_download):
        print(f"   - {f}")

    # --------------------------------------------------------
    # Demander confirmation pour les .pt
    # --------------------------------------------------------

    pt_files = [f for f in to_download if f.endswith(".pt")]

    if pt_files:

        print(f"\n⚠️ {len(pt_files)} checkpoint(s) .pt détecté(s)")
        print("   Les checkpoints sont volumineux (~200 Mo chacun)")

        confirm = "o"  # confirmation automatique pour environnement non-interactif

    # --------------------------------------------------------
    # Télécharger
    # --------------------------------------------------------

    print("\n📥 Téléchargement...\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for f in sorted(to_download):

        try:

            result = download_file(f)

            if result:
                downloaded += 1
            else:
                skipped += 1

        except Exception as error:

            print(f"   ❌ Erreur sur {f} : {error}")
            failed += 1

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TÉLÉCHARGEMENT TERMINÉ")
    print("=" * 70)

    print(f"\n✅ Téléchargés      : {downloaded}")
    print(f"⏭️ Déjà présents   : {skipped}")
    print(f"❌ Échecs          : {failed}")

    print("\n💡 Pour télécharger uniquement le checkpoint le plus récent :")
    print("   poetry run python -m scripts.bitnet.verify_checkpoint_in_hugging_face")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
