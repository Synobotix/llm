"""
Vérifie si le tokenizer existe en local.
Si absent, le télécharge depuis Hugging Face.

Le tokenizer est composé de 3 fichiers :
- tokenizer/tokenizer.json
- tokenizer/vocab.json
- tokenizer/config.json

Utilisation :
    poetry run python -m scripts.bitnet.verify_tokenizer_in_hugging_face
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import (list_repo_files, hf_hub_download)


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.bitnet.hub_config import (HF_REPO_ID, HF_REPO_TYPE)


# ============================================================
# CONFIGURATION
# ============================================================

TOKENIZER_FILES = [
    "tokenizer/tokenizer.json",
    "tokenizer/vocab.json",
    "tokenizer/config.json",
]

LOCAL_TOKENIZER_DIR = Path("tokenizer")


# ============================================================
# VÉRIFIER SI LE TOKENIZER EXISTE EN LOCAL
# ============================================================

def tokenizer_exists_locally():
    """
    Retourne True si les 3 fichiers tokenizer existent en local.
    """

    for file_path in TOKENIZER_FILES:

        local_path = Path(file_path)

        if not local_path.exists():
            return False

    return True


# ============================================================
# TÉLÉCHARGER DEPUIS HF
# ============================================================

def download_tokenizer():
    """
    Télécharge les 3 fichiers du tokenizer depuis HF.
    Retourne True si succès.
    """

    print("\n🔍 Recherche du tokenizer sur Hugging Face...")

    try:

        files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)

    except Exception as error:

        print(f"⚠️ Impossible de lister le repo HF : {error}")

        return False

    # --------------------------------------------------------
    # VÉRIFIER QUE LES 3 FICHIERS EXISTENT SUR HF
    # --------------------------------------------------------

    missing = []

    for file_path in TOKENIZER_FILES:

        if file_path not in files:
            missing.append(file_path)

    if missing:

        print("\n⚠️ Fichiers tokenizer manquants sur HF :")

        for f in missing:
            print(f"   - {f}")

        return False

    # --------------------------------------------------------
    # TÉLÉCHARGER
    # --------------------------------------------------------

    print(f"\n📥 Téléchargement de {len(TOKENIZER_FILES)} fichiers...")

    LOCAL_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in TOKENIZER_FILES:

        filename = Path(file_path).name

        print(f"\n   📥 {file_path}")

        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=file_path,
            repo_type=HF_REPO_TYPE,
        )

        target = LOCAL_TOKENIZER_DIR / filename
        shutil.copy(cached, target)

        size_kb = target.stat().st_size / 1024

        print(f"      ✅ {target} ({size_kb:.2f} Ko)")

    return True


# ============================================================
# VÉRIFICATION PRINCIPALE
# ============================================================

def verify_and_sync_tokenizer():
    """
    Vérifie si le tokenizer existe en local.
    Si absent, le télécharge depuis HF.

    Retourne True si le tokenizer est disponible après l'appel.
    """

    print("=" * 70)
    print("VÉRIFICATION DU TOKENIZER — LOCAL vs HF")
    print("=" * 70)

    # --------------------------------------------------------
    # VÉRIFIER LE LOCAL
    # --------------------------------------------------------

    if tokenizer_exists_locally():

        print("\n✅ Tokenizer présent en local")

        for file_path in TOKENIZER_FILES:

            local_path = Path(file_path)
            size_kb = local_path.stat().st_size / 1024

            print(f"   {local_path} ({size_kb:.2f} Ko)")

        return True

    # --------------------------------------------------------
    # TÉLÉCHARGER DEPUIS HF
    # --------------------------------------------------------

    print("\n⚠️ Tokenizer absent en local")

    success = download_tokenizer()

    if not success:

        print("\n❌ Impossible de télécharger le tokenizer")
        print("   → Lancez d'abord :")
        print("     poetry run python -m tokenizer.train_tokenizer")

        return False

    print("\n✅ Tokenizer téléchargé depuis HF")

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    verify_and_sync_tokenizer()
