"""
Vérifie si un checkpoint plus récent existe sur Hugging Face.

Compare le dernier checkpoint local avec le dernier checkpoint sur HF.
Si le checkpoint HF est plus récent (numéro plus élevé), le télécharge.

Utilisation :
    poetry run python -m scripts.bitnet.verify_checkpoint_in_hugging_face
"""

import re
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


from scripts.bitnet.hub_config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    HF_CHECKPOINT_FOLDER,
)


# ============================================================
# CONFIGURATION
# ============================================================

LOCAL_CHECKPOINT_FOLDER = Path("checkpoints/bitnet")

# Motif : entrainement_bitnet_<nombre>.pt
CHECKPOINT_PATTERN = re.compile(r"^entrainement_bitnet_(\d+)\.pt$")


# ============================================================
# DERNIER CHECKPOINT LOCAL
# ============================================================

def get_latest_local_checkpoint():
    """
    Retourne (numéro, chemin) du dernier checkpoint local,
    ou (None, None) si aucun.
    """

    if not LOCAL_CHECKPOINT_FOLDER.exists():
        return (None, None)

    checkpoints = []

    for file_path in LOCAL_CHECKPOINT_FOLDER.iterdir():

        if not file_path.is_file():
            continue

        match = CHECKPOINT_PATTERN.match(file_path.name)

        if match is None:
            continue

        number = int(match.group(1))

        checkpoints.append((number, file_path))

    if not checkpoints:
        return (None, None)

    checkpoints.sort(key=lambda x: x[0])

    return checkpoints[-1]


# ============================================================
# DERNIER CHECKPOINT SUR HF
# ============================================================

def get_latest_remote_checkpoint():
    """
    Retourne (numéro, chemin_distant) du dernier checkpoint sur HF,
    ou (None, None) si aucun.
    """

    print("\n🔍 Recherche des checkpoints sur Hugging Face...")

    try:

        files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)

    except Exception as error:

        print(f"⚠️ Impossible de lister le repo HF : {error}")

        return (None, None)

    checkpoints = []

    prefix = f"{HF_CHECKPOINT_FOLDER}/"

    for file_path in files:

        if not file_path.startswith(prefix):
            continue

        filename = file_path[len(prefix):]

        match = CHECKPOINT_PATTERN.match(filename)

        if match is None:
            continue

        number = int(match.group(1))

        checkpoints.append((number, file_path))

    if not checkpoints:
        return (None, None)

    checkpoints.sort(key=lambda x: x[0])

    return checkpoints[-1]


# ============================================================
# TÉLÉCHARGEMENT
# ============================================================

def download_checkpoint(remote_path):
    """
    Télécharge un checkpoint depuis HF vers le dossier local.
    """

    filename = Path(remote_path).name

    target = LOCAL_CHECKPOINT_FOLDER / filename

    print(f"\n📥 Téléchargement : {remote_path}")
    print(f"   Destination  : {target}")

    cached = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=remote_path,
        repo_type=HF_REPO_TYPE,
    )

    LOCAL_CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)

    shutil.copy(cached, target)

    size_mb = target.stat().st_size / (1024 * 1024)

    print(f"   ✅ Téléchargé ({size_mb:.2f} Mo)")

    return target


# ============================================================
# VÉRIFICATION PRINCIPALE
# ============================================================

def verify_and_sync():
    """
    Vérifie si un checkpoint HF est plus récent que le local.
    Si oui, le télécharge.

    Retourne le chemin du dernier checkpoint local après synchro.
    """

    print("=" * 70)
    print("VÉRIFICATION DES CHECKPOINTS — HF vs LOCAL")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECKPOINT LOCAL
    # --------------------------------------------------------

    local_number, local_path = get_latest_local_checkpoint()

    if local_path is not None:

        size_mb = local_path.stat().st_size / (1024 * 1024)

        print(
            f"\n📁 Dernier checkpoint local : "
            f"{local_path.name} "
            f"(numéro {local_number}, {size_mb:.2f} Mo)"
        )

    else:

        print("\n📁 Aucun checkpoint local")

    # --------------------------------------------------------
    # CHECKPOINT HF
    # --------------------------------------------------------

    remote_number, remote_path = get_latest_remote_checkpoint()

    if remote_path is not None:

        print(
            f"\n☁️  Dernier checkpoint HF    : "
            f"{remote_path} "
            f"(numéro {remote_number})"
        )

    else:

        print("\n☁️  Aucun checkpoint sur HF")

    # --------------------------------------------------------
    # DÉCISION
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DÉCISION")
    print("-" * 70)

    # Cas 1 : rien nulle part
    if remote_number is None and local_number is None:

        print("\n➡️ Aucun checkpoint existant — entraînement from scratch")
        return None

    # Cas 2 : rien sur HF → garder local
    if remote_number is None:

        print(f"\n➡️ Local uniquement — utilisation de {local_path.name}")
        return local_path

    # Cas 3 : rien en local → télécharger HF
    if local_number is None:

        print(f"\n➡️ HF uniquement — téléchargement de {remote_path}")

        return download_checkpoint(remote_path)

    # Cas 4 : les deux existent — comparer
    if remote_number > local_number:

        print(
            f"\n➡️ Le checkpoint HF est plus récent "
            f"({remote_number} > {local_number})"
        )
        print(f"   → Téléchargement de {remote_path}")

        return download_checkpoint(remote_path)

    elif remote_number == local_number:

        print(f"\n➡️ Checkpoints synchronisés " f"(numéro {local_number})")
        print(f"   → Utilisation du local : {local_path.name}")

        return local_path

    else:

        print(
            f"\n➡️ Le checkpoint local est plus récent "
            f"({local_number} > {remote_number})"
        )
        print(f"   → Utilisation du local : {local_path.name}")

        return local_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    verify_and_sync()
