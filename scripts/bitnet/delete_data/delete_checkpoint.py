"""
Supprime les anciens checkpoints step (local + HF).

Ne garde que le DERNIER checkpoint step enregistré dans le JSON.

Utilisation :
    poetry run python -m scripts.bitnet.delete_data.delete_checkpoint
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

JSON_FILE = Path("checkpoints/bitnet/training_state.json")


# ============================================================
# LECTURE DU JSON
# ============================================================

def load_state():

    if not JSON_FILE.exists():

        return None

    try:

        with JSON_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    except (json.JSONDecodeError, IOError):

        return None


def save_state(state):

    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)

    with JSON_FILE.open("w", encoding="utf-8") as f:

        json.dump(state, f, indent=2, ensure_ascii=False)


# ============================================================
# ANALYSE DU JSON
# ============================================================

def get_step_checkpoints(state):
    """
    Retourne la liste des checkpoints de type 'step'
    triés par step croissant.
    """

    if state is None:
        return []

    step_checkpoints = [
        ckpt for ckpt in state["checkpoints"]
        if ckpt.get("type") == "step"
    ]

    step_checkpoints.sort(key=lambda c: c.get("step", 0))

    return step_checkpoints


def get_old_step_checkpoints(state):
    """
    Retourne tous les checkpoints 'step' SAUF le dernier.
    Ce sont ceux à supprimer.
    """

    step_checkpoints = get_step_checkpoints(state)

    if len(step_checkpoints) <= 1:
        return []

    return step_checkpoints[:-1]


# ============================================================
# SUPPRESSION LOCALE
# ============================================================

def delete_local_checkpoint(checkpoint):
    """
    Supprime un checkpoint du disque local.
    """

    path = Path(checkpoint["path"])

    if not path.exists():

        print(f"   ⏭️ Absent : {path}")

        return False

    size_mb = path.stat().st_size / (1024 * 1024)

    path.unlink()

    print(f"   🗑️ {path} ({size_mb:.2f} Mo)")

    return True


# ============================================================
# SUPPRESSION HF
# ============================================================

def delete_hf_checkpoint(checkpoint):
    """
    Supprime un checkpoint de Hugging Face.
    """

    api = HfApi()

    # Chemin distant : checkpoints/bitnet/<nom>
    remote_path = f"{HF_CHECKPOINT_FOLDER}/{checkpoint['name']}"

    try:

        api.delete_file(
            path_in_repo=remote_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )

        print(f"   🗑️ HF : {remote_path}")

        return True

    except Exception as error:

        print(f"   ⚠️ HF échec : {remote_path} ({error})")

        return False


# ============================================================
# MISE À JOUR DU JSON
# ============================================================

def remove_checkpoints_from_state(state, checkpoints_to_remove):
    """
    Retire les checkpoints supprimés du JSON.
    """

    names_to_remove = {c["name"] for c in checkpoints_to_remove}

    state["checkpoints"] = [
        ckpt for ckpt in state["checkpoints"]
        if ckpt["name"] not in names_to_remove
    ]

    save_state(state)


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def clean_old_step_checkpoints():
    """
    Nettoie tous les anciens checkpoints step.
    Garde uniquement le plus récent.
    """

    print("=" * 70)
    print("NETTOYAGE DES CHECKPOINTS STEP")
    print("=" * 70)

    # --------------------------------------------------------
    # Lire le JSON
    # --------------------------------------------------------

    state = load_state()

    if state is None:

        print("\n⚠️ JSON introuvable ou invalide")
        return 0

    # --------------------------------------------------------
    # Trouver les checkpoints à supprimer
    # --------------------------------------------------------

    old_checkpoints = get_old_step_checkpoints(state)

    if not old_checkpoints:

        print("\n✅ Aucun ancien checkpoint à supprimer")
        print("   (le dernier step est conservé)")
        return 0

    print(f"\n📋 {len(old_checkpoints)} ancien(s) checkpoint(s) à supprimer :")

    for ckpt in old_checkpoints:
        print(f"   - {ckpt['name']} (step {ckpt.get('step', '?')})")

    # --------------------------------------------------------
    # Supprimer local + HF
    # --------------------------------------------------------

    print(f"\n🗑️ Suppression...")

    deleted_local = 0
    deleted_hf = 0

    for ckpt in old_checkpoints:

        print(f"\n▶ {ckpt['name']}")

        if delete_local_checkpoint(ckpt):
            deleted_local += 1

        if delete_hf_checkpoint(ckpt):
            deleted_hf += 1

    # --------------------------------------------------------
    # Mettre à jour le JSON
    # --------------------------------------------------------

    print(f"\n📝 Mise à jour du JSON...")

    remove_checkpoints_from_state(state, old_checkpoints)

    print(f"   ✅ JSON mis à jour")

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NETTOYAGE TERMINÉ")
    print("=" * 70)
    print(f"\n🗑️ Local   : {deleted_local} fichier(s)")
    print(f"🗑️ HF      : {deleted_hf} fichier(s)")
    print(f"\n💾 Checkpoint conservé :")

    remaining = get_step_checkpoints(state)

    for ckpt in remaining:
        print(f"   ✅ {ckpt['name']} (step {ckpt.get('step', '?')})")

    return deleted_local + deleted_hf


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    clean_old_step_checkpoints()
