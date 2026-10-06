"""
Reprend l'entraînement avec le chronométrage cumulé.

Lit training_state.json pour récupérer :
- Le dernier checkpoint (pour reprendre)
- La durée cumulée des runs précédents

Télécharge automatiquement le JSON depuis Hugging Face
s'il n'existe pas en local.
"""

import json
import shutil
import sys
from datetime import datetime
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

JSON_FILE = Path("checkpoints/bitnet/training_state.json")

HF_REMOTE_PATH = "checkpoints/bitnet/training_state.json"


# ============================================================
# TÉLÉCHARGEMENT DEPUIS HF
# ============================================================

def download_json_from_hf():
    """
    Télécharge training_state.json depuis HF
    s'il n'existe pas en local.

    Retourne True si le JSON est disponible.
    """

    # --------------------------------------------------------
    # Déjà présent en local
    # --------------------------------------------------------

    if JSON_FILE.exists():

        print(f"   ✅ JSON présent en local : {JSON_FILE}")

        return True

    # --------------------------------------------------------
    # Télécharger depuis HF
    # --------------------------------------------------------

    print(f"   📥 JSON absent en local — téléchargement depuis HF...")

    try:

        from huggingface_hub import hf_hub_download

        from scripts.bitnet.hub_config import (HF_REPO_ID, HF_REPO_TYPE)

        cached = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_REMOTE_PATH,
            repo_type=HF_REPO_TYPE,
        )

        JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, JSON_FILE)

        print(f"   ✅ JSON téléchargé : {JSON_FILE}")

        return True

    except Exception as error:

        print(f"   ⚠️ JSON introuvable sur HF : {error}")
        print(f"   → Démarrage avec un état vide")

        return False


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


# ============================================================
# DURÉE CUMULÉE
# ============================================================

def get_cumulative_duration():
    """
    Retourne la durée totale cumulée (en secondes)
    de tous les checkpoints d'époque enregistrés.

    Prend la plus grande durée parmi les checkpoints d'époque
    (car les durées sont cumulatives).
    """

    state = load_state()

    if state is None:
        return 0.0

    max_duration = 0.0

    for ckpt in state["checkpoints"]:

        if ckpt.get("type") == "epoch":

            duration = ckpt.get("duration_seconds")

            if duration and isinstance(duration, (int, float)):

                if duration > max_duration:
                    max_duration = duration

    return max_duration


# ============================================================
# DERNIER CHECKPOINT
# ============================================================

def get_last_epoch_checkpoint():
    """
    Retourne le dernier checkpoint d'époque.
    """

    state = load_state()

    if state is None:
        return None

    epoch_checkpoints = [
        c for c in state["checkpoints"]
        if c.get("type") == "epoch"
    ]

    if not epoch_checkpoints:
        return None

    epoch_checkpoints.sort(key=lambda c: c.get("epoch", 0))

    return epoch_checkpoints[-1]


def get_last_step_checkpoint():
    """
    Retourne le dernier checkpoint step.
    """

    state = load_state()

    if state is None:
        return None

    step_checkpoints = [
        c for c in state["checkpoints"]
        if c.get("type") == "step"
    ]

    if not step_checkpoints:
        return None

    step_checkpoints.sort(key=lambda c: c.get("step", 0))

    return step_checkpoints[-1]


# ============================================================
# ÉTAT DE REPRISE
# ============================================================

def get_resume_state():
    """
    Retourne un dictionnaire avec toutes les infos
    nécessaires pour reprendre l'entraînement.
    """

    state = load_state()

    if state is None:

        return {
            "cumulative_duration_seconds": 0.0,
            "cumulative_duration_human": "00h 00min 00s",
            "last_epoch": 0,
            "last_step": 0,
            "training_finished": False,
            "total_epochs_done": 0,
            "last_checkpoint": None,
        }

    cumulative = get_cumulative_duration()

    last_epoch_ckpt = get_last_epoch_checkpoint()
    last_step_ckpt = get_last_step_checkpoint()

    last_epoch = last_epoch_ckpt["epoch"] if last_epoch_ckpt else 0
    last_step = last_step_ckpt["step"] if last_step_ckpt else 0

    total_epochs_done = len([
        c for c in state["checkpoints"]
        if c.get("type") == "epoch"
    ])

    # Déterminer le dernier checkpoint utilisé
    if last_epoch_ckpt and last_step_ckpt:

        if last_epoch_ckpt["step"] > last_step_ckpt["step"]:
            last_checkpoint = last_epoch_ckpt["name"]
        else:
            last_checkpoint = last_step_ckpt["name"]

    elif last_epoch_ckpt:
        last_checkpoint = last_epoch_ckpt["name"]

    elif last_step_ckpt:
        last_checkpoint = last_step_ckpt["name"]

    else:
        last_checkpoint = None

    return {
        "cumulative_duration_seconds": cumulative,
        "cumulative_duration_human": format_duration(cumulative),
        "last_epoch": last_epoch,
        "last_step": last_step,
        "training_finished": state.get("training_finished", False),
        "total_epochs_done": total_epochs_done,
        "last_checkpoint": last_checkpoint,
    }


# ============================================================
# FORMATAGE
# ============================================================

def format_duration(seconds):
    """
    Formate une durée en secondes en "XXh XXmin XX.XXs".
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}h {minutes:02d}min {secs:05.2f}s"


# ============================================================
# AFFICHAGE
# ============================================================

def print_resume_state():

    state = get_resume_state()

    print("=" * 70)
    print("ÉTAT DE REPRISE")
    print("=" * 70)

    print(f"\n🏁 Training terminé      : {state['training_finished']}")
    print(f"📊 Époques terminées     : {state['total_epochs_done']}")
    print(f"📅 Dernière époque       : {state['last_epoch']}")
    print(f"🔢 Dernier step          : {state['last_step']}")
    print(f"💾 Dernier checkpoint    : {state['last_checkpoint']}")

    print(f"\n⏱️  Durée cumulée (avant) : {state['cumulative_duration_human']}")
    print(f"   ({state['cumulative_duration_seconds']:.2f} secondes)")

    return state


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print_resume_state()
