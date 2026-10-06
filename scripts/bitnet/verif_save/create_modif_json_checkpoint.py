"""
Crée et met à jour un fichier JSON qui suit l'état complet
de l'entraînement BitNet.

Le fichier JSON contient :
- La liste de tous les checkpoints (fin d'époque + steps périodiques)
- La durée de chaque étape d'entraînement
- Le statut training_finished (True/False)
- Les métriques (loss, PPL, etc.)
- Les métriques GPU (VRAM, utilisation, température)

Le JSON est synchronisé avec train_bitnet.py à chaque sauvegarde
ET uploadé sur Hugging Face.
"""

import json
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
# LECTURE / ÉCRITURE
# ============================================================

def load_state():
    """
    Charge le JSON existant ou retourne un état vide.
    """

    if not JSON_FILE.exists():

        return {
            "training_finished": False,
            "last_update": None,
            "checkpoints": [],
        }

    try:

        with JSON_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    except (json.JSONDecodeError, IOError) as error:

        print(f"⚠️ Erreur lecture JSON : {error}")
        print("   → Réinitialisation du fichier.")

        return {
            "training_finished": False,
            "last_update": None,
            "checkpoints": [],
        }


def save_state(state):
    """
    Sauvegarde l'état dans le JSON (local + HF).
    """

    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)

    state["last_update"] = datetime.now().isoformat()

    # --------------------------------------------------------
    # Sauvegarde locale
    # --------------------------------------------------------

    with JSON_FILE.open("w", encoding="utf-8") as f:

        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"   📝 JSON sauvegardé localement : {JSON_FILE}")

    # --------------------------------------------------------
    # Upload HF
    # --------------------------------------------------------

    try:

        from scripts.hub.hub_sync import upload_file_to_hub

        upload_file_to_hub(JSON_FILE, remote_path=HF_REMOTE_PATH)

        print(f"   📤 JSON uploadé sur HF")

    except Exception as error:

        print(f"   ⚠️ Upload JSON échoué : {error}")


# ============================================================
# ENREGISTREMENT D'UN CHECKPOINT
# ============================================================

def register_checkpoint(
    checkpoint_path,
    checkpoint_type,
    epoch,
    step,
    duration_seconds=None,
    train_loss=None,
    validation_loss=None,
    train_ppl=None,
    validation_ppl=None,
    gradient_norm=None,
    gpu_metrics=None,
):
    """
    Enregistre un checkpoint dans le JSON.

    Paramètres :
        checkpoint_path    : chemin du fichier .pt
        checkpoint_type    : "epoch" ou "step"
        epoch              : numéro d'époque
        step               : numéro de step global
        duration_seconds   : durée de l'entraînement (secondes)
        train_loss         : loss d'entraînement
        validation_loss    : loss de validation
        train_ppl          : perplexité d'entraînement
        validation_ppl     : perplexité de validation
        gradient_norm      : norme du gradient
        gpu_metrics        : dict des métriques GPU (optionnel)
    """

    checkpoint_path = Path(checkpoint_path)

    state = load_state()

    # --------------------------------------------------------
    # Créer l'entrée
    # --------------------------------------------------------

    entry = {
        "name": checkpoint_path.name,
        "type": checkpoint_type,
        "path": str(checkpoint_path),
        "epoch": epoch,
        "step": step,
        "duration_seconds": duration_seconds,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_ppl": train_ppl,
        "validation_ppl": validation_ppl,
        "gradient_norm": gradient_norm,
        "gpu_metrics": gpu_metrics if gpu_metrics is not None else {},
        "created_at": datetime.now().isoformat(),
    }

    # --------------------------------------------------------
    # Vérifier si le checkpoint existe déjà
    # --------------------------------------------------------

    existing_index = None

    for i, ckpt in enumerate(state["checkpoints"]):

        if ckpt["name"] == checkpoint_path.name:
            existing_index = i
            break

    if existing_index is not None:

        state["checkpoints"][existing_index] = entry

        print(f"   🔄 Checkpoint mis à jour : {checkpoint_path.name}")

    else:

        state["checkpoints"].append(entry)

        print(f"   ✅ Checkpoint enregistré : {checkpoint_path.name}")

    # --------------------------------------------------------
    # Affichage des métriques GPU si présentes
    # --------------------------------------------------------

    if gpu_metrics and gpu_metrics.get("devices"):

        d = gpu_metrics["devices"][0]

        print(
            f"      🧠 GPU : "
            f"{d['memory_used_mb']}/{d['memory_total_mb']} Mo "
            f"({d['memory_percent']}%) "
            f"| Util {d['utilization_gpu_percent']}% "
            f"| {d['temperature_c']}°C"
        )

    # --------------------------------------------------------
    # Trier par step
    # --------------------------------------------------------

    state["checkpoints"].sort(
        key=lambda c: (c.get("epoch", 0), c.get("step", 0))
    )

    save_state(state)

    return entry


# ============================================================
# MARQUER L'ENTRAÎNEMENT COMME EN COURS
# ============================================================

def start_training():
    """
    Marque l'entraînement comme EN COURS.

    - training_finished = false
    - Conserve les checkpoints existants
    """

    state = load_state()

    state["training_finished"] = False

    save_state(state)

    print("   🚀 Entraînement marqué comme EN COURS")


# ============================================================
# MARQUER L'ENTRAÎNEMENT COMME TERMINÉ
# ============================================================

def mark_training_finished():
    """
    Marque l'entraînement comme terminé.
    """

    state = load_state()

    state["training_finished"] = True

    save_state(state)

    print("   🏁 Entraînement marqué comme TERMINÉ")


# ============================================================
# RÉINITIALISER
# ============================================================

def reset_state():
    """
    Réinitialise complètement le JSON.
    """

    state = {
        "training_finished": False,
        "last_update": None,
        "checkpoints": [],
    }

    save_state(state)

    print("   🔄 JSON réinitialisé")


# ============================================================
# AFFICHER L'ÉTAT
# ============================================================

def print_state():

    state = load_state()

    print("=" * 70)
    print("ÉTAT DE L'ENTRAÎNEMENT")
    print("=" * 70)

    print(f"\n🏁 Training finished : {state['training_finished']}")
    print(f"📅 Dernière mise à jour : {state['last_update']}")
    print(f"💾 Nombre de checkpoints : {len(state['checkpoints'])}")

    if state["checkpoints"]:

        print("\n📋 Checkpoints enregistrés :")
        print("-" * 70)

        for ckpt in state["checkpoints"]:

            print(
                f"  [{ckpt['type']:5}] "
                f"{ckpt['name']:35} "
                f"epoch={ckpt.get('epoch', '?'):>3} "
                f"step={ckpt.get('step', '?'):>7} "
                f"loss={ckpt.get('train_loss', '?')}"
            )

        print("-" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print_state()
