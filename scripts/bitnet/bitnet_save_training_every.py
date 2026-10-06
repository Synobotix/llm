# scripts/bitnet/bitnet_save_training_every.py

"""
Sauvegarde périodique du checkpoint BitNet.

Sauvegarde tous les N steps (valeur lue depuis parameters.py) :
- En local  : checkpoints/bitnet/checkpoint_step<N>.pt
- Sur HF    : checkpoints/bitnet/checkpoint_step<N>.pt

Enregistre chaque checkpoint dans training_state.json.

Nettoie automatiquement les anciens checkpoints step
(local + HF) pour ne garder que le dernier.

Objectif : ne pas perdre l'entraînement si le serveur coupe,
sans saturer le disque.
"""

import math
import sys
from pathlib import Path


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION (depuis parameters.py)
# ============================================================

from llm.config.parameters import (
    SAVE_EVERY_N_STEPS,
    AUTO_CLEAN_STEP_CHECKPOINTS,
)

from scripts.bitnet.heartbeat.gpu_monitor import (get_gpu_metrics)


# Dossier local des checkpoints
LOCAL_CHECKPOINT_DIR = Path("checkpoints/bitnet")

# Dossier distant sur HF
HF_CHECKPOINT_FOLDER = "checkpoints/bitnet"


# ============================================================
# UTILITAIRES
# ============================================================

def safe_perplexity(loss):
    """
    Calcule la perplexité en gérant les cas None / inf / 0.
    """

    if loss is None:
        return None

    try:
        return math.exp(loss)
    except (OverflowError, ValueError):
        return float("inf")


# ============================================================
# CLASSE DE SUIVI
# ============================================================

class StepCheckpointSaver:
    """
    Gère la sauvegarde périodique du checkpoint à intervalles
    réguliers de steps.

    - Sauvegarde local + HF
    - Enregistre dans training_state.json
    - Nettoie les anciens checkpoints step
    """

    def __init__(
        self,
        save_every_n_steps=SAVE_EVERY_N_STEPS,
        local_dir=LOCAL_CHECKPOINT_DIR,
        hf_folder=HF_CHECKPOINT_FOLDER,
        enabled=True,
        auto_clean=AUTO_CLEAN_STEP_CHECKPOINTS,
    ):
        self.save_every_n_steps = save_every_n_steps
        self.local_dir = Path(local_dir)
        self.hf_folder = hf_folder
        self.enabled = enabled
        self.auto_clean = auto_clean

        # Compteur interne
        self.last_saved_step = 0

        # Créer le dossier local
        self.local_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Faut-il sauvegarder maintenant ?
    # --------------------------------------------------------

    def should_save(self, current_step):
        """
        Retourne True si on doit sauvegarder à ce step.
        """

        if not self.enabled:
            return False

        if current_step < self.save_every_n_steps:
            return False

        next_milestone = (
            (self.last_saved_step // self.save_every_n_steps) + 1
        ) * self.save_every_n_steps

        return current_step >= next_milestone

    # --------------------------------------------------------
    # Nom du fichier
    # --------------------------------------------------------

    def get_checkpoint_path(self, step):
        """
        Retourne le chemin local du checkpoint pour ce step.
        """

        filename = f"checkpoint_step{step}.pt"

        return self.local_dir / filename

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    def save(self, step, save_checkpoint_fn, **kwargs):
        """
        Sauvegarde le checkpoint via la fonction fournie.

        Toutes les métriques sont injectées dans le checkpoint :
        - train_loss, validation_loss
        - train_ppl, validation_ppl
        - gradient_norm
        - duration, duration_seconds
        - gpu_metrics, cpu_metrics
        - epoch, step
        """

        if not self.should_save(step):
            return None

        checkpoint_path = self.get_checkpoint_path(step)

        print()
        print("=" * 70)
        print(f"💾 SAUVEGARDE PÉRIODIQUE — STEP {step}")
        print("=" * 70)

        # ----------------------------------------------------
        # 🔥 Collecte des métriques système (GPU + CPU + RAM)
        # ----------------------------------------------------

        gpu_metrics = get_gpu_metrics()

        if gpu_metrics.get("available") and gpu_metrics.get("devices"):
            d = gpu_metrics["devices"][0]
            print(
                f"\n🧠 GPU : {d['memory_used_mb']}/{d['memory_total_mb']} Mo "
                f"({d['memory_percent']}%) "
                f"| Util {d['utilization_gpu_percent']}% "
                f"| {d['temperature_c']}°C"
            )
        else:
            print("\n🧠 GPU : non disponible (CPU-only)")

        # ----------------------------------------------------
        # Injection dans les kwargs si non fournis
        # ----------------------------------------------------

        save_kwargs = dict(kwargs)
        save_kwargs.setdefault("gpu_metrics", gpu_metrics)

        # ----------------------------------------------------
        # Calcul des PPL si non fournies
        # ----------------------------------------------------

        train_loss = save_kwargs.get("train_loss")
        validation_loss = save_kwargs.get("validation_loss")

        if save_kwargs.get("train_ppl") is None:
            save_kwargs["train_ppl"] = safe_perplexity(train_loss)

        if save_kwargs.get("validation_ppl") is None:
            save_kwargs["validation_ppl"] = safe_perplexity(validation_loss)

        # ----------------------------------------------------
        # Affichage des métriques live
        # ----------------------------------------------------

        print(f"\n📊 Métriques live au step {step}:")
        print(f"   epoch           : {save_kwargs.get('epoch')}")
        print(f"   train_loss      : {save_kwargs.get('train_loss')}")
        print(f"   validation_loss : {save_kwargs.get('validation_loss')}")
        print(f"   train_ppl       : {save_kwargs.get('train_ppl')}")
        print(f"   validation_ppl  : {save_kwargs.get('validation_ppl')}")
        print(f"   gradient_norm   : {save_kwargs.get('gradient_norm')}")
        print(f"   duration_sec    : {save_kwargs.get('duration_seconds')}")

        # ----------------------------------------------------
        # Sauvegarde locale
        # ----------------------------------------------------

        try:

            save_checkpoint_fn(checkpoint_path=checkpoint_path, **save_kwargs)

            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

            print(f"\n✅ Checkpoint local : {checkpoint_path}")
            print(f"   Taille          : {size_mb:.2f} Mo")

        except Exception as error:

            print(f"\n❌ Échec sauvegarde locale : {error}")

            return None

        # ----------------------------------------------------
        # Upload HF
        # ----------------------------------------------------

        try:

            from scripts.hub.hub_sync import upload_file_to_hub

            remote_path = f"{self.hf_folder}/{checkpoint_path.name}"

            upload_file_to_hub(checkpoint_path, remote_path=remote_path)

            print(f"\n✅ Upload HF : {remote_path}")

        except Exception as error:

            print(f"\n⚠️ Upload HF échoué : {error}")
            print("   (le checkpoint local est intact)")

        # ----------------------------------------------------
        # Enregistrer dans le JSON
        # ----------------------------------------------------

        try:

            from scripts.bitnet.verif_save.create_modif_json_checkpoint import (
                register_checkpoint,
            )

            register_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_type="step",
                epoch=save_kwargs.get("epoch", 0),
                step=step,
                duration_seconds=save_kwargs.get("duration_seconds"),
                train_loss=save_kwargs.get("train_loss"),
                validation_loss=save_kwargs.get("validation_loss"),
                train_ppl=save_kwargs.get("train_ppl"),
                validation_ppl=save_kwargs.get("validation_ppl"),
                gradient_norm=save_kwargs.get("gradient_norm"),
                gpu_metrics=gpu_metrics,
            )

        except Exception as error:

            print(f"\n⚠️ Enregistrement JSON échoué : {error}")

        # ----------------------------------------------------
        # Nettoyer les anciens checkpoints step
        # ----------------------------------------------------

        if self.auto_clean:

            try:

                from scripts.bitnet.delete_data.delete_checkpoint import (
                    clean_old_step_checkpoints,
                )

                print()
                print("=" * 70)
                print("NETTOYAGE DES ANCIENS CHECKPOINTS STEP")
                print("=" * 70)

                clean_old_step_checkpoints()

            except Exception as error:

                print(f"\n⚠️ Nettoyage échoué : {error}")

        # ----------------------------------------------------
        # Mettre à jour le compteur
        # ----------------------------------------------------

        self.last_saved_step = step

        print()
        print("=" * 70)
        print(f"PROCHAIN CHECKPOINT : step {step + self.save_every_n_steps}")
        print("=" * 70)

        return checkpoint_path

    # --------------------------------------------------------
    # État actuel
    # --------------------------------------------------------

    def print_status(self):

        print(
            f"💾 Sauvegarde périodique : "
            f"tous les {self.save_every_n_steps:,} steps"
        )

        print(
            f"   Nettoyage auto : "
            f"{'✅ activé' if self.auto_clean else '❌ désactivé'}"
        )

        print(
            f"   Prochain : step "
            f"{self.last_saved_step + self.save_every_n_steps:,}"
        )
