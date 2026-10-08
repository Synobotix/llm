# scripts/bitnet/train_bitnet.py

from scripts.bitnet.reprendre_entrainement.reprendre_entrainement import (
    get_resume_state,
    download_json_from_hf,
)
from scripts.bitnet.verif_save.create_modif_json_checkpoint import (
    register_checkpoint,
    mark_training_finished,
    start_training,
    print_state,
)
from scripts.bitnet.heartbeat.gpu_monitor import (
    get_gpu_metrics,
    write_monitoring_json,
)
from scripts.bitnet.heartbeat.heartbeat import (Heartbeat)
from scripts.bitnet.bitnet_save_training_every import (StepCheckpointSaver)
from scripts.bitnet.save_result_bitnet import (save_training_result)
from scripts.bitnet.checkpoint_path_bitnet import (get_checkpoint_paths)
from scripts.duration import TrainingTimer
from llm.bitnet_model.lm_head import (LMHead)
from llm.bitnet_model.bit_transformer import (BitTransformer)
from llm.data.dataloader import (create_validation_dataloader)
from llm.config.parameters import (
    MAX_DOCUMENT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    MAX_SEQUENCE_LENGTH,
    TOKENIZER_FILE,
    D_MODEL,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_BLOCKS,
)
from tokenizers import Tokenizer
from torch.optim import AdamW
import torch.nn as nn
import torch
import math
import os
import sys
from pathlib import Path


# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CONSTANTES MONITORING
# ============================================================

MONITORING_JSON_PATH = Path("monitoring.json")

TRAINING_STATE_PATH = Path("training_state.json")

# Fréquence d'upload de monitoring.json vers Hugging Face.
# ⚠️ Ne pas descendre en dessous de 20 (rate limits HF).
MONITORING_UPLOAD_INTERVAL = 1000

# Fréquence d'appel de nvidia-smi (sous-processus coûteux).
# Tous les N steps. Entre, on utilise uniquement torch.cuda.
NVIDIA_SMI_INTERVAL = 1000


# ============================================================
# DATASET
# ============================================================

TRAIN_DATASET_FILE = Path("data/tokenized/train.jsonl")


# ============================================================
# UTILITAIRES STATE_DICT / PARALLÉLISME
# ============================================================

def unwrap_state_dict(state_dict):
    """
    Retire le préfixe 'module.' des clés si le modèle
    a été enveloppé dans DataParallel.

    Cela garantit que les checkpoints sauvegardés sont
    compatibles avec ou sans DataParallel.
    """

    new_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        new_state_dict[new_key] = value

    return new_state_dict


def adapt_state_dict_for_model(state_dict, model):
    """
    Ajoute le préfixe 'module.' si le modèle est enveloppé
    dans DataParallel, sinon retourne le state_dict tel quel.

    Permet de charger un checkpoint sauvegardé avec ou
    sans DataParallel dans n'importe quelle configuration.
    """

    is_data_parallel = isinstance(model, nn.DataParallel)

    # Détecter si le state_dict a déjà le préfixe
    has_module_prefix = any(
        key.startswith("module.")
        for key in state_dict.keys()
    )

    if is_data_parallel and not has_module_prefix:

        # Ajouter le préfixe
        new_state_dict = {}

        for key, value in state_dict.items():
            new_state_dict[f"module.{key}"] = value

        return new_state_dict

    elif not is_data_parallel and has_module_prefix:

        # Retirer le préfixe
        return unwrap_state_dict(state_dict)

    else:

        # Rien à faire
        return state_dict


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():
    """
    Charge le tokenizer utilisé pour le dataset.
    """

    if not TOKENIZER_FILE.exists():

        raise FileNotFoundError(
            f"Tokenizer introuvable : "
            f"{TOKENIZER_FILE}"
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    return tokenizer


# ============================================================
# DATASET TRAIN
# ============================================================

class TokenizedDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch pour le fichier JSONL tokenisé.
    """

    def __init__(self, file_path: Path):
        import json

        self.samples = []

        if not file_path.exists():

            raise FileNotFoundError(
                f"Dataset train introuvable : "
                f"{file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                try:

                    sample = json.loads(line)

                except json.JSONDecodeError as error:

                    raise ValueError(
                        f"Ligne {line_number} invalide "
                        f"dans {file_path}: {error}"
                    )

                input_ids = sample.get("input_ids")

                labels = sample.get("labels")

                if not isinstance(input_ids, list):

                    raise ValueError(
                        f"input_ids invalide "
                        f"à la ligne {line_number}"
                    )

                if not isinstance(labels, list):

                    raise ValueError(
                        f"labels invalides "
                        f"à la ligne {line_number}"
                    )

                if len(input_ids) != MAX_SEQUENCE_LENGTH:

                    raise ValueError(
                        f"input_ids doit contenir "
                        f"{MAX_SEQUENCE_LENGTH} tokens "
                        f"à la ligne {line_number}"
                    )

                if len(labels) != MAX_SEQUENCE_LENGTH:

                    raise ValueError(
                        f"labels doit contenir "
                        f"{MAX_SEQUENCE_LENGTH} tokens "
                        f"à la ligne {line_number}"
                    )

                self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        return {
            "input_ids": torch.tensor(
                sample["input_ids"],
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                sample["labels"],
                dtype=torch.long,
            ),
        }


# ============================================================
# TRAIN DATALOADER (OPTIMISÉ)
# ============================================================

def create_train_dataloader():
    """
    Crée le DataLoader du dataset d'entraînement.

    Optimisations :
    - num_workers : parallélise le chargement CPU
    - pin_memory  : accélère les transferts CPU → GPU
    - persistent_workers : évite de recréer les workers
    - prefetch_factor : nombre de batchs préchargés
    """

    dataset = TokenizedDataset(TRAIN_DATASET_FILE)

    # Nombre de workers optimal :
    # - 0 si CPU-only (évite les surcoûts)
    # - min(4, cpu_count) si GPU
    if DEVICE.type == "cuda":
        num_workers = min(4, os.cpu_count() or 1)
        pin_memory = True
        persistent_workers = num_workers > 0
        prefetch_factor = 2 if num_workers > 0 else None
    else:
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None

    print(f"\n📦 DataLoader config (train) :")
    print(f"   num_workers        : {num_workers}")
    print(f"   pin_memory         : {pin_memory}")
    print(f"   persistent_workers : {persistent_workers}")
    print(f"   prefetch_factor    : {prefetch_factor}")

    dataloader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    return dataloader


# ============================================================
# PERPLEXITY
# ============================================================

def calculate_perplexity(loss):
    """
    Calcule la perplexité à partir de la loss.
    """

    if loss is None:
        return None

    try:

        return math.exp(loss)

    except OverflowError:

        return float("inf")


# ============================================================
# GRADIENT NORM
# ============================================================

def calculate_gradient_norm(model):
    """
    Calcule la norme globale des gradients.

    Compatible avec DataParallel : on unwrap le modèle
    pour accéder aux paramètres réels.
    """

    if isinstance(model, nn.DataParallel):
        target = model.module
    else:
        target = model

    total_norm = 0.0

    for parameter in target.parameters():

        if parameter.grad is None:
            continue

        parameter_norm = (parameter.grad.detach() .data.norm(2))

        total_norm += (parameter_norm.item() ** 2)

    return total_norm ** 0.5


# ============================================================
# UPLOAD MONITORING.JSON + TRAINING_STATE.JSON VERS HF
# ============================================================

def upload_monitoring_to_hub():
    """
    Upload monitoring.json et training_state.json vers Hugging Face.

    Ne bloque jamais l'entraînement en cas d'erreur.
    """

    try:

        from scripts.hub.hub_sync import (upload_file_to_hub)

        upload_file_to_hub(MONITORING_JSON_PATH, remote_path="monitoring.json")

        if TRAINING_STATE_PATH.exists():
            upload_file_to_hub(
                TRAINING_STATE_PATH,
                remote_path="training_state.json",
            )

        return True

    except Exception as error:

        print(f"\n⚠️ Upload monitoring.json échoué : " f"{error}")

        return False


# ============================================================
# PARAMÈTRES ENTRAÎNABLES
# ============================================================

def print_trainable_parameters(model, model_name="MODEL"):
    """
    Affiche et retourne les paramètres entraînables du modèle.

    Compatible avec DataParallel.
    """

    # Unwrap pour accéder au modèle réel
    if isinstance(model, nn.DataParallel):
        target = model.module
    else:
        target = model

    trainable_parameters = [p for p in target.parameters() if p.requires_grad]

    frozen_parameters = [p for p in target.parameters() if not p.requires_grad]

    total_trainable = sum(p.numel() for p in trainable_parameters)

    total_frozen = sum(p.numel() for p in frozen_parameters)

    total_parameters = (total_trainable + total_frozen)

    trainable_ratio = (
        (total_trainable / total_parameters * 100.0)
        if total_parameters > 0
        else 0.0
    )

    modules_info = []

    for name, module in target.named_children():

        module_total = sum(p.numel() for p in module.parameters())

        module_trainable = sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )

        modules_info.append(
            {
                "name": name,
                "total": module_total,
                "trainable": module_trainable,
            }
        )

    # ---------- Affichage terminal ----------

    print("\n" + "=" * 70)

    print(f"PARAMÈTRES ENTRAÎNABLES — {model_name}")

    print("=" * 70)

    print(f"\nTenseurs entraînables  : " f"{len(trainable_parameters)}")

    print(f"Tenseurs gelés         : " f"{len(frozen_parameters)}")

    print(f"\nParamètres entraînables : " f"{total_trainable:,}")

    print(f"Paramètres gelés        : " f"{total_frozen:,}")

    print(f"Paramètres totaux       : " f"{total_parameters:,}")

    if total_parameters > 0:

        print(f"Ratio entraînable       : " f"{trainable_ratio:.4f} %")

    print("\nDétail par sous-module :")

    print("-" * 70)

    for module in modules_info:

        print(
            f"  {module['name']:<30} "
            f"total = {module['total']:>12,}  "
            f"entraînable = {module['trainable']:>12,}"
        )

    print("-" * 70)

    return {
        "trainable_tensors": len(trainable_parameters),
        "frozen_tensors": len(frozen_parameters),
        "trainable_parameters": total_trainable,
        "frozen_parameters": total_frozen,
        "total_parameters": total_parameters,
        "trainable_ratio": f"{trainable_ratio:.4f} %",
        "modules": modules_info,
    }


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    transformer,
    lm_head,
    optimizer,
    vocab_size,
):
    """
    Charge le checkpoint précédent et restaure :

    - les poids du Transformer
    - les poids du LM Head
    - l'état de l'optimizer
    - l'époque précédente
    - les métriques précédentes

    Compatible avec/sans DataParallel grâce à
    adapt_state_dict_for_model().
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint précédent introuvable : "
            f"{checkpoint_path}"
        )

    print("\nChargement du checkpoint précédent...")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if "config" not in checkpoint:

        raise ValueError("Le checkpoint ne contient pas " "de configuration.")

    checkpoint_config = (checkpoint["config"])

    checkpoint_vocab_size = (checkpoint_config.get("vocab_size"))

    checkpoint_d_model = (checkpoint_config.get("d_model"))

    checkpoint_num_heads = (checkpoint_config.get("num_heads"))

    checkpoint_hidden_dim = (checkpoint_config.get("hidden_dim"))

    checkpoint_num_blocks = (checkpoint_config.get("num_blocks"))

    checkpoint_max_sequence_length = (
        checkpoint_config.get(
            "max_sequence_length"
        )
    )

    if checkpoint_vocab_size != vocab_size:

        raise ValueError(
            "\nLe vocabulaire du checkpoint "
            "ne correspond pas au tokenizer actuel.\n"
            f"Checkpoint : {checkpoint_vocab_size}\n"
            f"Tokenizer  : {vocab_size}"
        )

    if checkpoint_d_model != D_MODEL:

        raise ValueError(
            "\nD_MODEL différent du checkpoint.\n"
            f"Checkpoint : {checkpoint_d_model}\n"
            f"Actuel     : {D_MODEL}"
        )

    if checkpoint_num_heads != NUM_HEADS:

        raise ValueError(
            "\nNUM_HEADS différent du checkpoint.\n"
            f"Checkpoint : {checkpoint_num_heads}\n"
            f"Actuel     : {NUM_HEADS}"
        )

    if checkpoint_hidden_dim != HIDDEN_DIM:

        raise ValueError(
            "\nHIDDEN_DIM différent du checkpoint.\n"
            f"Checkpoint : {checkpoint_hidden_dim}\n"
            f"Actuel     : {HIDDEN_DIM}"
        )

    if checkpoint_num_blocks != NUM_BLOCKS:

        raise ValueError(
            "\nNUM_BLOCKS différent du checkpoint.\n"
            f"Checkpoint : {checkpoint_num_blocks}\n"
            f"Actuel     : {NUM_BLOCKS}"
        )

    if (checkpoint_max_sequence_length != MAX_SEQUENCE_LENGTH):

        raise ValueError(
            "\nMAX_SEQUENCE_LENGTH différent "
            "du checkpoint.\n"
            f"Checkpoint : "
            f"{checkpoint_max_sequence_length}\n"
            f"Actuel     : "
            f"{MAX_SEQUENCE_LENGTH}"
        )

    # ----------------------------------------------------
    # Adapter les state_dict au parallélisme actuel
    # ----------------------------------------------------

    transformer_state_dict = adapt_state_dict_for_model(
        checkpoint["transformer_state_dict"],
        transformer,
    )

    lm_head_state_dict = adapt_state_dict_for_model(
        checkpoint["lm_head_state_dict"],
        lm_head,
    )

    transformer.load_state_dict(transformer_state_dict)

    lm_head.load_state_dict(lm_head_state_dict)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    previous_epoch = checkpoint.get("epoch", 0)

    previous_train_loss = checkpoint.get("train_loss", None)

    previous_validation_loss = checkpoint.get("validation_loss", None)

    print(f"Checkpoint chargé : " f"{checkpoint_path}")

    print(f"Époque précédente : " f"{previous_epoch}")

    if previous_train_loss is not None:

        print(f"Ancienne train loss : " f"{previous_train_loss:.6f}")

    if previous_validation_loss is not None:

        print(f"Ancienne validation loss : " f"{previous_validation_loss:.6f}")

    return previous_epoch


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    transformer,
    lm_head,
    dataloader,
    optimizer,
    criterion,
    step_saver=None,
    global_step=0,
    save_checkpoint_fn=None,
    save_checkpoint_kwargs=None,
    heartbeat=None,
    epoch=None,
    monitoring_path=MONITORING_JSON_PATH,
    monitoring_upload_interval=MONITORING_UPLOAD_INTERVAL,
    live_metrics=None,
    training_timer=None,
):
    """
    Effectue une époque complète d'entraînement.

    Retourne (loss_moyenne, gradient_norm, global_step).
    """

    import time as _time

    transformer.train()

    lm_head.train()

    total_loss = 0.0

    total_tokens = 0

    last_gradient_norm = 0.0

    _last_heartbeat_time = _time.time()

    for batch_index, batch in enumerate(dataloader, start=1):

        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)

        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        hidden_states = transformer(input_ids)

        logits = lm_head(hidden_states)

        batch_size = logits.size(0)

        sequence_length = (logits.size(1))

        vocab_size = logits.size(2)

        logits = logits.reshape(batch_size * sequence_length, vocab_size)

        labels = labels.reshape(batch_size * sequence_length)

        loss = criterion(logits, labels)

        loss.backward()

        transformer_gradient_norm = (calculate_gradient_norm(transformer))

        lm_head_gradient_norm = (calculate_gradient_norm(lm_head))

        last_gradient_norm = (
            transformer_gradient_norm ** 2
            + lm_head_gradient_norm ** 2
        ) ** 0.5

        torch.nn.utils.clip_grad_norm_(
            list(
                transformer.parameters()
            )
            + list(
                lm_head.parameters()
            ),
            GRADIENT_CLIP,
        )

        optimizer.step()

        number_of_tokens = (batch_size * sequence_length)

        total_loss += (loss.item() * number_of_tokens)

        total_tokens += (number_of_tokens)

        # ------------------------------------------------
        # Incrémenter le step global
        # ------------------------------------------------

        global_step += 1

        # ------------------------------------------------
        # 💓 HEARTBEAT PRINT (toutes les 15s)
        # ------------------------------------------------

        if _time.time() - _last_heartbeat_time > 15.0:

            print(
                f"\n💓 Heartbeat step {global_step} "
                f"@ {_time.strftime('%H:%M:%S')}",
                flush=True,
            )

            _last_heartbeat_time = _time.time()

        # ------------------------------------------------
        # 📊 Mise à jour des métriques live
        # ------------------------------------------------

        if live_metrics is not None:

            live_metrics["step"] = global_step
            live_metrics["epoch"] = epoch
            live_metrics["train_loss"] = loss.item()
            live_metrics["gradient_norm"] = last_gradient_norm

            if training_timer is not None:
                live_metrics["duration_seconds"] = training_timer.elapsed()
                live_metrics["duration"] = training_timer.format_duration(
                    training_timer.elapsed()
                )

        # ------------------------------------------------
        # 💓 HEARTBEAT
        # ------------------------------------------------

        if heartbeat is not None:

            heartbeat.beat(step=global_step, loss=loss.item())

        # ------------------------------------------------
        # 📊 MONITORING SYSTÈME (CPU + GPU + RAM)
        # ------------------------------------------------

        try:

            write_monitoring_json(
                step=global_step,
                loss=loss.item(),
                gradient_norm=last_gradient_norm,
                epoch=epoch,
                path=monitoring_path,
                use_nvidia_smi=(
                    global_step % NVIDIA_SMI_INTERVAL == 0
                ),
            )

        except Exception as error:

            print(f"\n⚠️ Monitoring échoué : " f"{error}")

        # ------------------------------------------------
        # 📤 UPLOAD PÉRIODIQUE DE monitoring.json
        # ------------------------------------------------

        if (
            monitoring_upload_interval > 0
            and global_step % monitoring_upload_interval == 0
        ):

            print(f"\n📤 Upload monitoring.json " f"(step {global_step})...")

            upload_monitoring_to_hub()

        # ------------------------------------------------
        # Sauvegarde périodique
        # ------------------------------------------------

        if (step_saver is not None and save_checkpoint_fn is not None):

            # Construire les kwargs à jour
            kwargs = dict(save_checkpoint_kwargs or {})

            if live_metrics is not None:

                kwargs.update({
                    "epoch": live_metrics["epoch"],
                    "train_loss": live_metrics["train_loss"],
                    "validation_loss": live_metrics.get("validation_loss", 0.0),
                    "gradient_norm": live_metrics["gradient_norm"],
                    "duration_seconds": live_metrics.get("duration_seconds"),
                    "duration": live_metrics.get("duration", "N/A"),
                })

            step_saver.save(
                step=global_step,
                save_checkpoint_fn=save_checkpoint_fn,
                **kwargs,
            )

        # ------------------------------------------------
        # Affichage de la progression
        # ------------------------------------------------

        print(
            f"\rBatch "
            f"{batch_index:>4}/"
            f"{len(dataloader):<4}"
            f" | Loss : "
            f"{loss.item():.6f}",
            end="",
            flush=True,
        )

    print()

    average_loss = (
        total_loss / total_tokens
        if total_tokens > 0
        else float("inf")
    )

    return (average_loss, last_gradient_norm, global_step)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(transformer, lm_head, dataloader, criterion):
    """
    Évalue le modèle sur le dataset de validation.
    """

    transformer.eval()

    lm_head.eval()

    total_loss = 0.0

    total_tokens = 0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)

        labels = batch["labels"].to(DEVICE, non_blocking=True)

        hidden_states = transformer(input_ids)

        logits = lm_head(hidden_states)

        batch_size = logits.size(0)

        sequence_length = (logits.size(1))

        vocab_size = logits.size(2)

        logits = logits.reshape(batch_size * sequence_length, vocab_size)

        labels = labels.reshape(batch_size * sequence_length)

        loss = criterion(logits, labels)

        number_of_tokens = (batch_size * sequence_length)

        total_loss += (loss.item() * number_of_tokens)

        total_tokens += (number_of_tokens)

    average_loss = (
        total_loss / total_tokens
        if total_tokens > 0
        else float("inf")
    )

    return average_loss


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    checkpoint_path,
    transformer,
    lm_head,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    gradient_norm,
    vocab_size,
    duration="N/A",
    duration_seconds="N/A",
    batch_size="N/A",
    gpu_metrics=None,
    train_ppl=None,
    validation_ppl=None,
):
    """
    Sauvegarde le nouveau checkpoint.

    Compatible avec/sans DataParallel grâce à
    unwrap_state_dict() : les checkpoints sont toujours
    sauvegardés SANS le préfixe 'module.'.
    """

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Gestion des valeurs None (au cas où)
    if train_loss is None:
        train_loss = 0.0
    if validation_loss is None:
        validation_loss = 0.0
    if gradient_norm is None:
        gradient_norm = 0.0

    # Calcul des PPL si non fournies
    if train_ppl is None:
        train_ppl = calculate_perplexity(train_loss)

    if validation_ppl is None:
        validation_ppl = calculate_perplexity(validation_loss)

    # ------------------------------------------------
    # Unwrap des state_dict pour compatibilité
    # ------------------------------------------------

    transformer_state_dict = unwrap_state_dict(transformer.state_dict())

    lm_head_state_dict = unwrap_state_dict(lm_head.state_dict())

    checkpoint = {

        "epoch":
            epoch,

        "step":
            None,   # rempli par le saver si besoin

        "transformer_state_dict":
            transformer_state_dict,

        "lm_head_state_dict":
            lm_head_state_dict,

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        "train_ppl":
            train_ppl,

        "validation_ppl":
            validation_ppl,

        "train_pll_per_token":
            train_loss,

        "validation_pll_per_token":
            validation_loss,

        "gradient_norm":
            gradient_norm,

        "duration":
            duration,

        "duration_seconds":
            duration_seconds,

        # ------------------------------------------------
        # 🔥 MÉTRIQUES GPU / CPU AU MOMENT DE LA SAUVEGARDE
        # ------------------------------------------------

        "gpu_metrics":
            gpu_metrics if gpu_metrics is not None else {},

        "config": {

            "vocab_size":
                vocab_size,

            "d_model":
                transformer.module.d_model if isinstance(
                    transformer, nn.DataParallel) else transformer.d_model,

            "num_heads":
                transformer.module.num_heads if isinstance(
                    transformer, nn.DataParallel) else transformer.num_heads,

            "hidden_dim":
                transformer.module.hidden_dim if isinstance(
                    transformer, nn.DataParallel) else transformer.hidden_dim,

            "num_blocks":
                transformer.module.num_blocks if isinstance(
                    transformer, nn.DataParallel) else transformer.num_blocks,

            "max_sequence_length":
                transformer.module.max_sequence_length if isinstance(
                    transformer, nn.DataParallel) else transformer.max_sequence_length,

            "batch_size":
                batch_size,

            "epochs":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip":
                GRADIENT_CLIP,
        },
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"\nNouveau checkpoint sauvegardé : " f"{checkpoint_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print("ENTRAÎNEMENT BITNET")

    print("=" * 70)

    # ========================================================
    # SYNCHRONISATION CHECKPOINT HF → LOCAL
    # ========================================================

    try:

        from scripts.bitnet.verify_checkpoint_in_hugging_face import (
            verify_and_sync,
        )

        print("\n" + "=" * 70)
        print("SYNCHRONISATION DES CHECKPOINTS — HF vs LOCAL")
        print("=" * 70)

        verify_and_sync()

    except Exception as error:

        print(f"\n⚠️ Synchronisation échouée : {error}")
        print("   → L'entraînement continue avec les checkpoints locaux.")

    # ========================================================
    # SYNCHRONISATION TOKENIZER HF → LOCAL
    # ========================================================

    try:

        from scripts.bitnet.verify_tokenizer_in_hugging_face import (
            verify_and_sync_tokenizer,
        )

        print("\n" + "=" * 70)
        print("VÉRIFICATION DU TOKENIZER")
        print("=" * 70)

        verify_and_sync_tokenizer()

    except Exception as error:

        print(f"\n⚠️ Vérification tokenizer échouée : {error}")
        print("   → L'entraînement continue avec le tokenizer local.")

    # ========================================================
    # TÉLÉCHARGEMENT DU JSON DE SUIVI DEPUIS HF
    # ========================================================

    print("\n" + "=" * 70)
    print("VÉRIFICATION DU JSON DE SUIVI")
    print("=" * 70)

    try:

        download_json_from_hf()

    except Exception as error:

        print(f"\n⚠️ Téléchargement JSON échoué : {error}")
        print("   → Démarrage avec un état vide")

    # ========================================================
    # DEVICE
    # ========================================================

    print(f"\nDevice : {DEVICE}")

    if DEVICE.type == "cuda":

        print(f"GPU : " f"{torch.cuda.get_device_name(0)}")

    # ========================================================
    # CHECKPOINT PATHS
    # ========================================================

    (previous_checkpoint, new_checkpoint) = get_checkpoint_paths()

    print("\nCheckpoint à charger : " f"{previous_checkpoint}")

    print("Checkpoint à sauvegarder : " f"{new_checkpoint}")

    # ========================================================
    # TOKENIZER
    # ========================================================

    print("\nChargement du tokenizer...")

    tokenizer = load_tokenizer()

    vocab_size = (tokenizer.get_vocab_size())

    print(f"Vocabulaire réel : " f"{vocab_size}")

    # ========================================================
    # DATALOADERS
    # ========================================================

    print("\nCréation des DataLoaders...")

    train_loader = (create_train_dataloader())

    validation_loader = (create_validation_dataloader())

    print(f"\nTrain batches      : " f"{len(train_loader)}")

    print(f"Validation batches : " f"{len(validation_loader)}")

    print(f"Batch size         : " f"{BATCH_SIZE}")

    print(f"Sequence length    : " f"{MAX_SEQUENCE_LENGTH}")

    # ========================================================
    # TRANSFORMER
    # ========================================================

    print("\nCréation du BitTransformer...")

    transformer = BitTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    ).to(
        DEVICE
    )

    # ========================================================
    # LM HEAD
    # ========================================================

    print("Création du LM Head...")

    lm_head = LMHead(d_model=D_MODEL, vocab_size=vocab_size).to(DEVICE)

    # ========================================================
    # 🖥️ PARALLÉLISME MULTI-GPU
    # ========================================================

    GPU_COUNT = (torch.cuda.device_count() if DEVICE.type == "cuda" else 0)

    USE_DATA_PARALLEL = GPU_COUNT > 1

    if USE_DATA_PARALLEL:

        print(f"\n🖥️  Parallélisme activé : " f"{GPU_COUNT} GPU détectés")

        print(
            f"   → nn.DataParallel utilisé "
            f"sur les GPU 0..{GPU_COUNT - 1}"
        )

        transformer = nn.DataParallel(
            transformer,
            device_ids=list(range(GPU_COUNT)),
        )

        lm_head = nn.DataParallel(lm_head, device_ids=list(range(GPU_COUNT)))

    elif GPU_COUNT == 1:

        print(
            f"\n🖥️  Un seul GPU détecté "
            f"({torch.cuda.get_device_name(0)})"
        )

        print("   → Pas de parallélisme")

    else:

        print("\n🖥️  Aucun GPU détecté")

        print("   → Entraînement sur CPU")

    # ========================================================
    # PARAMETERS
    # ========================================================

    # Unwrap pour compter les paramètres
    transformer_raw = (
        transformer.module
        if isinstance(transformer, nn.DataParallel)
        else transformer
    )

    lm_head_raw = (
        lm_head.module
        if isinstance(lm_head, nn.DataParallel)
        else lm_head
    )

    transformer_parameters = sum(
        parameter.numel()
        for parameter in transformer_raw.parameters()
    )

    lm_head_parameters = sum(
        parameter.numel()
        for parameter in lm_head_raw.parameters()
    )

    total_parameters = (transformer_parameters + lm_head_parameters)

    print(f"\nParamètres Transformer : " f"{transformer_parameters:,}")

    print(f"Paramètres LM Head    : " f"{lm_head_parameters:,}")

    print(f"Paramètres totaux     : " f"{total_parameters:,}")

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = AdamW(
        list(
            transformer.parameters()
        )
        + list(
            lm_head.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # LOAD PREVIOUS CHECKPOINT
    # ========================================================

    if previous_checkpoint is not None:

        previous_epoch = (
            load_checkpoint(
                checkpoint_path=previous_checkpoint,
                transformer=transformer,
                lm_head=lm_head,
                optimizer=optimizer,
                vocab_size=vocab_size,
            )
        )

    else:

        previous_epoch = 0

        print("\nAucun checkpoint précédent.")

        print("Entraînement à partir de zéro.")

    # ========================================================
    # ÉTAT DE REPRISE (DURÉE CUMULÉE)
    # ========================================================

    print("\n" + "=" * 70)
    print("ÉTAT DE REPRISE")
    print("=" * 70)

    resume_state = get_resume_state()

    print(f"\n🏁 Training terminé      : {resume_state['training_finished']}")
    print(f"📊 Époques terminées     : {resume_state['total_epochs_done']}")
    print(f"📅 Dernière époque       : {resume_state['last_epoch']}")
    print(f"🔢 Dernier step          : {resume_state['last_step']}")
    print(f"💾 Dernier checkpoint    : {resume_state['last_checkpoint']}")
    print(
        f"\n⏱️  Durée cumulée (avant) : {resume_state['cumulative_duration_human']}")

    # ========================================================
    # MARQUER L'ENTRAÎNEMENT COMME EN COURS
    # ========================================================

    print("\n" + "=" * 70)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("=" * 70)

    start_training()

    print_state()

    # ========================================================
    # CONFIGURATION
    # ========================================================

    print("\nConfiguration :")

    print(f"  Documents           = " f"{MAX_DOCUMENT}")

    print(f"  Vocabulaire         = " f"{vocab_size}")

    print(f"  D model             = " f"{D_MODEL}")

    print(f"  Num heads           = " f"{NUM_HEADS}")

    print(f"  Hidden dim          = " f"{HIDDEN_DIM}")

    print(f"  Num blocks          = " f"{NUM_BLOCKS}")

    print(f"  Max sequence        = " f"{MAX_SEQUENCE_LENGTH}")

    print(f"  Batch size          = " f"{BATCH_SIZE}")

    print(f"  Nouvelles époques   = " f"{EPOCHS}")

    print(f"  Learning rate       = " f"{LEARNING_RATE}")

    print(f"  Weight decay        = " f"{WEIGHT_DECAY}")

    print(f"  Gradient clip       = " f"{GRADIENT_CLIP}")

    print(f"  Ancien checkpoint   = " f"{previous_checkpoint}")

    print(f"  Nouveau checkpoint  = " f"{new_checkpoint}")

    print(
        f"  DataParallel        = "
        f"{'✅ activé' if USE_DATA_PARALLEL else '❌ désactivé'}"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 70)

    print("REPRISE DE L'ENTRAÎNEMENT")

    print("=" * 70)

    print(f"\nReprise après l'époque " f"{previous_epoch}")

    print(f"Nouvelles époques : " f"{EPOCHS}")

    # ========================================================
    # CHRONOMÉTRAGE
    # ========================================================

    print("\n" + "=" * 70)

    print("CHRONOMÉTRAGE DE L'ENTRAÎNEMENT")

    print("=" * 70)

    training_timer = TrainingTimer(
        initial_duration=resume_state["cumulative_duration_seconds"]
    )

    training_timer.start()

    print(
        f"\n⏱️  Timer initialisé avec : "
        f"{resume_state['cumulative_duration_human']}"
    )

    # ========================================================
    # SAUVEGARDE PÉRIODIQUE
    # ========================================================

    step_saver = StepCheckpointSaver(enabled=True)

    step_saver.print_status()

    # ========================================================
    # 💓 HEARTBEAT
    # ========================================================

    heartbeat = Heartbeat(interval=30.0, enabled=True, prefix="💓")

    print(f"\n💓 Heartbeat activé " f"(intervalle = {heartbeat.interval}s)")

    # ========================================================
    # INITIALISATION DES DERNIÈRES MÉTRIQUES
    # ========================================================

    train_loss = None

    validation_loss = None

    gradient_norm = None

    # --------------------------------------------------------
    # Reprendre le global_step depuis le JSON
    # --------------------------------------------------------

    global_step = resume_state["last_step"]

    print(f"\n🔢 Global step repris : {global_step}")

    # ========================================================
    # MÉTRIQUES LIVE (partagées entre steps et époques)
    # ========================================================

    live_metrics = {
        "step": global_step,
        "epoch": previous_epoch,
        "train_loss": 0.0,
        "validation_loss": resume_state.get("last_validation_loss", 0.0),
        "train_ppl": None,
        "validation_ppl": None,
        "gradient_norm": 0.0,
        "duration_seconds": resume_state["cumulative_duration_seconds"],
        "duration": resume_state["cumulative_duration_human"],
    }

    # ========================================================
    # MONITORING INITIAL
    # ========================================================

    try:

        write_monitoring_json(
            step=global_step,
            loss=None,
            epoch=previous_epoch,
            path=MONITORING_JSON_PATH,
            use_nvidia_smi=True,
        )

        print(f"\n📊 monitoring.json initialisé : " f"{MONITORING_JSON_PATH}")

        upload_monitoring_to_hub()

    except Exception as error:

        print(f"\n⚠️ Monitoring initial échoué : " f"{error}")

    # ========================================================
    # EPOCHS
    # ========================================================

    for local_epoch in range(1, EPOCHS + 1):

        global_epoch = (previous_epoch + local_epoch)

        print(
            f"\nÉPOQUE "
            f"{global_epoch} "
            f"(nouvelle époque "
            f"{local_epoch}/{EPOCHS})"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        (
            train_loss,
            gradient_norm,
            global_step,
        ) = train_one_epoch(
            transformer=transformer,
            lm_head=lm_head,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            step_saver=step_saver,
            global_step=global_step,
            save_checkpoint_fn=save_checkpoint,
            save_checkpoint_kwargs={
                "transformer": transformer,
                "lm_head": lm_head,
                "optimizer": optimizer,
                "vocab_size": vocab_size,
                "batch_size": BATCH_SIZE,
            },
            heartbeat=heartbeat,
            epoch=global_epoch,
            monitoring_path=MONITORING_JSON_PATH,
            monitoring_upload_interval=MONITORING_UPLOAD_INTERVAL,
            live_metrics=live_metrics,
            training_timer=training_timer,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_loss = validate(
            transformer=transformer,
            lm_head=lm_head,
            dataloader=validation_loader,
            criterion=criterion,
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        train_ppl = calculate_perplexity(train_loss)

        validation_ppl = (calculate_perplexity(validation_loss))

        # ----------------------------------------------------
        # Mise à jour des métriques live (post-validation)
        # ----------------------------------------------------

        live_metrics["epoch"] = global_epoch
        live_metrics["train_loss"] = train_loss
        live_metrics["validation_loss"] = validation_loss
        live_metrics["train_ppl"] = train_ppl
        live_metrics["validation_ppl"] = validation_ppl
        live_metrics["gradient_norm"] = gradient_norm

        print(f"\nTrain loss       : " f"{train_loss:.6f}")

        print(f"Validation loss  : " f"{validation_loss:.6f}")

        print(f"Train PPL        : " f"{train_ppl:.4f}")

        print(f"Validation PPL   : " f"{validation_ppl:.4f}")

        print(f"Gradient norm    : " f"{gradient_norm:.6f}")

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_checkpoint(
            checkpoint_path=new_checkpoint,
            transformer=transformer,
            lm_head=lm_head,
            optimizer=optimizer,
            epoch=global_epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            gradient_norm=gradient_norm,
            vocab_size=vocab_size,
            batch_size=BATCH_SIZE,
            gpu_metrics=get_gpu_metrics(use_nvidia_smi=True),
            train_ppl=train_ppl,
            validation_ppl=validation_ppl,
        )

        # ----------------------------------------------------
        # ENREGISTRER DANS LE JSON (upload HF automatique)
        # ----------------------------------------------------

        register_checkpoint(
            checkpoint_path=new_checkpoint,
            checkpoint_type="epoch",
            epoch=global_epoch,
            step=global_step,
            duration_seconds=training_timer.elapsed(),
            train_loss=train_loss,
            validation_loss=validation_loss,
            train_ppl=train_ppl,
            validation_ppl=validation_ppl,
            gradient_norm=gradient_norm,
            gpu_metrics=get_gpu_metrics(use_nvidia_smi=True),
        )

        # ----------------------------------------------------
        # UPLOAD CHECKPOINT À CHAQUE ÉPOQUE
        # ----------------------------------------------------

        try:

            from scripts.hub.hub_sync import (upload_file_to_hub)

            upload_file_to_hub(
                new_checkpoint,
                remote_path=(
                    f"checkpoints/bitnet/"
                    f"{Path(new_checkpoint).name}"
                ),
            )

        except Exception as error:

            print(f"\n⚠️ Upload checkpoint échoué : " f"{error}")

        # ----------------------------------------------------
        # 📤 UPLOAD monitoring.json EN FIN D'ÉPOQUE
        # ----------------------------------------------------

        print(f"\n📤 Upload monitoring.json " f"(fin époque {global_epoch})...")

        upload_monitoring_to_hub()

    # ========================================================
    # DURÉE
    # ========================================================

    training_duration = (training_timer.stop())

    print("\n" + "=" * 70)

    print("DURÉE TOTALE DE L'ENTRAÎNEMENT")

    print("=" * 70)

    print(
        f"\nDurée totale de l'entraînement : "
        f"{training_timer.format_duration(training_duration)}"
    )

    print(f"Durée totale en secondes : " f"{training_duration:.2f}s")

    # ========================================================
    # RE-SAVE DU CHECKPOINT AVEC LA DURÉE
    # ========================================================

    if train_loss is not None:

        save_checkpoint(
            checkpoint_path=new_checkpoint,
            transformer=transformer,
            lm_head=lm_head,
            optimizer=optimizer,
            epoch=previous_epoch + EPOCHS,
            train_loss=train_loss,
            validation_loss=validation_loss,
            gradient_norm=gradient_norm,
            vocab_size=vocab_size,
            duration=training_timer.format_duration(
                training_duration
            ),
            duration_seconds=training_duration,
            batch_size=BATCH_SIZE,
            gpu_metrics=get_gpu_metrics(use_nvidia_smi=True),
            train_ppl=calculate_perplexity(train_loss),
            validation_ppl=calculate_perplexity(validation_loss),
        )

    # ========================================================
    # PARAMÈTRES ENTRAÎNABLES
    # ========================================================

    trainable_info = print_trainable_parameters(
        transformer,
        model_name="BitTransformer",
    )

    # ========================================================
    # RAPPORT MARKDOWN
    # ========================================================

    gpu_name = (
        torch.cuda.get_device_name(0)
        if DEVICE.type == "cuda"
        else None
    )

    result_path = save_training_result(
        checkpoint_path=new_checkpoint,
        trainable_parameters=trainable_info,
        device=str(DEVICE),
        gpu_name=gpu_name,
        duration=training_timer.format_duration(
            training_duration
        ),
        duration_seconds=training_duration,
        batch_size=BATCH_SIZE,
    )

    # ========================================================
    # UPLOAD FINAL (checkpoint + résultat + monitoring)
    # ========================================================

    try:

        from scripts.hub.hub_sync import (upload_file_to_hub)

        print("\n" + "=" * 70)

        print("UPLOAD FINAL VERS HUGGING FACE")

        print("=" * 70)

        upload_file_to_hub(
            new_checkpoint,
            remote_path=(
                f"checkpoints/bitnet/"
                f"{Path(new_checkpoint).name}"
            ),
        )

        upload_file_to_hub(
            result_path,
            remote_path=(
                f"documentation/result/bitnet/"
                f"{Path(result_path).name}"
            ),
        )

        # 📤 Upload final de monitoring.json
        upload_monitoring_to_hub()

    except Exception as error:

        print(f"\n⚠️ Upload final échoué : {error}")

    # ========================================================
    # MARQUER L'ENTRAÎNEMENT COMME TERMINÉ
    # ========================================================

    mark_training_finished()
    print_state()

    # ========================================================
    # FIN
    # ========================================================

    print("\n" + "=" * 70)

    print("POURSUITE DE L'ENTRAÎNEMENT TERMINÉE")

    print("=" * 70)

    print(f"\nDernière époque : " f"{previous_epoch + EPOCHS}")

    print(f"Ancien checkpoint : " f"{previous_checkpoint}")

    print(f"Nouveau checkpoint : " f"{new_checkpoint}")


if __name__ == "__main__":

    import traceback
    from datetime import datetime

    try:

        main()

    except Exception as error:

        crash_time = datetime.now().isoformat()

        print("\n" + "=" * 70)
        print(f"❌ CRASH À {crash_time}")
        print("=" * 70)

        traceback.print_exc()

        # Écrire dans un fichier pour analyse post-mortem
        with open("crash_log.txt", "w", encoding="utf-8") as f:
            f.write(f"Crash à {crash_time}\n\n")
            traceback.print_exc(file=f)

        print("\n📝 Trace complète écrite dans crash_log.txt")

        raise
