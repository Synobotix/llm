import re
import sys
from pathlib import Path

import torch


# ============================================================
# RACINE DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.bitnet.checkpoint_path_bitnet import (get_checkpoint_paths)


# ============================================================
# DOSSIER DES RÉSULTATS
# ============================================================

RESULT_DIRECTORY = (PROJECT_ROOT / "documentation" / "result" / "bitnet")


# ============================================================
# NOM DU FICHIER DE RÉSULTAT
# ============================================================

RESULT_FILE_PREFIX = "result_entrainement_bitnet"

RESULT_FILE_EXTENSION = ".md"

RESULT_FILE_PATTERN = re.compile(
    rf"^{re.escape(RESULT_FILE_PREFIX)}(\d+)"
    rf"{re.escape(RESULT_FILE_EXTENSION)}$"
)


# ============================================================
# PROCHAIN NUMÉRO DE FICHIER
# ============================================================

def get_next_result_number(result_directory: Path) -> int:

    if not result_directory.exists():

        return 1

    max_number = 0

    for file_path in result_directory.iterdir():

        if not file_path.is_file():

            continue

        match = RESULT_FILE_PATTERN.match(file_path.name)

        if match is None:

            continue

        number = int(match.group(1))

        if number > max_number:

            max_number = number

    return max_number + 1


def get_next_result_path(result_directory: Path) -> Path:

    number = get_next_result_number(result_directory)

    filename = (f"{RESULT_FILE_PREFIX}" f"{number}" f"{RESULT_FILE_EXTENSION}")

    return result_directory / filename


# ============================================================
# CONSTRUCTION DU CONTENU MARKDOWN
# ============================================================

def build_markdown_report(
    checkpoint_path,
    result_path,
    epoch,
    training_results,
    configuration,
    trainable_parameters,
    device,
    gpu_name,
):

    fence = "`" * 3

    lines = []

    # --------------------------------------------------------
    # TITRE
    # --------------------------------------------------------

    lines.append("# Résultats — BitNet")
    lines.append("")

    # --------------------------------------------------------
    # INFORMATIONS GÉNÉRALES
    # --------------------------------------------------------

    lines.append("## Informations générales")
    lines.append("")
    lines.append("| Information | Valeur |")
    lines.append("|---|---|")
    lines.append(f"| Device | `{device}` |")

    if gpu_name is not None:

        lines.append(f"| GPU | `{gpu_name}` |")

    lines.append(f"| Checkpoint | `{checkpoint_path}` |")
    lines.append(f"| Époque finale | `{epoch}` |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --------------------------------------------------------
    # RÉSULTATS D'ENTRAÎNEMENT
    # --------------------------------------------------------

    lines.append("## Résultats d'entraînement")
    lines.append("")
    lines.append("| Mesure | Valeur |")
    lines.append("|---|---:|")
    lines.append(
        f"| Train loss | `{training_results.get('train_loss', 'N/A')}` |"
    )
    lines.append(
        f"| Validation loss | `{training_results.get('validation_loss', 'N/A')}` |"
    )
    lines.append(
        f"| Train PPL | `{training_results.get('train_ppl', 'N/A')}` |"
    )
    lines.append(
        f"| Validation PPL | `{training_results.get('validation_ppl', 'N/A')}` |"
    )
    lines.append(
        f"| Gradient norm | `{training_results.get('gradient_norm', 'N/A')}` |"
    )
    lines.append(
        f"| Learning rate | `{training_results.get('learning_rate', 'N/A')}` |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # --------------------------------------------------------
    # DURÉE DE L'ENTRAÎNEMENT
    # --------------------------------------------------------

    lines.append("## Durée de l'entraînement")
    lines.append("")
    lines.append("| Mesure | Valeur |")
    lines.append("|---|---:|")
    lines.append(
        f"| Durée totale | `{training_results.get('duration', 'N/A')}` |"
    )
    lines.append(
        f"| Durée en secondes | `{training_results.get('duration_seconds', 'N/A')}` |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # --------------------------------------------------------
    # CONFIGURATION DU MODÈLE
    # --------------------------------------------------------

    lines.append("## Configuration du modèle")
    lines.append("")
    lines.append("| Paramètre | Valeur |")
    lines.append("|---|---:|")
    lines.append(
        f"| Vocabulaire | `{configuration.get('vocab_size', 'N/A')}` |"
    )
    lines.append(f"| D model | `{configuration.get('d_model', 'N/A')}` |")
    lines.append(f"| Num heads | `{configuration.get('num_heads', 'N/A')}` |")
    lines.append(
        f"| Hidden dim | `{configuration.get('hidden_dim', 'N/A')}` |"
    )
    lines.append(
        f"| Num blocks | `{configuration.get('num_blocks', 'N/A')}` |"
    )
    lines.append(
        f"| Max sequence length | `{configuration.get('max_sequence_length', 'N/A')}` |"
    )
    lines.append(
        f"| Batch size | `{configuration.get('batch_size', 'N/A')}` |"
    )
    lines.append(
        f"| Learning rate | `{configuration.get('learning_rate', 'N/A')}` |"
    )
    lines.append(
        f"| Weight decay | `{configuration.get('weight_decay', 'N/A')}` |"
    )
    lines.append(
        f"| Gradient clip | `{configuration.get('gradient_clip', 'N/A')}` |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # --------------------------------------------------------
    # PARAMÈTRES ENTRAÎNABLES
    # --------------------------------------------------------

    lines.append("## Paramètres entraînables")
    lines.append("")

    if trainable_parameters is not None:

        lines.append("| Mesure | Valeur |")
        lines.append("|---|---:|")
        lines.append(
            f"| Tenseurs entraînables | `{trainable_parameters.get('trainable_tensors', 'N/A')}` |"
        )
        lines.append(
            f"| Tenseurs gelés | `{trainable_parameters.get('frozen_tensors', 'N/A')}` |"
        )

        total_trainable = trainable_parameters.get(
            "trainable_parameters",
            "N/A",
        )

        total_frozen = trainable_parameters.get("frozen_parameters", "N/A")

        total_parameters = trainable_parameters.get("total_parameters", "N/A")

        if isinstance(total_trainable, int):

            lines.append(
                f"| Paramètres entraînables | `{total_trainable:,}` |"
            )

        else:

            lines.append(f"| Paramètres entraînables | `{total_trainable}` |")

        if isinstance(total_frozen, int):

            lines.append(f"| Paramètres gelés | `{total_frozen:,}` |")

        else:

            lines.append(f"| Paramètres gelés | `{total_frozen}` |")

        if isinstance(total_parameters, int):

            lines.append(f"| Paramètres totaux | `{total_parameters:,}` |")

        else:

            lines.append(f"| Paramètres totaux | `{total_parameters}` |")

        lines.append(
            f"| Ratio entraînable | `{trainable_parameters.get('trainable_ratio', 'N/A')}` |"
        )
        lines.append("")

        modules = trainable_parameters.get("modules", [])

        if modules:

            lines.append("### Détail par sous-module")
            lines.append("")
            lines.append("| Sous-module | Total | Entraînable |")
            lines.append("|---|---:|---:|")

            for module in modules:

                lines.append(
                    f"| `{module.get('name', 'N/A')}` "
                    f"| `{module.get('total', 0):,}` "
                    f"| `{module.get('trainable', 0):,}` |"
                )

            lines.append("")

    else:

        lines.append("Aucune information disponible.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    lines.append("## Checkpoint")
    lines.append("")
    lines.append("Le checkpoint utilisé pour générer ce rapport est :")
    lines.append("")
    lines.append(fence + "text")
    lines.append(f"{checkpoint_path}")
    lines.append(fence)
    lines.append("")
    lines.append("Le rapport Markdown est enregistré dans :")
    lines.append("")
    lines.append(fence + "text")
    lines.append(f"{result_path}")
    lines.append(fence)
    lines.append("")
    lines.append(
        "Ce fichier a été généré automatiquement "
        "à partir du checkpoint BitNet."
    )
    lines.append("")

    return "\n".join(lines)


# ============================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================

def save_training_result(
    checkpoint_path=None,
    trainable_parameters=None,
    device=None,
    gpu_name=None,
    duration=None,
    duration_seconds=None,
    batch_size=None,
):

    # --------------------------------------------------------
    # RÉCUPÉRATION DU CHECKPOINT
    # --------------------------------------------------------

    if checkpoint_path is None:
        (_, checkpoint_path) = get_checkpoint_paths()

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    # --------------------------------------------------------
    # CHARGEMENT DU CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    epoch = checkpoint.get("epoch", "N/A")

    configuration = checkpoint.get("config", {})

    # --------------------------------------------------------
    # RÉSULTATS D'ENTRAÎNEMENT
    # --------------------------------------------------------
    # Priorité : argument passé > checkpoint > N/A
    # --------------------------------------------------------

    training_results = {

        "train_loss":
            checkpoint.get("train_loss", "N/A"),

        "validation_loss":
            checkpoint.get("validation_loss", "N/A"),

        "train_ppl":
            checkpoint.get("train_ppl", "N/A"),

        "validation_ppl":
            checkpoint.get("validation_ppl", "N/A"),

        "gradient_norm":
            checkpoint.get("gradient_norm", "N/A"),

        "learning_rate":
            configuration.get("learning_rate", "N/A"),

        "duration":
            duration
            if duration is not None
            else checkpoint.get("duration", "N/A"),

        "duration_seconds":
            duration_seconds
            if duration_seconds is not None
            else checkpoint.get("duration_seconds", "N/A"),
    }

    # --------------------------------------------------------
    # BATCH SIZE
    # --------------------------------------------------------

    if batch_size is None:

        batch_size = configuration.get("batch_size", "N/A")

    # --------------------------------------------------------
    # INJECTION DANS LA CONFIGURATION
    # --------------------------------------------------------

    configuration = dict(configuration)

    configuration["batch_size"] = batch_size

    # --------------------------------------------------------
    # CRÉATION DU DOSSIER
    # --------------------------------------------------------

    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # CHEMIN DU FICHIER MARKDOWN (NUMÉROTÉ)
    # --------------------------------------------------------

    result_path = get_next_result_path(RESULT_DIRECTORY)

    # --------------------------------------------------------
    # CONSTRUCTION DU MARKDOWN
    # --------------------------------------------------------

    markdown = build_markdown_report(
        checkpoint_path=checkpoint_path,
        result_path=result_path,
        epoch=epoch,
        training_results=training_results,
        configuration=configuration,
        trainable_parameters=trainable_parameters,
        device=device if device is not None else "N/A",
        gpu_name=gpu_name,
    )

    # --------------------------------------------------------
    # ÉCRITURE DU FICHIER
    # --------------------------------------------------------

    result_path.write_text(markdown, encoding="utf-8")

    # --------------------------------------------------------
    # AFFICHAGE TERMINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAUVEGARDE DES RÉSULTATS")
    print("=" * 70)
    print(f"\nCheckpoint utilisé : {checkpoint_path}")
    print(f"Résultats Markdown : {result_path}")
    print("\nRésultats sauvegardés avec succès.")

    return result_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    save_training_result()
