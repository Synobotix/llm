from pathlib import Path
import re


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_DIRECTORY = Path("checkpoints/bitnet")


# ============================================================
# RÉCUPÉRATION DES CHECKPOINTS
# ============================================================

def get_checkpoint_paths():
    """
    Détermine automatiquement :

    - le dernier checkpoint à charger
    - le prochain checkpoint à sauvegarder

    Si aucun checkpoint n'existe :

        load = None
        save = entrainement_bitnet_1.pt

    Sinon :

        load = entrainement_bitnet_N.pt
        save = entrainement_bitnet_N+1.pt
    """

    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    checkpoints = []

    pattern = re.compile(r"^entrainement_bitnet_(\d+)\.pt$")

    for checkpoint in CHECKPOINT_DIRECTORY.iterdir():

        if not checkpoint.is_file():
            continue

        match = pattern.match(checkpoint.name)

        if match is None:
            continue

        number = int(match.group(1))

        checkpoints.append((number, checkpoint))

    # --------------------------------------------------------
    # Aucun checkpoint
    # --------------------------------------------------------

    if not checkpoints:

        return (None, CHECKPOINT_DIRECTORY / "entrainement_bitnet_1.pt")

    # --------------------------------------------------------
    # Dernier checkpoint
    # --------------------------------------------------------

    checkpoints.sort(key=lambda item: item[0])

    last_number, last_checkpoint = (checkpoints[-1])

    # --------------------------------------------------------
    # Nouveau checkpoint
    # --------------------------------------------------------

    next_number = (last_number + 1)

    next_checkpoint = (
        CHECKPOINT_DIRECTORY
        / f"entrainement_bitnet_{next_number}.pt"
    )

    return (last_checkpoint, next_checkpoint)
