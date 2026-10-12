"""
Upload du checkpoint BitNet + du résultat Markdown
vers Hugging Face Hub.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_file


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
    HF_RESULT_FOLDER,
)


# ============================================================
# UPLOAD D'UN FICHIER
# ============================================================

def upload_file_to_hub(
    local_path,
    remote_folder,
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
):

    local_path = Path(local_path)

    if not local_path.exists():

        raise FileNotFoundError(f"Fichier introuvable : {local_path}")

    remote_path = f"{remote_folder}/{local_path.name}"

    print(f"\n📤 Upload : {local_path.name}")
    print(f"   → {repo_id}/{remote_path}")

    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    url = (
        f"https://huggingface.co/{repo_type}s/"
        f"{repo_id}/resolve/main/{remote_path}"
    )

    print(f"   ✅ {url}")

    return url


# ============================================================
# UPLOAD CHECKPOINT + RÉSULTAT
# ============================================================

def upload_training_artifacts(
    checkpoint_path,
    result_path,
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
):

    checkpoint_path = Path(checkpoint_path)
    result_path = Path(result_path)

    # --------------------------------------------------------
    # CRÉATION DU REPO SI BESOIN
    # --------------------------------------------------------

    api = HfApi()

    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    # --------------------------------------------------------
    # UPLOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint_url = upload_file_to_hub(
        local_path=checkpoint_path,
        remote_folder=HF_CHECKPOINT_FOLDER,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    # --------------------------------------------------------
    # UPLOAD RÉSULTAT
    # --------------------------------------------------------

    result_url = upload_file_to_hub(
        local_path=result_path,
        remote_folder=HF_RESULT_FOLDER,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    return {"checkpoint_url": checkpoint_url, "result_url": result_url}


# ============================================================
# MAIN (test en local)
# ============================================================

if __name__ == "__main__":

    from scripts.bitnet.checkpoint_path_bitnet import (get_checkpoint_paths)

    (_, next_checkpoint) = get_checkpoint_paths()

    # Trouver le dernier checkpoint existant
    checkpoint_dir = Path("checkpoints/bitnet")

    checkpoints = sorted(
        checkpoint_dir.glob("entrainement_bitnet_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )

    if not checkpoints:

        raise FileNotFoundError(
            "Aucun checkpoint trouvé dans checkpoints/bitnet/"
        )

    last_checkpoint = checkpoints[-1]
    number = last_checkpoint.stem.split("_")[-1]

    result_path = (
        Path("documentation/result/bitnet")
        / f"result_entrainement_bitnet{number}.md"
    )

    if not result_path.exists():

        raise FileNotFoundError(f"Résultat introuvable : {result_path}")

    print("=" * 70)
    print("UPLOAD VERS HUGGING FACE")
    print("=" * 70)

    upload_training_artifacts(
        checkpoint_path=last_checkpoint,
        result_path=result_path,
    )

    print("\n✅ Upload terminé")
