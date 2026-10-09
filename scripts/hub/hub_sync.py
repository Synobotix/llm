"""
Synchronisation des artefacts vers Hugging Face.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_file, upload_folder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.hub.sync_config import (HF_REPO_ID, HF_REPO_TYPE, SYNC_FOLDERS)


def _ensure_repo():

    api = HfApi()

    api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, exist_ok=True)


def _is_hf_available():

    try:

        import huggingface_hub  # noqa

        return True

    except ImportError:

        return False


def upload_file_to_hub(local_path, remote_path=None):

    if not _is_hf_available():

        print("⚠️ huggingface_hub non installé — upload ignoré")

        return None

    local_path = Path(local_path)

    if not local_path.exists():

        print(f"⚠️ Fichier introuvable — upload ignoré : " f"{local_path}")

        return None

    if remote_path is None:
        remote_path = local_path.name

    _ensure_repo()

    print(f"\n📤 Upload : {local_path.name}")
    print(f"   → {HF_REPO_ID}/{remote_path}")

    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=str(remote_path),
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
    )

    url = (
        f"https://huggingface.co/{HF_REPO_TYPE}s/"
        f"{HF_REPO_ID}/resolve/main/{remote_path}"
    )

    print(f"   ✅ {url}")

    return url


def upload_folder_to_hub(local_folder, remote_folder=None):

    if not _is_hf_available():

        print("⚠️ huggingface_hub non installé — upload ignoré")

        return None

    local_folder = Path(local_folder)

    if not local_folder.exists():

        print(f"⚠️ Dossier introuvable — upload ignoré : " f"{local_folder}")

        return None

    if remote_folder is None:
        remote_folder = str(local_folder)

    _ensure_repo()

    print(f"\n📤 Upload dossier : {local_folder}")
    print(f"   → {HF_REPO_ID}/{remote_folder}")

    upload_folder(
        folder_path=str(local_folder),
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        path_in_repo=str(remote_folder),
    )

    print(f"   ✅ Terminé")

    return True


def sync_all_to_hub():

    if not _is_hf_available():

        print("⚠️ huggingface_hub non installé — sync ignorée")

        return

    print("=" * 70)
    print("SYNCHRONISATION VERS HUGGING FACE")
    print("=" * 70)

    for local_folder, remote_folder in SYNC_FOLDERS:

        local_path = Path(local_folder)

        if not local_path.exists():

            print(f"\n⚠️ Ignoré (absent) : {local_folder}")

            continue

        upload_folder_to_hub(
            local_folder=local_path,
            remote_folder=remote_folder,
        )

    print("\n" + "=" * 70)
    print("SYNCHRONISATION TERMINÉE")
    print("=" * 70)
