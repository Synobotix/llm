"""
Télécharge les artefacts depuis Hugging Face
vers le workspace local.
"""

import sys
from pathlib import Path

from huggingface_hub import (snapshot_download, list_repo_files)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.hub.sync_config import (HF_REPO_ID, HF_REPO_TYPE, SYNC_FOLDERS)


def download_all_from_hub():

    print("=" * 70)
    print("TÉLÉCHARGEMENT DEPUIS HUGGING FACE")
    print("=" * 70)

    files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)

    if not files:

        print("\n⚠️ Aucun fichier disponible sur HF")
        print("   Départ de zéro.")
        return

    remote_folders = [remote for _, remote in SYNC_FOLDERS]

    to_download = [
        f for f in files
        if any(
            f.startswith(folder + "/")
            for folder in remote_folders
        )
    ]

    if not to_download:

        print("\n⚠️ Aucun fichier dans les dossiers cibles")
        return

    print(f"\n{len(to_download)} fichier(s) à télécharger")

    for remote_folder in remote_folders:

        print(f"\n📥 Dossier : {remote_folder}")

        try:

            local_dir = snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                allow_patterns=[f"{remote_folder}/*"],
                local_dir=".",
            )

            print(f"   ✅ Téléchargé dans {local_dir}")

        except Exception as error:

            print(f"   ⚠️ Ignoré : {error}")

    print("\n" + "=" * 70)
    print("TÉLÉCHARGEMENT TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    download_all_from_hub()
