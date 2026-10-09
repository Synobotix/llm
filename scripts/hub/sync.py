"""
Synchronisation complète vers Hugging Face.

Usage :
    python -m scripts.hub.sync
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.hub.hub_sync import sync_all_to_hub


if __name__ == "__main__":
    sync_all_to_hub()
