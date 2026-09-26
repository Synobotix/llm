import json

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def write_jsonl(tmp_path):
    """Écrit des lignes brutes (str) ou des dicts dans un fichier JSONL."""

    def _write(lines, name="data.jsonl"):
        path = tmp_path / name
        with path.open("w", encoding="utf-8") as file:
            for line in lines:
                if not isinstance(line, str):
                    line = json.dumps(line)
                file.write(line + "\n")
        return path

    return _write
