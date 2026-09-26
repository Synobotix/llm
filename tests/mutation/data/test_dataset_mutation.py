import json
from pathlib import Path

import torch

from llm.data.dataset import CausalLanguageModelingDataset


def test_samples_are_read_as_utf8_text(tmp_path, monkeypatch):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"input_ids": [1], "labels": [2]}) + "\n",
        encoding="utf-8",
    )
    original_open = Path.open
    calls = []

    def spy_open(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)

    CausalLanguageModelingDataset(path)

    assert calls == [(("r",), {"encoding": "utf-8"})]


def test_labels_are_converted_to_long(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"input_ids": [1.0, 2.0], "labels": [3.0, 4.0]}) + "\n",
        encoding="utf-8",
    )

    item = CausalLanguageModelingDataset(path)[0]

    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype == torch.long
