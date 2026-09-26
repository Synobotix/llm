import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CausalLanguageModelingDataset(Dataset):
    """
    Dataset PyTorch pour l'entraînement d'un
    Transformer causal de type GPT.

    Chaque ligne du JSONL contient `input_ids` et `labels`.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {self.file_path}")

        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict]:
        """
        Charge les séquences depuis le JSONL.

        Les lignes vides, le JSON invalide et les lignes sans listes
        `input_ids` / `labels` sont ignorés. Des longueurs différentes
        lèvent une ValueError.
        """
        samples = []

        with self.file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                sample = self._parse_line(line, line_number)

                if sample is not None:
                    samples.append(sample)

        return samples

    @staticmethod
    def _parse_line(line: str, line_number: int) -> dict | None:
        line = line.strip()

        if not line:
            return None

        try:
            sample = json.loads(line)
        except json.JSONDecodeError as error:
            print(f"[WARNING] Ligne {line_number} ignorée : {error}")
            return None

        input_ids = sample.get("input_ids")
        labels = sample.get("labels")

        if not isinstance(input_ids, list) or not isinstance(labels, list):
            return None

        if len(input_ids) != len(labels):
            raise ValueError(f"Longueurs différentes à la ligne {line_number}")

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }
