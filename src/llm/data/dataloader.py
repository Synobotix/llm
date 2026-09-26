from pathlib import Path

from torch.utils.data import DataLoader

from llm.data.dataset import CausalLanguageModelingDataset


TRAIN_FILE = Path("data/tokenized/train.jsonl")
VALIDATION_FILE = Path("data/tokenized/validation.jsonl")

BATCH_SIZE = 2
NUM_WORKERS = 0


def _create_dataloader(file_path: Path, shuffle: bool) -> DataLoader:
    return DataLoader(
        CausalLanguageModelingDataset(file_path),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


def create_train_dataloader() -> DataLoader:
    return _create_dataloader(TRAIN_FILE, shuffle=True)


def create_validation_dataloader() -> DataLoader:
    return _create_dataloader(VALIDATION_FILE, shuffle=False)
