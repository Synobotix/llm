import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

import llm.data.dataloader as dl


@pytest.fixture
def files(write_jsonl, monkeypatch):
    rows = [{"input_ids": [i, i + 1], "labels": [i + 1, i + 2]}
            for i in range(5)]
    train = write_jsonl(rows, "train.jsonl")
    val = write_jsonl(rows[:3], "val.jsonl")
    monkeypatch.setattr(dl, "TRAIN_FILE", train)
    monkeypatch.setattr(dl, "VALIDATION_FILE", val)


def test_train_loader_config(files):
    loader = dl.create_train_dataloader()
    assert loader.batch_size == dl.BATCH_SIZE
    assert isinstance(loader.sampler, RandomSampler)
    assert len(loader.dataset) == 5


def test_validation_loader_not_shuffled(files):
    loader = dl.create_validation_dataloader()
    assert isinstance(loader.sampler, SequentialSampler)
    assert len(loader.dataset) == 3


def test_batches_are_stacked_and_last_is_partial(files):
    loader = dl.create_validation_dataloader()
    batches = list(loader)
    assert [b["input_ids"].shape for b in batches] == [(2, 2), (1, 2)]
    assert batches[0]["input_ids"].dtype == torch.long
    assert batches[0]["input_ids"].tolist() == [[0, 1], [1, 2]]


def test_missing_train_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "TRAIN_FILE", tmp_path / "x.jsonl")
    with pytest.raises(FileNotFoundError):
        dl.create_train_dataloader()


def test_missing_validation_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "VALIDATION_FILE", tmp_path / "x.jsonl")
    with pytest.raises(FileNotFoundError):
        dl.create_validation_dataloader()


def test_empty_dataset_gives_no_batches(write_jsonl, monkeypatch):
    monkeypatch.setattr(dl, "VALIDATION_FILE", write_jsonl([]))
    assert list(dl.create_validation_dataloader()) == []


def test_variable_length_batch_fails_collate(write_jsonl, monkeypatch):
    path = write_jsonl([{"input_ids": [1], "labels": [1]},
                        {"input_ids": [1, 2], "labels": [1, 2]}])
    monkeypatch.setattr(dl, "VALIDATION_FILE", path)
    with pytest.raises(RuntimeError):
        list(dl.create_validation_dataloader())
