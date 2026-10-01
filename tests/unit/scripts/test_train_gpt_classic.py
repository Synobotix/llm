import json
import math
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.gpt_classic import train as train_script

VOCAB = 10
SEQUENCE = 5


class TinyDataset(Dataset):
    def __init__(self, size=4):
        generator = torch.Generator().manual_seed(0)
        ids = torch.randint(0, VOCAB, (size, SEQUENCE + 1), generator=generator)
        self.items = [
            {"input_ids": row[:-1], "labels": row[1:]} for row in ids
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_model():
    from llm.model.transformer import Transformer

    return Transformer(VOCAB, 8, 2, 16, 1, SEQUENCE)


def make_training_objects(scheduler=True):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        if scheduler
        else None
    )
    return model, optimizer, lr_scheduler


# ------------------------------------------------------------
# get_vocab_size
# ------------------------------------------------------------

@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(train_script, "TOKENIZER_CONFIG_FILE", path)
    return path


def test_vocab_size_reads_actual_vocab_size(config_file):
    config_file.write_text(json.dumps({"actual_vocab_size": 321}))

    assert train_script.get_vocab_size() == 321


def test_vocab_size_missing_file_raises(config_file):
    with pytest.raises(FileNotFoundError):
        train_script.get_vocab_size()


def test_vocab_size_missing_key_raises(config_file):
    config_file.write_text(json.dumps({"other": 1}))

    with pytest.raises(KeyError):
        train_script.get_vocab_size()


@pytest.mark.parametrize("value", ["12", 1.5, None, "x"])
def test_vocab_size_non_integer_raises(config_file, value):
    config_file.write_text(json.dumps({"actual_vocab_size": value}))

    with pytest.raises(TypeError):
        train_script.get_vocab_size()


@pytest.mark.parametrize("value", [0, -3])
def test_vocab_size_non_positive_raises(config_file, value):
    config_file.write_text(json.dumps({"actual_vocab_size": value}))

    with pytest.raises(ValueError):
        train_script.get_vocab_size()


# ------------------------------------------------------------
# set_seed / count_parameters
# ------------------------------------------------------------

def test_set_seed_makes_all_generators_reproducible():
    train_script.set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    train_script.set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_count_parameters_sums_all_elements():
    model = torch.nn.Linear(3, 2)

    assert train_script.count_parameters(model) == 3 * 2 + 2


def test_count_parameters_includes_frozen_ones():
    model = torch.nn.Linear(3, 2)
    model.weight.requires_grad_(False)

    assert train_script.count_parameters(model) == 8


# ------------------------------------------------------------
# perplexité / norme du gradient
# ------------------------------------------------------------

def test_perplexity_is_exp_of_loss():
    assert train_script.calculate_perplexity(0.0) == 1.0
    assert train_script.calculate_perplexity(2.0) == pytest.approx(math.exp(2))


def test_perplexity_overflow_returns_infinity():
    assert train_script.calculate_perplexity(1e6) == float("inf")


def test_gradient_norm_without_gradients_is_zero():
    assert train_script.calculate_gradient_norm(torch.nn.Linear(2, 2)) == 0.0


def test_gradient_norm_is_global_l2_norm():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])

    assert train_script.calculate_gradient_norm(model) == pytest.approx(5.0)


def test_gradient_norm_skips_parameters_without_gradient():
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    model.bias.grad = None

    assert train_script.calculate_gradient_norm(model) == pytest.approx(5.0)


# ------------------------------------------------------------
# train_one_epoch
# ------------------------------------------------------------

def run_epoch(dataloader, gradient_clip=1.0, scheduler=True, global_step=0):
    model, optimizer, lr_scheduler = make_training_objects(scheduler)
    result = train_script.train_one_epoch(
        model,
        dataloader,
        optimizer,
        lr_scheduler,
        torch.device("cpu"),
        gradient_clip,
        global_step,
    )
    return model, lr_scheduler, result


def test_train_one_epoch_returns_metrics_and_advances_global_step():
    _, _, (loss, perplexity, gradient_norm, step) = run_epoch(
        DataLoader(TinyDataset(4), batch_size=2), global_step=10
    )

    assert loss > 0
    assert perplexity == pytest.approx(math.exp(loss))
    assert gradient_norm > 0
    assert step == 12


def test_train_one_epoch_updates_weights_and_scheduler():
    model = make_model()
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

    train_script.train_one_epoch(
        model,
        DataLoader(TinyDataset(4), batch_size=2),
        optimizer,
        scheduler,
        torch.device("cpu"),
        1.0,
        0,
    )

    assert not torch.equal(before, next(model.parameters()))
    assert scheduler.last_epoch == 2
    assert model.training


def test_train_one_epoch_works_without_scheduler_and_clip():
    _, _, (loss, _, _, step) = run_epoch(
        DataLoader(TinyDataset(2), batch_size=2),
        gradient_clip=None,
        scheduler=False,
    )

    assert loss > 0
    assert step == 1


def test_train_one_epoch_empty_dataloader_raises():
    with pytest.raises(RuntimeError):
        run_epoch([])


# ------------------------------------------------------------
# validate
# ------------------------------------------------------------

def test_validate_returns_loss_and_perplexity_without_training():
    model = make_model()
    before = [p.detach().clone() for p in model.parameters()]

    loss, perplexity = train_script.validate(
        model, DataLoader(TinyDataset(4), batch_size=2), torch.device("cpu")
    )

    assert loss > 0
    assert perplexity == pytest.approx(math.exp(loss))
    assert not model.training
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_validate_is_deterministic():
    model = make_model()
    loader = DataLoader(TinyDataset(4), batch_size=2)
    device = torch.device("cpu")

    assert train_script.validate(model, loader, device) == \
        train_script.validate(model, loader, device)


def test_validate_empty_dataloader_raises():
    with pytest.raises(RuntimeError):
        train_script.validate(make_model(), [], torch.device("cpu"))


# ------------------------------------------------------------
# save_checkpoint
# ------------------------------------------------------------

def test_save_checkpoint_stores_full_training_state(tmp_path):
    model, optimizer, scheduler = make_training_objects()
    path = tmp_path / "nested" / "ckpt.pt"

    train_script.save_checkpoint(
        path, model, optimizer, scheduler, 2, 17,
        {"train_loss": 0.5}, {"d_model": 8},
    )

    checkpoint = torch.load(path)
    assert checkpoint["epoch"] == 2
    assert checkpoint["global_step"] == 17
    assert checkpoint["training_results"] == {"train_loss": 0.5}
    assert checkpoint["configuration"] == {"d_model": 8}
    assert set(checkpoint) >= {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    }


def test_save_checkpoint_can_be_loaded_back_into_a_model(tmp_path):
    model, optimizer, scheduler = make_training_objects()
    path = tmp_path / "ckpt.pt"
    train_script.save_checkpoint(
        path, model, optimizer, scheduler, 1, 1, {}, {}
    )

    other = make_model()
    other.load_state_dict(torch.load(path)["model_state_dict"])

    assert all(
        torch.equal(a, b)
        for a, b in zip(model.parameters(), other.parameters())
    )
