import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from llm.training.loss import CausalLanguageModelingLoss
from scripts import evaluate as evaluate_script

VOCAB = 8


class TinyDataset(Dataset):
    def __init__(self, size=4):
        generator = torch.Generator().manual_seed(0)
        ids = torch.randint(0, VOCAB, (size, 5), generator=generator)
        self.items = [
            {"input_ids": row[:-1], "labels": row[1:]} for row in ids
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class ConstantModel(torch.nn.Module):
    """Logits uniformes : la loss vaut log(VOCAB) quel que soit le batch."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        return torch.zeros(*input_ids.shape, VOCAB) + self.anchor


def run(model, dataloader):
    return evaluate_script.evaluate(
        model, dataloader, CausalLanguageModelingLoss(), "cpu"
    )


def test_returns_mean_loss_over_batches():
    loss = run(ConstantModel(), DataLoader(TinyDataset(4), batch_size=2))

    assert loss == pytest.approx(math.log(VOCAB))


def test_switches_model_to_eval_mode():
    model = ConstantModel().train()

    run(model, DataLoader(TinyDataset(2), batch_size=2))

    assert not model.training


def test_does_not_modify_weights_or_build_gradients():
    model = ConstantModel()
    before = model.anchor.detach().clone()

    run(model, DataLoader(TinyDataset(2), batch_size=2))

    assert torch.equal(before, model.anchor)
    assert model.anchor.grad is None


def test_averages_per_batch_not_per_sample():
    class FirstBatchPerfect(ConstantModel):
        calls = 0

        def forward(self, input_ids):
            type(self).calls += 1
            logits = super().forward(input_ids)
            return logits * 0 if type(self).calls == 1 else logits

    loss = run(FirstBatchPerfect(), DataLoader(TinyDataset(4), batch_size=2))

    assert loss == pytest.approx(math.log(VOCAB))


def test_empty_dataloader_raises():
    with pytest.raises(RuntimeError):
        run(ConstantModel(), [])


def test_real_transformer_gives_finite_positive_loss():
    from llm.model.transformer import Transformer

    model = Transformer(VOCAB, 8, 2, 16, 1, 4)

    loss = run(model, DataLoader(TinyDataset(4), batch_size=2))

    assert math.isfinite(loss)
    assert loss > 0
