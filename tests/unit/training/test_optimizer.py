import torch

from llm.training.optimizer import create_optimizer


def test_returns_adamw_with_given_hyperparameters():
    model = torch.nn.Linear(2, 2)

    optimizer = create_optimizer(model, learning_rate=0.1, weight_decay=0.05)

    assert isinstance(optimizer, torch.optim.AdamW)
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.1
    assert group["weight_decay"] == 0.05


def test_optimizes_every_model_parameter():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))

    optimizer = create_optimizer(model, 0.1, 0.0)

    optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert optimized == {id(p) for p in model.parameters()}


def test_step_updates_weights():
    model = torch.nn.Linear(2, 1)
    before = model.weight.detach().clone()
    optimizer = create_optimizer(model, learning_rate=0.1, weight_decay=0.0)

    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    assert not torch.equal(before, model.weight)
