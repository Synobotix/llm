import pytest
import torch

from llm.training.scheduler import create_scheduler


def make(total_steps=10, min_lr=1e-5, lr=0.1):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=lr)
    return optimizer, create_scheduler(optimizer, total_steps, min_lr)


def run(optimizer, scheduler, steps):
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    return rates


def test_starts_at_initial_learning_rate():
    optimizer, _ = make(lr=0.1)
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_reaches_min_learning_rate_after_total_steps():
    optimizer, scheduler = make(total_steps=10, min_lr=0.01)
    assert run(optimizer, scheduler, 10)[-1] == pytest.approx(0.01)


def test_decreases_monotonically():
    optimizer, scheduler = make(total_steps=10)
    rates = run(optimizer, scheduler, 10)
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_midpoint_is_halfway_between_bounds():
    optimizer, scheduler = make(total_steps=10, min_lr=0.0, lr=0.1)
    assert run(optimizer, scheduler, 5)[-1] == pytest.approx(0.05)


def test_default_min_learning_rate_is_1e_minus_5():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = create_scheduler(optimizer, total_steps=4)

    assert scheduler.eta_min == 1e-5
