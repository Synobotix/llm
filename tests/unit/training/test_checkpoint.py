import pytest
import torch

from llm.training.checkpoint import load_checkpoint, save_checkpoint


def build():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return model, optimizer, scheduler


def train_one_step(model, optimizer, scheduler):
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    scheduler.step()


def test_round_trip_restores_model_optimizer_scheduler_and_step(tmp_path):
    model, optimizer, scheduler = build()
    train_one_step(model, optimizer, scheduler)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, scheduler, 7, path)

    new_model, new_optimizer, new_scheduler = build()
    step = load_checkpoint(new_model, new_optimizer, new_scheduler, path)

    assert step == 7
    assert torch.equal(model.weight, new_model.weight)
    assert new_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert new_scheduler.last_epoch == scheduler.last_epoch
    assert new_optimizer.state_dict()["state"].keys() == \
        optimizer.state_dict()["state"].keys()


def test_save_creates_missing_parent_directories(tmp_path):
    model, optimizer, scheduler = build()
    path = tmp_path / "a" / "b" / "ckpt.pt"

    save_checkpoint(model, optimizer, scheduler, 0, path)

    assert path.is_file()


def test_save_without_scheduler_omits_its_state(tmp_path):
    model, optimizer, _ = build()
    path = tmp_path / "ckpt.pt"

    save_checkpoint(model, optimizer, None, 3, path)

    assert "scheduler_state_dict" not in torch.load(path)


def test_load_without_scheduler_in_file_keeps_scheduler_untouched(tmp_path):
    model, optimizer, _ = build()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, None, 3, path)

    _, new_optimizer, new_scheduler = build()
    step = load_checkpoint(build()[0], new_optimizer, new_scheduler, path)

    assert step == 3
    assert new_scheduler.last_epoch == 0


def test_load_ignores_scheduler_when_none_is_given(tmp_path):
    model, optimizer, scheduler = build()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, scheduler, 5, path)

    new_model, new_optimizer, _ = build()

    assert load_checkpoint(new_model, new_optimizer, None, path) == 5


def test_load_defaults_global_step_to_zero(tmp_path):
    model, optimizer, _ = build()
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

    assert load_checkpoint(model, optimizer, None, path) == 0


def test_load_missing_file_raises(tmp_path):
    model, optimizer, scheduler = build()

    with pytest.raises(FileNotFoundError):
        load_checkpoint(model, optimizer, scheduler, tmp_path / "nope.pt")


def test_accepts_string_paths(tmp_path):
    model, optimizer, scheduler = build()
    path = str(tmp_path / "ckpt.pt")

    save_checkpoint(model, optimizer, scheduler, 1, path)

    assert load_checkpoint(model, optimizer, scheduler, path) == 1
