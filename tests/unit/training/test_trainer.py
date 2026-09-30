import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from llm.model.transformer import Transformer
from llm.training import trainer as trainer_module
from llm.training.optimizer import create_optimizer
from llm.training.scheduler import create_scheduler
from llm.training.trainer import FINAL_CHECKPOINT_NAME, Trainer

VOCAB = 10
SEQUENCE = 5


class TinyDataset(Dataset):
    def __init__(self, size):
        generator = torch.Generator().manual_seed(0)
        ids = torch.randint(0, VOCAB, (size, SEQUENCE + 1), generator=generator)
        self.items = [
            {"input_ids": row[:-1], "labels": row[1:]} for row in ids
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def loader(size=4, batch_size=2):
    return DataLoader(TinyDataset(size), batch_size=batch_size)


def make_trainer(tmp_path, validation=True, scheduler=False, **kwargs):
    model = Transformer(VOCAB, 8, 2, 16, 1, SEQUENCE)
    optimizer = create_optimizer(model, 1e-2, 0.0)
    kwargs.setdefault("save_every", 0)
    return Trainer(
        model=model,
        train_dataloader=loader(),
        validation_dataloader=loader() if validation else None,
        optimizer=optimizer,
        scheduler=create_scheduler(optimizer, 100) if scheduler else None,
        checkpoint_dir=tmp_path / "ckpt",
        **kwargs,
    )


def test_init_creates_checkpoint_directory(tmp_path):
    make_trainer(tmp_path)
    assert (tmp_path / "ckpt").is_dir()


def test_train_step_updates_weights_and_returns_float_loss(tmp_path):
    trainer = make_trainer(tmp_path)
    first_param = next(trainer.model.parameters()).detach().clone()

    loss = trainer._train_step(next(iter(trainer.train_dataloader)))

    assert isinstance(loss, float)
    assert not torch.equal(first_param, next(trainer.model.parameters()))


def test_train_step_steps_the_scheduler(tmp_path):
    trainer = make_trainer(tmp_path, scheduler=True)

    trainer._train_step(next(iter(trainer.train_dataloader)))

    assert trainer.scheduler.last_epoch == 1


def test_train_step_clips_gradients(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, gradient_clip=0.5)
    calls = []
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda parameters, max_norm: calls.append(max_norm),
    )

    trainer._train_step(next(iter(trainer.train_dataloader)))

    assert calls == [0.5]


def test_train_epoch_counts_one_global_step_per_batch(tmp_path):
    trainer = make_trainer(tmp_path)

    average = trainer._train_epoch(1)

    assert trainer.global_step == 2
    assert average > 0


def test_train_epoch_on_empty_dataloader_returns_zero(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train_dataloader = []

    assert trainer._train_epoch(1) == 0.0
    assert trainer.global_step == 0


def test_loss_decreases_over_epochs(tmp_path):
    trainer = make_trainer(tmp_path, epochs=1)
    first = trainer._train_epoch(1)
    for _ in range(30):
        last = trainer._train_epoch(1)

    assert last < first


def test_evaluate_leaves_weights_untouched_and_uses_eval_mode(tmp_path):
    trainer = make_trainer(tmp_path)
    before = [p.detach().clone() for p in trainer.model.parameters()]

    loss = trainer.evaluate()

    assert loss > 0
    assert not trainer.model.training
    assert all(
        torch.equal(a, b) for a, b in zip(before, trainer.model.parameters())
    )


def test_evaluate_is_deterministic(tmp_path):
    trainer = make_trainer(tmp_path)
    assert trainer.evaluate() == trainer.evaluate()


def test_evaluate_without_validation_dataloader_raises(tmp_path):
    trainer = make_trainer(tmp_path, validation=False)

    with pytest.raises(ValueError):
        trainer.evaluate()


def test_train_saves_final_checkpoint(tmp_path, capsys):
    trainer = make_trainer(tmp_path, epochs=2)

    trainer.train()

    assert (tmp_path / "ckpt" / FINAL_CHECKPOINT_NAME).is_file()
    assert trainer.global_step == 4
    output = capsys.readouterr().out
    assert "Epoch 2/2" in output
    assert "Validation Loss" in output
    assert "ENTRAÎNEMENT TERMINÉ" in output


def test_train_without_validation_skips_validation_report(tmp_path, capsys):
    trainer = make_trainer(tmp_path, validation=False)

    trainer.train()

    assert "Validation Loss" not in capsys.readouterr().out


def test_periodic_checkpoint_saved_every_n_steps(tmp_path):
    trainer = make_trainer(tmp_path, epochs=1, save_every=2)

    trainer.train()

    names = {p.name for p in (tmp_path / "ckpt").iterdir()}
    assert "checkpoint_step_2.pt" in names
    assert "checkpoint_step_1.pt" not in names


def test_save_every_zero_disables_periodic_checkpoints(tmp_path):
    trainer = make_trainer(tmp_path, epochs=1, save_every=0)

    trainer.train()

    names = {p.name for p in (tmp_path / "ckpt").iterdir()}
    assert names == {FINAL_CHECKPOINT_NAME}


def test_saved_checkpoint_records_global_step(tmp_path):
    trainer = make_trainer(tmp_path, epochs=1)
    trainer.train()

    checkpoint = torch.load(tmp_path / "ckpt" / FINAL_CHECKPOINT_NAME)

    assert checkpoint["global_step"] == 2


def test_unpack_batch_moves_tensors_to_trainer_device(tmp_path):
    trainer = make_trainer(tmp_path, device="cpu")

    input_ids, labels = trainer._unpack_batch(
        {"input_ids": torch.zeros(1, 2), "labels": torch.ones(1, 2)}
    )

    assert input_ids.device.type == "cpu"
    assert labels.sum() == 2


def test_log_step_reports_learning_rate(tmp_path, capsys):
    trainer = make_trainer(tmp_path)
    trainer.global_step = 3

    trainer._log_step(1, 0.5)

    output = capsys.readouterr().out
    assert "Step 0003" in output
    assert "LR 0.01000000" in output


def test_module_constants():
    assert FINAL_CHECKPOINT_NAME == "student_v1_final.pt"
    assert trainer_module.SEPARATOR == "=" * 60
