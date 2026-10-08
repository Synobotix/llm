import json
import math
import sys
import types

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llm.bitnet_model.bit_transformer import BitTransformer
from llm.bitnet_model.lm_head import LMHead
from scripts.bitnet import train_bitnet as module

VOCAB = 12
D_MODEL = 8
HEADS = 2
HIDDEN = 16
BLOCKS = 1
SEQUENCE = 5


@pytest.fixture(autouse=True)
def tiny_configuration(monkeypatch):
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(module, "D_MODEL", D_MODEL)
    monkeypatch.setattr(module, "NUM_HEADS", HEADS)
    monkeypatch.setattr(module, "HIDDEN_DIM", HIDDEN)
    monkeypatch.setattr(module, "NUM_BLOCKS", BLOCKS)
    monkeypatch.setattr(module, "MAX_SEQUENCE_LENGTH", SEQUENCE)
    monkeypatch.setattr(module, "GRADIENT_CLIP", 1.0)
    # Pas d'écriture monitoring.json / training_state.json dans le dépôt.
    monkeypatch.setattr(module, "write_monitoring_json", lambda **kw: None)


def build_model():
    transformer = BitTransformer(VOCAB, D_MODEL, HEADS, HIDDEN, BLOCKS, SEQUENCE)
    lm_head = LMHead(D_MODEL, VOCAB)
    optimizer = torch.optim.AdamW(
        list(transformer.parameters()) + list(lm_head.parameters()), lr=1e-2
    )
    return transformer, lm_head, optimizer


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


def loader(size=4, batch_size=2):
    return DataLoader(TinyDataset(size), batch_size=batch_size)


# ------------------------------------------------------------
# state_dict / DataParallel
# ------------------------------------------------------------

def test_unwrap_removes_module_prefix_only_where_present():
    state = {"module.a": 1, "b": 2, "module.module.c": 3}

    assert module.unwrap_state_dict(state) == {"a": 1, "b": 2, "module.c": 3}


def test_adapt_adds_prefix_for_data_parallel_model():
    model = nn.DataParallel(nn.Linear(2, 2))

    adapted = module.adapt_state_dict_for_model({"weight": 1, "bias": 2}, model)

    assert set(adapted) == {"module.weight", "module.bias"}


def test_adapt_removes_prefix_for_plain_model():
    adapted = module.adapt_state_dict_for_model(
        {"module.weight": 1}, nn.Linear(2, 2)
    )

    assert set(adapted) == {"weight"}


def test_adapt_leaves_matching_state_dict_untouched():
    state = {"weight": 1}
    assert module.adapt_state_dict_for_model(state, nn.Linear(2, 2)) is state

    wrapped = {"module.weight": 1}
    assert module.adapt_state_dict_for_model(
        wrapped, nn.DataParallel(nn.Linear(2, 2))
    ) is wrapped


# ------------------------------------------------------------
# tokenizer
# ------------------------------------------------------------

def test_load_tokenizer_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "TOKENIZER_FILE", tmp_path / "absent.json")

    with pytest.raises(FileNotFoundError):
        module.load_tokenizer()


def test_load_tokenizer_reads_tokenizers_file(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, "<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    monkeypatch.setattr(module, "TOKENIZER_FILE", path)

    assert module.load_tokenizer().encode("hello").ids == [1]


# ------------------------------------------------------------
# TokenizedDataset / dataloader
# ------------------------------------------------------------

def sample(start=0):
    ids = list(range(start, start + SEQUENCE + 1))
    return {"input_ids": ids[:-1], "labels": ids[1:]}


def write_jsonl(path, items):
    path.write_text(
        "\n".join(i if isinstance(i, str) else json.dumps(i) for i in items)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_dataset_loads_samples_as_long_tensors(tmp_path):
    path = write_jsonl(tmp_path / "t.jsonl", [sample(0), sample(1)])

    dataset = module.TokenizedDataset(path)

    assert len(dataset) == 2
    item = dataset[1]
    assert item["input_ids"].dtype == torch.long
    assert item["input_ids"].tolist() == sample(1)["input_ids"]
    assert item["labels"].tolist() == sample(1)["labels"]


def test_dataset_ignores_blank_lines(tmp_path):
    path = write_jsonl(tmp_path / "t.jsonl", ["", sample(), ""])

    assert len(module.TokenizedDataset(path)) == 1


def test_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.TokenizedDataset(tmp_path / "absent.jsonl")


@pytest.mark.parametrize(
    "bad_line",
    [
        "{broken",
        {"input_ids": "x", "labels": [1] * SEQUENCE},
        {"input_ids": [1] * SEQUENCE, "labels": None},
        {"input_ids": [1] * (SEQUENCE - 1), "labels": [1] * SEQUENCE},
        {"input_ids": [1] * SEQUENCE, "labels": [1] * (SEQUENCE + 1)},
        {"labels": [1] * SEQUENCE},
    ],
)
def test_dataset_rejects_invalid_lines(tmp_path, bad_line):
    path = write_jsonl(tmp_path / "t.jsonl", [sample(), bad_line])

    with pytest.raises(ValueError, match="ligne 2|Ligne 2|à la ligne 2"):
        module.TokenizedDataset(path)


def test_train_dataloader_on_cpu_uses_no_workers_and_batches(
    tmp_path, monkeypatch
):
    path = write_jsonl(tmp_path / "t.jsonl", [sample(i) for i in range(5)])
    monkeypatch.setattr(module, "TRAIN_DATASET_FILE", path)
    monkeypatch.setattr(module, "BATCH_SIZE", 2)

    dataloader = module.create_train_dataloader()

    assert dataloader.num_workers == 0
    assert not dataloader.pin_memory
    batches = list(dataloader)
    assert [b["input_ids"].shape[0] for b in batches] == [2, 2, 1]


# ------------------------------------------------------------
# perplexité / gradient
# ------------------------------------------------------------

def test_perplexity_none_exp_and_overflow():
    assert module.calculate_perplexity(None) is None
    assert module.calculate_perplexity(0.0) == 1.0
    assert module.calculate_perplexity(2.0) == pytest.approx(math.exp(2))
    assert module.calculate_perplexity(1e6) == float("inf")


def test_gradient_norm_is_global_l2_norm():
    layer = nn.Linear(2, 1, bias=False)
    layer.weight.grad = torch.tensor([[3.0, 4.0]])

    assert module.calculate_gradient_norm(layer) == pytest.approx(5.0)


def test_gradient_norm_without_gradients_is_zero():
    assert module.calculate_gradient_norm(nn.Linear(2, 2)) == 0.0


def test_gradient_norm_unwraps_data_parallel():
    layer = nn.Linear(2, 1, bias=False)
    layer.weight.grad = torch.tensor([[3.0, 4.0]])

    assert module.calculate_gradient_norm(nn.DataParallel(layer)) == \
        pytest.approx(5.0)


# ------------------------------------------------------------
# upload monitoring
# ------------------------------------------------------------

def test_upload_monitoring_returns_false_when_hub_unavailable(
    monkeypatch, capsys
):
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", None)

    assert module.upload_monitoring_to_hub() is False
    assert "Upload monitoring.json échoué" in capsys.readouterr().out


def test_upload_monitoring_uploads_both_files_when_state_exists(
    tmp_path, monkeypatch
):
    state = tmp_path / "training_state.json"
    state.write_text("{}")
    monkeypatch.setattr(module, "TRAINING_STATE_PATH", state)
    monkeypatch.setattr(module, "MONITORING_JSON_PATH", tmp_path / "m.json")
    uploads = []
    hub = types.ModuleType("hub")
    hub.upload_file_to_hub = lambda path, remote_path: uploads.append(remote_path)
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", hub)

    assert module.upload_monitoring_to_hub() is True
    assert uploads == ["monitoring.json", "training_state.json"]


def test_upload_monitoring_skips_missing_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "TRAINING_STATE_PATH", tmp_path / "absent.json")
    uploads = []
    hub = types.ModuleType("hub")
    hub.upload_file_to_hub = lambda path, remote_path: uploads.append(remote_path)
    monkeypatch.setitem(sys.modules, "scripts.hub.hub_sync", hub)

    module.upload_monitoring_to_hub()

    assert uploads == ["monitoring.json"]


# ------------------------------------------------------------
# paramètres entraînables
# ------------------------------------------------------------

def test_trainable_parameters_summary():
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    model[1].weight.requires_grad_(False)

    info = module.print_trainable_parameters(model, "TEST")

    assert info["trainable_parameters"] == (3 * 2 + 2) + 1
    assert info["frozen_parameters"] == 2
    assert info["total_parameters"] == 11
    assert info["trainable_tensors"] == 3 and info["frozen_tensors"] == 1
    assert info["trainable_ratio"] == f"{9 / 11 * 100:.4f} %"
    assert [m["name"] for m in info["modules"]] == ["0", "1"]


def test_trainable_parameters_unwraps_data_parallel():
    info = module.print_trainable_parameters(nn.DataParallel(nn.Linear(2, 2)))

    assert info["total_parameters"] == 6


# ------------------------------------------------------------
# validate
# ------------------------------------------------------------

def test_validate_returns_token_weighted_mean_loss_in_eval_mode():
    transformer, lm_head, _ = build_model()

    loss = module.validate(transformer, lm_head, loader(), nn.CrossEntropyLoss())

    assert 0 < loss < 10
    assert not transformer.training and not lm_head.training


def test_validate_does_not_change_weights():
    transformer, lm_head, _ = build_model()
    before = [p.detach().clone() for p in transformer.parameters()]

    module.validate(transformer, lm_head, loader(), nn.CrossEntropyLoss())

    assert all(
        torch.equal(a, b) for a, b in zip(before, transformer.parameters())
    )


def test_validate_empty_dataloader_returns_infinity():
    transformer, lm_head, _ = build_model()

    assert module.validate(transformer, lm_head, [], nn.CrossEntropyLoss()) \
        == float("inf")


# ------------------------------------------------------------
# train_one_epoch
# ------------------------------------------------------------

def train(**kwargs):
    transformer, lm_head, optimizer = build_model()
    arguments = dict(
        transformer=transformer,
        lm_head=lm_head,
        dataloader=loader(),
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        monitoring_upload_interval=0,
    )
    arguments.update(kwargs)
    return transformer, module.train_one_epoch(**arguments)


def test_train_epoch_returns_loss_gradient_norm_and_step():
    _, (loss, gradient_norm, step) = train(global_step=10)

    assert loss > 0
    assert gradient_norm > 0
    assert step == 12


def test_train_epoch_updates_weights_and_sets_train_mode():
    transformer, lm_head, optimizer = build_model()
    before = next(transformer.parameters()).detach().clone()

    module.train_one_epoch(
        transformer, lm_head, loader(), optimizer, nn.CrossEntropyLoss(),
        monitoring_upload_interval=0,
    )

    assert not torch.equal(before, next(transformer.parameters()))
    assert transformer.training and lm_head.training


def test_train_epoch_empty_dataloader_returns_infinity():
    _, (loss, gradient_norm, step) = train(dataloader=[])

    assert loss == float("inf")
    assert gradient_norm == 0.0
    assert step == 0


def test_train_epoch_fills_live_metrics_and_timer_values():
    timer = types.SimpleNamespace(
        elapsed=lambda: 12.0, format_duration=lambda s: f"{s}s"
    )
    live = {}

    train(live_metrics=live, epoch=3, training_timer=timer)

    assert live["step"] == 2 and live["epoch"] == 3
    assert live["train_loss"] > 0 and live["gradient_norm"] > 0
    assert live["duration_seconds"] == 12.0
    assert live["duration"] == "12.0s"


def test_train_epoch_calls_heartbeat_every_step():
    beats = []
    heartbeat = types.SimpleNamespace(beat=lambda **kw: beats.append(kw["step"]))

    train(heartbeat=heartbeat)

    assert beats == [1, 2]


def test_train_epoch_passes_live_metrics_to_step_saver():
    saved = []
    saver = types.SimpleNamespace(save=lambda **kw: saved.append(kw))
    live = {"validation_loss": 0.5}

    train(
        step_saver=saver,
        save_checkpoint_fn=lambda **kw: None,
        save_checkpoint_kwargs={"vocab_size": VOCAB},
        live_metrics=live,
        epoch=2,
    )

    assert [s["step"] for s in saved] == [1, 2]
    assert saved[0]["vocab_size"] == VOCAB
    assert saved[0]["epoch"] == 2
    assert saved[0]["validation_loss"] == 0.5


def test_train_epoch_without_save_function_never_saves():
    saved = []
    saver = types.SimpleNamespace(save=lambda **kw: saved.append(kw))

    train(step_saver=saver, save_checkpoint_fn=None)

    assert saved == []


def test_train_epoch_uploads_monitoring_at_interval(monkeypatch):
    uploads = []
    monkeypatch.setattr(
        module, "upload_monitoring_to_hub", lambda: uploads.append(1)
    )

    train(monitoring_upload_interval=2)

    assert uploads == [1]


def test_train_epoch_survives_monitoring_failure(monkeypatch, capsys):
    def boom(**kwargs):
        raise RuntimeError("monitoring down")

    monkeypatch.setattr(module, "write_monitoring_json", boom)

    _, (_, _, step) = train()

    assert step == 2
    assert "Monitoring échoué" in capsys.readouterr().out


# ------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ------------------------------------------------------------

def save(path, transformer, lm_head, optimizer, **overrides):
    arguments = dict(
        checkpoint_path=path,
        transformer=transformer,
        lm_head=lm_head,
        optimizer=optimizer,
        epoch=2,
        train_loss=1.5,
        validation_loss=2.0,
        gradient_norm=0.3,
        vocab_size=VOCAB,
    )
    arguments.update(overrides)
    module.save_checkpoint(**arguments)


def test_save_checkpoint_stores_state_metrics_and_config(tmp_path):
    transformer, lm_head, optimizer = build_model()
    path = tmp_path / "nested" / "ckpt.pt"

    save(path, transformer, lm_head, optimizer, batch_size=4)

    checkpoint = torch.load(path)
    assert checkpoint["epoch"] == 2
    assert checkpoint["train_loss"] == 1.5
    assert checkpoint["train_ppl"] == pytest.approx(math.exp(1.5))
    assert checkpoint["validation_ppl"] == pytest.approx(math.exp(2.0))
    assert checkpoint["gpu_metrics"] == {}
    assert checkpoint["config"]["vocab_size"] == VOCAB
    assert checkpoint["config"]["d_model"] == D_MODEL
    assert checkpoint["config"]["max_sequence_length"] == SEQUENCE
    assert checkpoint["config"]["batch_size"] == 4


def test_save_checkpoint_replaces_none_metrics_by_zero(tmp_path):
    transformer, lm_head, optimizer = build_model()
    path = tmp_path / "ckpt.pt"

    save(path, transformer, lm_head, optimizer,
         train_loss=None, validation_loss=None, gradient_norm=None)

    checkpoint = torch.load(path)
    assert checkpoint["train_loss"] == 0.0
    assert checkpoint["gradient_norm"] == 0.0
    assert checkpoint["train_ppl"] == 1.0


def test_save_checkpoint_keeps_given_perplexities(tmp_path):
    transformer, lm_head, optimizer = build_model()
    path = tmp_path / "ckpt.pt"

    save(path, transformer, lm_head, optimizer, train_ppl=7.0, validation_ppl=8.0)

    checkpoint = torch.load(path)
    assert (checkpoint["train_ppl"], checkpoint["validation_ppl"]) == (7.0, 8.0)


def test_save_checkpoint_removes_data_parallel_prefix(tmp_path):
    transformer, lm_head, optimizer = build_model()
    path = tmp_path / "ckpt.pt"

    save(path, nn.DataParallel(transformer), lm_head, optimizer)

    keys = torch.load(path)["transformer_state_dict"]
    assert not any(k.startswith("module.") for k in keys)


def test_load_checkpoint_restores_weights_and_returns_epoch(tmp_path):
    transformer, lm_head, optimizer = build_model()
    path = tmp_path / "ckpt.pt"
    save(path, transformer, lm_head, optimizer, epoch=5)

    new_transformer, new_head, new_optimizer = build_model()
    epoch = module.load_checkpoint(
        path, new_transformer, new_head, new_optimizer, VOCAB
    )

    assert epoch == 5
    assert all(
        torch.equal(a, b)
        for a, b in zip(transformer.parameters(), new_transformer.parameters())
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(lm_head.parameters(), new_head.parameters())
    )


def test_load_checkpoint_into_data_parallel_model(tmp_path):
    transformer, lm_head, optimizer = build_model()
    path = tmp_path / "ckpt.pt"
    save(path, transformer, lm_head, optimizer)

    new_transformer, new_head, new_optimizer = build_model()
    wrapped = nn.DataParallel(new_transformer)

    module.load_checkpoint(path, wrapped, new_head, new_optimizer, VOCAB)

    assert torch.equal(
        next(transformer.parameters()), next(new_transformer.parameters())
    )


def test_load_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.load_checkpoint(tmp_path / "x.pt", *build_model(), VOCAB)


def test_load_checkpoint_without_config_raises(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"epoch": 1}, path)

    with pytest.raises(ValueError, match="configuration"):
        module.load_checkpoint(path, *build_model(), VOCAB)


def test_load_checkpoint_rejects_different_vocabulary(tmp_path):
    transformer, lm_head, optimizer = build_model()
    path = tmp_path / "ckpt.pt"
    save(path, transformer, lm_head, optimizer)

    with pytest.raises(ValueError, match="vocabulaire"):
        module.load_checkpoint(path, transformer, lm_head, optimizer, VOCAB + 1)


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("D_MODEL", "D_MODEL"),
        ("NUM_HEADS", "NUM_HEADS"),
        ("HIDDEN_DIM", "HIDDEN_DIM"),
        ("NUM_BLOCKS", "NUM_BLOCKS"),
        ("MAX_SEQUENCE_LENGTH", "MAX_SEQUENCE_LENGTH"),
    ],
)
def test_load_checkpoint_rejects_different_architecture(
    tmp_path, monkeypatch, attribute, message
):
    transformer, lm_head, optimizer = build_model()
    path = tmp_path / "ckpt.pt"
    save(path, transformer, lm_head, optimizer)
    monkeypatch.setattr(module, attribute, getattr(module, attribute) + 1)

    with pytest.raises(ValueError, match=message):
        module.load_checkpoint(path, transformer, lm_head, optimizer, VOCAB)
